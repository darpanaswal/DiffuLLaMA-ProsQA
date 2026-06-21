#!/usr/bin/env python3
# eval_prosqa.py   (place at the ROOT of the cloned DiffuLLaMA repo)
"""
Evaluate a ddm-sft finetuned DiffuGPT-M on ProsQA.

STANDALONE: depends ONLY on the DiffuLLaMA repo's own inference (model.py:
DiscreteDiffusionModel + generate_samples) and HF transformers. No DLIG / DLIG
backend imports. Run from the repo root, same place as inf_diffugpt.py.

Pipeline per item:
  1. prompt = ProsQA question (BOS + encoded, like inf_diffugpt.py prefix gen).
  2. build  input_ids = prefix + [mask]*gen_len  and  src_mask = 1 on prefix, 0 on gen.
  3. generate_samples() denoises the masked suffix (diffusion_steps, shift=True).
  4. decode the generated suffix, parse the answer after '###', exact-match gold.

ProsQA answers are binary ("X or Y"), chance ~50%, CoT-GPT2 reference 77.5%.
"""

import re
import os
import sys
import json
import argparse
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoTokenizer

# eval_prosqa.py lives in scripts/ but model.py is at the repo ROOT.
# When run as `python scripts/eval_prosqa.py`, Python puts scripts/ on sys.path,
# NOT the root -> add the parent (repo root) so `from model import ...` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
# also expose the LLaMA-Factory package so we can import the SAME tokenizer
# wrapper the trainer used (MaskTokenWrapper). The checkpoint's tokenizer_config
# records tokenizer_class=MaskTokenWrapper, which AutoTokenizer can't resolve.
sys.path.insert(0, os.path.join(_REPO_ROOT, "LLaMA-Factory", "src"))

# from the DiffuLLaMA repo root (same import as inf_diffugpt.py)
from model import DiscreteDiffusionModel, generate_samples
# the trainer wraps gpt2-medium's tokenizer in this class (model/loader.py),
# which sets mask_token_id = vocab_size (50257) and the ====== sep token.
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from tokenizers.pre_tokenizers import Digits
from itertools import chain
# the trainer wraps gpt2-medium's tokenizer in this class (model/loader.py),
# which sets mask_token_id = vocab_size (50257) and the ====== sep token.


ANSWER_PREFIX = "###"


class MaskTokenWrapper(GPT2TokenizerFast):
    def __init__(self, tokenizer):
        EOS_TOKEN = tokenizer.eos_token
        PAD_TOKEN = "¨"
        SEP_TOKEN = "======"
        MASK_TOKEN_ID = tokenizer.vocab_size
        self.tokenizer = tokenizer
        self.digit_tokenizer = Digits(individual_digits=True)
        self.token2id = {token:id for token, id in self.tokenizer.vocab.items()}
        self.tokenizer.add_special_tokens({
            'pad_token': PAD_TOKEN, 
            'eos_token': EOS_TOKEN, 
            'sep_token': SEP_TOKEN,
            'mask_token': "[¨M¨]"
        })
        self.__dict__.update(self.tokenizer.__dict__.items())
        self.eos_token_id = self.token2id[EOS_TOKEN]
        self.pad_token_id = self.token2id[PAD_TOKEN]
        self.sep_token_id = self.token2id[SEP_TOKEN]
        self.mask_token_id = MASK_TOKEN_ID

    def encode(self, text, digit=True, **kwargs):
        if digit:
            chunks = self.digit_tokenizer.pre_tokenize_str(text)
            res = self.encode_batch([i[0] for i in chunks], digit=False, **kwargs)
            return res
        return self.tokenizer(text)


    def encode_batch(self, texts, digit=True, **kwargs):
        if digit:
            return [self.encode(text, digit=True, **kwargs) for text in texts]
        return list(chain.from_iterable([self.tokenizer.encode(text, **kwargs) for text in texts]))


def normalize(s):
    s = s.strip().lower().rstrip(".")
    return re.sub(r"\s+", " ", s)


def parse_answer(text):
    # Targets look like "<CoT> ### <Subject> is a <concept>." With gen_len padding,
    # the model often runs on AFTER the real answer (repeats junk). Take the FIRST
    # statement right after the FIRST "###", up to the first period, not the last.
    if ANSWER_PREFIX in text:
        after = text.split(ANSWER_PREFIX, 1)[1].strip()
        # first sentence only (cut at the first period)
        first = after.split(".", 1)[0].strip()
        return first if first else after
    sents = [s for s in re.split(r"(?<=\.)\s+", text.strip()) if s.strip()]
    return sents[-1] if sents else text.strip()


def final_concept(s):
    w = re.findall(r"[a-zA-Z]+", s)
    return w[-1].lower() if w else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True,
                    help="path to the ddm-sft finetuned DiffuGPT-M checkpoint dir "
                         "(LLaMA-Factory output_dir), or a HF model id.")
    ap.add_argument("--base_model_name", default="gpt2-medium",
                    help="AR base for CONFIG ONLY (DiffuGPT-M = gpt2-medium).")
    ap.add_argument("--data", default="prosqa_test.json",
                    help="COCONUT-format ProsQA json (question/answer/steps).")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32,
                    help="eval batch size. generate_samples is fully batched; "
                         "32 on a 48GB card is safe for gpt2-medium at this seq len. "
                         "main speedup lever — sequential (bs=1) is very slow.")
    ap.add_argument("--gen_len", type=int, default=100,
                    help="length of the masked generation region (CoT+answer). "
                         "ProsQA CoT+answer is short; can drop to ~64 to save compute.")
    ap.add_argument("--diffusion_steps", type=int, default=64,
                    help="denoising steps. 64 = train default; 32 is usually fine "
                         "for eval and ~2x faster. lower if still slow.")
    ap.add_argument("--logits_temp", type=float, default=0.95)
    ap.add_argument("--topp_temp", type=float, default=0.9)
    ap.add_argument("--shift", type=bool, default=True)   # do not change (DiffuGPT)
    ap.add_argument("--out_file", default="prosqa_eval_preds.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # fail loudly with a clear message if the finetuned checkpoint dir is absent
    # (otherwise HF treats the path as a hub repo id and throws a confusing
    # HFValidationError about slashes).
    if not os.path.isdir(args.model_name):
        raise FileNotFoundError(
            f"checkpoint dir not found: {args.model_name}\n"
            f"  -> training likely did not save. Check the training log "
            f"(runs/.../train_*.log) and the yaml's output_dir.")

    config = AutoConfig.from_pretrained(args.model_name)
    # Build the tokenizer the SAME way training did: wrap the base gpt2-medium
    # GPT2TokenizerFast in MaskTokenWrapper. We can't AutoTokenizer.from_pretrained
    # the checkpoint dir because its tokenizer_config.json records
    # tokenizer_class="MaskTokenWrapper", which AutoTokenizer cannot resolve
    # (ValueError: Tokenizer class MaskTokenWrapper does not exist). The wrapper
    # is deterministic from the base vocab, so loading from base is equivalent:
    # mask_token_id = vocab_size = 50257 (matches the 50257->50258 train resize).
    base_tok = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
    tokenizer = MaskTokenWrapper(base_tok)
    assert tokenizer.mask_token_id == 50257, (
        f"unexpected mask_token_id={tokenizer.mask_token_id} (expected 50257)")
    # sep_token ("======", id 50155 in train) separates question from CoT; the eval
    # prefix appends it to match training. Guard against a tokenizer mismatch.
    assert tokenizer.sep_token_id == 50155, (
        f"unexpected sep_token_id={tokenizer.sep_token_id} (expected 50155 for ======)")
    # Build the model the way from_pretrained would (construct from base config,
    # then load weights), but load weights MANUALLY from pytorch_model.bin.
    # DiscreteDiffusionModel uses PyTorchModelHubMixin, whose _from_pretrained
    # hardcodes loading model.safetensors. Training was run with
    # save_safetensors: false, so only pytorch_model.bin exists -> from_pretrained
    # raises FileNotFoundError on model.safetensors. We replicate its two steps:
    #   1. instantiate with the base model (same as from_pretrained's __init__ call)
    #   2. load_state_dict from the .bin checkpoint
    model = DiscreteDiffusionModel(
        model=args.base_model_name,   # base gpt2-medium; weights overwritten below
        config=config,
        tokenizer=tokenizer,
        device="cuda",
    )
    # locate the weights file (prefer .bin since save_safetensors=False; fall back
    # to safetensors if a future run saves that instead).
    bin_path = os.path.join(args.model_name, "pytorch_model.bin")
    sft_path = os.path.join(args.model_name, "model.safetensors")
    if os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    elif os.path.isfile(sft_path):
        from safetensors.torch import load_file as _load_sft
        state_dict = _load_sft(sft_path, device="cpu")
    else:
        raise FileNotFoundError(
            f"no weights in {args.model_name} (looked for pytorch_model.bin / "
            f"model.safetensors). Check the training output_dir.")
    # the saved keys (embed_tokens.*, denoise_model.h.*, denoise_model.wpe.*,
    # lm_head.*) match this module's attributes exactly -> direct load.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # GPT2 ties lm_head.weight <-> wte(embed_tokens).weight, so the checkpoint may
    # store the embedding ONCE and omit lm_head.weight. If so, re-tie after load so
    # logits use the trained embedding rather than a stale/re-init head.
    if any(k.startswith("lm_head") for k in missing) and \
       not any(k.startswith("lm_head") for k in state_dict):
        model.lm_head.weight = model.embed_tokens.weight
        missing = [k for k in missing if not k.startswith("lm_head")]
        print("[info] re-tied lm_head.weight <- embed_tokens.weight (GPT2 tied weights)")
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    model = model.to("cuda")
    model.eval()

    # generate_samples reads sampling params off a namespace (like inf_diffugpt's args)
    gen_args = SimpleNamespace(
        logits_temp=args.logits_temp,
        topp_temp=args.topp_temp,
        diffusion_steps=args.diffusion_steps,
        shift=args.shift,
    )

    data = json.load(open(args.data))
    if args.n_samples > 0:
        data = data[:args.n_samples]

    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    bos_id = tokenizer.bos_token_id
    bs = max(1, args.batch_size)

    # pre-encode all prefixes so we can left-pad each batch to its own max length.
    # batching gives a large speedup: generate_samples is fully batched ([B,L]),
    # so B examples share the diffusion_steps forward passes instead of 1-at-a-time.
    items = []
    for ex in data:
        q = ex["question"].strip()
        # TRAINING FORMAT (from the tokenized train-log example):
        #   <bos> question ====== <CoT> ### <answer> <eos>
        # The "======" separator (sep_token_id 50155) sits BETWEEN question and CoT.
        # Eval must reproduce it: without ======, the model is off-distribution and
        # never enters CoT-generation mode (this is what tanked accuracy to ~4%).
        prefix = [bos_id] + tokenizer.encode(q) + [tokenizer.sep_token_id]
        items.append({"q": q, "gold": ex["answer"].strip(), "prefix": prefix})

    n, n_exact, n_concept = 0, 0, 0
    with open(args.out_file, "w") as fout:
        for start in range(0, len(items), bs):
            batch = items[start:start + bs]
            max_pref = max(len(it["prefix"]) for it in batch)
            seq_len = max_pref + args.gen_len

            input_ids, src_masks, pad_lens = [], [], []
            for it in batch:
                p = it["prefix"]
                n_pad = max_pref - len(p)          # LEFT-pad so prefix ends flush
                # row: [pad]*n_pad + prefix + [mask]*gen_len
                row = [pad_id] * n_pad + p + [mask_id] * args.gen_len
                # src=1 on pad+prefix (fixed context, NOT denoised), 0 on gen region
                sm  = [1] * (n_pad + len(p)) + [0] * args.gen_len
                input_ids.append(row)
                src_masks.append(sm)
                pad_lens.append(n_pad)

            inputs = {
                "input_ids": torch.tensor(input_ids),
                "src_mask":  torch.tensor(src_masks),
            }
            with torch.no_grad():
                res = generate_samples(model, gen_args, tokenizer, inputs, verbose=False)

            res = res.tolist()
            for it, row, n_pad in zip(batch, res, pad_lens):
                # SHIFT off-by-one: generate_samples returns a length seq_len-1
                # sequence where returned[j] == original position j+1 (it does
                # x0 = x0[:,1:] at the end when shift=True). The gen region starts
                # at original position max_pref -> returned index max_pref-1.
                # (slicing at max_pref instead drops the first gen token, i.e. the
                # subject name "Sally"/"Bob", which is exactly what we saw.)
                cut = (max_pref - 1) if gen_args.shift else max_pref
                gen_ids = row[cut:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                pred = parse_answer(gen_text)
                gold = it["gold"]
                exact = normalize(pred) == normalize(gold)
                concept = final_concept(pred) == final_concept(gold)
                n += 1; n_exact += int(exact); n_concept += int(concept)

                fout.write(json.dumps({
                    "question": it["q"], "gold": gold,
                    "gen": gen_text, "pred_answer": pred,
                    "exact": exact, "concept_match": concept,
                }) + "\n")

            print(f"  [{n}/{len(items)}] running: exact {100*n_exact/n:.1f}%  "
                  f"concept {100*n_concept/n:.1f}%", flush=True)

    print(f"\n[RESULT] n={n}")
    print(f"  answer_exact   : {100*n_exact/n:.2f}%")
    print(f"  target_concept : {100*n_concept/n:.2f}%   (chance ~50%, CoT-GPT2 ref 77.5%)")
    print(f"  preds -> {args.out_file}")


if __name__ == "__main__":
    main()