#!/usr/bin/env python3
# scripts/wic_to_diffusft.py
"""
Convert Word-in-Context (WiC) -> DiffuGPT ddm-sft (LLaMA-Factory) data.

Mirrors scripts/prosqa_to_diffusft.py exactly in output shape and conventions,
so the same dataset_info.json registration, template: empty, shift: true, and
the "###" answer-prefix eval parser all work unchanged.

INPUT: super_glue/wic from HF (or a local jsonl via --wic_jsonl), rows with
    {sentence1, sentence2, word, label}   label: 1 = same sense, 0 = different

OUTPUT (alpaca-style JSONL that LLaMA-Factory reads via dataset_info.json):
    {"instruction": <prompt>, "input": "", "output": <phrase> " ### " <Answer>.}

Unlike ProsQA there is NO reasoning chain, so the target is a short fixed
carrier phrase terminated by the answer word. Two reasons for the carrier
phrase rather than a bare "Yes":
  (1) diffusion SFT learns better from a few target tokens than from one;
  (2) DLIG later needs a committed answer span to score, and the "### <Answer>."
      layout gives a clean, fixed answer position (matches eval_prosqa parsing:
      split on "###", take first sentence up to the period).

The pivot word is echoed into the carrier phrase so the answer token is NOT
lexically adjacent to the disambiguating context — keeps the attribution target
disjoint from context vocab (the whole point of choosing WiC over free-form
sense definitions).

ANSWER SET: {"Yes", "No"} mapped from label {1, 0}. The forced-choice readout in
eval_wic.py aggregates yes/same/true vs no/different/false, so "Yes"/"No" is the
canonical surface here.

Prints the GPT-2 token-length distribution so cutoff_len can be set from the
REAL tokenizer. WiC is short (2 sentences), so cutoff is small — expect p99 well
under 128.
"""

import os
import json
import argparse

ANSWER_PREFIX = "###"           # keep aligned with eval_prosqa.py / eval_wic.py
POS_ANSWER = "Yes"              # label 1 = same sense
NEG_ANSWER = "No"               # label 0 = different sense
EOS_TOKEN = "<|endoftext|>"     # GPT-2 EOS; teaches the model to stop after the answer


def build_prompt(ex):
    """
    Instruction = the WiC question. Fixed template, empty system (template: empty),
    so the prompt tokens are a contiguous prefix the baseline-masking / attribution
    can locate — same contract as ProsQA's raw question format.
    """
    return (
        f'Sentence 1: {ex["sentence1"].strip()}\n'
        f'Sentence 2: {ex["sentence2"].strip()}\n'
        f'Does the word "{ex["word"].strip()}" have the same meaning in both '
        f'sentences?'
    )


def build_output(ex):
    """
    Answer only, EOS-terminated: "### Yes.<|endoftext|>" / "### No.<|endoftext|>".
    The explicit EOS teaches the model to halt after the answer instead of
    rambling into the remaining generation budget.
    """
    ans = POS_ANSWER if int(ex["label"]) == 1 else NEG_ANSWER
    return f'{ANSWER_PREFIX} {ans}.{EOS_TOKEN}'


def load_wic_split(split, wic_jsonl=None):
    if wic_jsonl:
        rows = []
        with open(wic_jsonl) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.append({
                        "sentence1": r["sentence1"],
                        "sentence2": r["sentence2"],
                        "word": r.get("word") or r.get("lemma") or r.get("target"),
                        "label": int(r["label"]),
                    })
        return rows
    from datasets import load_dataset
    ds = load_dataset("super_glue", "wic", split=split)
    return [{"sentence1": r["sentence1"], "sentence2": r["sentence2"],
             "word": r["word"], "label": int(r["label"])} for r in ds]


def convert(rows, out_path):
    n = 0
    with open(out_path, "w") as f:
        for ex in rows:
            rec = {
                "instruction": build_prompt(ex),
                "input": "",
                "output": build_output(ex),
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n


def convert_raw(rows, out_path):
    """
    Dump the RAW WiC fields {sentence1, sentence2, word, label} — the format
    eval_wic.py's load_wic() expects. The SFT `convert()` above bakes the label
    into text (### Yes/No) under the training template, which is the wrong input
    for the forced-choice eval; this keeps the held-out test set readable by the
    gate with its own Answer: template and gold labels intact.
    """
    n = 0
    with open(out_path, "w") as f:
        for ex in rows:
            rec = {
                "sentence1": ex["sentence1"],
                "sentence2": ex["sentence2"],
                "word": ex["word"],
                "label": int(ex["label"]),
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n


def length_report(rows, name):
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception as e:
        print(f"[WARN] could not load gpt2 tokenizer ({e}); skipping length report.")
        return
    import numpy as np
    full = []
    for ex in rows:
        instr = tok.encode(build_prompt(ex))
        out = tok.encode(" " + build_output(ex))
        full.append(len(instr) + len(out))
    full = np.array(full)
    p = lambda x: int(np.percentile(full, x))
    print(f"[LEN] {name}: n={len(full)} mean={full.mean():.0f} "
          f"p50={p(50)} p95={p(95)} p99={p(99)} max={full.max()}")
    for cut in (64, 96, 128, 192, 256):
        print(f"      fit<= {cut}: {(full <= cut).mean()*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="LLaMA-Factory/data",
                    help="where LLaMA-Factory reads datasets")
    ap.add_argument("--wic_jsonl_train", default=None,
                    help="optional local WiC train jsonl; else HF super_glue/wic train")
    ap.add_argument("--wic_jsonl_valid", default=None,
                    help="optional local WiC valid jsonl; else HF super_glue/wic validation")
    ap.add_argument("--holdout_n", type=int, default=500,
                    help="carve this many examples out of the WiC train split as a "
                         "held-out TEST set (wic_test.jsonl). SuperGLUE WiC test labels "
                         "are hidden, so the test set must come from train. 0 disables.")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the deterministic, stratified train/test carve-out")
    ap.add_argument("--report", action="store_true",
                    help="print real GPT-2 token-length distribution")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- TRAIN split: load, then carve a held-out TEST set out of it. ---
    # SuperGLUE WiC test labels are hidden, so we cannot use the official test
    # split. Instead: train_full -> (train, test). The official validation split
    # (638) stays untouched as the SFT eval set. Result: three disjoint sets,
    #   wic_train  = train_full minus holdout   (fit on this)
    #   wic_valid  = official validation         (SFT eval / model selection)
    #   wic_test   = holdout from train_full     (final inference number, never tuned on)
    train_full = load_wic_split("train", wic_jsonl=args.wic_jsonl_train)

    if args.holdout_n > 0:
        import random
        # Stratified by label so the carved test set keeps WiC's ~50/50 balance.
        pos = [r for r in train_full if int(r["label"]) == 1]
        neg = [r for r in train_full if int(r["label"]) == 0]
        rng = random.Random(args.seed)
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_pos = args.holdout_n // 2
        n_neg = args.holdout_n - n_pos
        if n_pos > len(pos) or n_neg > len(neg):
            raise SystemExit(f"[ERROR] holdout_n={args.holdout_n} too large for "
                             f"train (pos={len(pos)}, neg={len(neg)}).")
        test_rows = pos[:n_pos] + neg[:n_neg]
        train_rows = pos[n_pos:] + neg[n_neg:]
        rng.shuffle(test_rows)
        rng.shuffle(train_rows)
    else:
        test_rows = []
        train_rows = train_full

    valid_rows = load_wic_split("validation", wic_jsonl=args.wic_jsonl_valid)

    outputs = [
        ("train", "wic_train.jsonl", train_rows),
        ("validation", "wic_valid.jsonl", valid_rows),
    ]
    if test_rows:
        outputs.append(("test(holdout from train)", "wic_test.jsonl", test_rows))

    for split, dst, rows in outputs:
        op = os.path.join(args.out_dir, dst)
        n = convert(rows, op)
        pos_rate = sum(int(r["label"]) for r in rows) / max(len(rows), 1)
        print(f"[OK] WiC {split} -> {op}  ({n} examples, pos_rate={pos_rate:.3f})")
        if args.report:
            length_report(rows, dst)

    if test_rows:
        # raw-fields dump of the holdout, for eval_wic.py (--wic_jsonl)
        raw_op = os.path.join(args.out_dir, "wic_test_raw.jsonl")
        convert_raw(test_rows, raw_op)
        print(f"[OK] WiC test (raw fields for eval) -> {raw_op}  ({len(test_rows)} examples)")

        print(f"\n[SPLIT] train={len(train_rows)}  valid={len(valid_rows)}  "
              f"test={len(test_rows)}  (train+test = {len(train_full)} original train)")
        print("[SPLIT] eval the gate on the holdout with:")
        print(f"        python -m experiments.eval_wic --model_path <ckpt> "
              f"--wic_jsonl {raw_op}")


if __name__ == "__main__":
    main()