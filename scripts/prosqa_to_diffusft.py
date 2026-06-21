#!/usr/bin/env python3
# scripts/prosqa_to_diffusft.py
"""
Convert COCONUT-format ProsQA -> DiffuGPT ddm-sft (LLaMA-Factory) data.

INPUT  (data/prosqa_{train,valid,test}.json), one JSON list of:
    {
      "question": str,           # DAG statements + the query
      "answer":   str,           # final concept (e.g. "Tom is a zhorpus.")
      "steps":    [str, ...],    # NL reasoning chain (the path)
      "edges","root","target","neg_target","idx_to_symbol": ...  # DAG (kept for DLIG later)
    }

OUTPUT (alpaca-style JSONL that LLaMA-Factory reads via dataset_info.json):
    {"instruction": <question>, "input": "", "output": <CoT trace> + " ### " + <answer>}

The CoT target is the explicit reasoning chain followed by the answer, so the
diffusion model learns to GENERATE the reasoning tokens (which DLIG then
attributes back to the DAG premises). The "### " answer-prefix matches the
eval parser convention.

Also prints the GPT-2 token-length distribution so cutoff_len can be set from
the REAL tokenizer (not an estimate). Run on the cluster where gpt2 vocab is
available.
"""

import os
import json
import argparse

ANSWER_PREFIX = "###"   # keep aligned with eval_prosqa.py parsing


def build_output(ex):
    steps = " ".join(s.strip() for s in ex["steps"])
    return f"{steps} {ANSWER_PREFIX} {ex['answer'].strip()}"


def convert(in_path, out_path):
    data = json.load(open(in_path))
    n = 0
    with open(out_path, "w") as f:
        for ex in data:
            rec = {
                "instruction": ex["question"].strip(),
                "input": "",
                "output": build_output(ex),
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n


def length_report(in_path):
    """Exact GPT-2 token lengths (real tokenizer). Prints percentiles + fit rates."""
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception as e:
        print(f"[WARN] could not load gpt2 tokenizer ({e}); skipping length report.")
        return
    import numpy as np
    data = json.load(open(in_path))
    full = []
    for ex in data:
        instr = tok.encode(ex["question"].strip())
        out = tok.encode(" " + build_output(ex))   # leading space, BPE-consistent
        full.append(len(instr) + len(out))
    full = np.array(full)
    p = lambda x: int(np.percentile(full, x))
    print(f"[LEN] {os.path.basename(in_path)}: n={len(full)} "
          f"mean={full.mean():.0f} p50={p(50)} p95={p(95)} p99={p(99)} max={full.max()}")
    for cut in (256, 384, 512, 640, 768):
        print(f"      fit<= {cut}: {(full <= cut).mean()*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data", help="dir with prosqa_*.json")
    ap.add_argument("--out_dir", default="LLaMA-Factory/data",
                    help="where LLaMA-Factory reads datasets")
    ap.add_argument("--report", action="store_true",
                    help="print real GPT-2 token-length distribution (needs gpt2 vocab).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mapping = {
        "prosqa_train.json": "prosqa_train.jsonl",
        "prosqa_valid.json": "prosqa_valid.jsonl",
        "prosqa_test.json":  "prosqa_test.jsonl",
    }
    for src, dst in mapping.items():
        ip = os.path.join(args.in_dir, src)
        if not os.path.exists(ip):
            print(f"[SKIP] missing {ip}")
            continue
        op = os.path.join(args.out_dir, dst)
        n = convert(ip, op)
        print(f"[OK] {src} -> {op}  ({n} examples)")
        if args.report:
            length_report(ip)


if __name__ == "__main__":
    main()