#!/bin/bash
set -euo pipefail
# fix_answer_upweight_patch.sh
#
# PROBLEM: WiC finetune loss flatlines at ~step 50/1500. The ddm-sft loss averages
# cross-entropy equally over all masked tokens. The answer target is 4 tokens
# (### / Yes|No / . / <eos>); once the 3 format tokens are trivially learned, the
# ONE token carrying the label (Yes vs No) is 1/4 of the loss and its gradient is
# swamped. Result: model nails the format, never learns discrimination, defaults
# to "Yes" (balanced acc 0.569, diff-sense acc 0.162).
#
# FIX: multiply the loss on the ANSWER token (the token right after "###", id 21017)
# by a factor (default 5x), so training keeps optimizing the label bit after the
# format is learned. Opt-in via env ANSWER_LOSS_WEIGHT (default 1.0 = unchanged),
# so ProsQA runs are unaffected unless the flag is set.
#
# Run from the DiffuLLaMA repo root:  bash fix_answer_upweight_patch.sh

TRAINER="LLaMA-Factory/src/llamafactory/train/ddm/trainer.py"

if [ ! -f "${TRAINER}" ]; then
  echo "ERROR: ${TRAINER} not found. Run from the DiffuLLaMA repo root." >&2
  exit 1
fi

if grep -q "ANSWER_LOSS_WEIGHT" "${TRAINER}"; then
  echo "Already patched (ANSWER_LOSS_WEIGHT present). Nothing to do."
  exit 0
fi

cp "${TRAINER}" "${TRAINER}.bak"
echo "backed up -> ${TRAINER}.bak"

python3 - "$TRAINER" <<'PY'
import sys, re
path = sys.argv[1]
src = open(path).read()

# Inject answer-token upweighting right AFTER the loss_mask fill and BEFORE the
# final_loss reduction, in inner_forward. Anchor on the exact masked_fill line.
anchor = '''        loss = loss.masked_fill(~loss_mask, 0)
        
        
        final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()   # avg token loss
        unweighted_loss = (loss).sum() / loss_mask.sum()'''

inject = '''        loss = loss.masked_fill(~loss_mask, 0)

        # --- answer-token upweighting (WiC) ---------------------------------
        # The label token (Yes/No) is the token right after "###" (id 21017).
        # Once the format tokens are learned the loss flatlines and the label
        # gradient is swamped; upweight it so training keeps optimizing it.
        # Opt-in: ANSWER_LOSS_WEIGHT=1.0 (default) leaves behaviour unchanged.
        _alw = float(os.environ.get("ANSWER_LOSS_WEIGHT", "1.0"))
        if _alw != 1.0:
            _hash_id = 21017  # "###"
            # x here is already shifted if self.diff_args.shift (x = x[:,1:]),
            # so x aligns with logits/loss positions. Answer token = position
            # immediately following a "###" token in the (shifted) targets.
            _is_hash = (x == _hash_id)
            _ans_pos = torch.zeros_like(loss, dtype=torch.bool)
            _ans_pos[:, 1:] = _is_hash[:, :-1]   # one position after "###"
            _ans_pos = _ans_pos & loss_mask       # only where loss is active
            _w = torch.ones_like(loss)
            _w = _w.masked_fill(_ans_pos, _alw)
            loss = loss * _w
        # --------------------------------------------------------------------

        final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()   # avg token loss
        unweighted_loss = (loss).sum() / loss_mask.sum()'''

assert anchor in src, "anchor block not found; trainer.py layout changed — patch manually."
src = src.replace(anchor, inject, 1)

# ensure os is imported (it is used already via os.environ elsewhere, but be safe)
if "\nimport os" not in src and "^import os" not in src:
    src = "import os\n" + src

open(path, "w").write(src)
print("patched", path)
PY

echo "=== patched region ==="
grep -n "ANSWER_LOSS_WEIGHT\|answer-token upweighting\|_ans_pos" "${TRAINER}"
echo "======================"
echo "Done. To use: set ANSWER_LOSS_WEIGHT=5 (or similar) in train_wic.sh before"
echo "the training launch. Leave unset / =1.0 for ProsQA (unchanged behaviour)."