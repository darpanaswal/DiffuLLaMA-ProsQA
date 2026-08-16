#!/bin/bash
set -euo pipefail
# fix_class_weight_patch.sh
#
# Class-conditional answer-token loss weighting to attack the Yes-bias directly.
#
# Supersedes fix_answer_upweight_patch.sh (which upweighted the answer token
# symmetrically, regardless of class). Here the answer token is upweighted MORE
# when the gold answer is "No" (id 1400) than when it is "Yes" (id 3363), so the
# model pays extra for defaulting to Yes on different-sense examples.
#
# Env controls (all opt-in; defaults leave behaviour unchanged):
#   ANSWER_LOSS_WEIGHT_YES  weight on the answer token when gold = Yes (default 1.0)
#   ANSWER_LOSS_WEIGHT_NO   weight on the answer token when gold = No  (default 1.0)
# e.g. ANSWER_LOSS_WEIGHT_YES=5 ANSWER_LOSS_WEIGHT_NO=15 penalises missed-No 3x
# harder than missed-Yes on top of the base answer-token upweight.
#
# NOTE: if fix_answer_upweight_patch.sh was already applied, REVERT it first
# (restore trainer.py.bak) so the two patches don't stack. This patch replaces it.
#
# Run from the DiffuLLaMA repo root:  bash fix_class_weight_patch.sh

TRAINER="LLaMA-Factory/src/llamafactory/train/ddm/trainer.py"

if [ ! -f "${TRAINER}" ]; then
  echo "ERROR: ${TRAINER} not found. Run from the DiffuLLaMA repo root." >&2
  exit 1
fi

if grep -q "ANSWER_LOSS_WEIGHT_NO" "${TRAINER}"; then
  echo "Already patched (ANSWER_LOSS_WEIGHT_NO present). Nothing to do."
  exit 0
fi

if grep -q "ANSWER_LOSS_WEIGHT\b" "${TRAINER}" && ! grep -q "ANSWER_LOSS_WEIGHT_NO" "${TRAINER}"; then
  echo "WARNING: the symmetric ANSWER_LOSS_WEIGHT patch appears to be applied."
  echo "         Restore ${TRAINER}.bak first, then re-run this. Aborting to avoid stacking."
  exit 1
fi

cp "${TRAINER}" "${TRAINER}.bak"
echo "backed up -> ${TRAINER}.bak"

python3 - "$TRAINER" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()

anchor = '''        loss = loss.masked_fill(~loss_mask, 0)
        
        
        final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()   # avg token loss
        unweighted_loss = (loss).sum() / loss_mask.sum()'''

inject = '''        loss = loss.masked_fill(~loss_mask, 0)

        # --- class-conditional answer-token weighting (WiC yes-bias) ---------
        # Upweight the answer token (token right after "###", id 21017) more when
        # the gold answer is "No" (1400) than "Yes" (3363), to penalise defaulting
        # to Yes on different-sense examples. Opt-in via env; defaults = 1.0.
        _w_yes = float(os.environ.get("ANSWER_LOSS_WEIGHT_YES", "1.0"))
        _w_no  = float(os.environ.get("ANSWER_LOSS_WEIGHT_NO",  "1.0"))
        if _w_yes != 1.0 or _w_no != 1.0:
            _HASH, _YES, _NO = 21017, 3363, 1400
            # answer position = one after a "###" token in the (shifted) targets x
            _is_hash = (x == _HASH)
            _ans_pos = torch.zeros_like(loss, dtype=torch.bool)
            _ans_pos[:, 1:] = _is_hash[:, :-1]
            _ans_pos = _ans_pos & loss_mask
            # what is the gold answer token sitting at those positions?
            _is_yes = _ans_pos & (x == _YES)
            _is_no  = _ans_pos & (x == _NO)
            _w = torch.ones_like(loss)
            _w = _w.masked_fill(_is_yes, _w_yes)
            _w = _w.masked_fill(_is_no,  _w_no)
            loss = loss * _w
        # --------------------------------------------------------------------

        final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()   # avg token loss
        unweighted_loss = (loss).sum() / loss_mask.sum()'''

assert anchor in src, "anchor block not found; trainer.py layout changed — patch manually."
src = src.replace(anchor, inject, 1)
open(path, "w").write(src)
print("patched", path)
PY

echo "=== patched region ==="
grep -n "ANSWER_LOSS_WEIGHT_NO\|ANSWER_LOSS_WEIGHT_YES\|_is_no\|_is_yes" "${TRAINER}"
echo "======================"
echo "Done. Example use in train_wic.sh launch line:"
echo "  ANSWER_LOSS_WEIGHT_YES=5 ANSWER_LOSS_WEIGHT_NO=15 \\"
echo "    python -c \"from llamafactory.cli import main; main()\" train \${CONFIG}"