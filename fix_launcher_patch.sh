#!/bin/bash
set -euo pipefail
# fix_launcher_patch.sh
# ROOT CAUSE of the recurring 4d-mask crash:
#   The DiffuGPT attention patch (llamafactory/attention_patch.py:replace_attention_mask)
#   is applied in cli.py at import. But with FORCE_TORCHRUN=1, cli.py re-spawns workers
#   via torchrun, and each worker runs launcher.py -> run_exp() WITHOUT importing cli.py.
#   So the patch never executes in the workers that do the forward pass, and stock
#   GPT2Model.forward runs -> the [B,1,L,L] anneal mask collides in _attn / SDPA mask-prep.
#
#   The "Use provided 4d attn-mask" line you saw is the REPO-ROOT inference patch firing
#   in the parent process only; the training-side patch (prints "logging....attention-mask
#   for 4d", which never appears in any worker log) does not run.
#
# FIX: apply the patch inside launcher.py, which every torchrun worker actually runs.
#
# Run from the DiffuLLaMA repo root:  bash fix_launcher_patch.sh

LAUNCHER="LLaMA-Factory/src/llamafactory/launcher.py"

if [ ! -f "${LAUNCHER}" ]; then
  echo "ERROR: ${LAUNCHER} not found. Run from the DiffuLLaMA repo root." >&2
  exit 1
fi

# idempotent: skip if already patched
if grep -q "attention_patch.replace_attention_mask" "${LAUNCHER}"; then
  echo "launcher.py already applies the attention patch. Nothing to do."
  exit 0
fi

cp "${LAUNCHER}" "${LAUNCHER}.bak"
echo "backed up -> ${LAUNCHER}.bak"

# Insert the patch import+call ABOVE the run_exp import, so it runs before any
# transformers GPT2 forward is bound at model-build time.
python3 - "$LAUNCHER" <<'PY'
import sys, io
path = sys.argv[1]
src = open(path).read()

anchor = "from llamafactory.train.tuner import run_exp  # use absolute import"
inject = (
    "# --- DiffuGPT fix: apply the 4d-attention-mask patch in EVERY torchrun worker.\n"
    "# cli.py applies it at import, but FORCE_TORCHRUN re-spawns workers that run this\n"
    "# launcher directly and never import cli.py, so the patch must be applied here too.\n"
    "from llamafactory import attention_patch\n"
    "attention_patch.replace_attention_mask()\n"
    "\n"
    + anchor
)
assert anchor in src, "anchor import line not found; launcher.py layout changed."
src = src.replace(anchor, inject, 1)
open(path, "w").write(src)
print("patched", path)
PY

echo "=== new launcher.py ==="
cat "${LAUNCHER}"
echo "========================"
echo "Done. Re-run: bash train_prosqa.sh probe"
echo "Worker logs should now print 'logging....attention-mask for 4d' and NOT crash at the mask add."