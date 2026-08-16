#!/bin/bash
set -euo pipefail
# fix_torchrun_patch.sh
# ROOT CAUSE of "/bin/sh: 1: torchrun: not found":
#   cli.py launches distributed training with subprocess.run("torchrun ...", shell=True).
#   The bare name "torchrun" is resolved via /bin/sh's PATH, NOT the active venv's PATH.
#   If the torch console-script isn't on the child shell's PATH (common under OAR, where
#   the subprocess doesn't inherit diffu/bin), the launch dies before any worker starts.
#
# FIX: replace the literal "torchrun" with "{python} -m torch.distributed.run", where
#   {python} is sys.executable — the SAME interpreter already running cli.py, i.e. the
#   diffu venv's python. This is immune to a missing console-script and to PATH not being
#   inherited by the shell subprocess. Functionally identical: torchrun IS the console
#   entry point for torch.distributed.run.
#
# Run from the DiffuLLaMA repo root:  bash fix_torchrun_patch.sh

CLI="LLaMA-Factory/src/llamafactory/cli.py"

if [ ! -f "${CLI}" ]; then
  echo "ERROR: ${CLI} not found. Run from the DiffuLLaMA repo root." >&2
  exit 1
fi

# idempotent: skip if already patched
if grep -q "torch.distributed.run" "${CLI}"; then
  echo "cli.py already uses 'python -m torch.distributed.run'. Nothing to do."
  exit 0
fi

cp "${CLI}" "${CLI}.bak"
echo "backed up -> ${CLI}.bak"

python3 - "$CLI" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()

# The command string starts with the literal 'torchrun ' — swap it for the
# module invocation using the current interpreter. We inject a Python f-string
# expression, so we also need sys imported (it already is in cli.py).
old = '"torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "'
new = '"{python} -m torch.distributed.run --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "'
assert old in src, "expected torchrun command string not found; cli.py layout changed."
src = src.replace(old, new, 1)

# add python=sys.executable to the .format(...) kwargs
anchor = "nnodes=os.environ.get(\"NNODES\", \"1\"),"
inject = "python=sys.executable,\n                    " + anchor
assert anchor in src, "expected .format kwargs not found; cli.py layout changed."
src = src.replace(anchor, inject, 1)

open(path, "w").write(src)
print("patched", path)
PY

echo "=== patched launch line ==="
grep -n "torch.distributed.run\|python=sys.executable" "${CLI}"
echo "==========================="
echo "Done. Re-run your training. The launch now uses the diffu venv's own python,"
echo "so it no longer depends on 'torchrun' being on the shell PATH."