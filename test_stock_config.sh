#!/bin/bash
set -euo pipefail
# test_stock_config.sh
# Diagnostic: run HKUNLP's OWN working gpt2_full_ddm-sft.yaml (GSM) with only
# checkpoint_dir + max_steps edited, to isolate whether the 4d-mask crash is in
# OUR config or in the ENVIRONMENT (patch not wired / transformers version).
#
# Run from the DiffuLLaMA repo root:  bash test_stock_config.sh
#
# Interpretation:
#   - crashes with the SAME 4d-mask error -> environment problem (the attention
#     patch isn't replacing GPT2Model.forward in this install). Fix the install,
#     not the config.
#   - trains fine -> our prosqa config has the offending delta; diff it.

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p runs/stock_test
    oarsub \
        -n "stock_ddmsft_test" \
        -p "network_address='lig-gpu9.imag.fr'" \
        -l /host=1/gpu=4,walltime=00:30:00 \
        -O runs/stock_test/test.txt \
        -E runs/stock_test/test.txt \
        "bash ${SCRIPT_PATH}"
    exit 0
fi

cd "$(dirname "${SCRIPT_PATH}")"
source diffu/bin/activate

# make a temp copy of THEIR config with only checkpoint_dir + max_steps changed
TEST_CFG="LLaMA-Factory/examples/train_full/_stocktest_ddm-sft.yaml"
cp LLaMA-Factory/examples/train_full/gpt2_full_ddm-sft.yaml "${TEST_CFG}"

# point at the local DiffuGPT-M ckpt (training runs inside LLaMA-Factory/)
sed -i 's|^checkpoint_dir:.*|checkpoint_dir: ../models/diffugpt-m|' "${TEST_CFG}"
# cap to 2 steps; keep everything else (cutoff_len 256, deepspeed) as theirs
grep -q '^max_steps:' "${TEST_CFG}" && sed -i 's/^max_steps:.*/max_steps: 2/' "${TEST_CFG}" || echo 'max_steps: 2' >> "${TEST_CFG}"
# Turing RTX 8000 has no bf16 -> flip to fp16 (otherwise we'd hit a bf16 error,
# not the result we're actually testing for).
sed -i 's/^bf16: true/bf16: false/' "${TEST_CFG}"
grep -q '^fp16:' "${TEST_CFG}" || echo 'fp16: true' >> "${TEST_CFG}"
# their config uses dataset: gsm (may be unregistered here) -> use our registered
# prosqa_train so a missing-dataset error doesn't confound the mask test.
sed -i 's/^dataset:.*/dataset: prosqa_train/' "${TEST_CFG}"
# DeepSpeed isn't installed in this venv -> strip it so the test reaches the
# attention/mask path. (If the mask crash then appears WITHOUT deepspeed, deepspeed
# wrapping is what makes their setup work -> install it for the real runs.)
sed -i '/^deepspeed:/d' "${TEST_CFG}"
# remove deepspeed line: not installed, and we want to test the PLAIN-DDP mask path
# (same as our prosqa config) rather than deepspeed's wrapping.
sed -i '/^deepspeed:/d' "${TEST_CFG}"

echo "=== running THEIR stock config (only checkpoint_dir + max_steps changed) ==="
cat "${TEST_CFG}"
echo "================================================================"

cd LLaMA-Factory
FORCE_TORCHRUN=1 NPROC_PER_NODE=4 \
    python -c "from llamafactory.cli import main; main()" train "examples/train_full/_stocktest_ddm-sft.yaml" \
    2>&1 | tee ../runs/stock_test/stock_train.log

echo "=== DONE. If this crashed with the 4d-mask error too -> environment/patch issue. ==="