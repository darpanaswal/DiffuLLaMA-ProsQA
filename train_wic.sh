#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE RUN PROPERTIES
########################################
EXPERIMENT="train_wic_ddmsft"
N_GPUS=4
WALLTIME="24:00:00"

# Answer-token loss upweight (requires fix_answer_upweight_patch.sh applied).
# 1.0 = unchanged. >1 upweights the Yes/No token so training keeps optimizing
# the label after the format is learned. Sweep if needed.
ANSWER_LOSS_WEIGHT="${ANSWER_LOSS_WEIGHT:-5}"

# MODE: "probe" (small subset, viability gate) or "full"
RAW_MODE="${1:-probe}"
MODE="${RAW_MODE}"

LF_DIR="LLaMA-Factory"
if [ "${MODE}" = "probe" ]; then
    CONFIG="examples/train_full/gpt2_wic_ddm-sft_probe.yaml"
    CKPT_OUT="output/diffugpt-m-wic-probe"
else
    CONFIG="examples/train_full/gpt2_wic_ddm-sft.yaml"
    CKPT_OUT="output/diffugpt-m-wic"
fi
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT}_${MODE}.txt"
# absolute path to THIS script, so the OAR job can invoke it from any cwd
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# --- OAR self-submit ---
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    oarsub \
        -n "${EXPERIMENT}_${MODE}" \
        -p "network_address='lig-gpu9.imag.fr'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "bash ${SCRIPT_PATH} ${RAW_MODE}"
    exit 0
fi

# ensure cwd = repo root (dir containing this script), so all relative paths
# (diffu/, LLaMA-Factory/, models/, scripts/, runs/) resolve regardless of OAR's cwd.
cd "$(dirname "${SCRIPT_PATH}")"

source diffu/bin/activate    # working env (all successful runs use diffu, NOT dlig)

echo "Experiment    : ${EXPERIMENT}"
echo "Mode          : ${MODE}"
echo "Config        : ${CONFIG}"
echo "GPUs          : ${N_GPUS}"
echo "Answer weight : ${ANSWER_LOSS_WEIGHT}"
echo "Ckpt out      : ${LF_DIR}/${CKPT_OUT}"
echo "----------------------------------------"

REPO_ROOT="$(pwd)"           # DiffuLLaMA repo root

# ------------------------------------------------------------------ #
# STEP 0: build WiC ddm-sft data (idempotent; seconds). Writes
# LLaMA-Factory/data/wic_{train,valid}.jsonl (+ wic_test_raw.jsonl for eval).
# Register wic_train / wic_valid in LLaMA-Factory/data/dataset_info.json ONCE
# (see note at bottom of this file).
# ------------------------------------------------------------------ #
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Building WiC ddm-sft data + length report..."
python -u scripts/wic_to_diffusft.py \
    --out_dir "${LF_DIR}/data" \
    --report \
    2>&1 | tee "${LOG_DIR}/data_${MODE}.log"

# ------------------------------------------------------------------ #
# TRAIN: llamafactory-cli launches torchrun across N_GPUS internally.
# Same python -c workaround as train_prosqa.sh (cli.py has no __main__ guard).
# ANSWER_LOSS_WEIGHT is inlined so every torchrun worker inherits it.
# ------------------------------------------------------------------ #
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting ddm-sft training (${MODE})..."
cd "${LF_DIR}"
FORCE_TORCHRUN=1 NPROC_PER_NODE=${N_GPUS} ANSWER_LOSS_WEIGHT=${ANSWER_LOSS_WEIGHT} \
    python -c "from llamafactory.cli import main; main()" train "${CONFIG}" \
    > "${REPO_ROOT}/${LOG_DIR}/train_${MODE}.log" 2>&1
cd "${REPO_ROOT}"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training done. Checkpoint -> ${LF_DIR}/${CKPT_OUT}"

echo "----------------------------------------"
echo "Data log  : ${LOG_DIR}/data_${MODE}.log"
echo "Train log : ${LOG_DIR}/train_${MODE}.log"
echo ""
echo "NEXT: run eval in the DLIG repo (dlig venv) on the held-out test set:"
echo "  bash eval_wic.sh    # from DiffusionLayerIntegratedGradients, ckpt = ${LF_DIR}/${CKPT_OUT}"
echo ""
echo "ONE-TIME SETUP: register the WiC datasets in LLaMA-Factory/data/dataset_info.json"
echo "  (same schema as prosqa_train), e.g.:"
echo '  "wic_train": {"file_name":"wic_train.jsonl","columns":{"prompt":"instruction","query":"input","response":"output"}},'
echo '  "wic_valid": {"file_name":"wic_valid.jsonl","columns":{"prompt":"instruction","query":"input","response":"output"}}'