#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE RUN PROPERTIES
########################################
EXPERIMENT="train_prosqa_ddmsft"
N_GPUS=4
WALLTIME="24:00:00"

# MODE: "probe" (small subset, viability gate), "full", or
#       "eval-probe" / "eval-full" (skip training, just gen-eval an existing ckpt)
RAW_MODE="${1:-probe}"
MODE="${RAW_MODE}"

SKIP_TRAIN=0
case "${MODE}" in
    eval-probe) SKIP_TRAIN=1; MODE="probe" ;;
    eval-full)  SKIP_TRAIN=1; MODE="full"  ;;
esac

LF_DIR="LLaMA-Factory"
if [ "${MODE}" = "probe" ]; then
    CONFIG="examples/train_full/gpt2_prosqa_ddm-sft_probe.yaml"
    CKPT_OUT="output/diffugpt-m-prosqa-probe"
    EVAL_N=300
else
    CONFIG="examples/train_full/gpt2_prosqa_ddm-sft.yaml"
    CKPT_OUT="output/diffugpt-m-prosqa"
    EVAL_N=500
fi

EVAL_DATA="data/prosqa_test.json"     # COCONUT-format test (question/answer/steps)
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

source diffu/bin/activate   # working env (all successful runs use diffu, NOT dlig)

echo "Experiment : ${EXPERIMENT}"
echo "Mode       : ${MODE}"
echo "Config     : ${CONFIG}"
echo "GPUs       : ${N_GPUS}"
echo "Ckpt out   : ${LF_DIR}/${CKPT_OUT}"
echo "----------------------------------------"

# ------------------------------------------------------------------ #
# TRAIN: llamafactory-cli launches torchrun across N_GPUS internally
# (FORCE_TORCHRUN=1). Pascal 1080 Ti -> bf16:false is set in the yaml.
# Skipped entirely in eval-only mode (eval-probe / eval-full).
# ------------------------------------------------------------------ #
REPO_ROOT="$(pwd)"   # DiffuLLaMA repo root (where model.py + eval_prosqa.py live)

if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting ddm-sft training (${MODE})..."
    cd "${LF_DIR}"

    # NOTE: llamafactory/cli.py has NO `if __name__ == '__main__'` guard, so
    # `python -m llamafactory.cli` imports and exits WITHOUT running (silent no-op,
    # which is why training "finished" in seconds and saved nothing). The console
    # entry point calls llamafactory.cli:main directly -> replicate that here.
    FORCE_TORCHRUN=1 NPROC_PER_NODE=${N_GPUS} \
        python -c "from llamafactory.cli import main; main()" train "${CONFIG}" \
        > "${REPO_ROOT}/${LOG_DIR}/train_${MODE}.log" 2>&1

    cd "${REPO_ROOT}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training done. Checkpoint -> ${LF_DIR}/${CKPT_OUT}"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] SKIP_TRAIN set -> evaluating existing checkpoint ${LF_DIR}/${CKPT_OUT}"
fi

# ------------------------------------------------------------------ #
# EVAL: viability gate. Uses the DiffuLLaMA repo's OWN inference
# (model.py: generate_samples). eval_prosqa.py lives in scripts/ but does
# `from model import ...`, so it must run with the REPO ROOT as cwd (where
# model.py is). Already at REPO_ROOT here.
# ------------------------------------------------------------------ #
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Evaluating on ProsQA test (n=${EVAL_N})..."

# Run gen-eval. `tee` so accuracy lands in BOTH the eval log AND the main OAR
# log (this file), and don't let a non-zero exit kill the script before we can
# report — temporarily relax `set -e` around the eval call.
set +e
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_prosqa.py \
    --model_name "${LF_DIR}/${CKPT_OUT}" \
    --base_model_name gpt2-medium \
    --data "${LF_DIR}/${EVAL_DATA}" \
    --n_samples ${EVAL_N} \
    --batch_size 4 \
    --gen_len 80 \
    --diffusion_steps 64 \
    --out_file "${LOG_DIR}/prosqa_eval_${MODE}.jsonl" \
    2>&1 | tee "${LOG_DIR}/eval_${MODE}.log"
EVAL_RC=${PIPESTATUS[0]}
set -e

echo "----------------------------------------"
if [ "${EVAL_RC}" -ne 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] EVAL FAILED (exit ${EVAL_RC}). Training is fine;"
    echo "  see ${LOG_DIR}/eval_${MODE}.log for the traceback (common causes: wrong venv,"
    echo "  checkpoint saved as pytorch_model.bin vs safetensors, or model.py loader args)."
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] COMPLETE (${MODE}). Accuracy is in the [RESULT] block above."
fi
echo "Train log : ${LOG_DIR}/train_${MODE}.log"
echo "Eval log  : ${LOG_DIR}/eval_${MODE}.log   <-- accuracy here"
echo "Preds     : ${LOG_DIR}/prosqa_eval_${MODE}.jsonl"
echo ""
echo "GATE: probe answer_exact clearly > 50% (chance) and trending to 77.5% (CoT-GPT2 ref)"
echo "      -> run full:  bash train_prosqa.sh full"