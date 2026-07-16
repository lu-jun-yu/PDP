#!/bin/bash
# =============================================================
#  scripts/eval_RQ3_gspo.sh - RQ3 GSPO checkpoint evaluation
#
#  Mirrors scripts/eval_RQ3.sh, but resolves GSPO run names from
#  scripts/train_RQ3_gspo.sh:
#    RQ3_GSPO_${MODEL_NAME}_${GROUP_SAFE}_seed${SEED}_${PROMPT_VARIANT}
#
#  Common usage:
#    bash scripts/eval_RQ3_gspo.sh
#    bash scripts/eval_RQ3_gspo.sh balanced
#    RQ3_GROUPS="balanced DNP-40" bash scripts/eval_RQ3_gspo.sh
#    CHECKPOINT_PATH=checkpoints/RQ3_GSPO/.../checkpoint-30 bash scripts/eval_RQ3_gspo.sh
#    SPLIT=test_rq2 NUM_REPEATS=8 bash scripts/eval_RQ3_gspo.sh balanced
#    DRY_RUN=1 bash scripts/eval_RQ3_gspo.sh
# =============================================================

set -euo pipefail

# ---- Naming parameters aligned with scripts/train_RQ3_gspo.sh ----
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
SEED="${SEED:-42}"
PROMPT_VARIANT="${PROMPT_VARIANT:-original}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/RQ3_GSPO}"

# Keep this aligned with the current GSPO training default.
DEFAULT_GROUPS="${DEFAULT_GROUPS:-balanced DNP-40 DNP-55 SNP-40 SNP-55 IENP-40 IENP-55}"
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    RQ3_GROUPS="__direct_checkpoint__"
elif [[ $# -gt 0 ]]; then
    RQ3_GROUPS="$*"
else
    RQ3_GROUPS="${RQ3_GROUPS:-$DEFAULT_GROUPS}"
fi

# ---- Evaluation parameters aligned with scripts/eval_RQ3.sh ----
DATA_PATH="${DATA_PATH:-data/pdp4k}"
SPLIT="${SPLIT:-test}"
NUM_REPEATS="${NUM_REPEATS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TP_SIZE="${TP_SIZE:-1}"
GPU_UTIL="${GPU_UTIL:-0.92}"
BATCH_SIZE="${BATCH_SIZE:-200}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-results/RQ3_GSPO}"

# vLLM may create CUDA worker subprocesses.  If the parent process has already
# touched CUDA, the default fork method can fail with "Cannot re-initialize CUDA
# in forked subprocess"; spawn is the safe default for this evaluation path.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo "============================================================="
echo "RQ3 GSPO checkpoint evaluation (vLLM)"
echo "  model_name:      $MODEL_NAME"
echo "  groups:          $RQ3_GROUPS"
echo "  seed:            $SEED"
echo "  prompt_variant:  $PROMPT_VARIANT"
echo "  checkpoint_root: $CHECKPOINT_ROOT"
echo "  checkpoint_path: ${CHECKPOINT_PATH:-<auto-resolve-by-group>}"
echo "  data:            $DATA_PATH [$SPLIT]"
echo "  num_repeats:     $NUM_REPEATS"
echo "  max_model_len:   $MAX_MODEL_LEN, max_tokens: $MAX_TOKENS"
echo "  TP:              $TP_SIZE, gpu_util: $GPU_UTIL"
echo "  batch_size:      $BATCH_SIZE, dtype: $DTYPE"
echo "  output_dir:      $OUTPUT_DIR"
echo "  vllm_mp_method:  $VLLM_WORKER_MULTIPROC_METHOD"
echo "============================================================="

if [[ ! -d "$DATA_PATH" ]]; then
    echo "Error: dataset directory not found: $DATA_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

find_last_inner_checkpoint() {
    local dir="$1"
    local last="" last_step=-1
    for sub in "$dir"/checkpoint-*/; do
        [[ -d "$sub" ]] || continue
        local name="${sub%/}"
        local step="${name##*-}"
        if [[ "$step" =~ ^[0-9]+$ ]] && (( step > last_step )); then
            last_step=$step
            last="$name"
        fi
    done
    printf '%s' "$last"
}

has_model_weights() {
    local dir="$1"
    [[ -f "$dir/model.safetensors" ]] \
        || [[ -f "$dir/model.safetensors.index.json" ]] \
        || [[ -f "$dir/pytorch_model.bin" ]] \
        || [[ -f "$dir/pytorch_model.bin.index.json" ]]
}

resolve_model_path() {
    local ckpt_dir="$1"
    if [[ ! -d "$ckpt_dir" ]]; then
        return 1
    fi
    if has_model_weights "$ckpt_dir"; then
        printf '%s' "$ckpt_dir"
        return 0
    fi

    local inner
    inner="$(find_last_inner_checkpoint "$ckpt_dir")"
    if [[ -n "$inner" ]] && has_model_weights "$inner"; then
        printf '%s' "$inner"
        return 0
    fi
    return 1
}

eval_one() {
    local label="$1"
    local model_path="$2"

    echo
    echo "-------------------------------------------------------------"
    echo "[start] $label"
    echo "  checkpoint: $model_path"
    echo "  split:      $SPLIT  (repeats=$NUM_REPEATS)"
    echo "-------------------------------------------------------------"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN=1; command:"
        echo "python eval/evaluate_vllm.py --model-path \"$model_path\" --data-path \"$DATA_PATH\" --split \"$SPLIT\" --num-repeats \"$NUM_REPEATS\" --seed \"$SEED\" --max-model-len \"$MAX_MODEL_LEN\" --max-tokens \"$MAX_TOKENS\" --tensor-parallel-size \"$TP_SIZE\" --gpu-memory-utilization \"$GPU_UTIL\" --batch-size \"$BATCH_SIZE\" --dtype \"$DTYPE\" --output-dir \"$OUTPUT_DIR\" --prompt-variant \"$PROMPT_VARIANT\""
        return 0
    fi

    python eval/evaluate_vllm.py \
        --model-path "$model_path" \
        --data-path "$DATA_PATH" \
        --split "$SPLIT" \
        --num-repeats "$NUM_REPEATS" \
        --seed "$SEED" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-tokens "$MAX_TOKENS" \
        --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --batch-size "$BATCH_SIZE" \
        --dtype "$DTYPE" \
        --output-dir "$OUTPUT_DIR" \
        --prompt-variant "$PROMPT_VARIANT"
}

overall_fail=0

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    if ! MODEL_PATH="$(resolve_model_path "$CHECKPOINT_PATH")"; then
        echo "Error: CHECKPOINT_PATH has no model weights: $CHECKPOINT_PATH"
        exit 1
    fi
    eval_one "direct_checkpoint" "$MODEL_PATH" || overall_fail=1
else
    for GROUP in $RQ3_GROUPS; do
        GROUP_SAFE="${GROUP//-/_}"
        RUN_NAME="RQ3_GSPO_${MODEL_NAME}_${GROUP_SAFE}_seed${SEED}_${PROMPT_VARIANT}"
        CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"

        if ! MODEL_PATH="$(resolve_model_path "$CKPT_DIR")"; then
            echo
            echo "[skip] group=$GROUP - no model weights found under: $CKPT_DIR"
            overall_fail=1
            continue
        fi

        eval_one "group=$GROUP" "$MODEL_PATH" || overall_fail=1
    done
fi

if [[ "$overall_fail" -ne 0 ]]; then
    echo
    echo "Some GSPO evaluations failed or checkpoints were missing."
    exit 1
fi

echo
echo "[done] RQ3 GSPO checkpoint evaluation finished -> $OUTPUT_DIR"
