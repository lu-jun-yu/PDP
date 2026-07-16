#!/bin/bash
# =============================================================
#  scripts/train_RQ3_gspo.sh -- RQ3 GSPO 类别占比干预训练
#
#  GSPO 在官方 TRL 中不是新的 Trainer，而是基于 GRPOTrainer：
#    - loss_type="grpo"
#    - importance_sampling_level="sequence"
#    - 配合 num_iterations > 1 或 steps_per_generation > grad_accum
#
#  默认设置：
#    - 数据集: data/pdp2k_rq3
#    - prompt: original
#    - 训练组: natural / balanced / P-40 / P-55 / DNP-40 / DNP-55 /
#             SNP-40 / SNP-55 / IENP-40 / IENP-55
#    - seed: 42
#
#  用法：
#    bash scripts/train_RQ3_gspo.sh
#    bash scripts/train_RQ3_gspo.sh DNP-40 DNP-55
#    RQ3_GROUPS="balanced DNP-40" bash scripts/train_RQ3_gspo.sh
#
#  常用覆盖：
#    MODEL_PATH=/path/to/Qwen3-8B DATA_PATH=data/pdp2k_rq3 bash scripts/train_RQ3_gspo.sh
#    PROMPT_VARIANT=original MAX_SAMPLES=32 DRY_RUN=1 bash scripts/train_RQ3_gspo.sh balanced
# =============================================================

set -euo pipefail

if [[ "${GSPO_LOG_ACTIVE:-0}" != "1" ]]; then
    LOG_DIR="${LOG_DIR:-logs}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_FILE:-$LOG_DIR/train_RQ3_gspo_$(date +%Y%m%d_%H%M%S).log}"
    export GSPO_LOG_ACTIVE=1
    export LOG_FILE
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

    if command -v stdbuf >/dev/null 2>&1; then
        exec > >(stdbuf -oL -eL tee -a "$LOG_FILE") 2>&1
    else
        exec > >(tee -a "$LOG_FILE") 2>&1
    fi
    echo "[train_RQ3_gspo] log_file=$LOG_FILE"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- GPU / DeepSpeed ----
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    DETECTED_GPUS=0
fi
NUM_GPUS="${NUM_GPUS:-$DETECTED_GPUS}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "错误: 未检测到 GPU；如需手动指定，请设置 NUM_GPUS。"
    exit 1
fi

_default_deepspeed="configs/ds_zero1.json"
if [[ "${OFFLOAD_OPTIMIZER:-0}" == "1" ]]; then
  _default_deepspeed="configs/ds_zero1_offload.json"
fi
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$_default_deepspeed}"

# ---- RQ3 数据与模型 ----
MODEL_PATH="${MODEL_PATH:-models/Qwen3-8B}"
DATA_PATH="${DATA_PATH:-data/pdp2k_rq3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/RQ3_GSPO}"
PROMPT_VARIANT="${PROMPT_VARIANT:-original}"

DEFAULT_GROUPS="balanced DNP-40 DNP-55 SNP-40 SNP-55 IENP-40 IENP-55" # natural
if [[ $# -gt 0 ]]; then
    RQ3_GROUPS="$*"
else
    RQ3_GROUPS="${RQ3_GROUPS:-$DEFAULT_GROUPS}"
fi
echo "[train_RQ3_gspo] argv_count=$# argv=[$*] resolved_RQ3_GROUPS=[$RQ3_GROUPS]"
SEED="${SEED:-42}"

# ---- 生成 ----
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1536}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"

# ---- vLLM ----
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.50}"
VLLM_TP="${VLLM_TP:-1}"

# ---- 训练 ----
NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
LR="${LR:-1e-5}"

# ---- LoRA ----
USE_LORA="${USE_LORA:-0}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_BIAS="${LORA_BIAS:-none}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"

# ---- GSPO 超参 ----
EPSILON="${EPSILON:-0.2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-}"

# ---- 日志与保存 ----
LOGGING_STEPS="${LOGGING_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-30}"
WANDB_PROJECT="${WANDB_PROJECT:-PDP-RQ3}"

mkdir -p "$OUTPUT_ROOT"

echo "============================================================="
echo "RQ3 GSPO training"
echo "  model:                $MODEL_PATH"
echo "  data:                 $DATA_PATH"
echo "  groups:               $RQ3_GROUPS"
echo "  seed:                 $SEED"
echo "  prompt_variant:       $PROMPT_VARIANT"
echo "  num_gpus:             $NUM_GPUS"
echo "  deepspeed:            $DEEPSPEED_CONFIG  (OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-0})"
echo "  output_root:          $OUTPUT_ROOT"
echo "  batch/accum:          $BATCH_SIZE / $GRAD_ACCUM"
echo "  num_gen:              $NUM_GENERATIONS"
echo "  vllm_gpu_mem:         $VLLM_GPU_MEM_UTIL"
echo "  use_lora:             $USE_LORA"
echo "  lora_r/alpha/drop:    $LORA_R / $LORA_ALPHA / $LORA_DROPOUT"
echo "  gspo_epsilon:         $EPSILON"
echo "  gspo_num_iterations:  $NUM_ITERATIONS"
echo "  steps_per_generation: ${STEPS_PER_GENERATION:-<trl-default>}"
echo "============================================================="

if [[ ! -d "$DATA_PATH" ]]; then
    echo "错误: 数据集目录不存在: $DATA_PATH"
    echo "请先确认 data/pdp2k_rq3 已生成。"
    exit 1
fi

MODEL_NAME="$(basename "$MODEL_PATH")"

for GROUP in $RQ3_GROUPS; do
    GROUP_SAFE="${GROUP//-/_}"
    RUN_NAME="RQ3_GSPO_${MODEL_NAME}_${GROUP_SAFE}_seed${SEED}_${PROMPT_VARIANT}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

    CMD=(
        deepspeed --num_gpus "$NUM_GPUS"
        train/gspo.py
        --deepspeed "$DEEPSPEED_CONFIG"
        --model-path "$MODEL_PATH"
        --data-path "$DATA_PATH"
        --split "$GROUP"
        --prompt-variant "$PROMPT_VARIANT"
        --output-dir "$OUTPUT_DIR"
        --max-completion-length "$MAX_COMPLETION_LENGTH"
        --max-prompt-length "$MAX_PROMPT_LENGTH"
        --num-generations "$NUM_GENERATIONS"
        --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
        --vllm-tensor-parallel-size "$VLLM_TP"
        --num-train-epochs "$NUM_EPOCHS"
        --per-device-train-batch-size "$BATCH_SIZE"
        --gradient-accumulation-steps "$GRAD_ACCUM"
        --learning-rate "$LR"
        --seed "$SEED"
        --epsilon "$EPSILON"
        --num-iterations "$NUM_ITERATIONS"
        --logging-steps "$LOGGING_STEPS"
        --save-steps "$SAVE_STEPS"
        --wandb-project "$WANDB_PROJECT"
        --run-name "$RUN_NAME"
    )

    if [[ -n "$STEPS_PER_GENERATION" ]]; then
        CMD+=(--steps-per-generation "$STEPS_PER_GENERATION")
    fi

    if [[ "$USE_LORA" == "1" ]]; then
        CMD+=(
            --use-lora
            --lora-r "$LORA_R"
            --lora-alpha "$LORA_ALPHA"
            --lora-dropout "$LORA_DROPOUT"
            --lora-bias "$LORA_BIAS"
            --lora-target-modules
        )
        for module in $LORA_TARGET_MODULES; do
            CMD+=("$module")
        done
    fi

    if [[ -n "${MAX_SAMPLES:-}" ]]; then
        CMD+=(--max-samples "$MAX_SAMPLES")
    fi

    echo
    echo "-------------------------------------------------------------"
    echo "Start: group=$GROUP seed=$SEED"
    echo "Output: $OUTPUT_DIR"
    echo "Command: ${CMD[*]}"
    echo "-------------------------------------------------------------"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
    fi

    "${CMD[@]}"
done

echo
echo "[完成] RQ3 GSPO training jobs finished."
