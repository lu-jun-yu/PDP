#!/bin/bash
# =============================================================
#  scripts/train_RQ3_sft.sh — RQ3 CoT SFT cold-start training
#
#  默认设置：
#    - conda env: pytorch
#    - model: models/Qwen3-8B
#    - data: data/pdp2k_rq3_sft
#    - split: train
#    - target: cot_data
#    - output: checkpoints/RQ3_SFT_Qwen3-8B
#
#  用法：
#    conda activate pytorch
#    bash scripts/train_RQ3_sft.sh
#
#  快速检查：
#    DRY_RUN=1 bash scripts/train_RQ3_sft.sh
#    MAX_SAMPLES=32 DRY_RUN=1 bash scripts/train_RQ3_sft.sh
#
#  常用覆盖：
#    MODEL_PATH=models/Qwen3-8B DATA_PATH=data/pdp2k_rq3_sft bash scripts/train_RQ3_sft.sh
#    NUM_GPUS=2 DEEPSPEED_CONFIG=configs/ds_zero2.json bash scripts/train_RQ3_sft.sh
# =============================================================

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Limit CPU saturation from tokenizers, BLAS/OpenMP, DataLoader workers, and
# DeepSpeed CPU offload threads. Override with MAX_CPU_CORES=16, etc.
MAX_CPU_CORES="${MAX_CPU_CORES:-48}"
if ! [[ "$MAX_CPU_CORES" =~ ^[0-9]+$ ]] || [[ "$MAX_CPU_CORES" -lt 1 ]]; then
    echo "错误: MAX_CPU_CORES 必须为正整数，当前值: $MAX_CPU_CORES"
    exit 1
fi
if command -v nproc >/dev/null 2>&1; then
    AVAILABLE_CPU_CORES="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
    AVAILABLE_CPU_CORES="$(getconf _NPROCESSORS_ONLN)"
else
    AVAILABLE_CPU_CORES="$MAX_CPU_CORES"
fi
if [[ "$MAX_CPU_CORES" -gt "$AVAILABLE_CPU_CORES" ]]; then
    MAX_CPU_CORES="$AVAILABLE_CPU_CORES"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$MAX_CPU_CORES}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$MAX_CPU_CORES}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$MAX_CPU_CORES}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$MAX_CPU_CORES}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$MAX_CPU_CORES}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---- 环境提示 ----
if [[ "${CONDA_DEFAULT_ENV:-}" != "pytorch" ]]; then
    echo "警告: 当前 conda 环境不是 pytorch (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-未设置})。"
    echo "      大模型相关依赖默认在 pytorch 环境中；建议先运行: conda activate pytorch"
fi

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

_default_deepspeed="configs/ds_zero2.json"
if [[ "${OFFLOAD_OPTIMIZER:-0}" == "1" ]]; then
    _default_deepspeed="configs/ds_zero2_offload.json"
fi
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$_default_deepspeed}"

# ---- 数据与模型 ----
MODEL_PATH="${MODEL_PATH:-models/Qwen3-8B}"
DATA_PATH="${DATA_PATH:-data/pdp2k_rq3_sft}"
SPLIT="${SPLIT:-train}"
COT_FIELD="${COT_FIELD:-cot_data}"
PROMPT_VARIANT="${PROMPT_VARIANT:-original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints}"
MODEL_NAME="$(basename "$MODEL_PATH")"
RUN_NAME="${RUN_NAME:-RQ3_SFT_${MODEL_NAME}_${SPLIT}_seed${SEED:-42}_${PROMPT_VARIANT}}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"

# ---- 训练 ----
MAX_LENGTH="${MAX_LENGTH:-3072}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_BIAS="${LORA_BIAS:-none}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant_with_warmup}"
SEED="${SEED:-42}"

# ---- 日志与保存 ----
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-PDP-RQ3-SFT}"
REPORT_TO="${REPORT_TO:-wandb}"

if [[ ! -d "$DATA_PATH" ]]; then
    echo "错误: SFT 数据集目录不存在: $DATA_PATH"
    echo "请先运行 data_process/RQ3/generate_pdp2k_rq3_sft.py 生成 CoT 数据。"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

CMD=(
    deepspeed --num_gpus "$NUM_GPUS"
    train/sft.py
    --deepspeed "$DEEPSPEED_CONFIG"
    --model-path "$MODEL_PATH"
    --data-path "$DATA_PATH"
    --split "$SPLIT"
    --cot-field "$COT_FIELD"
    --prompt-variant "$PROMPT_VARIANT"
    --output-dir "$OUTPUT_DIR"
    --max-length "$MAX_LENGTH"
    --num-train-epochs "$NUM_EPOCHS"
    --per-device-train-batch-size "$BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --learning-rate "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-ratio "$WARMUP_RATIO"
    --lr-scheduler-type "$LR_SCHEDULER_TYPE"
    --seed "$SEED"
    --logging-steps "$LOGGING_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --wandb-project "$WANDB_PROJECT"
    --report-to "$REPORT_TO"
    --run-name "$RUN_NAME"
)

if [[ "$USE_LORA" == "1" ]]; then
    CMD+=(
        --use-lora
        --lora-r "$LORA_R"
        --lora-alpha "$LORA_ALPHA"
        --lora-dropout "$LORA_DROPOUT"
        --lora-bias "$LORA_BIAS"
    )
    read -r -a LORA_TARGET_MODULES_ARR <<< "$LORA_TARGET_MODULES"
    CMD+=(--lora-target-modules "${LORA_TARGET_MODULES_ARR[@]}")
else
    CMD+=(--no-use-lora)
fi

if command -v taskset >/dev/null 2>&1; then
    CPU_LAST=$((MAX_CPU_CORES - 1))
    CMD=(taskset -c "0-${CPU_LAST}" "${CMD[@]}")
fi

if [[ -n "${MAX_SAMPLES:-}" ]]; then
    CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "${BF16:-1}" == "0" ]]; then
    CMD+=(--no-bf16)
fi
if [[ "${FP16:-0}" == "1" ]]; then
    CMD+=(--fp16 --no-bf16)
fi
if [[ "${GRADIENT_CHECKPOINTING:-1}" == "0" ]]; then
    CMD+=(--no-gradient-checkpointing)
fi

echo "============================================================="
echo "RQ3 SFT cold-start training"
echo "  model:          $MODEL_PATH"
echo "  data:           $DATA_PATH [$SPLIT]"
echo "  cot_field:      $COT_FIELD"
echo "  prompt_variant: $PROMPT_VARIANT"
echo "  seed:           $SEED"
echo "  num_gpus:       $NUM_GPUS"
echo "  deepspeed:      $DEEPSPEED_CONFIG  (OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-0})"
echo "  output_dir:     $OUTPUT_DIR"
echo "  max_length:     $MAX_LENGTH"
echo "  batch/accum:    $BATCH_SIZE / $GRAD_ACCUM"
echo "  max_cpu_cores:  $MAX_CPU_CORES"
echo "  epochs/lr:      $NUM_EPOCHS / $LR"
echo "  use_lora:       $USE_LORA"
if [[ "$USE_LORA" == "1" ]]; then
    echo "  lora:           r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT bias=$LORA_BIAS"
    echo "  lora_targets:   $LORA_TARGET_MODULES"
fi
echo "============================================================="
echo "Command: ${CMD[*]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] command not executed."
    exit 0
fi

"${CMD[@]}"

echo
echo "[完成] RQ3 SFT training finished: $OUTPUT_DIR"
