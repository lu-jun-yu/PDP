#!/bin/bash
# =============================================================
#  scripts/run_dapo.sh — PDP DAPO 多卡训练启动脚本 (DeepSpeed ZeRO-1)
#  用法: bash scripts/run_dapo.sh
# =============================================================

set -euo pipefail

# ---- 多卡配置 ----
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "检测到 ${NUM_GPUS} 张 GPU"

# ---- 参数配置（直接在此修改） ----
MODEL_PATH="models/Qwen3-0.6B"
DATA_PATH="data/pdp10k"
OUTPUT_DIR="checkpoints/DAPO-0.6B-0320"

# 生成
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=2048
NUM_GENERATIONS=8

# vLLM
VLLM_GPU_MEM_UTIL=0.5

# 训练
NUM_EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=32
LR=1e-6

# DAPO 超参
EPSILON=0.2
EPSILON_HIGH=0.28

# 日志与保存
LOGGING_STEPS=16
SAVE_STEPS=1024

# wandb
WANDB_PROJECT="PDP"
RUN_NAME="DAPO-0.6B-0320"

# ---- 运行训练 ----
deepspeed --num_gpus "$NUM_GPUS" \
    train/dapo.py \
    --deepspeed configs/ds_zero1.json \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-completion-length "$MAX_COMPLETION_LENGTH" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --num-generations "$NUM_GENERATIONS" \
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
    --num-train-epochs "$NUM_EPOCHS" \
    --per-device-train-batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate "$LR" \
    --epsilon "$EPSILON" \
    --epsilon-high "$EPSILON_HIGH" \
    --logging-steps "$LOGGING_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "$RUN_NAME"
