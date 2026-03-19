#!/bin/bash
# =============================================================
#  scripts/run_dapo.sh — PDP DAPO 训练启动脚本
#  用法: bash scripts/run_dapo.sh
# =============================================================

set -euo pipefail

# ---- 参数配置（直接在此修改） ----
MODEL_PATH="models/Qwen3-4B"
DATA_PATH="data/pdp10k"
OUTPUT_DIR="results/DAPO-4B-0319"

# 生成
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=3072
NUM_GENERATIONS=8

# vLLM
VLLM_GPU_MEM_UTIL=0.5

# 训练
NUM_EPOCHS=1
BATCH_SIZE=4
GRAD_ACCUM=4
LR=5e-6

# DAPO 超参
EPSILON=0.2
EPSILON_HIGH=0.28

# 日志与保存
LOGGING_STEPS=10
SAVE_STEPS=100

# wandb
WANDB_PROJECT="PDP"
RUN_NAME="DAPO-4B-0319"

# ---- 运行训练 ----
python train/dapo.py \
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
