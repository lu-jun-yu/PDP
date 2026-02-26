#!/bin/bash
# =============================================================
#  scripts/run_grpo.sh — PDP GRPO 训练启动脚本
#  用法: bash scripts/run_grpo.sh
# =============================================================

set -euo pipefail

# ---- 参数配置（直接在此修改） ----
MODEL_PATH="models/Qwen3-4B"
DATA_PATH="data/pdp25k"
OUTPUT_DIR="results/grpo"

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

# 日志与保存
LOGGING_STEPS=10
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=3

# ---- 运行训练 ----
python train/grpo.py \
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
    --logging-steps "$LOGGING_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT"
