#!/bin/bash
# =============================================================
#  scripts/eval.sh — PDP 评估启动脚本
#  用法: bash scripts/eval.sh
# =============================================================

set -euo pipefail

# ---- 参数配置（直接在此修改） ----
MODEL_PATH="models/Qwen3-4B"
DATA_PATH="data/pdp25k"
SPLIT="ood"
MAX_MODEL_LEN=4096
MAX_TOKENS=2048
TEMPERATURE=0.0
TP_SIZE=1                  # 张量并行数
GPU_UTIL=0.9
OUTPUT_DIR="results"
BATCH_SIZE=500             # 分批推理批次大小，0 表示一次性全部推理

# ---- 运行评估 ----
python eval/evaluate.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --split "$SPLIT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE"
