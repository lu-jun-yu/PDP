#!/bin/bash
# =============================================================
#  scripts/eval.sh — PDP 评估启动脚本
#  用法: bash scripts/eval.sh
# =============================================================

set -euo pipefail

# ---- 参数配置（直接在此修改） ----
MODEL_PATH="checkpoints/DAPO-0.6B-0320" # results/grpo
DATA_PATH="data/pdp4k"
MAX_MODEL_LEN=4096
MAX_TOKENS=4096
TP_SIZE=1                  # 张量并行数
GPU_UTIL=0.9
OUTPUT_DIR="results"
BATCH_SIZE=100             # 分批推理批次大小，和 OpenRouter 脚本保持一致

# ---- 运行三种消融实验 ----
for VARIANT in original definitions one_shot; do
    echo "============================================================="
    echo "运行消融实验: $VARIANT"
    echo "============================================================="

    python eval/evaluate_vllm.py \
        --model-path "$MODEL_PATH" \
        --data-path "$DATA_PATH" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-tokens "$MAX_TOKENS" \
        --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --prompt-variant "$VARIANT"
done
