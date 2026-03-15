#!/bin/bash
# =============================================================
#  scripts/eval_openrouter.sh — PDP 评估启动脚本 (OpenRouter API)
#
#  用法:
#    bash scripts/eval_openrouter.sh
#
#  前置条件:
#    1. 设置环境变量 OPENROUTER_API_KEY
#    2. 安装依赖: pip install openai datasets
# =============================================================

set -euo pipefail

# ---- API Key ----
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
    export "$(grep -E '^OPENROUTER_API_KEY=' .env | head -1)"
fi

# 检查 API Key
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "错误: 请设置 OPENROUTER_API_KEY 环境变量"
    echo "  方式一: export OPENROUTER_API_KEY=\"sk-or-v1-...\""
    echo "  方式二: 在项目根目录创建 .env 文件，写入 OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
fi

# ---- 参数配置（直接在此修改） ----
MODEL="qwen/qwen3-30b-a3b-thinking-2507"         # OpenRouter 模型标识符
DATA_PATH="data/pdp25k"
MAX_TOKENS=2048
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MIN_P=0.0
CONCURRENCY=5                  # 并发请求数
BATCH_SIZE=100                 # 分批大小
OUTPUT_DIR="results"

# ---- 运行评估 (baseline) ----
python eval/evaluate_openrouter.py \
    --model "$MODEL" \
    --data-path "$DATA_PATH" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    --concurrency "$CONCURRENCY" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR"

# ---- 运行评估 (with definitions) ----
python eval/evaluate_openrouter.py \
    --model "$MODEL" \
    --data-path "$DATA_PATH" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    --concurrency "$CONCURRENCY" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --with-definitions
