#!/bin/bash
# =============================================================
#  scripts/eval_RQ1.sh — RQ1 多模型并行评估脚本 (OpenRouter API)
#
#  用法:
#    bash scripts/eval_RQ1.sh
# =============================================================

set -euo pipefail

VARIANT="original"
REASONING_EFFORT="auto"

if [[ $# -ne 0 ]]; then
    echo "错误: scripts/eval_RQ1.sh 不接受任何参数"
    echo "用法: bash scripts/eval_RQ1.sh"
    exit 1
fi

# ---- API Key ----
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
    export "$(grep -E '^OPENROUTER_API_KEY=' .env | head -1)"
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "错误: 请设置 OPENROUTER_API_KEY 环境变量"
    echo "  方式一: export OPENROUTER_API_KEY=\"sk-or-v1-...\""
    echo "  方式二: 在项目根目录创建 .env 文件，写入 OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
fi

# ---- 参数配置 ----
MODELS=(
    "openai/gpt-5.5"
    "google/gemini-3.1-pro-preview"
    "deepseek/deepseek-v4-pro"
    "qwen/qwen3.6-max-preview"
    "moonshotai/kimi-k2.6"
)
DATA_PATH="data/pdp4k"
MAX_TOKENS=8192
CONCURRENCY=20
BATCH_SIZE=100
OUTPUT_DIR="results/RQ1"

echo "============================================================="
echo "RQ1 并行评估"
echo "模型: ${MODELS[*]}"
echo "prompt variant: $VARIANT"
echo "reasoning-effort: $REASONING_EFFORT"
echo "============================================================="

overall_fail=0
echo
echo "-------------------------------------------------------------"
echo "当前变体: $VARIANT（在当前终端后台并行启动所有模型，不新开窗口）"
echo "-------------------------------------------------------------"

pids=()
names=()

for MODEL in "${MODELS[@]}"; do
    echo "[启动] model=$MODEL variant=$VARIANT"
    (
    python eval/evaluate_openrouter.py \
        --model "$MODEL" \
        --data-path "$DATA_PATH" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONCURRENCY" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        --prompt-variant "$VARIANT" \
        --reasoning-effort "$REASONING_EFFORT"
    ) &
    pids+=("$!")
    names+=("$MODEL")
done

for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    model="${names[$i]}"
    if wait "$pid"; then
        echo "[完成] $model (variant=$VARIANT)"
    else
        echo "[失败] $model (variant=$VARIANT)"
        overall_fail=1
    fi
done

if [[ "$overall_fail" -ne 0 ]]; then
    echo "存在模型评估失败，请检查日志。"
    exit 1
fi

echo "全部模型评估完成。"
