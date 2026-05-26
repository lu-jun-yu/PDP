#!/bin/bash
# =============================================================
#  scripts/eval_openrouter.sh — PDP 评估启动脚本 (OpenRouter API)
#
#  用法:
#    bash scripts/eval_openrouter.sh
#    bash scripts/eval_openrouter.sh --reasoning-effort high
#    bash scripts/eval_openrouter.sh --variants original,one_shot --reasoning-effort none
#
#  前置条件:
#    1. 设置环境变量 OPENROUTER_API_KEY
#    2. 安装依赖: pip install openai datasets
# =============================================================

set -euo pipefail

VARIANTS=("original" "definitions" "one_shot")
REASONING_EFFORT="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variants)
            IFS=',' read -r -a VARIANTS <<< "$2"
            shift 2
            ;;
        --reasoning-effort)
            REASONING_EFFORT="$2"
            shift 2
            ;;
        *)
            echo "错误: 未知参数 $1"
            echo "用法: bash scripts/eval_openrouter.sh [--variants original,definitions,one_shot] [--reasoning-effort auto|none|low|medium|high]"
            exit 1
            ;;
    esac
done

VALID_VARIANTS=("original" "definitions" "one_shot")
for variant in "${VARIANTS[@]}"; do
    is_valid=false
    for valid in "${VALID_VARIANTS[@]}"; do
        if [[ "$variant" == "$valid" ]]; then
            is_valid=true
            break
        fi
    done
    if [[ "$is_valid" == false ]]; then
        echo "错误: 未知 prompt variant: $variant"
        echo "可选值: ${VALID_VARIANTS[*]}"
        exit 1
    fi
done

case "$REASONING_EFFORT" in
    auto|none|low|medium|high) ;;
    *)
        echo "错误: --reasoning-effort 只能是 auto|none|low|medium|high"
        exit 1
        ;;
esac

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
DATA_PATH="data/pdp4k"
MAX_TOKENS=4096
CONCURRENCY=20                 # 并发请求数
BATCH_SIZE=100                 # 分批大小
OUTPUT_DIR="results"

# ---- 运行消融实验 ----
for VARIANT in "${VARIANTS[@]}"; do
    echo "============================================================="
    echo "运行消融实验: $VARIANT"
    echo "============================================================="

    python eval/evaluate_openrouter.py \
        --model "$MODEL" \
        --data-path "$DATA_PATH" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONCURRENCY" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        --prompt-variant "$VARIANT" \
        --reasoning-effort "$REASONING_EFFORT"
done
