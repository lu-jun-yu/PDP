#!/bin/bash
# =============================================================
#  scripts/eval_RQ2_1.sh — RQ2.1 推理预算并行评估 (OpenRouter API)
#
#  固定模型，并行启动多组推理预算；数据为 pdp4k 的 test_rq2 split（100 条），
#  每条样本重复 5 次。
#
#  预算模式（与 OpenRouter API 一致，二选一）:
#    - effort: 对每个 REASONING_EFFORTS 传 --reasoning-effort
#    - max_tokens: 对每个 REASONING_MAX_TOKENS_LIST 传 --reasoning-max-tokens
#
#  用法:
#    bash scripts/eval_RQ2_1.sh
# =============================================================

set -euo pipefail

MODEL="qwen/qwen3.6-max-preview"
VARIANT="original"

# "effort" | "max_tokens"
REASONING_BUDGET_MODE="effort"
REASONING_EFFORTS=("none" "low" "medium" "high" "xhigh")
# OpenRouter 要求 completion 的 --max-tokens 严格大于 reasoning.max_tokens；勿让下列最大值 >= MAX_TOKENS。
REASONING_MAX_TOKENS_LIST=(4096 8192 16384)

if [[ $# -ne 0 ]]; then
    echo "错误: scripts/eval_RQ2_1.sh 不接受任何参数"
    echo "用法: bash scripts/eval_RQ2_1.sh"
    exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ -f .env ]]; then
    export "$(grep -E '^OPENROUTER_API_KEY=' .env | head -1)"
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "错误: 请设置 OPENROUTER_API_KEY 环境变量"
    echo "  方式一: export OPENROUTER_API_KEY=\"sk-or-v1-...\""
    echo "  方式二: 在项目根目录创建 .env 文件，写入 OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
fi

DATA_PATH="data/pdp4k"
SPLIT="test_rq2"
MAX_TOKENS=16384
CONCURRENCY=20
BATCH_SIZE=100
OUTPUT_DIR="results/RQ2_1"
NUM_REPEATS=5

echo "============================================================="
echo "RQ2.1 并行评估（固定模型，不同推理预算）"
echo "模型: $MODEL"
echo "prompt variant: $VARIANT"
echo "reasoning budget mode: $REASONING_BUDGET_MODE"
if [[ "$REASONING_BUDGET_MODE" == "effort" ]]; then
    echo "reasoning efforts: ${REASONING_EFFORTS[*]}"
elif [[ "$REASONING_BUDGET_MODE" == "max_tokens" ]]; then
    echo "reasoning max_tokens: ${REASONING_MAX_TOKENS_LIST[*]}"
else
    echo "错误: REASONING_BUDGET_MODE 须为 effort 或 max_tokens"
    exit 1
fi
echo "split: $SPLIT, repeats per sample: $NUM_REPEATS"
echo "============================================================="

overall_fail=0
echo
echo "-------------------------------------------------------------"
echo "在当前终端后台并行启动不同推理预算档位（不新开窗口）"
echo "-------------------------------------------------------------"

pids=()
labels=()

if [[ "$REASONING_BUDGET_MODE" == "effort" ]]; then
    for EFFORT in "${REASONING_EFFORTS[@]}"; do
        echo "[启动] model=$MODEL variant=$VARIANT reasoning-effort=$EFFORT"
        (
        python eval/evaluate_openrouter_rq2.py \
            --model "$MODEL" \
            --data-path "$DATA_PATH" \
            --split "$SPLIT" \
            --num-repeats "$NUM_REPEATS" \
            --max-tokens "$MAX_TOKENS" \
            --concurrency "$CONCURRENCY" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$OUTPUT_DIR" \
            --prompt-variant "$VARIANT" \
            --reasoning-effort "$EFFORT"
        ) &
        pids+=("$!")
        labels+=("effort=$EFFORT")
    done
else
    for MT in "${REASONING_MAX_TOKENS_LIST[@]}"; do
        echo "[启动] model=$MODEL variant=$VARIANT reasoning-max-tokens=$MT"
        (
        python eval/evaluate_openrouter_rq2.py \
            --model "$MODEL" \
            --data-path "$DATA_PATH" \
            --split "$SPLIT" \
            --num-repeats "$NUM_REPEATS" \
            --max-tokens "$MAX_TOKENS" \
            --concurrency "$CONCURRENCY" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$OUTPUT_DIR" \
            --prompt-variant "$VARIANT" \
            --reasoning-max-tokens "$MT"
        ) &
        pids+=("$!")
        labels+=("max_tokens=$MT")
    done
fi

for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    label="${labels[$i]}"
    if wait "$pid"; then
        echo "[完成] $label"
    else
        echo "[失败] $label"
        overall_fail=1
    fi
done

if [[ "$overall_fail" -ne 0 ]]; then
    echo "存在评估失败，请检查日志。"
    exit 1
fi

echo "全部推理预算档位评估完成。"
