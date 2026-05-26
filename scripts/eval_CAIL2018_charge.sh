#!/bin/bash
# =============================================================
#  scripts/eval_CAIL2018_charge.sh — CAIL2018-Small charge eval
#
#  Default dataset:
#    data/CAIL2018/exercise_contest/data_test_charge_4k_seed42.json
#
#  Usage:
#    MODEL="google/gemini-3.1-pro-preview" bash scripts/eval_CAIL2018_charge.sh
#    INPUT_FILE=data/CAIL2018/exercise_contest/data_test.json MODEL="openai/gpt-5.4" bash scripts/eval_CAIL2018_charge.sh
# =============================================================

set -euo pipefail

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
    export "$(grep -E '^OPENROUTER_API_KEY=' .env | head -1)"
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "错误: 请设置 OPENROUTER_API_KEY"
    exit 1
fi

MODEL="${MODEL:-qwen/qwen3.6-max-preview}"
INPUT_FILE="${INPUT_FILE:-data/CAIL2018/exercise_contest/data_test_charge_4k_seed42.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/RQ1_CAIL2018_charge}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
CONCURRENCY="${CONCURRENCY:-20}"
BATCH_SIZE="${BATCH_SIZE:-100}"
REASONING_EFFORT="${REASONING_EFFORT:-auto}"

CMD=(
    python eval/evaluate_cail_charge_openrouter.py
    --model "$MODEL"
    --input-file "$INPUT_FILE"
    --output-dir "$OUTPUT_DIR"
    --max-tokens "$MAX_TOKENS"
    --concurrency "$CONCURRENCY"
    --batch-size "$BATCH_SIZE"
    --reasoning-effort "$REASONING_EFFORT"
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
    CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "${SINGLE_ONLY:-0}" == "1" ]]; then
    CMD+=(--single-only)
fi

if [[ "${NO_RESUME:-0}" == "1" ]]; then
    CMD+=(--no-resume)
fi

echo "============================================================="
echo "CAIL2018-Small charge prediction"
echo "  model:       $MODEL"
echo "  input_file:  $INPUT_FILE"
echo "  output_dir:  $OUTPUT_DIR"
echo "  max_samples: ${MAX_SAMPLES:-ALL}"
echo "  reasoning:   $REASONING_EFFORT"
echo "============================================================="

"${CMD[@]}"
