#!/bin/bash
# =============================================================
#  scripts/eval_RQ2_2.sh — RQ2.2 Legal LLM Transfer 单卡 vLLM 评估
#
#  对 pdp4k 的 test split (4630 条) × 8 次重复评估单个本地模型，
#  目标硬件: 单卡 4090 48GB (TP=1)。每次仅评估一个模型，
#  RQ2.2 的 6 个 base/legal 模型需循环调用本脚本 6 次。
#
#  说明:
#    - num_repeats=8 时启用 (sample_index, run_id) 断点；
#      每个 run 使用 seed=base_seed+run_id，独立但可复现。
#    - prefix caching 默认开启，重复 run 共享 prefill。
#    - 解码参数采用 vLLM 默认值（不指定 --temperature 等），
#      避免推理模型在 temperature=0 下出现退化。
#
#  用法:
#    bash scripts/eval_RQ2_2.sh /path/to/Qwen2.5-7B-Instruct
#    bash scripts/eval_RQ2_2.sh /path/to/LegalDelta
#
#  可选环境变量覆盖默认值:
#    NUM_REPEATS=8  MAX_MODEL_LEN=8192  MAX_TOKENS=4096
#    GPU_UTIL=0.92  BATCH_SIZE=200      DTYPE=auto
#    OUTPUT_DIR=results/RQ2_2           SEED=42
#    SPLIT=test                          VARIANT=original
#
#  快速 sanity check（先用小重复跑通流程）:
#    NUM_REPEATS=1 BATCH_SIZE=50 bash scripts/eval_RQ2_2.sh /path/to/model
# =============================================================

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "错误: 需要且仅需要一个模型路径参数"
    echo "用法: bash scripts/eval_RQ2_2.sh /path/to/model"
    exit 1
fi

MODEL_PATH="$1"
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "警告: 模型路径不是目录: $MODEL_PATH"
    echo "  （如果是 HuggingFace hub id，请确认权重已通过 HF_HOME 缓存到本地）"
fi

# ---- 参数配置（环境变量可覆盖） ----
DATA_PATH="${DATA_PATH:-data/pdp4k}"
SPLIT="${SPLIT:-test}"
NUM_REPEATS="${NUM_REPEATS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TP_SIZE="${TP_SIZE:-1}"
GPU_UTIL="${GPU_UTIL:-0.92}"
BATCH_SIZE="${BATCH_SIZE:-200}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-results/RQ2_2}"
SEED="${SEED:-42}"
VARIANT="${VARIANT:-original}"

MODEL_NAME="$(basename "$MODEL_PATH")"

echo "============================================================="
echo "RQ2.2 Legal LLM Transfer 评估 (单卡 4090 48GB)"
echo "  模型:          $MODEL_PATH ($MODEL_NAME)"
echo "  数据:          $DATA_PATH [$SPLIT]"
echo "  num_repeats:   $NUM_REPEATS"
echo "  max_model_len: $MAX_MODEL_LEN, max_tokens: $MAX_TOKENS"
echo "  TP:            $TP_SIZE, gpu_util: $GPU_UTIL"
echo "  batch_size:    $BATCH_SIZE, dtype: $DTYPE"
echo "  prompt_variant:$VARIANT"
echo "  base_seed:     $SEED  (run i 使用 seed=$SEED+i)"
echo "  output_dir:    $OUTPUT_DIR"
echo "============================================================="

python eval/evaluate_vllm.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --split "$SPLIT" \
    --num-repeats "$NUM_REPEATS" \
    --seed "$SEED" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --batch-size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-variant "$VARIANT"

echo "[完成] $MODEL_NAME → $OUTPUT_DIR"
