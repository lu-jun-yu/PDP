#!/bin/bash
# =============================================================
#  scripts/eval_RQ3_sft.sh - RQ3 SFT checkpoint 评估 (单卡 vLLM)
#
#  参考 scripts/eval_RQ3.sh 的结构与参数配置，用于评估
#  scripts/train_RQ3_sft.sh 产出的 SFT cold-start checkpoint。
#
#  默认参数与训练脚本对齐：
#    - model basename:  Qwen3-8B
#    - train split:     train
#    - seed:            42
#    - prompt_variant:  original
#    - default run:     RQ3_SFT_Qwen3-8B_train_seed42_original
#
#  默认评估 data/pdp4k 的 test split；通过 SPLIT=test_rq2 可切换到
#  类别均衡诊断集。若训练目录根部缺少最终权重，则回退到最新 checkpoint-N。
#
#  Usage:
#    bash scripts/eval_RQ3_sft.sh
#    bash scripts/eval_RQ3_sft.sh RQ3_SFT_Qwen3-8B_train_seed42_original
#    SFT_RUNS="run_a run_b" bash scripts/eval_RQ3_sft.sh
#    CHECKPOINT_PATH=checkpoints/RQ3_SFT_Qwen3-8B_train_seed42_original bash scripts/eval_RQ3_sft.sh
#    SPLIT=test_rq2 NUM_REPEATS=8 bash scripts/eval_RQ3_sft.sh
#    DRY_RUN=1 bash scripts/eval_RQ3_sft.sh
# =============================================================

set -euo pipefail

# ---- Naming parameters aligned with scripts/train_RQ3_sft.sh ----
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
SEED="${SEED:-42}"
PROMPT_VARIANT="${PROMPT_VARIANT:-original}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"

DEFAULT_RUN_NAME="RQ3_SFT_${MODEL_NAME}_${TRAIN_SPLIT}_seed${SEED}_${PROMPT_VARIANT}"

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    SFT_RUNS=("$CHECKPOINT_PATH")
elif [[ $# -gt 0 ]]; then
    SFT_RUNS=("$@")
elif [[ -n "${SFT_RUNS:-}" ]]; then
    read -r -a SFT_RUNS <<< "${SFT_RUNS}"
else
    SFT_RUNS=("$DEFAULT_RUN_NAME")
fi

# ---- Evaluation parameters aligned with scripts/eval_RQ3.sh ----
DATA_PATH="${DATA_PATH:-data/pdp4k}"
SPLIT="${SPLIT:-test}"
NUM_REPEATS="${NUM_REPEATS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TP_SIZE="${TP_SIZE:-1}"
GPU_UTIL="${GPU_UTIL:-0.92}"
BATCH_SIZE="${BATCH_SIZE:-200}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-results/RQ3_SFT}"

echo "============================================================="
echo "RQ3 SFT checkpoint evaluation (单卡 vLLM)"
echo "  model_name:      $MODEL_NAME"
echo "  train_split:     $TRAIN_SPLIT"
echo "  runs:            ${SFT_RUNS[*]}"
echo "  seed:            $SEED  (run i 使用 seed=$SEED+i)"
echo "  prompt_variant:  $PROMPT_VARIANT"
echo "  checkpoint_root: $CHECKPOINT_ROOT"
echo "  data:            $DATA_PATH [$SPLIT]"
echo "  num_repeats:     $NUM_REPEATS"
echo "  max_model_len:   $MAX_MODEL_LEN, max_tokens: $MAX_TOKENS"
echo "  TP:              $TP_SIZE, gpu_util: $GPU_UTIL"
echo "  batch_size:      $BATCH_SIZE, dtype: $DTYPE"
echo "  output_dir:      $OUTPUT_DIR"
echo "============================================================="

if [[ ! -d "$DATA_PATH" ]]; then
    echo "错误: 数据集目录不存在: $DATA_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

find_last_inner_checkpoint() {
    local dir="$1"
    local last="" last_step=-1
    for sub in "$dir"/checkpoint-*/; do
        [[ -d "$sub" ]] || continue
        local name="${sub%/}"
        local step="${name##*-}"
        if [[ "$step" =~ ^[0-9]+$ ]] && (( step > last_step )); then
            last_step=$step
            last="$name"
        fi
    done
    printf '%s' "$last"
}

has_model_weights() {
    local dir="$1"
    [[ -f "$dir/model.safetensors" ]] \
        || [[ -f "$dir/model.safetensors.index.json" ]] \
        || [[ -f "$dir/pytorch_model.bin" ]] \
        || [[ -f "$dir/pytorch_model.bin.index.json" ]]
}

resolve_checkpoint_dir() {
    local run="$1"
    if [[ -d "$run" ]]; then
        printf '%s' "$run"
    else
        printf '%s' "${CHECKPOINT_ROOT}/${run}"
    fi
}

overall_fail=0
for RUN in "${SFT_RUNS[@]}"; do
    CKPT_DIR="$(resolve_checkpoint_dir "$RUN")"

    if [[ ! -d "$CKPT_DIR" ]]; then
        echo
        echo "[跳过] run=$RUN - checkpoint 目录不存在: $CKPT_DIR"
        overall_fail=1
        continue
    fi

    if has_model_weights "$CKPT_DIR"; then
        RESOLVED_MODEL_PATH="$CKPT_DIR"
    else
        RESOLVED_MODEL_PATH="$(find_last_inner_checkpoint "$CKPT_DIR")"
        if [[ -z "$RESOLVED_MODEL_PATH" ]]; then
            echo
            echo "[跳过] run=$RUN - $CKPT_DIR 中未找到权重文件或 checkpoint-* 子目录"
            overall_fail=1
            continue
        fi
        echo "[提示] run=$RUN 根部无最终权重，回退到中间 checkpoint: $RESOLVED_MODEL_PATH"
    fi

    echo
    echo "-------------------------------------------------------------"
    echo "[启动] run=$RUN"
    echo "  checkpoint: $RESOLVED_MODEL_PATH"
    echo "  split:      $SPLIT  (repeats=$NUM_REPEATS)"
    echo "-------------------------------------------------------------"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN=1，仅打印不执行:"
        echo "python eval/evaluate_vllm.py --model-path \"$RESOLVED_MODEL_PATH\" --data-path \"$DATA_PATH\" --split \"$SPLIT\" --num-repeats \"$NUM_REPEATS\" --seed \"$SEED\" --max-model-len \"$MAX_MODEL_LEN\" --max-tokens \"$MAX_TOKENS\" --tensor-parallel-size \"$TP_SIZE\" --gpu-memory-utilization \"$GPU_UTIL\" --batch-size \"$BATCH_SIZE\" --dtype \"$DTYPE\" --output-dir \"$OUTPUT_DIR\" --prompt-variant \"$PROMPT_VARIANT\""
        continue
    fi

    if python eval/evaluate_vllm.py \
        --model-path "$RESOLVED_MODEL_PATH" \
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
        --prompt-variant "$PROMPT_VARIANT"; then
        echo "[完成] run=$RUN"
    else
        echo "[失败] run=$RUN"
        overall_fail=1
    fi
done

if [[ "$overall_fail" -ne 0 ]]; then
    echo
    echo "存在评估失败或缺失 checkpoint，请检查上方日志。"
    exit 1
fi

echo
echo "[完成] RQ3 SFT 全部 checkpoint 评估完成 -> $OUTPUT_DIR"
