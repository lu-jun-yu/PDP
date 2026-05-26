#!/bin/bash
# =============================================================
#  scripts/train_RQ3_dapo.sh — RQ3 DAPO 类别占比干预训练
#
#  默认设置：
#    - 数据集: data/pdp2k_rq3
#    - prompt: original
#    - 训练组: natural / balanced / P-40 / P-55 / DNP-40 / DNP-55 /
#             SNP-40 / SNP-55 / IENP-40 / IENP-55
#    - seed: 42
#
#  用法：
#    bash scripts/train_RQ3_dapo.sh
#    bash scripts/train_RQ3_dapo.sh DNP-40 DNP-55
#    RQ3_GROUPS="balanced DNP-40" bash scripts/train_RQ3_dapo.sh
#
#  常用覆盖：
#    MODEL_PATH=/path/to/Qwen3-8B DATA_PATH=data/pdp2k_rq3 bash scripts/train_RQ3_dapo.sh
#    PROMPT_VARIANT=original MAX_SAMPLES=32 DRY_RUN=1 bash scripts/train_RQ3_dapo.sh balanced
#
#  PyTorch 前向 OOM：减小 BATCH_SIZE / NUM_GENERATIONS，或加大 GRAD_ACCUM。
#  TRL + vLLM colocate：勿用 ZeRO-3（参数被分片为占位 tensor，vLLM load_weights 会报 Size([4096]) vs Size([0])）。
#  默认 ZeRO-2；仅训练、无 colocate vLLM 时再考虑 DEEPSPEED_CONFIG=configs/ds_zero3.json。
#  ZeRO-2 在 optimizer.step 处 CUDA OOM：可 OFFLOAD_OPTIMIZER=1（ZeRO-2 + 优化器 CPU offload），
#    需加主机内存或 swap（建议系统可用内存+swap 合计 ≥128GB 更稳；两 rank 仍会占不少 RAM）。
#    OFFLOAD_OPTIMIZER=1 bash scripts/train_RQ3_dapo.sh balanced
#  ZeRO-1 也支持优化器 offload：DEEPSPEED_CONFIG=configs/ds_zero1_offload.json（只分片优化器，与 ZeRO-2 offload 的 CPU/GPU 行为略有差别，可二选一试）。
#  vLLM “No available memory for the cache blocks”：略提高 VLLM_GPU_MEM_UTIL。
#    BATCH_SIZE=1 GRAD_ACCUM=64 NUM_GENERATIONS=4 bash scripts/train_RQ3_dapo.sh balanced
# =============================================================

set -euo pipefail

# 缓解显存碎片（PyTorch 日志提示 reserved 很大时可开）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- GPU / DeepSpeed ----
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    DETECTED_GPUS=0
fi
NUM_GPUS="${NUM_GPUS:-$DETECTED_GPUS}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "错误: 未检测到 GPU；如需手动指定，请设置 NUM_GPUS。"
    exit 1
fi

# 2×A100 + colocate vLLM：默认 ZeRO-2（与 vLLM 权重拷贝兼容）。ZeRO-3 与此路径不兼容，见文件头注释。
# OFFLOAD_OPTIMIZER=1 时默认改用 configs/ds_zero2_offload.json；仍可用 DEEPSPEED_CONFIG=... 完全覆盖。
_default_deepspeed="configs/ds_zero2.json"
if [[ "${OFFLOAD_OPTIMIZER:-0}" == "1" ]]; then
  _default_deepspeed="configs/ds_zero2_offload.json"
fi
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$_default_deepspeed}"

# ---- RQ3 数据与模型 ----
MODEL_PATH="${MODEL_PATH:-checkpoints/Qwen3-8B}"
DATA_PATH="${DATA_PATH:-data/pdp2k_rq3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/RQ3_DAPO}"
PROMPT_VARIANT="${PROMPT_VARIANT:-original}"

DEFAULT_GROUPS="DNP-40 DNP-55 SNP-40 SNP-55 IENP-40 IENP-55 natural" # balanced
# 勿用变量名 GROUPS：与 bash 内建只读数组 $GROUPS（用户 gid 列表）冲突，赋值会被忽略。
if [[ $# -gt 0 ]]; then
    RQ3_GROUPS="$*"
else
    RQ3_GROUPS="${RQ3_GROUPS:-$DEFAULT_GROUPS}"
fi
echo "[train_RQ3_dapo] argv_count=$# argv=[$*] resolved_RQ3_GROUPS=[$RQ3_GROUPS]"
SEED="${SEED:-42}"

# ---- 生成 ----
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1536}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
# GRPO 显存随 num_generations 近似线性放大；8B + vLLM colocate 在 80GB 上默认用 4 更稳
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"

# ---- vLLM ----
# 过低会触发 vLLM “No available memory for the cache blocks”；默认 0.25 与 ZeRO-2 折中。
# 若仍 PyTorch OOM 再降到 0.22 并减 BATCH_SIZE；若仍 vLLM cache 报错再升到 0.28～0.30。
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.50}"
# vLLM 张量并行度：colocate 模式下，每个 vLLM 实例的权重切到 TP 张卡，腾出空间给 KV cache。
# 约束：NUM_GPUS 必须能被 VLLM_TP 整除；NUM_GENERATIONS 必须是 (NUM_GPUS/VLLM_TP) 的倍数。
# 例：8 卡 + VLLM_TP=2 → DP=4，NUM_GENERATIONS ∈ {4, 8, 12, ...}。
VLLM_TP="${VLLM_TP:-2}"

# ---- 训练 ----
# ZeRO-2：优化器+梯度多分片，权重仍整卡一份（与 vLLM colocate 一致）。
# 2×A100 80GB：全局 batch ≈ BATCH_SIZE × NUM_GPUS × GRAD_ACCUM（默认 2×2×32=128）。
NUM_EPOCHS="${NUM_EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
LR="${LR:-1e-5}"

# ---- DAPO 超参 ----
EPSILON="${EPSILON:-0.2}"
EPSILON_HIGH="${EPSILON_HIGH:-0.28}"

# ---- 日志与保存 ----
LOGGING_STEPS="${LOGGING_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-30}"
WANDB_PROJECT="${WANDB_PROJECT:-PDP-RQ3}"

mkdir -p "$OUTPUT_ROOT"

echo "============================================================="
echo "RQ3 DAPO training"
echo "  model:          $MODEL_PATH"
echo "  data:           $DATA_PATH"
echo "  groups:         $RQ3_GROUPS"
echo "  seed:           $SEED"
echo "  prompt_variant: $PROMPT_VARIANT"
echo "  num_gpus:       $NUM_GPUS"
echo "  deepspeed:      $DEEPSPEED_CONFIG  (OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-0})"
echo "  output_root:    $OUTPUT_ROOT"
echo "  batch/accum:    $BATCH_SIZE / $GRAD_ACCUM  num_gen: $NUM_GENERATIONS  vllm_gpu_mem: $VLLM_GPU_MEM_UTIL"
echo "============================================================="

if [[ ! -d "$DATA_PATH" ]]; then
    echo "错误: 数据集目录不存在: $DATA_PATH"
    echo "请先确认 data/pdp2k_rq3 已生成。"
    exit 1
fi

MODEL_NAME="$(basename "$MODEL_PATH")"

for GROUP in $RQ3_GROUPS; do
    GROUP_SAFE="${GROUP//-/_}"
    RUN_NAME="RQ3_DAPO_${MODEL_NAME}_${GROUP_SAFE}_seed${SEED}_${PROMPT_VARIANT}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

    CMD=(
        deepspeed --num_gpus "$NUM_GPUS"
        train/dapo.py
        --deepspeed "$DEEPSPEED_CONFIG"
        --model-path "$MODEL_PATH"
        --data-path "$DATA_PATH"
        --split "$GROUP"
        --prompt-variant "$PROMPT_VARIANT"
        --output-dir "$OUTPUT_DIR"
        --max-completion-length "$MAX_COMPLETION_LENGTH"
        --max-prompt-length "$MAX_PROMPT_LENGTH"
        --num-generations "$NUM_GENERATIONS"
        --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
        --vllm-tensor-parallel-size "$VLLM_TP"
        --num-train-epochs "$NUM_EPOCHS"
        --per-device-train-batch-size "$BATCH_SIZE"
        --gradient-accumulation-steps "$GRAD_ACCUM"
        --learning-rate "$LR"
        --seed "$SEED"
        --epsilon "$EPSILON"
        --epsilon-high "$EPSILON_HIGH"
        --logging-steps "$LOGGING_STEPS"
        --save-steps "$SAVE_STEPS"
        --wandb-project "$WANDB_PROJECT"
        --run-name "$RUN_NAME"
    )

    if [[ -n "${MAX_SAMPLES:-}" ]]; then
        CMD+=(--max-samples "$MAX_SAMPLES")
    fi

    echo
    echo "-------------------------------------------------------------"
    echo "Start: group=$GROUP seed=$SEED"
    echo "Output: $OUTPUT_DIR"
    echo "Command: ${CMD[*]}"
    echo "-------------------------------------------------------------"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
    fi

    "${CMD[@]}"
done

echo
echo "[完成] RQ3 DAPO training jobs finished."
