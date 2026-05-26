#!/bin/bash
# =============================================================
#  scripts/eval_RQ3.sh — RQ3 DAPO checkpoint 评估 (单卡 vLLM)
#
#  对 scripts/train_RQ3_dapo.sh 产出的 RQ3 训练组 checkpoint 逐组评估，
#  用于复现 §4.3 "Class-wise F1 under Target-Ratio Intervention" 主图。
#
#  默认评估自然测试集 (data/pdp4k 的 test split，4630 条)，对应
#  §4.3.3 中的「自然测试集」结果；通过 SPLIT=test_rq2 可切换至 100 条
#  类别均衡诊断集（25 条/类），用于减少测试分布对类别级比较的干扰。
#
#  默认参数与 scripts/train_RQ3_dapo.sh 对齐：
#    - model basename:  Qwen3-8B
#    - seed:            42
#    - prompt_variant:  original
#    - groups:          natural balanced DNP-40 DNP-55 SNP-40 SNP-55 IENP-40 IENP-55
#
#  单卡 4090 48GB (TP=1)，按组顺序评估；trl 训练末尾会调用
#  `trainer.save_model(output_dir)` 将最终权重落到 checkpoint 目录根部，
#  脚本直接将该目录作为 --model-path。若根部缺少权重（训练中断），
#  则回退到最新的 checkpoint-N 子目录。
#
#  用法：
#    bash scripts/eval_RQ3.sh                              # 默认 8 组 × test split
#    bash scripts/eval_RQ3.sh DNP-40 DNP-55                # 仅评估指定组
#    RQ3_GROUPS="balanced DNP-40" bash scripts/eval_RQ3.sh
#    SPLIT=test_rq2 NUM_REPEATS=8 bash scripts/eval_RQ3.sh # 均衡诊断 + 多重复
#
#  常用覆盖：
#    MODEL_NAME=Qwen3-8B CHECKPOINT_ROOT=checkpoints/RQ3_DAPO \
#      SEED=42 PROMPT_VARIANT=original bash scripts/eval_RQ3.sh
#    DRY_RUN=1 bash scripts/eval_RQ3.sh                    # 只打印命令、不执行
# =============================================================

set -euo pipefail

# ---- 与训练脚本对齐的命名参数（影响 checkpoint 路径解析） ----
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
SEED="${SEED:-42}"
PROMPT_VARIANT="${PROMPT_VARIANT:-original}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/RQ3_DAPO}"

# ---- 训练组（与 train_RQ3_dapo.sh DEFAULT_GROUPS 保持一致） ----
# 如需扩展 P-40 / P-55，先在训练脚本里跑通，再 RQ3_GROUPS="P-40 P-55" 调用本脚本。
# 勿用 GROUPS：与 bash 内建 $GROUPS（用户 gid 数组）冲突。
DEFAULT_GROUPS="natural balanced DNP-40 DNP-55 SNP-40 SNP-55 IENP-40 IENP-55"
if [[ $# -gt 0 ]]; then
    RQ3_GROUPS="$*"
else
    RQ3_GROUPS="${RQ3_GROUPS:-$DEFAULT_GROUPS}"
fi

# ---- 评估参数（环境变量可覆盖） ----
DATA_PATH="${DATA_PATH:-data/pdp4k}"
SPLIT="${SPLIT:-test}"                       # test=自然测试集；test_rq2=均衡诊断
NUM_REPEATS="${NUM_REPEATS:-1}"              # 单测自然集；test_rq2 上推荐 8
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TP_SIZE="${TP_SIZE:-1}"
GPU_UTIL="${GPU_UTIL:-0.92}"
BATCH_SIZE="${BATCH_SIZE:-200}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-results/RQ3}"

echo "============================================================="
echo "RQ3 DAPO checkpoint evaluation (单卡 vLLM)"
echo "  model_name:      $MODEL_NAME"
echo "  groups:          $RQ3_GROUPS"
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

# 在某个 checkpoint 目录下查找最新的 checkpoint-N 子目录（按 step 数值排序）
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

overall_fail=0
for GROUP in $RQ3_GROUPS; do
    GROUP_SAFE="${GROUP//-/_}"
    RUN_NAME="RQ3_DAPO_${MODEL_NAME}_${GROUP_SAFE}_seed${SEED}_${PROMPT_VARIANT}"
    CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"

    if [[ ! -d "$CKPT_DIR" ]]; then
        echo
        echo "[跳过] group=$GROUP — checkpoint 目录不存在: $CKPT_DIR"
        overall_fail=1
        continue
    fi

    # trl 训练末尾 `save_model(output_dir)` 会把最终权重落到 CKPT_DIR 根部；
    # 中间 checkpoint-N 仅作恢复用。若根部缺少权重则回退到最新 checkpoint-N。
    if [[ -f "$CKPT_DIR/model.safetensors" ]] \
        || [[ -f "$CKPT_DIR/model.safetensors.index.json" ]] \
        || [[ -f "$CKPT_DIR/pytorch_model.bin" ]] \
        || [[ -f "$CKPT_DIR/pytorch_model.bin.index.json" ]]; then
        MODEL_PATH="$CKPT_DIR"
    else
        MODEL_PATH="$(find_last_inner_checkpoint "$CKPT_DIR")"
        if [[ -z "$MODEL_PATH" ]]; then
            echo "[跳过] group=$GROUP — $CKPT_DIR 中未找到权重文件或 checkpoint-* 子目录"
            overall_fail=1
            continue
        fi
        echo "[提示] group=$GROUP 根部无最终权重，回退到中间 checkpoint: $MODEL_PATH"
    fi

    echo
    echo "-------------------------------------------------------------"
    echo "[启动] group=$GROUP"
    echo "  checkpoint: $MODEL_PATH"
    echo "  split:      $SPLIT  (repeats=$NUM_REPEATS)"
    echo "-------------------------------------------------------------"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN=1，仅打印不执行"
        continue
    fi

    if python eval/evaluate_vllm.py \
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
        --prompt-variant "$PROMPT_VARIANT"; then
        echo "[完成] group=$GROUP"
    else
        echo "[失败] group=$GROUP"
        overall_fail=1
    fi
done

if [[ "$overall_fail" -ne 0 ]]; then
    echo
    echo "存在评估失败或缺失 checkpoint，请检查上方日志。"
    exit 1
fi

echo
echo "[完成] RQ3 全部 checkpoint 评估完成 → $OUTPUT_DIR"
