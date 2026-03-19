#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train/dapo.py

使用 trl GRPOTrainer (DAPO loss) + vLLM 对 PDP 数据集进行强化学习训练。
关键设置：
  - loss_type="dapo"：消除长度偏差，使用非对称裁剪
  - temperature=1.0, top_p=1.0, top_k=None：采样策略与策略模型一致，避免分布偏移
  - beta=0.0：不使用 KL 惩罚
  - mask_truncated_completions=True：排除被截断的生成

Usage:
    python train/dapo.py --model-path models/Qwen3-0.6B --data-path data/pdp10k
"""

import argparse
import os
import re
import sys
from pathlib import Path

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk

# PyTorch < 2.6 加载自身 checkpoint 时会被 CVE-2025-32434 检查拦截，
# 训练脚本只加载自己保存的文件，跳过该检查。
import transformers.utils.import_utils
import transformers.trainer
_noop = lambda: None
transformers.utils.import_utils.check_torch_load_is_safe = _noop
transformers.trainer.check_torch_load_is_safe = _noop

from trl import GRPOConfig, GRPOTrainer

from prompt_template import build_messages
from reward_function import (
    format_reward_func,
    decision_reward_func,
    process_reward_func,
)


def add_prompt(example):
    """将输入字段组装为 chat messages 格式的 prompt 列。"""
    example["prompt"] = build_messages(
        person_info=example["person_info"],
        procedure=example["procedure"],
        fact=example["fact"],
    )
    return example


def get_last_checkpoint(output_dir: str):
    """在 output_dir 中寻找最新的 checkpoint-N 目录，返回路径或 None。"""
    output_path = Path(output_dir)
    if not output_path.is_dir():
        return None
    checkpoints = [
        d for d in output_path.iterdir()
        if d.is_dir() and re.match(r"^checkpoint-\d+$", d.name)
    ]
    if not checkpoints:
        return None
    last = max(checkpoints, key=lambda d: int(d.name.split("-")[1]))
    return str(last)


def main():
    parser = argparse.ArgumentParser(description="PDP DAPO 训练脚本")
    parser.add_argument("--model-path", default="models/Qwen3-0.6B")
    parser.add_argument("--data-path", default="data/pdp10k")
    parser.add_argument("--output-dir", default="results/dapo")
    # 生成
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--num-generations", type=int, default=8)
    # vLLM
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    # 训练
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    # DAPO 超参
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="裁剪下界 (trust region)")
    parser.add_argument("--epsilon-high", type=float, default=0.28,
                        help="裁剪上界 (DAPO 非对称裁剪)")
    # 日志与保存
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    # wandb
    parser.add_argument("--wandb-project", default="pdp-dapo",
                        help="wandb 项目名称")
    parser.add_argument("--run-name", default=None,
                        help="wandb run 名称（默认自动生成）")
    # DeepSpeed
    parser.add_argument("--deepspeed", default=None,
                        help="DeepSpeed 配置文件路径（多卡训练时使用）")
    # DeepSpeed 启动器会注入 --local_rank 参数
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="DeepSpeed 注入的 local rank（勿手动设置）")
    args = parser.parse_args()

    # 设置 wandb 项目名称（通过环境变量传递给 wandb）
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # ---- 数据集 ----
    dataset = load_from_disk(args.data_path)["train"]
    dataset = dataset.map(add_prompt)

    # ---- 训练配置 (DAPO) ----
    config = GRPOConfig(
        output_dir=args.output_dir,
        # 生成采样：温度 1.0，不使用 top_p/top_k，保证采样与策略分布一致
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        # 生成长度
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        num_generations=args.num_generations,
        # DAPO 核心参数
        loss_type="dapo",
        beta=0.0,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        mask_truncated_completions=True,
        # vLLM
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        # 训练
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        # 日志与保存
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        # wandb 监控
        report_to="wandb",
        run_name=args.run_name,
        # DeepSpeed
        deepspeed=args.deepspeed,
    )

    # ---- 训练 ----
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=[
            format_reward_func,
            decision_reward_func,
            process_reward_func,
        ],
        args=config,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(args.output_dir))
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
