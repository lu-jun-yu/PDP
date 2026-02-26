#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train/grpo.py

使用 trl GRPOTrainer + vLLM 对 PDP 数据集进行 GRPO (RLVR) 训练。

Usage:
    python train/grpo.py --model-path models/Qwen3-0.6B --data-path data/pdp25k
"""

import argparse
import sys
from pathlib import Path

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

from prompt_template import build_messages
from reward_function import (
    format_reward_func,
    decision_reward_func,
    process_reward_func,
    citation_reward_func,
)


def add_prompt(example):
    """将输入字段组装为 chat messages 格式的 prompt 列。"""
    example["prompt"] = build_messages(
        person_info=example["person_info"],
        procedure=example["procedure"],
        fact=example["fact"],
    )
    return example


def main():
    parser = argparse.ArgumentParser(description="PDP GRPO 训练脚本")
    parser.add_argument("--model-path", default="models/Qwen3-0.6B")
    parser.add_argument("--data-path", default="data/pdp25k")
    parser.add_argument("--output-dir", default="results/grpo")
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
    # 日志与保存
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    args = parser.parse_args()

    # ---- 数据集 ----
    dataset = load_from_disk(args.data_path)["train"]
    dataset = dataset.map(add_prompt)

    # ---- 训练配置 ----
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        num_generations=args.num_generations,
        use_vllm=True,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )

    # ---- 训练 ----
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=[
            format_reward_func,
            decision_reward_func,
            process_reward_func,
            citation_reward_func,
        ],
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
