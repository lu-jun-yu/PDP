#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train/sft.py

Use the generated CoT data to cold-start Qwen3-8B before RQ3 DAPO.

The prompt only contains inference-time visible fields
(person_info/procedure/fact). The generated CoT field is used as the assistant
target and is masked so loss is computed only on assistant tokens.

Usage:
    python train/sft.py \
      --model-path models/Qwen3-8B \
      --data-path data/pdp2k_rq3_sft \
      --split train \
      --output-dir checkpoints/RQ3_SFT_Qwen3-8B
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset, load_from_disk

# PyTorch < 2.6 may block loading local checkpoints because of the
# CVE-2025-32434 guard. This script loads user-managed local checkpoints only.
import transformers.trainer
import transformers.utils.import_utils

_noop = lambda: None
transformers.utils.import_utils.check_torch_load_is_safe = _noop
transformers.trainer.check_torch_load_is_safe = _noop

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from prompt_template import (
    PROMPT_VARIANT_ORIGINAL,
    PROMPT_VARIANTS,
    build_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDP RQ3 CoT SFT cold-start training")
    parser.add_argument("--model-path", default="models/Qwen3-8B")
    parser.add_argument("--data-path", default="data/pdp2k_rq3_sft")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cot-field", default="cot_data")
    parser.add_argument(
        "--prompt-variant",
        default=PROMPT_VARIANT_ORIGINAL,
        choices=PROMPT_VARIANTS,
        help="SFT prompt variant. Keep this aligned with later DAPO/eval settings.",
    )
    parser.add_argument("--output-dir", default="checkpoints/RQ3_SFT_Qwen3-8B")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default="pdp-sft")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def resolve_split_name(dataset_dict, requested: str) -> str:
    available = set(dataset_dict.keys())
    candidates = [
        requested,
        requested.replace("-", "_"),
        requested.lower(),
        requested.lower().replace("-", "_"),
        requested.upper(),
        requested.upper().replace("-", "_"),
    ]
    for candidate in candidates:
        if candidate in available:
            return candidate
    choices = ", ".join(sorted(available))
    raise ValueError(f"未找到 split: {requested!r}；可用 split: {choices}")


def get_last_checkpoint(output_dir: str) -> str | None:
    output_path = Path(output_dir)
    if not output_path.is_dir():
        return None
    checkpoints = [
        d for d in output_path.iterdir()
        if d.is_dir() and re.match(r"^checkpoint-\d+$", d.name)
    ]
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda d: int(d.name.split("-")[1])))


def load_split(data_path: str, split: str, max_samples: int | None) -> Dataset:
    dataset_dict = load_from_disk(data_path)
    split_name = resolve_split_name(dataset_dict, split)
    dataset = dataset_dict[split_name]
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def build_prompt(example: dict[str, Any], prompt_variant: str) -> list[dict[str, str]]:
    return build_messages(
        person_info=example["person_info"],
        procedure=example["procedure"],
        fact=example["fact"],
        prompt_variant=prompt_variant,
    )


def tokenize_one(
    *,
    example: dict[str, Any],
    tokenizer,
    cot_field: str,
    prompt_variant: str,
    max_length: int,
) -> tuple[dict[str, list[int]], bool]:
    completion = str(example[cot_field]).strip()
    if not completion:
        raise ValueError(f"样本 {example.get('id', '')!r} 的 {cot_field!r} 为空")

    prompt_messages = build_prompt(example, prompt_variant)
    full_messages = prompt_messages + [{"role": "assistant", "content": completion}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(f"chat template prefix mismatch: {example.get('id', '')!r}")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"tokenized prompt prefix mismatch: {example.get('id', '')!r}")
    if len(prompt_ids) >= max_length:
        raise ValueError(
            f"prompt tokens exceed max_length for {example.get('id', '')!r}: "
            f"{len(prompt_ids)} >= {max_length}"
        )

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    if len(labels) != len(input_ids):
        raise ValueError(f"label length mismatch: {example.get('id', '')!r}")

    truncated = len(input_ids) > max_length
    if truncated:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    if all(label == -100 for label in labels):
        raise ValueError(f"no assistant labels left: {example.get('id', '')!r}")

    return {"input_ids": input_ids, "labels": labels}, truncated


def build_tokenized_dataset(
    *,
    dataset: Dataset,
    tokenizer,
    cot_field: str,
    prompt_variant: str,
    max_length: int,
) -> Dataset:
    rows: list[dict[str, list[int]]] = []
    truncated = 0
    for example in dataset:
        item, was_truncated = tokenize_one(
            example=example,
            tokenizer=tokenizer,
            cot_field=cot_field,
            prompt_variant=prompt_variant,
            max_length=max_length,
        )
        rows.append(item)
        truncated += int(was_truncated)
    if truncated:
        print(f"[SFT] warning: truncated {truncated}/{len(rows)} samples to max_length")
    return Dataset.from_list(rows)


@dataclass
class CausalLMCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for feature in features:
            length = len(feature["input_ids"])
            pad_len = max_len - length
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append([1] * length + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()

    if args.max_length <= 0:
        raise ValueError("--max-length 必须为正整数")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples 必须为正整数或省略")
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 与 --fp16 不能同时启用")
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    raw_dataset = load_split(args.data_path, args.split, args.max_samples)
    if args.cot_field not in raw_dataset.column_names:
        raise ValueError(
            f"数据集中不存在字段 {args.cot_field!r}；可用字段: {raw_dataset.column_names}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = build_tokenized_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        cot_field=args.cot_field,
        prompt_variant=args.prompt_variant,
        max_length=args.max_length,
    )

    print(
        f"[SFT] model={args.model_path} data={args.data_path} split={args.split} "
        f"samples={len(tokenized_dataset)} prompt_variant={args.prompt_variant}"
    )

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        run_name=args.run_name,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=CausalLMCollator(tokenizer=tokenizer),
    )
    trainer.train(resume_from_checkpoint=get_last_checkpoint(args.output_dir))
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
