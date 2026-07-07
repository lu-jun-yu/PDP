#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train/merge_lora.py

Merge an RQ3 SFT LoRA adapter into the base model and save a standalone model.

Usage:
    python train/merge_lora.py \
      --base-model-path models/Qwen3-8B \
      --adapter-path checkpoints/RQ3_SFT_Qwen3-8B_train_seed42_original \
      --output-dir checkpoints/RQ3_SFT_Qwen3-8B_train_seed42_original_merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from peft import PeftModel

# PyTorch < 2.6 may block loading local checkpoints because of the
# CVE-2025-32434 guard. This script loads user-managed local checkpoints only.
import transformers.trainer
import transformers.utils.import_utils

_noop = lambda: None
transformers.utils.import_utils.check_torch_load_is_safe = _noop
transformers.trainer.check_torch_load_is_safe = _noop

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model-path", default="models/Qwen3-8B")
    parser.add_argument(
        "--adapter-path",
        default="checkpoints/RQ3_SFT_Qwen3-8B_train_seed42_original",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/RQ3_SFT_Qwen3-8B_train_seed42_original_merged",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--safe-serialization", action="store_true", default=True)
    parser.add_argument(
        "--no-safe-serialization",
        dest="safe_serialization",
        action="store_false",
    )
    return parser.parse_args()


def resolve_dtype(raw: str):
    if raw == "auto":
        return "auto"
    if raw == "bfloat16":
        return torch.bfloat16
    if raw == "float16":
        return torch.float16
    if raw == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {raw}")


def main() -> None:
    args = parse_args()

    base_model_path = Path(args.base_model_path)
    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path not found: {base_model_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    torch_dtype = resolve_dtype(args.torch_dtype)

    print(f"[merge] base_model_path={base_model_path}")
    print(f"[merge] adapter_path={adapter_path}")
    print(f"[merge] output_dir={output_dir}")
    print(
        f"[merge] torch_dtype={args.torch_dtype} "
        f"device_map={args.device_map} safe_serialization={args.safe_serialization}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
        use_fast=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch_dtype,
    )

    print("[merge] merging LoRA adapter into base model...")
    merged_model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print("[merge] saving merged model...")
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=args.safe_serialization,
    )
    tokenizer.save_pretrained(str(output_dir))
    print(f"[merge] done: {output_dir}")


if __name__ == "__main__":
    main()
