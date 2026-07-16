#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preview a few samples from data/pdp2k_rq3_sft.

Usage:
    python data_process/RQ3/preview_pdp2k_rq3_sft.py
    python data_process/RQ3/preview_pdp2k_rq3_sft.py --num-samples 5
    python data_process/RQ3/preview_pdp2k_rq3_sft.py --split train --full-text
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a few pdp2k_rq3_sft samples")
    parser.add_argument("--data-path", default="data/pdp2k_rq3_sft")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--text-limit", type=int, default=800)
    parser.add_argument("--full-text", action="store_true")
    return parser.parse_args()


def shorten(value, limit: int, full_text: bool):
    if full_text or not isinstance(value, str):
        return value
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + "\n...<truncated>"


def main() -> None:
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.text_limit < 0:
        raise ValueError("--text-limit must be non-negative")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    dataset_dict = load_from_disk(str(data_path))
    if args.split not in dataset_dict:
        raise ValueError(f"Split not found: {args.split}; available: {list(dataset_dict.keys())}")

    dataset = dataset_dict[args.split]
    count = min(args.num_samples, len(dataset))

    print(f"data_path: {data_path}")
    print(f"split: {args.split}")
    print(f"dataset_size: {len(dataset)}")
    print(f"preview_count: {count}")
    print("=" * 80)

    for i in range(count):
        sample = dict(dataset[i])
        sample = {
            key: shorten(value, args.text_limit, args.full_text)
            for key, value in sample.items()
        }
        print(f"[sample {i}]")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        print("=" * 80)


if __name__ == "__main__":
    main()
