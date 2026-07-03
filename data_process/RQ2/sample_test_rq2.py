#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 dataset_final.json 中按决定类别分层均匀采样，生成 RQ2 用 test_rq2 子集 JSON。

每类固定采样 per_class 条（默认 25），共 4 类 → 100 条。使用固定随机种子保证可复现。

Usage:
    python data_process/sample_test_rq2.py
    python data_process/sample_test_rq2.py --per-class 25 --seed 42 \\
        --output data/public_sources/Stage4/test_rq2_100.json
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSES = ("起诉", "相对不起诉", "法定不起诉", "存疑不起诉")


def main() -> None:
    parser = argparse.ArgumentParser(description="分层采样生成 test_rq2 子集")
    parser.add_argument(
        "--input",
        default="data/public_sources/Stage4/dataset_final.json",
        help="全量测试集 JSON",
    )
    parser.add_argument(
        "--output",
        default="data/public_sources/Stage4/test_rq2_100.json",
        help="输出 JSON（100 条列表）",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=25,
        help="每个决定类别采样条数 (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"输入不存在: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入应为 JSON 数组")

    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        dec = item.get("decision")
        if dec in CLASSES:
            by_class[dec].append(item)

    rng = random.Random(args.seed)
    sampled: list[dict] = []
    for cls in CLASSES:
        pool = by_class[cls]
        k = args.per_class
        if len(pool) < k:
            raise ValueError(
                f"类别「{cls}」仅有 {len(pool)} 条，不足每类采样 {k} 条"
            )
        chosen = rng.sample(pool, k)
        sampled.extend(chosen)
        logger.info("类别 %s: 从 %d 条中采样 %d 条", cls, len(pool), k)

    rng.shuffle(sampled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    logger.info("已写入 %d 条 → %s", len(sampled), output_path)


if __name__ == "__main__":
    main()
