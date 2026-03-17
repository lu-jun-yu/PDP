#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_hf_dataset.py

将 PDP_dataset/ 下 train/test 中的 dataset.json 转换为 HuggingFace datasets 格式，
保存为 Arrow 格式的数据集，命名为 "pdp10k"。

支持按 decision 类别进行分层采样过滤。

Usage:
    python convert_to_hf_dataset.py
    python convert_to_hf_dataset.py --input-dir data/PDP_dataset --output-dir data/pdp10k
"""

import os
import json
import random
import argparse
import logging
from collections import Counter

from datasets import Dataset, DatasetDict, Features, Value, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 数据集 schema 定义
FEATURES = Features({
    "id": Value("string"),
    "meta": {
        "year": Value("int32"),
        "province": Value("string"),
    },
    "person_info": Value("string"),
    "procedure": Value("string"),
    "fact": Value("string"),
    "relevant_articles": Sequence(Value("string")),
    "decision": Value("string"),
    "charges": Sequence(Value("string")),
    "raw_reasoning_and_decision": Value("string"),
})


# 每个 split 按 decision 类别的目标采样数
SPLIT_QUOTAS = {
    "train": {
        "起诉": 4096,
        "相对不起诉": 2048,
        "法定不起诉": 1024,
        "存疑不起诉": 1024,
    },
    "test": {
        "起诉": 1024,
        "相对不起诉": 512,
        "法定不起诉": 256,
        "存疑不起诉": 256,
    },
}


def stratified_sample(records: list[dict], quotas: dict[str, int], seed: int = 42) -> list[dict]:
    """按 decision 类别进行分层采样，返回采样后的记录列表。"""
    rng = random.Random(seed)

    # 按 decision 分组
    groups: dict[str, list[dict]] = {}
    for r in records:
        dec = r["decision"]
        groups.setdefault(dec, []).append(r)

    sampled = []
    for decision, quota in quotas.items():
        pool = groups.get(decision, [])
        if len(pool) < quota:
            logger.warning(
                f"  类别 '{decision}' 仅有 {len(pool)} 条，不足目标 {quota} 条，将全部保留"
            )
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, quota))

    # 检查是否有未在 quota 中出现的类别
    extra_decisions = set(groups.keys()) - set(quotas.keys())
    if extra_decisions:
        logger.warning(f"  以下类别不在过滤配额中，已丢弃: {extra_decisions}")

    rng.shuffle(sampled)
    return sampled


def load_json(json_path: str) -> list[dict]:
    """读取 dataset.json，处理缺失值。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        meta = item.get("meta", {})
        records.append({
            "id": item.get("id", ""),
            "meta": {
                "year": meta.get("year") or 0,
                "province": meta.get("province", ""),
            },
            "person_info": item.get("person_info", ""),
            "procedure": item.get("procedure", ""),
            "fact": item.get("fact", ""),
            "relevant_articles": item.get("relevant_articles", []),
            "decision": item.get("decision", ""),
            "charges": item.get("charges", []),
            "raw_reasoning_and_decision": item.get("raw_reasoning_and_decision", ""),
        })
    return records


def main():
    parser = argparse.ArgumentParser(
        description="将 PDP_dataset JSON 转换为 HuggingFace datasets 格式"
    )
    parser.add_argument(
        "--input-dir", default="data/PDP_dataset",
        help="输入目录，包含 train/test 子目录 (default: data/PDP_dataset)",
    )
    parser.add_argument(
        "--output-dir", default="data/pdp10k",
        help="输出目录 (default: data/pdp10k)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (default: 42)",
    )
    args = parser.parse_args()

    splits = {}
    split_names = ["train", "test"]

    for split in split_names:
        json_path = os.path.join(args.input_dir, split, "dataset.json")
        if not os.path.isfile(json_path):
            logger.warning(f"[{split}] 文件不存在: {json_path}，跳过")
            continue

        records = load_json(json_path)
        logger.info(f"[{split}] 原始数据 {len(records)} 条")

        # 分层采样过滤
        if split in SPLIT_QUOTAS:
            records = stratified_sample(records, SPLIT_QUOTAS[split], seed=args.seed)
            logger.info(f"[{split}] 过滤后 {len(records)} 条")

        ds = Dataset.from_list(records, features=FEATURES)
        splits[split] = ds

    if not splits:
        logger.error("未找到任何数据文件，退出")
        return

    dataset_dict = DatasetDict(splits)

    # 保存到磁盘
    dataset_dict.save_to_disk(args.output_dir)
    logger.info(f"数据集已保存至: {args.output_dir}")

    # 打印汇总
    print("\n" + "=" * 50)
    print("  pdp10k 数据集转换完成")
    print("=" * 50)
    for name, ds in dataset_dict.items():
        # 统计各类别分布
        decisions = [r["decision"] for r in ds]
        dist = Counter(decisions)
        print(f"  {name}: {len(ds)} 条")
        for dec, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {dec}: {cnt}")
    print(f"\n  保存路径: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
