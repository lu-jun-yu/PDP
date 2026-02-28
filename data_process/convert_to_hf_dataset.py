#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_hf_dataset.py

将 PDP_dataset/ 下 train/test 中的 dataset.json 转换为 HuggingFace datasets 格式，
保存为 Arrow 格式的数据集，命名为 "pdp25k"。

Usage:
    python convert_to_hf_dataset.py
    python convert_to_hf_dataset.py --input-dir data/PDP_dataset --output-dir data/pdp25k
"""

import os
import json
import argparse
import logging

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
    "relevant_articles_cl": Sequence(Value("string")),
    "relevant_articles_cpl": Sequence(Value("string")),
    "relevant_articles_cpr": Sequence(Value("string")),
    "decision": Value("string"),
    "charges": Sequence(Value("string")),
    "raw_reasoning_and_decision": Value("string"),
})


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
            "relevant_articles_cl": item.get("relevant_articles_cl", []),
            "relevant_articles_cpl": item.get("relevant_articles_cpl", []),
            "relevant_articles_cpr": item.get("relevant_articles_cpr", []),
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
        "--output-dir", default="data/pdp25k",
        help="输出目录 (default: data/pdp25k)",
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
        ds = Dataset.from_list(records, features=FEATURES)
        splits[split] = ds
        logger.info(f"[{split}] 加载 {len(ds)} 条记录")

    if not splits:
        logger.error("未找到任何数据文件，退出")
        return

    dataset_dict = DatasetDict(splits)

    # 保存到磁盘
    dataset_dict.save_to_disk(args.output_dir)
    logger.info(f"数据集已保存至: {args.output_dir}")

    # 打印汇总
    print("\n" + "=" * 50)
    print("  pdp25k 数据集转换完成")
    print("=" * 50)
    for name, ds in dataset_dict.items():
        print(f"  {name}: {len(ds)} 条")
    print(f"\n  保存路径: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
