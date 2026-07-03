#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_hf_dataset.py

将 data/public_sources/Stage4/dataset_final.json 转换为 HuggingFace datasets
Arrow 格式，保存为本地 PDP-Bench 数据集。默认包含 test split；若提供分层采样子集
（见 sample_test_rq2.py），额外包含 test_rq2 split。

Usage:
    python data_process/convert_to_hf_dataset.py
    python data_process/convert_to_hf_dataset.py \
        --input-file data/public_sources/Stage4/dataset_final.json \
        --output-dir data/PDP-Bench
    # 若存在 test_rq2 JSON（可先运行 sample_test_rq2.py），会额外写入 split=test_rq2
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

try:
    from datasets import Dataset, DatasetDict, Features, Sequence, Value
except ImportError as exc:
    raise SystemExit("请先安装依赖: pip install datasets") from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURES = Features(
    {
        "id": Value("string"),
        "meta": {
            "date": Value("string"),
            "province": Value("string"),
        },
        "person_info": Value("string"),
        "procedure": Value("string"),
        "fact": Value("string"),
        "relevant_articles": Sequence(Value("string")),
        "decision": Value("string"),
        "raw_reasoning_and_decision": Value("string"),
        "source_url": Value("string"),
    }
)


def _normalize_item(item: dict) -> dict:
    """将原始 JSON 记录规范化为固定 schema。"""
    meta = item.get("meta") or {}
    return {
        "id": item.get("id", ""),
        "meta": {
            "date": meta.get("date", ""),
            "province": meta.get("province", ""),
        },
        "person_info": item.get("person_info", ""),
        "procedure": item.get("procedure", ""),
        "fact": item.get("fact", ""),
        "relevant_articles": item.get("relevant_articles") or [],
        "decision": item.get("decision", ""),
        "raw_reasoning_and_decision": item.get("raw_reasoning_and_decision", ""),
        "source_url": item.get("source_url", ""),
    }


def load_json(input_file: Path) -> list[dict]:
    """读取 dataset_final.json 并处理缺失值。"""
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"输入文件应为 JSON list: {input_file}")

    return [_normalize_item(item) for item in data]


def print_summary(dataset_dict: DatasetDict, output_dir: Path) -> None:
    """打印转换后的数据集摘要。"""
    print("\n" + "=" * 60)
    print("  PDP-Bench 数据集转换完成")
    print("=" * 60)
    for split_name, dataset in dataset_dict.items():
        decisions = Counter(dataset["decision"])
        print(f"  {split_name}: {len(dataset)} 条")
        for decision, count in sorted(decisions.items(), key=lambda x: -x[1]):
            print(f"    {decision}: {count}")
    print(f"\n  保存路径: {output_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 dataset_final.json 转换为 PDP-Bench HuggingFace datasets 格式"
    )
    parser.add_argument(
        "--input-file",
        default="data/public_sources/Stage4/dataset_final.json",
        help=(
            "输入 JSON 文件 "
            "(default: data/public_sources/Stage4/dataset_final.json)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/PDP-Bench",
        help="输出目录 (default: data/PDP-Bench)",
    )
    parser.add_argument(
        "--test-rq2-file",
        default="data/public_sources/Stage4/test_rq2_100.json",
        help=(
            "RQ2 分层采样子集；若文件存在则写入 split=test_rq2 "
            "(default: data/public_sources/Stage4/test_rq2_100.json)"
        ),
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_file.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    logger.info(f"读取数据: {input_file}")
    records = load_json(input_file)
    logger.info(f"读取完成: {len(records)} 条")

    splits: dict[str, Dataset] = {
        "test": Dataset.from_list(records, features=FEATURES),
    }

    rq2_path = Path(args.test_rq2_file)
    if rq2_path.is_file():
        rq2_records = load_json(rq2_path)
        logger.info(
            "加载 test_rq2: %s (%d 条)",
            rq2_path,
            len(rq2_records),
        )
        splits["test_rq2"] = Dataset.from_list(rq2_records, features=FEATURES)
    else:
        logger.warning(
            "未找到 test_rq2 文件 %s，跳过 split=test_rq2（可运行 "
            "python data_process/RQ2/sample_test_rq2.py 生成）",
            rq2_path,
        )

    dataset_dict = DatasetDict(splits)

    dataset_dict.save_to_disk(str(output_dir))
    logger.info(f"数据集已保存至: {output_dir}")
    print_summary(dataset_dict, output_dir)


if __name__ == "__main__":
    main()

