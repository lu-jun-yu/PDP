#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 train/dataset.json 按标签分层随机抽样固定规模池子，再按各 split 规格从前缀切分，
构建 HuggingFace DatasetDict 并保存到本地（仿 convert_to_hf_dataset.py 的 schema）。

池子规模（随机打乱后取前 N 条）:
  起诉 1731；相对不起诉、法定不起诉、存疑不起诉各 1100。

各 split 从对应池子顺序取前 k 条（起诉, 相对不起诉, 法定不起诉, 存疑不起诉）。

Usage（建议在 pytorch conda 环境中运行，并已安装 datasets）:
    conda activate pytorch
    python data_process/build_pdp2k_rq3_dataset.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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

LABEL_ORDER = ("起诉", "相对不起诉", "法定不起诉", "存疑不起诉")

POOL_SIZES: dict[str, int] = {
    "起诉": 1731,
    "相对不起诉": 1100,
    "法定不起诉": 1100,
    "存疑不起诉": 1100,
}

# (起诉, 相对不起诉, 法定不起诉, 存疑不起诉)
SPLIT_SPECS: dict[str, tuple[int, int, int, int]] = {
    "natural": (1731, 207, 18, 44),
    "balanced": (500, 500, 500, 500),
    "P_40": (800, 400, 400, 400),
    "P_55": (1100, 300, 300, 300),
    "DNP_40": (400, 800, 400, 400),
    "DNP_55": (300, 1100, 300, 300),
    "SNP_40": (400, 400, 800, 400),
    "SNP_55": (300, 300, 1100, 300),
    "IENP_40": (400, 400, 400, 800),
    "IENP_55": (300, 300, 300, 1100),
}

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
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"输入文件应为 JSON list: {input_file}")
    return [_normalize_item(item) for item in data]


def stratify_pools(
    records: list[dict], rng: random.Random
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {lab: [] for lab in LABEL_ORDER}
    unknown: list[str] = []
    for item in records:
        dec = item.get("decision", "")
        if dec in buckets:
            buckets[dec].append(item)
        else:
            unknown.append(dec)
    if unknown:
        u = Counter(unknown)
        raise ValueError(f"存在未识别的 decision 标签: {dict(u)}")

    for lab in LABEL_ORDER:
        need = POOL_SIZES[lab]
        avail = len(buckets[lab])
        if avail < need:
            raise ValueError(
                f"标签「{lab}」可用样本 {avail} 条，少于池子需求 {need} 条"
            )
        rng.shuffle(buckets[lab])
        buckets[lab] = buckets[lab][:need]
    return buckets


def build_split_rows(
    pools: dict[str, list[dict]],
    counts: tuple[int, int, int, int],
) -> list[dict]:
    rows: list[dict] = []
    for lab, n in zip(LABEL_ORDER, counts, strict=True):
        chunk = pools[lab][:n]
        if len(chunk) < n:
            raise ValueError(
                f"池子「{lab}」仅有 {len(chunk)} 条，需要 {n} 条"
            )
        rows.extend(chunk)
    return rows


def print_summary(dataset_dict: DatasetDict, output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("  pdp2k_rq3 数据集构建完成")
    print("=" * 60)
    for split_name, dataset in dataset_dict.items():
        decisions = Counter(dataset["decision"])
        print(f"  {split_name}: {len(dataset)} 条")
        for decision in LABEL_ORDER:
            if decision in decisions:
                print(f"    {decision}: {decisions[decision]}")
        extra = set(decisions) - set(LABEL_ORDER)
        for decision in sorted(extra):
            print(f"    {decision}: {decisions[decision]}")
    print(f"\n  保存路径: {output_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 train dataset.json 构建 pdp2k_rq3（多 split 标签比例）"
    )
    parser.add_argument(
        "--input-file",
        default="data/PDP_dataset/train/dataset.json",
        help="输入 JSON（default: data/PDP_dataset/train/dataset.json）",
    )
    parser.add_argument(
        "--output-dir",
        default="data/pdp2k_rq3",
        help="输出目录 (default: data/pdp2k_rq3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="分层抽样 shuffle 随机种子 (default: 42)",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="可选：将池子/sample id 摘要写入该 JSON 路径；默认 output-dir/manifest.json",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_dir / "manifest.json"
    )

    if not input_file.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    rng = random.Random(args.seed)
    logger.info("读取数据: %s", input_file)
    records = load_json(input_file)
    logger.info("读取完成: %d 条", len(records))

    pools = stratify_pools(records, rng)
    logger.info(
        "分层池子: %s",
        ", ".join(f"{k}={len(v)}" for k, v in pools.items()),
    )

    splits: dict[str, Dataset] = {}
    manifest: dict = {
        "seed": args.seed,
        "input_file": str(input_file.resolve()),
        "pool_sizes": POOL_SIZES,
        "pools": {
            lab: [pools[lab][i]["id"] for i in range(min(5, len(pools[lab])))]
            + (["..."] if len(pools[lab]) > 5 else [])
            for lab in LABEL_ORDER
        },
        "splits": {},
    }

    for split_name, counts in SPLIT_SPECS.items():
        rows = build_split_rows(pools, counts)
        splits[split_name] = Dataset.from_list(rows, features=FEATURES)
        manifest["splits"][split_name] = {
            "counts": dict(zip(LABEL_ORDER, counts, strict=True)),
            "total": len(rows),
        }

    dataset_dict = DatasetDict(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("数据集已保存至: %s", output_dir)
    logger.info("manifest: %s", manifest_path)
    print_summary(dataset_dict, output_dir)


if __name__ == "__main__":
    main()
