#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_huggingface.py

将本地 pdp25k 数据集上传到 HuggingFace Hub。

前置条件:
    1. 先运行 convert_to_hf_dataset.py 生成 data/pdp25k
    2. 项目根目录下需有 token.json（含 hf_token 字段）或通过 --token 参数传入

Usage:
    python upload_to_huggingface.py --repo-id your-username/pdp25k
    python upload_to_huggingface.py --repo-id your-username/pdp25k --private
    python upload_to_huggingface.py --repo-id your-username/pdp25k --token hf_xxxxx
"""

import os
import json
import argparse
import logging

from datasets import DatasetDict
from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 数据集卡片模板
DATASET_CARD = """\
---
language:
- zh
license: apache-2.0
task_categories:
- text-classification
tags:
- legal
- chinese
- prosecution-decision
- criminal-law
size_categories:
- 10K<n<100K
---

# PDP-25K: Prosecution Decision Prediction Dataset

PDP-25K 是一个中文检察起诉决定预测数据集，包含约 25,000 条结构化的检察文书数据。

## 数据集结构

| Split | 样本数 | 说明 |
|-------|--------|------|
| train | 20k | 训练集 |
| test  | 4k | 测试集 |
| ood   | 1k | 域外测试集 |

## 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文书唯一标识 |
| `meta_year` | int | 文书年份 |
| `meta_province` | string | 省份 |
| `person_info` | string | 当事人信息（已脱敏） |
| `procedure` | string | 程序信息 |
| `fact` | string | 案件事实 |
| `relevant_articles_cl` | list[string] | 相关刑法条文 |
| `relevant_articles_cpl` | list[string] | 相关刑事诉讼法条文 |
| `decision` | string | 决定类型（起诉/相对不起诉/法定不起诉/存疑不起诉） |
| `charges` | list[string] | 罪名 |
| `raw_reasoning_and_decision` | string | 原始推理与决定文本 |
"""


def load_token(token_path: str) -> str | None:
    """从 token.json 中读取 hf_token。"""
    if not os.path.isfile(token_path):
        return None
    with open(token_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("hf_token")


def main():
    parser = argparse.ArgumentParser(
        description="将 pdp25k 数据集上传到 HuggingFace Hub"
    )
    parser.add_argument(
        "--dataset-dir", default="data/pdp25k",
        help="本地数据集目录 (default: data/pdp25k)",
    )
    parser.add_argument(
        "--repo-id", required=True,
        help="HuggingFace 仓库 ID，如 your-username/pdp25k",
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace API token（默认从 token.json 读取）",
    )
    parser.add_argument(
        "--token-file", default="token.json",
        help="token 文件路径 (default: token.json)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="是否将数据集设为私有",
    )
    args = parser.parse_args()

    # 获取 token
    token = args.token or load_token(args.token_file)
    if not token:
        logger.error(
            "未找到 HuggingFace token。"
            "请通过 --token 参数传入，或在 token.json 中配置 hf_token 字段"
        )
        return

    # 加载数据集
    if not os.path.isdir(args.dataset_dir):
        logger.error(f"数据集目录不存在: {args.dataset_dir}")
        logger.error("请先运行 convert_to_hf_dataset.py 生成数据集")
        return

    logger.info(f"加载数据集: {args.dataset_dir}")
    dataset = DatasetDict.load_from_disk(args.dataset_dir)

    for name, ds in dataset.items():
        logger.info(f"  {name}: {len(ds)} 条")

    # 创建仓库（如不存在）
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # 上传数据集卡片
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    logger.info("数据集卡片已上传")

    # 上传数据集
    logger.info(f"正在上传至 {args.repo_id} ...")
    dataset.push_to_hub(
        repo_id=args.repo_id,
        token=token,
        private=args.private,
    )

    print("\n" + "=" * 50)
    print("  上传完成!")
    print("=" * 50)
    print(f"  仓库: https://huggingface.co/datasets/{args.repo_id}")
    print("=" * 50)


if __name__ == "__main__":
    main()
