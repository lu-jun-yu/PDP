#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_huggingface.py

将本地 PDP-Bench 数据集上传到 HuggingFace Hub，并根据本地数据动态生成
dataset card。默认包含 test split，可附带 test_rq2 split。

前置条件:
    1. 先运行 data_process/convert_to_hf_dataset.py 生成 data/PDP-Bench
    2. 项目根目录下需有 token.json（含 hf_token 字段）或通过 --token 参数传入

Usage:
    python data_process/upload_to_huggingface.py --repo-id your-username/PDP-Bench
    python data_process/upload_to_huggingface.py --repo-id your-username/PDP-Bench --private
    python data_process/upload_to_huggingface.py --repo-id your-username/PDP-Bench --token hf_xxxxx
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

try:
    from datasets import DatasetDict
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit("请先安装依赖: pip install datasets huggingface_hub") from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DECISION_ORDER = ["起诉", "相对不起诉", "法定不起诉", "存疑不起诉"]


def load_token(token_path: str) -> str | None:
    """从 token.json 中读取 hf_token。"""
    token_file = Path(token_path)
    if not token_file.is_file():
        return None
    with token_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("hf_token")


def _split_summary(dataset: DatasetDict) -> list[dict]:
    """统计每个 split 的样本数和决定类别分布。"""
    summary = []
    for split_name, split_dataset in dataset.items():
        summary.append(
            {
                "split": split_name,
                "num_rows": len(split_dataset),
                "decisions": Counter(split_dataset["decision"]),
            }
        )
    return summary


def _meta_summary(dataset: DatasetDict) -> dict:
    """统计日期、省份、来源 URL 和法条前缀等数据卡信息。"""
    dates = []
    provinces = Counter()
    source_url_count = 0
    article_count = 0
    article_prefixes = Counter()

    for split_dataset in dataset.values():
        for row in split_dataset:
            meta = row.get("meta") or {}
            date = meta.get("date", "")
            province = meta.get("province", "")
            if date:
                dates.append(date)
            if province:
                provinces[province] += 1
            if row.get("source_url"):
                source_url_count += 1

            for article in row.get("relevant_articles") or []:
                article_count += 1
                prefix = article.split(":", 1)[0] if ":" in article else "unknown"
                article_prefixes[prefix] += 1

    return {
        "date_min": min(dates) if dates else "",
        "date_max": max(dates) if dates else "",
        "province_count": len(provinces),
        "top_provinces": provinces.most_common(10),
        "source_url_count": source_url_count,
        "article_count": article_count,
        "article_prefixes": article_prefixes,
    }


def _format_int(value: int) -> str:
    return f"{value:,}"


def _build_split_table(summary: list[dict]) -> str:
    lines = [
        "| Split | 样本数 | 起诉 | 相对不起诉 | 法定不起诉 | 存疑不起诉 |",
        "|-------|--------|------|-----------|-----------|-----------|",
    ]
    for item in summary:
        decisions = item["decisions"]
        row = [
            item["split"],
            _format_int(item["num_rows"]),
            *[_format_int(decisions.get(decision, 0)) for decision in DECISION_ORDER],
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _build_top_provinces_table(top_provinces: list[tuple[str, int]]) -> str:
    lines = [
        "| 省级地区 | 样本数 |",
        "|---------|--------|",
    ]
    for province, count in top_provinces:
        lines.append(f"| {province} | {_format_int(count)} |")
    return "\n".join(lines)


def _build_article_prefix_text(article_prefixes: Counter) -> str:
    if not article_prefixes:
        return "无"
    return "、".join(
        f"{prefix}: {_format_int(count)}"
        for prefix, count in sorted(article_prefixes.items())
    )


def build_dataset_card(dataset: DatasetDict) -> str:
    """根据本地 PDP-Bench 数据集动态生成 HuggingFace dataset card。"""
    split_summary = _split_summary(dataset)
    meta = _meta_summary(dataset)
    total_rows = sum(item["num_rows"] for item in split_summary)
    split_names = ", ".join(item["split"] for item in split_summary)
    split_table = _build_split_table(split_summary)
    top_provinces_table = _build_top_provinces_table(meta["top_provinces"])
    article_prefix_text = _build_article_prefix_text(meta["article_prefixes"])

    return f"""\
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
- public-prosecution
size_categories:
- 1K<n<10K
---

# PDP-Bench: Prosecution Decision Prediction Benchmark

PDP-Bench 是一个中文检察机关公诉决定预测数据集，包含 {_format_int(total_rows)} 条结构化检察文书样本。当前版本包含 `{split_names}` split。

## 任务

给定犯罪嫌疑人信息、案件程序和案件事实，模型需要预测：

- 适用法条：`relevant_articles`，使用 `cl:`、`cpl:`、`cpr:` 等前缀标识法律来源。
- 公诉决定：`decision`，四分类标签包括起诉、相对不起诉、法定不起诉、存疑不起诉。

## 数据集结构

{split_table}

## 数据来源与时间范围

- 来源字段覆盖：{_format_int(meta["source_url_count"])}/{_format_int(total_rows)} 条样本包含 `source_url`。
- 文书日期范围：{meta["date_min"]} 至 {meta["date_max"]}。
- 覆盖省级地区数量：{_format_int(meta["province_count"])}。
- 法条标注总数：{_format_int(meta["article_count"])}（{article_prefix_text}）。

### 样本量最多的省级地区

{top_provinces_table}

## 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文书唯一标识 |
| `meta` | dict | 元数据，含 `date` 和 `province` |
| `person_info` | string | 犯罪嫌疑人信息，已做必要脱敏 |
| `procedure` | string | 案件程序信息 |
| `fact` | string | 案件事实 |
| `relevant_articles` | list[string] | 相关法条，带 `cl:` / `cpl:` / `cpr:` 前缀 |
| `decision` | string | 公诉决定类型 |
| `raw_reasoning_and_decision` | string | 原始审查意见和决定文本 |
| `source_url` | string | 文书来源网页 URL |

## 标签说明

- `起诉`：犯罪事实清楚、证据确实充分，依法应当追究刑事责任。
- `相对不起诉`：已构成犯罪，但犯罪情节轻微，依照刑法规定不需要判处刑罚或者免除刑罚。
- `法定不起诉`：没有犯罪事实，或者存在依法不追究刑事责任的情形。
- `存疑不起诉`：经过补充侦查后仍证据不足，不符合起诉条件。

## 注意事项

- 本数据集用于法律 NLP 研究与模型评测，不构成法律意见。
- 文本来自公开检察文书并进行了结构化处理；使用时仍应注意个人信息保护和合规要求。
- PDP-Bench 适合零样本、提示工程、上下文学习和模型评测实验。
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 PDP-Bench 数据集上传到 HuggingFace Hub"
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/PDP-Bench",
        help="本地数据集目录 (default: data/PDP-Bench)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace 仓库 ID，如 your-username/PDP-Bench",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token（默认从 token.json 读取）",
    )
    parser.add_argument(
        "--token-file",
        default="token.json",
        help="token 文件路径 (default: token.json)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="是否将数据集设为私有",
    )
    args = parser.parse_args()

    token = args.token or load_token(args.token_file)
    if not token:
        logger.error(
            "未找到 HuggingFace token。"
            "请通过 --token 参数传入，或在 token.json 中配置 hf_token 字段"
        )
        return

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        logger.error(f"数据集目录不存在: {dataset_dir}")
        logger.error("请先运行 data_process/convert_to_hf_dataset.py 生成 data/PDP-Bench")
        return

    logger.info(f"加载数据集: {dataset_dir}")
    dataset = DatasetDict.load_from_disk(str(dataset_dir))
    allowed_splits = {"test", "test_rq2"}
    unexpected = set(dataset.keys()) - allowed_splits
    if unexpected:
        logger.warning("PDP-Bench 包含非预期 splits: %s", sorted(unexpected))

    for name, split_dataset in dataset.items():
        logger.info(f"  {name}: {len(split_dataset)} 条")

    dataset_card = build_dataset_card(dataset)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api.upload_file(
        path_or_fileobj=dataset_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    logger.info("数据集卡片已上传")

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

