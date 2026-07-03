#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_process/Stage3/apply_refinement.py

把法律专家复核精炼后的“有问题子集”覆盖回 Stage2 合并全集，产出最终 Stage3 数据。

默认输入：
- data/public_sources/Stage2/dataset_reviewed_merged.json
    合并全集（基底）。
- data/public_sources/Stage3/dataset_refinement_reviewed.json
    专家复核精炼后的子集；专家在 llm_filter_data_refinement.py 输出的
    dataset_refinement_flagged.json 上就地修订字段后另存得到，schema 与全集一致。

默认输出：
- data/public_sources/Stage3/dataset_reviewed.json

覆盖规则：
- 按案号匹配；子集中每条记录覆盖基底同案号记录的可精炼文本字段：
  person_info、procedure、fact、raw_reasoning_and_decision。
- 默认不改动 decision、relevant_articles、source_url；若子集在这些字段上与基底不一致，
  仅告警提示而不覆盖，除非显式 --override-all-fields。
- 子集中可能残留 llm_filter 注入的 refinement 注记，输出时一律剥除。
- 输出顺序与基底全集一致。

Usage:
    python data_process/Stage3/apply_refinement.py
    python data_process/Stage3/apply_refinement.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage2"
STAGE3_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage3"

DEFAULT_BASE = STAGE2_DIR / "dataset_reviewed_merged.json"
DEFAULT_REVIEWED = STAGE3_DIR / "dataset_refinement_reviewed.json"
DEFAULT_OUTPUT = STAGE3_DIR / "dataset_reviewed.json"

# 专家可精炼并覆盖回基底的文本字段
REVIEWABLE_FIELDS = (
    "person_info",
    "procedure",
    "fact",
    "raw_reasoning_and_decision",
)
# 默认保持不变的字段，仅在不一致时告警
PROTECTED_FIELDS = (
    "decision",
    "relevant_articles",
    "source_url",
)


def normalize_case_id(value: Any) -> str:
    return str(value or "").replace("〔", "[").replace("〕", "]").strip()


def normalize_decision(value: Any) -> str:
    decision = str(value or "").strip()
    return decision if decision else "（空）"


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"文件内容不是数组: {path}")
    bad_count = sum(1 for item in data if not isinstance(item, dict))
    if bad_count:
        logger.warning("%s 中跳过非对象记录: %s", path.name, bad_count)
    return [item for item in data if isinstance(item, dict)]


def build_records_index(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """按案号建立索引，保留输入顺序。"""
    records_by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    missing_id = 0
    for record in records:
        case_id = normalize_case_id(record.get("id"))
        if not case_id:
            missing_id += 1
            continue
        if case_id not in records_by_id:
            ordered_ids.append(case_id)
        records_by_id[case_id] = record
    if missing_id:
        logger.warning("基底中跳过缺少案号的记录: %s", missing_id)
    return records_by_id, ordered_ids


def apply_reviewed_subset(
    records_by_id: dict[str, dict[str, Any]],
    reviewed: list[dict[str, Any]],
    *,
    override_all_fields: bool,
) -> tuple[int, int, int]:
    """把专家子集覆盖回基底，返回 (updated, not_found, protected_conflicts)。"""
    updated = 0
    not_found = 0
    protected_conflicts = 0
    seen_ids: set[str] = set()

    for item in reviewed:
        case_id = normalize_case_id(item.get("id"))
        if not case_id:
            logger.warning("专家子集存在缺少案号的记录，跳过")
            continue
        if case_id in seen_ids:
            logger.warning("专家子集存在重复案号，后者继续覆盖: %s", item.get("id"))
        seen_ids.add(case_id)

        record = records_by_id.get(case_id)
        if record is None:
            not_found += 1
            logger.warning("专家子集案号未出现在基底中，跳过: %s", item.get("id"))
            continue

        for field in PROTECTED_FIELDS:
            if field not in item or item.get(field) == record.get(field):
                continue
            protected_conflicts += 1
            if override_all_fields:
                record[field] = item[field]
                logger.warning("受保护字段被覆盖(%s): %s", field, item.get("id"))
            else:
                logger.warning(
                    "受保护字段不一致但保留基底(%s): %s（如确需修改请用 --override-all-fields）",
                    field,
                    item.get("id"),
                )

        for field in REVIEWABLE_FIELDS:
            if field not in item:
                continue
            new_value = item.get(field)
            if not isinstance(new_value, str):
                logger.warning("字段 %s 非字符串，跳过覆盖: %s", field, item.get("id"))
                continue
            if str(record.get(field) or "").strip() and not new_value.strip():
                logger.warning("字段 %s 被清空，请确认是否有意: %s", field, item.get("id"))
            record[field] = new_value

        updated += 1

    logger.info(
        "覆盖完成: 子集=%s, updated=%s, not_found=%s, protected_conflicts=%s",
        len(reviewed),
        updated,
        not_found,
        protected_conflicts,
    )
    return updated, not_found, protected_conflicts


def strip_refinement(record: dict[str, Any]) -> dict[str, Any]:
    """剥除 refinement 注记（若存在）。"""
    if "refinement" in record:
        return {k: v for k, v in record.items() if k != "refinement"}
    return record


def log_decision_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts = Counter(normalize_decision(record.get("decision")) for record in records)
    logger.info("decision 类别统计:")
    for decision, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        logger.info("  %s: %s", decision, count)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把专家复核精炼后的子集覆盖回合并全集，产出最终 Stage3 数据"
    )
    parser.add_argument(
        "--base",
        type=str,
        default=str(DEFAULT_BASE),
        help="基底全集 JSON（默认 data/public_sources/Stage2/dataset_reviewed_merged.json）",
    )
    parser.add_argument(
        "--reviewed",
        type=str,
        default=str(DEFAULT_REVIEWED),
        help="专家复核精炼后的子集 JSON（默认 data/public_sources/Stage3/dataset_refinement_reviewed.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="输出 JSON（默认 data/public_sources/Stage3/dataset_reviewed.json）",
    )
    parser.add_argument(
        "--override-all-fields",
        action="store_true",
        help="连同 decision/relevant_articles/source_url 等字段也按专家子集覆盖",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写入输出文件")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base_path = Path(args.base)
    reviewed_path = Path(args.reviewed)
    output_path = Path(args.output)

    base_records = load_json_list(base_path)
    logger.info("读取基底全集: %s (%s 条)", base_path, len(base_records))
    records_by_id, ordered_ids = build_records_index(base_records)

    reviewed = load_json_list(reviewed_path)
    logger.info("读取专家子集: %s (%s 条)", reviewed_path, len(reviewed))

    apply_reviewed_subset(records_by_id, reviewed, override_all_fields=args.override_all_fields)

    final_records = [
        strip_refinement(records_by_id[case_id])
        for case_id in ordered_ids
        if case_id in records_by_id
    ]
    log_decision_counts(final_records)

    if args.dry_run:
        logger.info("dry-run: 不写入输出文件，预计输出记录数: %s", len(final_records))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("写入输出: %s (%s 条)", output_path, len(final_records))


if __name__ == "__main__":
    main()
