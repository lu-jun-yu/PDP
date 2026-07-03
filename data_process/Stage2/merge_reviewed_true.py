#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_process/Stage2/merge_reviewed_true.py

合并人工确认通过的 reviewed_true 数据：合并四份 reviewed_true 文件、
过滤 source_url 为空记录、按案号去重，输出合并全集。

默认输入（均位于 data/public_sources/Stage2/）：
- dataset_reviewed_true_mismatch_regex.json
- dataset_reviewed_true_only_llm.json
- dataset_reviewed_true_only_regex.json
- dataset_reviewed_true.json

默认输出：
- data/public_sources/Stage2/dataset_reviewed_merged.json

合并规则：
- dataset_reviewed_true_mismatch_llm.json 与 dataset_reviewed_true_mismatch_regex.json
  内容一致，默认只读取 regex 版本；
- 合并时过滤 source_url 为空的记录；
- 合并时按案号去重，若同案号重复，后出现的输入文件覆盖前者。

Usage:
    python data_process/Stage2/merge_reviewed_true.py
    python data_process/Stage2/merge_reviewed_true.py --dry-run
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

DEFAULT_INPUT_NAMES = [
    "dataset_reviewed_true_mismatch_regex.json",
    "dataset_reviewed_true_only_llm.json",
    "dataset_reviewed_true_only_regex.json",
    "dataset_reviewed_true.json",
]

DEFAULT_OUTPUT_NAME = "dataset_reviewed_merged.json"


def normalize_case_id(value: Any) -> str:
    return str(value or "").replace("〔", "[").replace("〕", "]").strip()


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


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


def has_source_url(record: dict[str, Any]) -> bool:
    return bool(str(record.get("source_url") or "").strip())


def normalize_decision(value: Any) -> str:
    decision = str(value or "").strip()
    return decision if decision else "（空）"


def merge_input_files(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    duplicate_identical = 0
    duplicate_conflict = 0
    missing_id = 0
    empty_source_url = 0

    for path in paths:
        records = load_json_list(path)
        path_empty_source_url = 0
        logger.info("读取输入: %s (%s 条)", path, len(records))
        for record in records:
            if not has_source_url(record):
                empty_source_url += 1
                path_empty_source_url += 1
                continue

            case_id = normalize_case_id(record.get("id"))
            if not case_id:
                missing_id += 1
                continue

            if case_id not in records_by_id:
                ordered_ids.append(case_id)
            elif records_by_id[case_id] == record:
                duplicate_identical += 1
            else:
                duplicate_conflict += 1
                logger.warning("重复案号内容不一致，采用后出现版本: %s (%s)", record.get("id"), path.name)

            records_by_id[case_id] = record

        if path_empty_source_url:
            logger.info("%s 跳过 source_url 为空记录: %s", path.name, path_empty_source_url)

    if missing_id:
        logger.warning("合并时跳过缺少案号的记录: %s", missing_id)
    if empty_source_url:
        logger.info("合并时跳过 source_url 为空记录总数: %s", empty_source_url)
    logger.info(
        "合并完成: unique=%s, duplicate_identical=%s, duplicate_conflict=%s",
        len(records_by_id),
        duplicate_identical,
        duplicate_conflict,
    )
    return records_by_id, ordered_ids


def log_decision_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts = Counter(normalize_decision(record.get("decision")) for record in records)
    logger.info("decision 类别统计:")
    for decision, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        logger.info("  %s: %s", decision, count)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 reviewed_true 数据，生成 Stage3 基底")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(STAGE2_DIR),
        help="Stage2 reviewed input directory",
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="输入 JSON 文件；可重复指定。相对路径按 --base-dir 解析。默认使用四份 reviewed_true 文件",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出 JSON 文件（默认 data/public_sources/Stage2/dataset_reviewed_merged.json）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写入输出文件")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base_dir = Path(args.base_dir)
    input_names = args.inputs or DEFAULT_INPUT_NAMES
    input_paths = [resolve_path(name, base_dir) for name in input_names]
    output_path = Path(args.output) if args.output else STAGE2_DIR / DEFAULT_OUTPUT_NAME

    records_by_id, ordered_ids = merge_input_files(input_paths)
    merged_records = [records_by_id[case_id] for case_id in ordered_ids if case_id in records_by_id]
    log_decision_counts(merged_records)

    if args.dry_run:
        logger.info("dry-run: 不写入输出文件，预计输出记录数: %s", len(merged_records))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("写入输出: %s (%s 条)", output_path, len(merged_records))


if __name__ == "__main__":
    main()
