#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按案号比对 dataset.json 与 dataset_llm_processed.json：
- 完全一致 -> dataset_reviewed_true.json
- 不一致（LLM 侧） -> dataset_reviewed_false_mismatch_llm.json
- 不一致（正则侧） -> dataset_reviewed_false_mismatch_regex.json
- 单独存在（LLM 侧） -> dataset_reviewed_false_only_llm.json
- 单独存在（正则侧） -> dataset_reviewed_false_only_regex.json

Usage:
    python data_process/split_dataset_reviewed_by_case_id.py
    python data_process/split_dataset_reviewed_by_case_id.py --base-dir data/public_sources/Stage2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage1"
STAGE2_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage2"


def normalize_case_id(value: str) -> str:
    """统一案号中的中括号写法，避免 〔〕 与 [] 导致误判。"""
    return value.replace("〔", "[").replace("〕", "]").strip()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"文件内容不是数组: {path}")
    return [x for x in data if isinstance(x, dict)]


def build_map(records: list[dict[str, Any]], source_name: str) -> tuple[dict[str, dict[str, Any]], int]:
    out: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for rec in records:
        rid = str(rec.get("id") or "").strip()
        key = normalize_case_id(rid)
        if not key:
            continue
        if key in out:
            duplicate_count += 1
            logger.warning("[%s] 检测到重复案号，后者覆盖前者: %s", source_name, rid)
        out[key] = rec
    return out, duplicate_count


def main() -> None:
    parser = argparse.ArgumentParser(description="按案号比对并拆分 reviewed_true 与 false 四类")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(STAGE2_DIR),
        help="Stage2 input/output directory",
    )
    parser.add_argument("--processed-json", type=str, default="", help="默认 <base-dir>/dataset_llm_processed.json")
    parser.add_argument("--dataset-json", type=str, default="", help="default data/public_sources/Stage1/dataset.json")
    parser.add_argument("--out-true", type=str, default="", help="默认 <base-dir>/dataset_reviewed_true.json")
    parser.add_argument(
        "--out-false-mismatch-llm",
        type=str,
        default="",
        help="默认 <base-dir>/dataset_reviewed_false_mismatch_llm.json",
    )
    parser.add_argument(
        "--out-false-mismatch-regex",
        type=str,
        default="",
        help="默认 <base-dir>/dataset_reviewed_false_mismatch_regex.json",
    )
    parser.add_argument(
        "--out-false-only-llm",
        type=str,
        default="",
        help="默认 <base-dir>/dataset_reviewed_false_only_llm.json",
    )
    parser.add_argument(
        "--out-false-only-regex",
        type=str,
        default="",
        help="默认 <base-dir>/dataset_reviewed_false_only_regex.json",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    base = Path(args.base_dir)
    processed_path = Path(args.processed_json) if args.processed_json else base / "dataset_llm_processed.json"
    dataset_path = Path(args.dataset_json) if args.dataset_json else STAGE1_DIR / "dataset.json"
    out_true = Path(args.out_true) if args.out_true else base / "dataset_reviewed_true.json"
    out_false_mismatch_llm = (
        Path(args.out_false_mismatch_llm)
        if args.out_false_mismatch_llm
        else base / "dataset_reviewed_false_mismatch_llm.json"
    )
    out_false_mismatch_regex = (
        Path(args.out_false_mismatch_regex)
        if args.out_false_mismatch_regex
        else base / "dataset_reviewed_false_mismatch_regex.json"
    )
    out_false_only_llm = (
        Path(args.out_false_only_llm)
        if args.out_false_only_llm
        else base / "dataset_reviewed_false_only_llm.json"
    )
    out_false_only_regex = (
        Path(args.out_false_only_regex)
        if args.out_false_only_regex
        else base / "dataset_reviewed_false_only_regex.json"
    )

    processed = load_json_list(processed_path)
    dataset = load_json_list(dataset_path)

    processed_map, processed_dup_count = build_map(processed, "dataset_llm_processed")
    dataset_map, dataset_dup_count = build_map(dataset, "dataset")

    all_ids = sorted(set(processed_map.keys()) | set(dataset_map.keys()))
    reviewed_true: list[dict[str, Any]] = []
    false_mismatch_llm: list[dict[str, Any]] = []
    false_mismatch_regex: list[dict[str, Any]] = []
    false_only_llm: list[dict[str, Any]] = []
    false_only_regex: list[dict[str, Any]] = []

    only_in_processed = 0
    only_in_dataset = 0
    mismatch = 0

    for rid in all_ids:
        p = processed_map.get(rid)
        d = dataset_map.get(rid)

        if p is None:
            only_in_dataset += 1
            false_only_regex.append(d)  # type: ignore[arg-type]
            continue
        if d is None:
            only_in_processed += 1
            false_only_llm.append(p)
            continue

        if p == d:
            reviewed_true.append(p)
        else:
            mismatch += 1
            false_mismatch_llm.append(p)
            false_mismatch_regex.append(d)

    out_true.parent.mkdir(parents=True, exist_ok=True)
    out_false_mismatch_llm.parent.mkdir(parents=True, exist_ok=True)
    out_false_mismatch_regex.parent.mkdir(parents=True, exist_ok=True)
    out_false_only_llm.parent.mkdir(parents=True, exist_ok=True)
    out_false_only_regex.parent.mkdir(parents=True, exist_ok=True)

    out_true.write_text(json.dumps(reviewed_true, ensure_ascii=False, indent=2), encoding="utf-8")
    out_false_mismatch_llm.write_text(json.dumps(false_mismatch_llm, ensure_ascii=False, indent=2), encoding="utf-8")
    out_false_mismatch_regex.write_text(json.dumps(false_mismatch_regex, ensure_ascii=False, indent=2), encoding="utf-8")
    out_false_only_llm.write_text(json.dumps(false_only_llm, ensure_ascii=False, indent=2), encoding="utf-8")
    out_false_only_regex.write_text(json.dumps(false_only_regex, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("输入统计: dataset=%s, processed=%s", len(dataset), len(processed))
    logger.info("案号去重后: dataset=%s, processed=%s", len(dataset_map), len(processed_map))
    logger.info(
        "比对结果: true=%s (mismatch=%s, only_in_processed=%s, only_in_dataset=%s)",
        len(reviewed_true),
        mismatch,
        only_in_processed,
        only_in_dataset,
    )
    logger.info(
        "false四类: mismatch_llm=%s, mismatch_regex=%s, only_llm=%s, only_regex=%s",
        len(false_mismatch_llm),
        len(false_mismatch_regex),
        len(false_only_llm),
        len(false_only_regex),
    )
    if processed_dup_count or dataset_dup_count:
        logger.info(
            "重复案号提示: processed侧=%s, dataset侧=%s",
            processed_dup_count,
            dataset_dup_count,
        )
    logger.info("输出文件: %s", out_true)
    logger.info("输出文件: %s", out_false_mismatch_llm)
    logger.info("输出文件: %s", out_false_mismatch_regex)
    logger.info("输出文件: %s", out_false_only_llm)
    logger.info("输出文件: %s", out_false_only_regex)


if __name__ == "__main__":
    main()


