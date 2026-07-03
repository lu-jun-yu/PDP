#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 results/RQ2_1/*_r{FROM}_{TIMESTAMP}/ 下的 details_*.json 合并成
results/RQ2_1/*_r{TO}_checkpoint.jsonl，供 eval/evaluate_openrouter_rq2.py
在更大的 num_repeats 下断点续跑。

机制：评估脚本运行时把每条 (sample_index, run_id) 写入 checkpoint JSONL，
跑完后清理 checkpoint 并把 details 按错误类别拆分到时间戳子目录。
本脚本反向操作——从 details_*.json 重建 checkpoint，文件名中的 r{FROM}
替换为 r{TO}。重跑评估脚本时会复用 run_id 0..FROM-1，只补跑 FROM..TO-1。

Usage:
    python data_process/upgrade_rq2_repeats.py
    python data_process/upgrade_rq2_repeats.py --from-repeats 5 --to-repeats 8
    python data_process/upgrade_rq2_repeats.py --results-dir results/RQ2_1
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TIMESTAMP_SUFFIX = re.compile(r"_(\d{8})_(\d{6})$")


def derive_checkpoint_name(dir_name: str, from_r: int, to_r: int) -> str | None:
    m = TIMESTAMP_SUFFIX.search(dir_name)
    if not m:
        return None
    base = dir_name[: m.start()]
    suffix = f"_r{from_r}"
    if not base.endswith(suffix):
        return None
    base = base[: -len(suffix)] + f"_r{to_r}"
    return f"{base}_checkpoint.jsonl"


def collect_records(result_dir: Path) -> list[dict]:
    records: dict[tuple[int, int], dict] = {}
    for path in sorted(result_dir.glob("details_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            key = (entry["sample_index"], entry["run_id"])
            if key in records:
                logger.warning("重复键 %s 在 %s，保留第一份", key, path.name)
                continue
            records[key] = entry
    return [records[k] for k in sorted(records)]


def main() -> None:
    parser = argparse.ArgumentParser(description="把 r5 details 重建为 r8 checkpoint")
    parser.add_argument("--results-dir", default="results/RQ2_1")
    parser.add_argument("--from-repeats", type=int, default=5)
    parser.add_argument("--to-repeats", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="目标 checkpoint 已存在时覆盖（默认跳过）",
    )
    args = parser.parse_args()

    if args.from_repeats >= args.to_repeats:
        logger.error("--from-repeats 必须小于 --to-repeats")
        sys.exit(1)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        logger.error("results dir 不存在: %s", results_dir)
        sys.exit(1)

    candidates = [
        p for p in sorted(results_dir.iterdir())
        if p.is_dir() and f"_r{args.from_repeats}_" in p.name
    ]
    if not candidates:
        logger.warning("未找到 r%d 子目录", args.from_repeats)
        return

    logger.info("发现 %d 个 r%d 子目录", len(candidates), args.from_repeats)
    migrated = 0
    skipped = 0
    for src in candidates:
        ckpt_name = derive_checkpoint_name(src.name, args.from_repeats, args.to_repeats)
        if ckpt_name is None:
            logger.warning("[skip] 命名不匹配: %s", src.name)
            skipped += 1
            continue
        ckpt_path = results_dir / ckpt_name
        if ckpt_path.exists() and not args.overwrite:
            logger.info("[skip] 已存在 %s（用 --overwrite 覆盖）", ckpt_path.name)
            skipped += 1
            continue

        records = collect_records(src)
        if not records:
            logger.warning("[skip] 无记录: %s", src.name)
            skipped += 1
            continue

        run_ids = {r["run_id"] for r in records}
        if max(run_ids) >= args.from_repeats:
            logger.warning(
                "[warn] %s 出现 run_id=%d，超出 r%d 预期",
                src.name, max(run_ids), args.from_repeats,
            )

        with open(ckpt_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n_samples = len({r["sample_index"] for r in records})
        logger.info(
            "[ok] %s -> %s (%d samples × %d runs = %d records)",
            src.name, ckpt_path.name, n_samples, len(run_ids), len(records),
        )
        migrated += 1

    logger.info("完成：迁移 %d，跳过 %d", migrated, skipped)


if __name__ == "__main__":
    main()
