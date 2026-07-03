#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample a fixed CAIL2018-Small charge-prediction calibration subset.

Default:
  input:  data/CAIL2018/exercise_contest/data_test.json
  output: data/CAIL2018/exercise_contest/data_test_charge_4k_seed42.json

The default sampling is simple random sampling without replacement, which keeps
the original test distribution in expectation. A manifest with label counts is
written next to the sampled file for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_source_index"] = line_no
            rows.append(obj)
    return rows


def accusation_list(row: dict[str, Any]) -> list[str]:
    meta = row.get("meta") or {}
    acc = meta.get("accusation") or []
    return [str(x) for x in acc]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counter: Counter[str] = Counter()
    label_len_counter: Counter[int] = Counter()
    for row in rows:
        labels = accusation_list(row)
        label_len_counter[len(labels)] += 1
        label_counter.update(labels)
    return {
        "num_rows": len(rows),
        "num_unique_accusations": len(label_counter),
        "num_single_accusation_rows": label_len_counter.get(1, 0),
        "num_multi_accusation_rows": sum(v for k, v in label_len_counter.items() if k > 1),
        "accusation_count_top50": label_counter.most_common(50),
        "accusation_count_all": dict(label_counter.most_common()),
        "accusation_length_count": dict(sorted(label_len_counter.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = dict(row)
            out.pop("_source_index", None)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a fixed CAIL2018 charge calibration subset"
    )
    parser.add_argument(
        "--input-file",
        default="data/CAIL2018/exercise_contest/data_test.json",
        help="Source CAIL2018 jsonl file",
    )
    parser.add_argument(
        "--output-file",
        default="data/CAIL2018/exercise_contest/data_test_charge_4k_seed42.json",
        help="Output sampled jsonl file",
    )
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Sample only rows with exactly one accusation label",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Manifest path; default is <output-file>.manifest.json",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    manifest_file = Path(args.manifest) if args.manifest else output_file.with_suffix(
        output_file.suffix + ".manifest.json"
    )

    rows = load_jsonl(input_file)
    if args.single_only:
        rows = [row for row in rows if len(accusation_list(row)) == 1]

    if args.sample_size > len(rows):
        raise ValueError(
            f"sample_size={args.sample_size} exceeds available rows={len(rows)}"
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(rows, args.sample_size)
    sampled.sort(key=lambda row: int(row["_source_index"]))

    write_jsonl(output_file, sampled)
    manifest = {
        "input_file": str(input_file.resolve()),
        "output_file": str(output_file.resolve()),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "single_only": args.single_only,
        "source_summary": summarize(rows),
        "sample_summary": summarize(sampled),
        "sample_source_indices_first100": [
            int(row["_source_index"]) for row in sampled[:100]
        ],
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("CAIL2018 charge calibration subset sampled")
    print(f"  input:    {input_file}")
    print(f"  output:   {output_file}")
    print(f"  manifest: {manifest_file}")
    print(f"  rows:     {len(sampled)}")
    print(f"  labels:   {manifest['sample_summary']['num_unique_accusations']}")
    print(f"  multi:    {manifest['sample_summary']['num_multi_accusation_rows']}")


if __name__ == "__main__":
    main()
