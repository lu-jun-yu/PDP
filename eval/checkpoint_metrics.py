#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build interim metrics from an evaluation checkpoint jsonl file.

Usage:
    python eval/checkpoint_metrics.py ^
      --checkpoint-file results/RQ3_SFT/RQ3_SFT_Qwen3-8B_train_seed42_original_merged_original_test_r8_checkpoint.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (
    build_metrics_markdown,
    build_rq2_result_f1_mean_se,
    build_rq2_stability_markdown,
    compute_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interim metrics from checkpoint jsonl")
    parser.add_argument(
        "--checkpoint-file",
        default="results/RQ3_SFT/RQ3_SFT_Qwen3-8B_train_seed42_original_merged_original_test_r8_checkpoint.jsonl",
    )
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def infer_output_file(checkpoint_file: Path) -> Path:
    name = checkpoint_file.name
    if name.endswith("_checkpoint.jsonl"):
        return checkpoint_file.with_name(name[:-len("_checkpoint.jsonl")] + "_metrics.md")
    if checkpoint_file.suffix == ".jsonl":
        return checkpoint_file.with_suffix(".metrics.md")
    return checkpoint_file.with_name(checkpoint_file.name + ".metrics.md")


def infer_model_variant(checkpoint_file: Path) -> tuple[str, str]:
    stem = checkpoint_file.stem
    if stem.endswith("_checkpoint"):
        stem = stem[:-len("_checkpoint")]
    return stem, "partial_checkpoint"


def record_key(record: dict) -> tuple[int | str, int]:
    sample_index = record.get("sample_index", record.get("index", record.get("id", "")))
    run_id = int(record.get("run_id", 0))
    return sample_index, run_id


def load_checkpoint_records(checkpoint_file: Path) -> tuple[list[dict], int]:
    latest_by_key: dict[tuple[int | str, int], dict] = {}
    skipped_lines = 0

    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_lines += 1
                continue
            if "prediction" not in record or "reference" not in record:
                skipped_lines += 1
                continue
            latest_by_key[record_key(record)] = record

    records = sorted(
        latest_by_key.values(),
        key=lambda record: (
            str(record.get("sample_index", record.get("index", record.get("id", "")))),
            int(record.get("run_id", 0)),
        ),
    )
    return records, skipped_lines


def resolve_checkpoint_file(raw_path: str) -> Path:
    checkpoint_file = Path(raw_path)
    if checkpoint_file.exists():
        return checkpoint_file

    candidate_name = checkpoint_file.name
    matches = list((PROJECT_ROOT / "results").rglob(candidate_name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Checkpoint file not found at given path: {raw_path}; multiple matches found: {matches}"
        )
    raise FileNotFoundError(f"Checkpoint file not found: {raw_path}")


def main() -> None:
    args = parse_args()

    checkpoint_file = resolve_checkpoint_file(args.checkpoint_file)

    output_file = Path(args.output_file) if args.output_file else infer_output_file(checkpoint_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records, skipped_lines = load_checkpoint_records(checkpoint_file)
    if not records:
        raise ValueError(f"No valid records found in checkpoint file: {checkpoint_file}")

    predictions = [record["prediction"] for record in records]
    references = [record["reference"] for record in records]
    metrics = compute_metrics(predictions, references)
    result_f1_mean_se = build_rq2_result_f1_mean_se(records)

    model, variant = infer_model_variant(checkpoint_file)
    output = build_metrics_markdown(
        metrics,
        model=model,
        variant=variant,
        num_samples=len(records),
        result_f1_mean_se=result_f1_mean_se,
    )

    unique_cases = len({record.get("sample_index", record.get("index", record.get("id", ""))) for record in records})
    unique_runs = sorted({int(record.get("run_id", 0)) for record in records})
    output += (
        "\n## Interim Checkpoint Summary\n\n"
        f"- checkpoint_file: `{checkpoint_file.as_posix()}`\n"
        f"- completed_records: `{len(records)}`\n"
        f"- unique_cases_seen: `{unique_cases}`\n"
        f"- run_ids_seen: `{unique_runs}`\n"
        f"- skipped_invalid_lines: `{skipped_lines}`\n"
        "- note: this is a partial checkpoint summary built from currently completed records only.\n"
    )

    stability_md = build_rq2_stability_markdown(records)
    if stability_md:
        output += stability_md

    with output_file.open("w", encoding="utf-8") as f:
        f.write(output)

    print(f"[checkpoint-metrics] checkpoint_file={checkpoint_file}")
    print(f"[checkpoint-metrics] records={len(records)} unique_cases={unique_cases} runs={unique_runs}")
    print(f"[checkpoint-metrics] skipped_invalid_lines={skipped_lines}")
    print(f"[checkpoint-metrics] output_file={output_file}")


if __name__ == "__main__":
    main()
