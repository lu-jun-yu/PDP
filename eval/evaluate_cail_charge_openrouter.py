#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CAIL2018 charge prediction with OpenRouter.

This script is intended as an LJP calibration experiment for RQ1:
use CAIL2018-Small test, predict accusations from facts, and report
charge Macro-F1 / Micro-F1 / exact match.

Default input:
  data/CAIL2018/exercise_contest/data_test_charge_4k_seed42.json

Default output:
  results/RQ1_CAIL2018_charge

The script supports multi-label accusations and keeps a candidate label
set in the prompt to reduce synonym / formatting drift.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path, *, single_only: bool = False, max_samples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("meta") or {}
            accusations = meta.get("accusation") or []
            if single_only and len(accusations) != 1:
                continue
            rows.append(
                {
                    "index": idx,
                    "fact": str(obj.get("fact") or ""),
                    "accusation": [str(x) for x in accusations],
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def load_label_space(paths: list[Path]) -> list[str]:
    labels: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for label in (obj.get("meta") or {}).get("accusation") or []:
                    labels.add(str(label))
    return sorted(labels)


def compact_label(label: str) -> str:
    text = str(label).strip()
    text = text.replace(" ", "").replace("\u3000", "")
    text = text.replace("“", "").replace("”", "").replace("\"", "")
    text = text.replace("《", "").replace("》", "")
    text = re.sub(r"[。；;，,、：:\s]+$", "", text)
    if text.endswith("罪"):
        text = text[:-1]
    return text


def expand_bracket_label(label: str) -> set[str]:
    """
    Expand CAIL compact labels such as:
      [走私、贩卖、运输、制造]毒品
      非法[持有、私藏][枪支、弹药]
    into likely surface forms for matching.
    """
    label = compact_label(label)
    aliases = {label, label.replace("[", "").replace("]", "")}
    variants = [""]
    pos = 0
    for match in re.finditer(r"\[([^\]]+)\]", label):
        prefix = label[pos:match.start()]
        options = [x for x in re.split(r"[、,，/]", match.group(1)) if x]
        variants = [base + prefix + opt for base in variants for opt in options]
        pos = match.end()
    suffix = label[pos:]
    variants = [base + suffix for base in variants]
    return set(variants) | aliases


def build_label_matcher(labels: list[str]) -> dict[str, str]:
    matcher: dict[str, str] = {}
    for label in labels:
        for variant in expand_bracket_label(label):
            matcher[compact_label(variant)] = label
    return matcher


def build_messages(fact: str, labels: list[str]) -> list[dict]:
    label_text = "、".join(labels)
    system = (
        "你是刑事法律判决预测助手。请根据案件事实预测被告人的罪名。"
        "只能从给定候选罪名中选择；如涉及多个罪名，可以输出多个。"
        "罪名应尽量原样复制候选列表中的标签，不要创造候选列表之外的罪名。"
        "可以在内部完成法律要件判断，但最终不要输出推理过程。"
        "请严格输出 JSON，不要输出解释文字。格式为："
        "{\"accusation\":[\"罪名1\",\"罪名2\"]}。"
    )
    user = (
        f"候选罪名列表：{label_text}\n\n"
        "案件事实：\n"
        f"{fact}\n\n"
        "请输出 JSON："
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def strip_think(text: str) -> str:
    end = text.find("</think>")
    return text[end + len("</think>"):].strip() if end >= 0 else text.strip()


def split_labels(text: str) -> list[str]:
    parts = re.split(r"[、,，;；/\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_prediction(text: str, matcher: dict[str, str]) -> tuple[list[str], str]:
    raw = strip_think(text)
    candidates: list[str] = []

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    json_text = fenced.group(1).strip() if fenced else raw
    try:
        obj = json.loads(json_text)
        if isinstance(obj, dict):
            val = obj.get("accusation") or obj.get("罪名") or obj.get("charges")
            if isinstance(val, list):
                candidates = [str(x) for x in val]
            elif isinstance(val, str):
                candidates = split_labels(val)
        elif isinstance(obj, list):
            candidates = [str(x) for x in obj]
    except Exception:
        pass

    if not candidates:
        match = re.search(r"(?:accusation|罪名|预测罪名)\s*[:：]\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
        if match:
            candidates = split_labels(match.group(1))
        else:
            candidates = split_labels(raw)

    normalized: list[str] = []
    for item in candidates:
        key = compact_label(item)
        if not key:
            continue
        if key in matcher:
            normalized.append(matcher[key])
            continue
        # Some models place multiple labels in one JSON-list item, e.g. ["盗窃、抢夺"].
        # Try exact matching first to preserve CAIL compact labels, then split as fallback.
        for part in split_labels(item):
            part_key = compact_label(part)
            if part_key in matcher:
                normalized.append(matcher[part_key])

    deduped = list(dict.fromkeys(normalized))
    return deduped, raw


def compute_multilabel_metrics(preds: list[list[str]], refs: list[list[str]], labels: list[str]) -> dict[str, Any]:
    label_set = set(labels)
    per_class: dict[str, dict[str, float | int]] = {}
    total_tp = total_fp = total_fn = 0

    for label in labels:
        tp = fp = fn = 0
        for pred, ref in zip(preds, refs):
            p = label in set(pred)
            g = label in set(ref)
            if p and g:
                tp += 1
            elif p and not g:
                fp += 1
            elif not p and g:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(1 for ref in refs if label in set(ref))
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    supported_labels = [label for label in labels if per_class[label]["support"] > 0]
    macro_f1 = (
        sum(float(per_class[label]["f1"]) for label in supported_labels) / len(supported_labels)
        if supported_labels else 0.0
    )
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    exact_match = sum(1 for p, r in zip(preds, refs) if set(p) == set(r)) / len(refs) if refs else 0.0
    parse_fail = sum(1 for p in preds if not p)
    invalid_pred = sum(1 for pred in preds for label in pred if label not in label_set)

    return {
        "num_samples": len(refs),
        "num_labels": len(labels),
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "exact_match": exact_match,
        "parse_fail_count": parse_fail,
        "invalid_prediction_count": invalid_pred,
        "per_class": per_class,
    }


def metrics_markdown(metrics: dict[str, Any], *, model: str, input_file: Path, labels: list[str]) -> str:
    top_by_support = sorted(
        metrics["per_class"].items(),
        key=lambda kv: int(kv[1]["support"]),
        reverse=True,
    )[:30]
    lines = [
        "# CAIL2018 Charge Prediction Metrics",
        "",
        f"- model: `{model}`",
        f"- input_file: `{input_file}`",
        f"- num_samples: `{metrics['num_samples']}`",
        f"- num_labels: `{metrics['num_labels']}`",
        f"- parse_fail_count: `{metrics['parse_fail_count']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Charge Macro-F1 | {metrics['macro_f1']:.4f} |",
        f"| Charge Micro-F1 | {metrics['micro_f1']:.4f} |",
        f"| Charge Micro-Precision | {metrics['micro_precision']:.4f} |",
        f"| Charge Micro-Recall | {metrics['micro_recall']:.4f} |",
        f"| Exact Match | {metrics['exact_match']:.4f} |",
        "",
        "## Top-30 Labels By Support",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, info in top_by_support:
        lines.append(
            f"| {label} | {float(info['precision']):.4f} | {float(info['recall']):.4f} | "
            f"{float(info['f1']):.4f} | {int(info['support'])} |"
        )
    lines.append("")
    return "\n".join(lines)


async def call_openrouter(client, *, model: str, messages: list[dict], max_tokens: int,
                          temperature: float | None, reasoning_effort: str,
                          semaphore: asyncio.Semaphore, max_retries: int = 5) -> dict[str, str]:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
                if temperature is not None:
                    kwargs["temperature"] = temperature
                extra = {"include_reasoning": True}
                if reasoning_effort != "auto":
                    extra["reasoning"] = {"effort": reasoning_effort}
                kwargs["extra_body"] = extra
                response = await client.chat.completions.create(**kwargs)
                choices = getattr(response, "choices", None) or []
                if not choices:
                    raise ValueError("empty choices")
                message = choices[0].message
                reasoning = (
                    getattr(message, "reasoning", None)
                    or getattr(message, "reasoning_content", None)
                    or ""
                )
                if not reasoning:
                    details = getattr(message, "reasoning_details", None) or []
                    parts: list[str] = []
                    for item in details:
                        if isinstance(item, dict):
                            summary = item.get("summary")
                            if isinstance(summary, list):
                                parts.extend(str(x).strip() for x in summary if str(x).strip())
                            elif isinstance(summary, str) and summary.strip():
                                parts.append(summary.strip())
                            elif isinstance(item.get("text"), str) and item["text"].strip():
                                parts.append(item["text"].strip())
                    reasoning = "\n".join(parts).strip()
                return {
                    "content": message.content or "",
                    "reasoning": reasoning,
                }
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                wait = min(2 ** attempt, 30)
                logger.warning("OpenRouter error: %s; retry in %ss", exc, wait)
                await asyncio.sleep(wait)
    return {"content": "", "reasoning": ""}


async def run_eval(args) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("请先安装依赖: pip install openai") from exc

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and Path(".env").is_file():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    if not api_key:
        raise SystemExit("请设置 OPENROUTER_API_KEY")

    input_file = Path(args.input_file)
    label_files = [Path(p) for p in args.label_files]
    labels = load_label_space(label_files)
    if not labels:
        labels = load_label_space([input_file])
    matcher = build_label_matcher(labels)
    rows = load_jsonl(input_file, single_only=args.single_only, max_samples=args.max_samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    scope_parts = ["cail2018_charge", input_file.stem, f"reasoning_{args.reasoning_effort}"]
    if args.single_only:
        scope_parts.append("single_only")
    if args.max_samples is not None:
        scope_parts.append(f"max{args.max_samples}")
    scope_tag = "_".join(scope_parts)
    checkpoint_file = output_dir / f"{safe_model}_{scope_tag}_checkpoint.jsonl"

    done: dict[int, dict] = {}
    if not args.no_resume and checkpoint_file.is_file():
        logger.info("Detected checkpoint: %s", checkpoint_file)
        with checkpoint_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "raw_output" in item:
                        item["prediction"], _ = parse_prediction(item["raw_output"], matcher)
                    item.setdefault("raw_reasoning", "")
                    done[int(item["index"])] = item
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    logger.warning("Skip malformed checkpoint line %d", line_no)
        logger.info("Resume: loaded %d/%d completed predictions", len(done), len(rows))
    elif args.no_resume and checkpoint_file.is_file():
        checkpoint_file.unlink()
        logger.info("Deleted old checkpoint because --no-resume is set")

    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=120.0,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    pending = [row for row in rows if int(row["index"]) not in done]
    logger.info("Input rows=%d, pending=%d, labels=%d", len(rows), len(pending), len(labels))

    batch_size = args.batch_size if args.batch_size > 0 else max(len(pending), 1)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        tasks = [
            call_openrouter(
                client,
                model=args.model,
                messages=build_messages(row["fact"], labels),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                reasoning_effort=args.reasoning_effort,
                semaphore=semaphore,
            )
            for row in batch
        ]
        outputs = await asyncio.gather(*tasks)
        with checkpoint_file.open("a", encoding="utf-8") as f:
            for row, output in zip(batch, outputs):
                pred, raw = parse_prediction(output["content"], matcher)
                item = {
                    "index": row["index"],
                    "fact": row["fact"],
                    "reference": row["accusation"],
                    "prediction": pred,
                    "raw_output": raw,
                    "raw_reasoning": output["reasoning"],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                done[int(row["index"])] = item
        logger.info("Progress: %d/%d", len(done), len(rows))

    missing = [int(row["index"]) for row in rows if int(row["index"]) not in done]
    if missing:
        raise RuntimeError(
            f"{len(missing)} samples are missing from checkpoint; rerun without --no-resume to continue"
        )

    ordered = [done[int(row["index"])] for row in rows]
    preds = [item["prediction"] for item in ordered]
    refs = [item["reference"] for item in ordered]
    metrics = compute_multilabel_metrics(preds, refs, labels)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = output_dir / f"{safe_model}_{scope_tag}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
        encoding="utf-8",
    )
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (result_dir / "metrics.md").write_text(
        metrics_markdown(metrics, model=args.model, input_file=input_file, labels=labels),
        encoding="utf-8",
    )
    logger.info("Done: %s", result_dir)
    if checkpoint_file.is_file():
        checkpoint_file.unlink()
        logger.info("Checkpoint cleaned: %s", checkpoint_file)
    print(f"Charge Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Charge Micro-F1: {metrics['micro_f1']:.4f}")
    print(f"Exact Match: {metrics['exact_match']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CAIL2018 charge prediction via OpenRouter")
    parser.add_argument("--model", required=True, help="OpenRouter model id")
    parser.add_argument(
        "--input-file",
        default="data/CAIL2018/exercise_contest/data_test_charge_4k_seed42.json",
        help="CAIL2018 jsonl input file",
    )
    parser.add_argument(
        "--label-files",
        nargs="*",
        default=[
            "data/CAIL2018/exercise_contest/data_train.json",
            "data/CAIL2018/exercise_contest/data_valid.json",
        ],
        help="Files used to build candidate accusation label space",
    )
    parser.add_argument("--output-dir", default="results/RQ1_CAIL2018_charge")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=["auto", "none", "low", "medium", "high", "xhigh"],
        default="auto",
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--single-only", action="store_true", help="Evaluate only single-accusation cases")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
