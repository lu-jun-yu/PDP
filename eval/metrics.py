#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/metrics.py

PDP 评估指标计算模块。

既可作为库被 evaluate_vllm.py 导入，也可独立运行，从已有的 details_*.json
重新计算并生成 metrics.md。

独立用法:
    python eval/metrics.py results/Qwen3-4B_test_baseline_20260312_055107
    python eval/metrics.py dir1 dir2 dir3   # 批量更新多个目录
"""

import argparse
import glob
import json
import logging
import math
import os
import re
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
#  评估指标
# ============================================================

def set_precision_recall_f1(pred: set, gold: set) -> tuple[float, float, float]:
    """计算集合级别的 precision, recall, F1。"""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 0.0, 0.0

    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


NON_PROSECUTION_TYPES = {"相对不起诉", "法定不起诉", "存疑不起诉"}
DECISION_LABELS = {"起诉", "相对不起诉", "法定不起诉", "存疑不起诉"}


def _to_level1(decision: str) -> str:
    """将四分类决定映射为二分类：起诉 / 不起诉。"""
    return "不起诉" if decision in NON_PROSECUTION_TYPES else "起诉"


def normalize_decision_label(label: str) -> str | None:
    """将模型输出归一化为四类决定标签；无法归一化则返回 None。"""
    if not isinstance(label, str):
        return None
    normalized = label.strip().replace("**", "")
    if normalized.endswith("。"):
        normalized = normalized[:-1].strip()
    if normalized in DECISION_LABELS:
        return normalized
    return None


def per_class_f1(pred_labels: list[str | None], gold_labels: list[str], classes: list[str]) -> dict[str, dict]:
    """按给定类别计算 one-vs-rest 的 precision / recall / f1。"""
    results = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred_labels, gold_labels) if p != cls and g == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for g in gold_labels if g == cls)
        results[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": support,
        }
    return results


def macro_f1(per_class_metrics: dict[str, dict], classes: list[str]) -> float:
    """计算给定类别集合上的宏平均 F1。"""
    if not classes:
        return 0.0
    return sum(per_class_metrics[cls]["f1"] for cls in classes) / len(classes)


def micro_prf1(pred_labels: list[str | None], gold_labels: list[str], classes: list[str]) -> dict[str, float]:
    """在给定类别集合上计算 micro precision/recall/F1。"""
    tp = 0
    fp = 0
    fn = 0
    for cls in classes:
        tp += sum(1 for p, g in zip(pred_labels, gold_labels) if p == cls and g == cls)
        fp += sum(1 for p, g in zip(pred_labels, gold_labels) if p == cls and g != cls)
        fn += sum(1 for p, g in zip(pred_labels, gold_labels) if p != cls and g == cls)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(predictions: list[dict], references: list[dict]) -> dict:
    """
    计算所有评估指标。

    过程评估 (relevant_articles): 法条 F1
    结果评估 (decision):
      - 第一级（二分类）：起诉 / 不起诉 Accuracy + 各类别 Precision/Recall/F1
      - 第二级（四分类）：起诉、相对不起诉、法定不起诉、存疑不起诉 Accuracy + 各类别 Precision/Recall/F1
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions 与 references 数量不一致: {len(predictions)} != {len(references)}"
        )

    metrics = defaultdict(list)
    level1_preds: list[str | None] = []
    level1_refs = []
    level2_preds: list[str | None] = []
    level2_refs = []
    decision_parse_fail_count = 0
    reference_parse_fail_count = 0

    for pred, ref in zip(predictions, references):
        # --- 过程评估：法条 F1（所有样本） ---
        pred_arts = set(pred["relevant_articles"])
        gold_arts = set(ref["relevant_articles"])
        p, r, f = set_precision_recall_f1(pred_arts, gold_arts)
        metrics["articles_precision"].append(p)
        metrics["articles_recall"].append(r)
        metrics["articles_f1"].append(f)

        pred_decision = normalize_decision_label(pred.get("decision", ""))
        ref_decision = normalize_decision_label(ref.get("decision", ""))
        if ref_decision is None:
            reference_parse_fail_count += 1
            continue
        if pred_decision is None:
            decision_parse_fail_count += 1

        # --- 结果评估：第一级（二分类）---
        pred_l1 = _to_level1(pred_decision) if pred_decision is not None else None
        ref_l1 = _to_level1(ref_decision)
        l1_correct = 1.0 if (pred_l1 is not None and pred_l1 == ref_l1) else 0.0
        metrics["decision_level1_accuracy"].append(l1_correct)
        level1_preds.append(pred_l1)
        level1_refs.append(ref_l1)

        # --- 结果评估：第二级（四分类）---
        correct = 1.0 if (pred_decision is not None and pred_decision == ref_decision) else 0.0
        metrics["decision_level2_accuracy"].append(correct)
        level2_preds.append(pred_decision)
        level2_refs.append(ref_decision)

    # 宏平均
    result = {}
    for key, values in metrics.items():
        result[key] = sum(values) / len(values) if values else 0.0

    level1_classes = ["起诉", "不起诉"]
    level2_classes = ["起诉", "相对不起诉", "法定不起诉", "存疑不起诉"]

    # 第一级（二分类）各类别的 precision / recall / f1
    result["level1_per_class"] = per_class_f1(level1_preds, level1_refs, level1_classes)

    # 第二级（四分类）各类别的 precision / recall / f1
    result["decision_per_class"] = per_class_f1(
        level2_preds,
        level2_refs,
        level2_classes,
    )
    result["decision_level1_macro_f1"] = macro_f1(result["level1_per_class"], level1_classes)
    result["decision_level2_macro_f1"] = macro_f1(result["decision_per_class"], level2_classes)
    result["decision_level1_micro"] = micro_prf1(level1_preds, level1_refs, level1_classes)
    result["decision_level2_micro"] = micro_prf1(level2_preds, level2_refs, level2_classes)
    result["decision_parse_fail_count"] = decision_parse_fail_count
    result["decision_eval_count"] = len(level2_refs)
    result["reference_parse_fail_count"] = reference_parse_fail_count

    return result


def _mean_se(values: list[float]) -> dict[str, float]:
    """Return mean and standard error for run-level values."""
    if not values:
        return {"mean": 0.0, "se": 0.0}
    mean = sum(values) / len(values)
    sd = _sample_std(values)
    se = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "se": se}


def _format_f1(value: float, mean_se: dict[str, float] | None = None) -> str:
    """Format F1 either as a plain value or as run-level mean ± SE."""
    if mean_se is None:
        return f"{value:.4f}"
    return f"{mean_se['mean']:.4f} ± {mean_se['se']:.4f}"


def build_rq2_result_f1_mean_se(entries: list[dict]) -> dict | None:
    """Compute run-level Mean ± SE for result F1 metrics in RQ2 repeats."""
    by_run: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        run_id = entry.get("run_id")
        if run_id is None:
            return None
        by_run[int(run_id)].append(entry)

    if len(by_run) < 2:
        return None

    run_metrics = []
    for run_id in sorted(by_run):
        run_entries = by_run[run_id]
        run_metrics.append(compute_metrics(
            [e["prediction"] for e in run_entries],
            [e["reference"] for e in run_entries],
        ))

    level1_classes = ["起诉", "不起诉"]
    level2_classes = ["起诉", "相对不起诉", "法定不起诉", "存疑不起诉"]
    return {
        "level1_macro_f1": _mean_se([m["decision_level1_macro_f1"] for m in run_metrics]),
        "level1_micro_f1": _mean_se([m["decision_level1_micro"]["f1"] for m in run_metrics]),
        "level2_macro_f1": _mean_se([m["decision_level2_macro_f1"] for m in run_metrics]),
        "level2_micro_f1": _mean_se([m["decision_level2_micro"]["f1"] for m in run_metrics]),
        "level1_per_class": {
            cls: _mean_se([m["level1_per_class"][cls]["f1"] for m in run_metrics])
            for cls in level1_classes
        },
        "level2_per_class": {
            cls: _mean_se([m["decision_per_class"][cls]["f1"] for m in run_metrics])
            for cls in level2_classes
        },
    }


def build_metrics_markdown(metrics: dict, *, model: str, variant: str,
                           num_samples: int,
                           result_f1_mean_se: dict | None = None) -> str:
    """将评估结果转为可读的 Markdown 表格。"""
    articles = {
        "precision": round(metrics["articles_precision"], 4),
        "recall": round(metrics["articles_recall"], 4),
        "f1": round(metrics["articles_f1"], 4),
    }
    decision = {
        "level1_macro_f1": metrics["decision_level1_macro_f1"],
        "level2_macro_f1": metrics["decision_level2_macro_f1"],
        "level1_micro_f1": metrics["decision_level1_micro"]["f1"],
        "level2_micro_f1": metrics["decision_level2_micro"]["f1"],
        "level1_per_class": metrics["level1_per_class"],
        "per_class": metrics["decision_per_class"],
    }
    level1_f1_stats = (result_f1_mean_se or {}).get("level1_per_class", {})
    level2_f1_stats = (result_f1_mean_se or {}).get("level2_per_class", {})

    lines = [
        "# 评估指标",
        "",
        f"- model: `{model}`",
        f"- variant: `{variant}`",
        f"- num_samples: `{num_samples}`",
        f"- decision_parse_fail_count: `{metrics['decision_parse_fail_count']}`",
        f"- reference_parse_fail_count: `{metrics['reference_parse_fail_count']}`",
        f"- decision_eval_count: `{metrics['decision_eval_count']}`",
        "",
        "## 过程指标（法条）",
        "",
        "| Precision | Recall | F1 |",
        "| ---: | ---: | ---: |",
        f"| {articles['precision']:.4f} | {articles['recall']:.4f} | {articles['f1']:.4f} |",
        "",
        "## 结果指标（决定）",
        "",
        "| Metric | F1 |",
        "| --- | ---: |",
        f"| Level1 Macro-F1 | {_format_f1(decision['level1_macro_f1'], (result_f1_mean_se or {}).get('level1_macro_f1'))} |",
        f"| Level1 Micro-F1 | {_format_f1(decision['level1_micro_f1'], (result_f1_mean_se or {}).get('level1_micro_f1'))} |",
        f"| Level2 Macro-F1 | {_format_f1(decision['level2_macro_f1'], (result_f1_mean_se or {}).get('level2_macro_f1'))} |",
        f"| Level2 Micro-F1 | {_format_f1(decision['level2_micro_f1'], (result_f1_mean_se or {}).get('level2_micro_f1'))} |",
        "",
        "### Level1 各类别（起诉 / 不起诉）",
        "",
        "| 类别 | Precision | Recall | F1 | Count |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cls in sorted(decision["level1_per_class"]):
        info = decision["level1_per_class"][cls]
        lines.append(
            f"| {cls} | {info['precision']:.4f} | {info['recall']:.4f} | "
            f"{_format_f1(info['f1'], level1_f1_stats.get(cls))} | {info['count']} |"
        )

    lines.extend([
        "",
        "### Level2 各类别（四分类）",
        "",
        "| 类别 | Precision | Recall | F1 | Count |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])

    for cls in sorted(decision["per_class"]):
        info = decision["per_class"][cls]
        lines.append(
            f"| {cls} | {info['precision']:.4f} | {info['recall']:.4f} | "
            f"{_format_f1(info['f1'], level2_f1_stats.get(cls))} | {info['count']} |"
        )

    lines.append("")
    return "\n".join(lines)


def _sample_std(values: list[float]) -> float:
    """样本标准差。"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def _t_critical_95(df: int) -> float:
    """双侧 95% CI 的 t 临界值；df 超出表时退化为正态近似。"""
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 1:
        return 0.0
    return 1.96


def build_rq2_stability_markdown(entries: list[dict]) -> str:
    """基于 RQ2 重复实验的 run_id，生成 run-level 稳定性指标表。"""
    by_run: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        run_id = entry.get("run_id")
        if run_id is None:
            return ""
        by_run[int(run_id)].append(entry)

    if len(by_run) < 2:
        return ""

    run_metrics = []
    for run_id in sorted(by_run):
        run_entries = by_run[run_id]
        metrics = compute_metrics(
            [e["prediction"] for e in run_entries],
            [e["reference"] for e in run_entries],
        )
        run_metrics.append({
            "run_id": run_id,
            "n": len(run_entries),
            "articles_f1": metrics["articles_f1"],
            "level1_macro_f1": metrics["decision_level1_macro_f1"],
            "level1_micro_f1": metrics["decision_level1_micro"]["f1"],
            "level2_macro_f1": metrics["decision_level2_macro_f1"],
            "level2_micro_f1": metrics["decision_level2_micro"]["f1"],
            "decision_parse_fail_count": metrics["decision_parse_fail_count"],
        })

    metric_specs = [
        ("Articles F1", "articles_f1"),
        ("Level1 Macro-F1", "level1_macro_f1"),
        ("Level1 Micro-F1", "level1_micro_f1"),
        ("Level2 Macro-F1", "level2_macro_f1"),
        ("Level2 Micro-F1", "level2_micro_f1"),
    ]

    lines = [
        "",
        "## RQ2 5 次重复稳定性",
        "",
        "### 各 run 指标",
        "",
        "| Run | N | Articles F1 | L1 Macro-F1 | L1 Micro-F1 | L2 Macro-F1 | L2 Micro-F1 | Parse Fail |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in run_metrics:
        lines.append(
            f"| {row['run_id']} | {row['n']} | "
            f"{row['articles_f1']:.4f} | "
            f"{row['level1_macro_f1']:.4f} | "
            f"{row['level1_micro_f1']:.4f} | "
            f"{row['level2_macro_f1']:.4f} | "
            f"{row['level2_micro_f1']:.4f} | "
            f"{row['decision_parse_fail_count']} |"
        )

    lines.extend([
        "",
        "### 稳定性汇总",
        "",
        "| Metric | Mean | SD | SE | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    n_runs = len(run_metrics)
    t_value = _t_critical_95(n_runs - 1)
    for label, key in metric_specs:
        values = [row[key] for row in run_metrics]
        mean = sum(values) / n_runs
        sd = _sample_std(values)
        se = sd / math.sqrt(n_runs) if n_runs > 1 else 0.0
        half_width = t_value * se
        lines.append(
            f"| {label} | {mean:.4f} | {sd:.4f} | {se:.4f} | "
            f"[{mean - half_width:.4f}, {mean + half_width:.4f}] |"
        )

    lines.extend([
        "",
        f"> 95% CI 基于 run-level 指标计算，使用 t 分布临界值（n={n_runs}, df={n_runs - 1}）。",
        "",
    ])
    return "\n".join(lines)


def infer_model_variant_from_dir(result_dir: str) -> tuple[str, str]:
    """从结果目录名推断 model 和 variant。"""
    name = os.path.basename(os.path.normpath(result_dir))
    # 去掉末尾时间戳，如 20260429_135541
    base = re.sub(r"_\d{8}_\d{6}$", "", name)

    # 常见 variant 起始标记
    variant_markers = ("original", "definitions", "one_shot")

    # OpenRouter/新命名: <model>_<variant>，variant 以 marker 开头
    parts = base.split("_")
    marker_idx = -1
    for i, p in enumerate(parts):
        if p in variant_markers:
            marker_idx = i
            break
    if marker_idx > 0:
        model = "_".join(parts[:marker_idx])
        variant = "_".join(parts[marker_idx:])
        return model, variant

    # vLLM 历史命名: <model>_test_<variant>
    if "_test_" in base:
        model, variant = base.split("_test_", 1)
        return model, variant

    # 兜底：无法识别则整串作为 model
    return base, ""


def _load_existing_metrics_metadata(result_dir: str) -> tuple[str | None, str | None, str]:
    """尽量从已有 metrics.md 中恢复 model / variant / 额外说明段。"""
    md_path = os.path.join(result_dir, "metrics.md")
    if not os.path.exists(md_path):
        return None, None, ""

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    model = None
    variant = None
    m = re.search(r"^- model: `([^`]+)`", text, re.MULTILINE)
    if m:
        model = m.group(1)
    m = re.search(r"^- variant: `([^`]+)`", text, re.MULTILINE)
    if m:
        variant = m.group(1)

    extra = ""
    extra_start = text.find("\n## RQ2 重复测试")
    if extra_start >= 0:
        next_section = text.find("\n## RQ2 5 次重复稳定性", extra_start + 1)
        if next_section >= 0:
            extra = text[extra_start:next_section].rstrip()
        else:
            extra = text[extra_start:].rstrip()

    return model, variant, extra


# ============================================================
#  独立运行：从 details_*.json 重算 metrics.md
# ============================================================

def _detail_sort_key(entry: dict) -> tuple:
    """兼容普通评估的 index 和 RQ2 重复评估的 sample_index/run_id。"""
    if "index" in entry:
        return (0, entry["index"], 0)
    if "sample_index" in entry:
        return (1, entry["sample_index"], entry.get("run_id", 0))
    return (2, str(entry.get("id", "")), 0)


def load_details(result_dir: str) -> list[dict]:
    """读取结果目录下所有 details_*.json，合并为一个列表。"""
    pattern = os.path.join(result_dir, "details_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在 {result_dir} 下未找到 details_*.json")

    all_entries = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            entries = json.load(f)
        all_entries.extend(entries)
        logger.info(f"  读取 {os.path.basename(fp)}: {len(entries)} 条")

    # 按样本索引排序，保持普通评估和 RQ2 重复评估的顺序一致
    all_entries.sort(key=_detail_sort_key)
    return all_entries


def recompute_for_dir(result_dir: str) -> None:
    """对单个结果目录重新计算 metrics.md。"""
    logger.info(f"处理目录: {result_dir}")

    # 优先沿用已有 metrics.md 的元信息；没有则从目录名推断
    model, variant, extra_section = _load_existing_metrics_metadata(result_dir)
    if model is None or variant is None:
        model_guess, variant_guess = infer_model_variant_from_dir(result_dir)
        model = model or model_guess
        variant = variant or variant_guess

    # 从 details 文件加载所有样本
    entries = load_details(result_dir)
    predictions = [e["prediction"] for e in entries]
    references = [e["reference"] for e in entries]
    logger.info(f"  共 {len(entries)} 个样本")

    # 计算指标
    metrics = compute_metrics(predictions, references)
    logger.info(
        f"  决定标签可评估样本 {metrics['decision_eval_count']} 个, "
        f"prediction 解析失败 {metrics['decision_parse_fail_count']} 个, "
        f"reference 解析失败 {metrics['reference_parse_fail_count']} 个"
    )
    md_content = build_metrics_markdown(
        metrics,
        model=model,
        variant=variant,
        num_samples=len(entries),
        result_f1_mean_se=build_rq2_result_f1_mean_se(entries),
    )
    if extra_section:
        md_content = md_content.rstrip() + "\n" + extra_section + "\n"
    stability_content = build_rq2_stability_markdown(entries)
    if stability_content:
        md_content = md_content.rstrip() + "\n" + stability_content

    # 只写入 Markdown 表格
    metrics_md_path = os.path.join(result_dir, "metrics.md")
    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"  已生成 {metrics_md_path}")

    # 打印摘要
    print(
        f"\n  Level1 Macro-F1: {round(metrics['decision_level1_macro_f1'], 4)} "
        f"(Micro-F1={round(metrics['decision_level1_micro']['f1'], 4)})"
    )
    for cls, info in metrics["level1_per_class"].items():
        print(
            f"    {cls}: F1={round(info['f1'], 4)} "
            f"(P={round(info['precision'], 4)}, R={round(info['recall'], 4)}, {info['count']} 样本)"
        )
    print(
        f"  Level2 Macro-F1: {round(metrics['decision_level2_macro_f1'], 4)} "
        f"(Micro-F1={round(metrics['decision_level2_micro']['f1'], 4)})"
    )
    for cls, info in metrics["decision_per_class"].items():
        print(
            f"    {cls}: F1={round(info['f1'], 4)} "
            f"(P={round(info['precision'], 4)}, R={round(info['recall'], 4)}, {info['count']} 样本)"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="从 details_*.json 重新计算 metrics.md"
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="一个或多个结果目录路径",
    )
    args = parser.parse_args()

    for d in args.result_dirs:
        recompute_for_dir(d)


if __name__ == "__main__":
    main()
