#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/metrics.py

PDP 评估指标计算模块。

既可作为库被 evaluate.py 导入，也可独立运行，从已有的 details_*.json
重新计算并更新 metrics.json。

独立用法:
    python eval/metrics.py results/Qwen3-4B_test_baseline_20260312_055107
    python eval/metrics.py dir1 dir2 dir3   # 批量更新多个目录
"""

import argparse
import glob
import json
import logging
import os
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


def _to_level1(decision: str) -> str:
    """将四分类决定映射为二分类：起诉 / 不起诉。"""
    return "不起诉" if decision in NON_PROSECUTION_TYPES else "起诉"


def compute_metrics(predictions: list[dict], references: list[dict]) -> dict:
    """
    计算所有评估指标。

    过程评估 (relevant_articles): 法条 F1
    结果评估 (decision):
      - 第一级（二分类）：起诉 / 不起诉 Accuracy
      - 第二级（四分类）：起诉、相对不起诉、法定不起诉、存疑不起诉 Accuracy
    """
    metrics = defaultdict(list)
    decision_per_class = defaultdict(list)  # 各决定类别的准确率（四分类）
    level1_per_class = defaultdict(list)    # 各决定类别的准确率（二分类）

    for pred, ref in zip(predictions, references):
        # --- 过程评估：法条 F1（所有样本） ---
        pred_arts = set(pred["relevant_articles"])
        gold_arts = set(ref["relevant_articles"])
        p, r, f = set_precision_recall_f1(pred_arts, gold_arts)
        metrics["articles_precision"].append(p)
        metrics["articles_recall"].append(r)
        metrics["articles_f1"].append(f)

        # --- 结果评估：第一级（二分类）---
        pred_l1 = _to_level1(pred["decision"])
        ref_l1 = _to_level1(ref["decision"])
        l1_correct = 1.0 if pred_l1 == ref_l1 else 0.0
        metrics["decision_level1_accuracy"].append(l1_correct)
        level1_per_class[ref_l1].append(l1_correct)

        # --- 结果评估：第二级（四分类）---
        correct = 1.0 if pred["decision"] == ref["decision"] else 0.0
        metrics["decision_level2_accuracy"].append(correct)
        decision_per_class[ref["decision"]].append(correct)

    # 宏平均
    result = {}
    for key, values in metrics.items():
        result[key] = sum(values) / len(values) if values else 0.0

    # 第一级（二分类）各类别的准确率
    result["level1_per_class"] = {}
    for cls, values in level1_per_class.items():
        result["level1_per_class"][cls] = {
            "accuracy": sum(values) / len(values) if values else 0.0,
            "count": len(values),
        }

    # 第二级（四分类）各类别的准确率
    result["decision_per_class"] = {}
    for cls, values in decision_per_class.items():
        result["decision_per_class"][cls] = {
            "accuracy": sum(values) / len(values) if values else 0.0,
            "count": len(values),
        }

    return result


def build_metrics_json(metrics: dict, *, model: str, variant: str,
                       num_samples: int, parse_fail_count: int) -> dict:
    """将 compute_metrics 的结果组装为 metrics.json 输出格式。"""
    level1_per_class_output = {}
    for cls, info in metrics["level1_per_class"].items():
        level1_per_class_output[cls] = {
            "accuracy": round(info["accuracy"], 4),
            "count": info["count"],
        }
    decision_per_class_output = {}
    for cls, info in metrics["decision_per_class"].items():
        decision_per_class_output[cls] = {
            "accuracy": round(info["accuracy"], 4),
            "count": info["count"],
        }
    return {
        "model": model,
        "variant": variant,
        "num_samples": num_samples,
        "parse_fail_count": parse_fail_count,
        "process_metrics": {
            "articles": {
                "precision": round(metrics["articles_precision"], 4),
                "recall": round(metrics["articles_recall"], 4),
                "f1": round(metrics["articles_f1"], 4),
            },
        },
        "result_metrics": {
            "decision": {
                "level1_accuracy": round(metrics["decision_level1_accuracy"], 4),
                "level1_per_class": level1_per_class_output,
                "level2_accuracy": round(metrics["decision_level2_accuracy"], 4),
                "per_class": decision_per_class_output,
            },
        },
    }


# ============================================================
#  独立运行：从 details_*.json 重算 metrics.json
# ============================================================

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

    # 按 index 排序，保持顺序一致
    all_entries.sort(key=lambda e: e["index"])
    return all_entries


def recompute_for_dir(result_dir: str) -> None:
    """对单个结果目录重新计算 metrics.json。"""
    logger.info(f"处理目录: {result_dir}")

    # 读取旧 metrics.json 以获取 model / variant 元信息
    old_metrics_path = os.path.join(result_dir, "metrics.json")
    if os.path.exists(old_metrics_path):
        with open(old_metrics_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        model = old.get("model", "")
        variant = old.get("variant", "")
    else:
        model, variant = "", ""

    # 从 details 文件加载所有样本
    entries = load_details(result_dir)
    predictions = [e["prediction"] for e in entries]
    references = [e["reference"] for e in entries]
    parse_fail_count = sum(1 for p in predictions if not p["decision"])

    logger.info(f"  共 {len(entries)} 个样本, 解析失败 {parse_fail_count} 个")

    # 计算指标
    metrics = compute_metrics(predictions, references)
    output = build_metrics_json(
        metrics,
        model=model,
        variant=variant,
        num_samples=len(entries),
        parse_fail_count=parse_fail_count,
    )

    # 写入
    with open(old_metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"  已更新 {old_metrics_path}")

    # 打印摘要
    d = output["result_metrics"]["decision"]
    print(f"\n  Level1 Accuracy: {d['level1_accuracy']}")
    for cls, info in d["level1_per_class"].items():
        print(f"    {cls}: {info['accuracy']} ({info['count']} 样本)")
    print(f"  Level2 Accuracy: {d['level2_accuracy']}")
    for cls, info in d["per_class"].items():
        print(f"    {cls}: {info['accuracy']} ({info['count']} 样本)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="从 details_*.json 重新计算 metrics.json"
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
