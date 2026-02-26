#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/evaluate.py

使用 vLLM 对 PDP 数据集进行评估。

模型预测四个目标：
  - relevant_articles (法条：刑法+刑诉法+刑事诉讼规则) — 过程评估
  - decision           (起诉决定)                       — 结果评估
  - charges            (罪名)                           — 结果评估

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --split ood
    python eval/evaluate.py --split test --model-path models/Qwen3-0.6B
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# 将项目根目录加入 sys.path，以便导入 prompt_template
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk
from vllm import LLM, SamplingParams

from prompt_template import build_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
#  输出解析
# ============================================================

def parse_answer(text: str) -> dict:
    """
    从模型输出中解析 <answer>...</answer> 块，提取结构化字段。

    返回:
        {
            "relevant_articles_cl": list[str],
            "relevant_articles_cpl": list[str],
            "relevant_articles_cpr": list[str],
            "decision": str,
            "charges": list[str],
        }
    """
    result = {
        "relevant_articles_cl": [],
        "relevant_articles_cpl": [],
        "relevant_articles_cpr": [],
        "decision": "",
        "charges": [],
    }

    # 提取 <answer> 块
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not answer_match:
        logger.warning("未找到 <answer> 标签，尝试全文解析")
        answer_block = text
    else:
        answer_block = answer_match.group(1)

    # --- 适用法条 ---
    articles_section = re.search(
        r"【适用法条】(.*?)(?=【审查分析】|【最终结论】|$)",
        answer_block,
        re.DOTALL,
    )
    if articles_section:
        section_text = articles_section.group(1)

        # 刑法法条
        cl_match = re.search(r"刑法[：:](.*?)(?:\n|$)", section_text)
        if cl_match:
            raw = cl_match.group(1).strip()
            result["relevant_articles_cl"] = _split_articles(raw)

        # 刑事诉讼法法条
        cpl_match = re.search(r"刑事诉讼法[：:](.*?)(?:\n|$)", section_text)
        if cpl_match:
            raw = cpl_match.group(1).strip()
            result["relevant_articles_cpl"] = _split_articles(raw)

        # 刑事诉讼规则法条
        cpr_match = re.search(r"刑事诉讼规则[：:](.*?)(?:\n|$)", section_text)
        if cpr_match:
            raw = cpr_match.group(1).strip()
            result["relevant_articles_cpr"] = _split_articles(raw)

    # --- 最终结论 ---
    conclusion_section = re.search(
        r"【最终结论】(.*?)(?=</answer>|$)", answer_block, re.DOTALL
    )
    if conclusion_section:
        section_text = conclusion_section.group(1)

        # 决定
        dec_match = re.search(r"决定[：:]\s*(.*?)(?:\n|$)", section_text)
        if dec_match:
            result["decision"] = dec_match.group(1).strip()

        # 罪名
        charges_match = re.search(r"罪名[：:]\s*(.*?)(?:\n|$)", section_text)
        if charges_match:
            raw = charges_match.group(1).strip()
            if raw and raw != "无":
                result["charges"] = _split_items(raw)

    return result


def _split_articles(raw: str) -> list[str]:
    """将 '第XXX条、第YYY条第Z款' 拆分为列表。"""
    items = re.split(r"[、，,;；]", raw)
    return [a.strip() for a in items if a.strip()]


def _split_items(raw: str) -> list[str]:
    """通用分隔：按 '、' '，' ',' 拆分。"""
    items = re.split(r"[、，,;；]", raw)
    return [a.strip() for a in items if a.strip()]


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


def _build_mixed_articles(cl: list, cpl: list, cpr: list) -> set:
    """将各法律的条文加前缀后合并为一个集合，避免不同法律的相同条号冲突。"""
    mixed = set()
    for a in cl:
        mixed.add(f"cl:{a}")
    for a in cpl:
        mixed.add(f"cpl:{a}")
    for a in cpr:
        mixed.add(f"cpr:{a}")
    return mixed


def compute_metrics(predictions: list[dict], references: list[dict]) -> dict:
    """
    计算所有评估指标。

    过程评估 (relevant_articles): 混合法条 F1
    结果评估 (decision): Accuracy
    结果评估 (charges): Precision / Recall / F1
    """
    metrics = defaultdict(list)

    for pred, ref in zip(predictions, references):
        # 判断参考答案的决定类型是否有罪名
        has_charges = ref["decision"] in ("起诉", "相对不起诉")

        # --- 过程评估：混合法条 F1（所有样本） ---
        pred_mixed = _build_mixed_articles(
            pred["relevant_articles_cl"],
            pred["relevant_articles_cpl"],
            pred["relevant_articles_cpr"],
        )
        gold_mixed = _build_mixed_articles(
            ref["relevant_articles_cl"],
            ref["relevant_articles_cpl"],
            ref["relevant_articles_cpr"],
        )
        p, r, f = set_precision_recall_f1(pred_mixed, gold_mixed)
        metrics["articles_precision"].append(p)
        metrics["articles_recall"].append(r)
        metrics["articles_f1"].append(f)

        # --- 结果评估：决定（所有样本） ---
        metrics["decision_accuracy"].append(
            1.0 if pred["decision"] == ref["decision"] else 0.0
        )

        # --- 结果评估：罪名（仅起诉/相对不起诉） ---
        if has_charges:
            pred_charges = set(pred["charges"])
            gold_charges = set(ref["charges"])
            p, r, f = set_precision_recall_f1(pred_charges, gold_charges)
            metrics["charges_precision"].append(p)
            metrics["charges_recall"].append(r)
            metrics["charges_f1"].append(f)

    # 宏平均
    result = {}
    for key, values in metrics.items():
        result[key] = sum(values) / len(values) if values else 0.0

    return result


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PDP 数据集评估脚本")
    parser.add_argument(
        "--model-path",
        default="models/Qwen3-0.6B",
        help="模型路径 (default: models/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--data-path",
        default="data/pdp25k",
        help="HuggingFace 数据集路径 (default: data/pdp25k)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "ood"],
        help="评估的数据集 split (default: test)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM 最大模型长度 (default: 4096)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="生成的最大 token 数 (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度 (default: 0.0，贪心解码)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行大小 (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU 显存利用率 (default: 0.9)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="结果输出目录 (default: results)",
    )
    args = parser.parse_args()

    # ---- 加载数据 ----
    logger.info(f"加载数据集: {args.data_path} [{args.split}]")
    try:
        dataset_dict = load_from_disk(args.data_path)
    except FileNotFoundError:
        from datasets import load_dataset
        dataset_dict = load_dataset(args.data_path)
    dataset = dataset_dict[args.split]

    logger.info(f"评估样本数: {len(dataset)}")

    # ---- 构建提示 ----
    logger.info("构建提示词...")
    conversations = []
    for sample in dataset:
        messages = build_messages(
            person_info=sample["person_info"],
            procedure=sample["procedure"],
            fact=sample["fact"],
        )
        conversations.append(messages)

    # ---- 加载模型 ----
    logger.info(f"加载模型: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # ---- 批量推理 ----
    logger.info("开始批量推理...")
    start_time = time.time()

    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
    )

    elapsed = time.time() - start_time
    logger.info(f"推理完成，耗时 {elapsed:.1f}s ({len(dataset) / elapsed:.1f} samples/s)")

    # ---- 解析预测结果 ----
    logger.info("解析模型输出...")
    predictions = []
    raw_outputs = []
    parse_fail_count = 0

    for output in outputs:
        generated_text = output.outputs[0].text
        raw_outputs.append(generated_text)

        parsed = parse_answer(generated_text)
        predictions.append(parsed)

        # 检查解析完整性
        if not parsed["decision"]:
            parse_fail_count += 1

    if parse_fail_count > 0:
        logger.warning(
            f"有 {parse_fail_count}/{len(predictions)} 个样本未成功解析出决定字段"
        )

    # ---- 构建参考答案 ----
    references = []
    for sample in dataset:
        references.append({
            "relevant_articles_cl": list(sample["relevant_articles_cl"]),
            "relevant_articles_cpl": list(sample["relevant_articles_cpl"]),
            "relevant_articles_cpr": list(sample["relevant_articles_cpr"]),
            "decision": sample["decision"],
            "charges": list(sample["charges"]),
        })

    # ---- 计算指标 ----
    logger.info("计算评估指标...")
    metrics = compute_metrics(predictions, references)

    # ---- 输出结果 ----
    print("\n" + "=" * 60)
    print(f"  PDP 评估结果 — {args.split} split ({len(dataset)} samples)")
    print(f"  模型: {args.model_path}")
    print("=" * 60)

    print("\n【过程评估】")
    print(f"  法条 (articles):")
    print(f"    Precision: {metrics['articles_precision']:.4f}")
    print(f"    Recall:    {metrics['articles_recall']:.4f}")
    print(f"    F1:        {metrics['articles_f1']:.4f}")

    print("\n【结果评估】")
    print(f"  决定 (decision):")
    print(f"    Accuracy:  {metrics['decision_accuracy']:.4f}")
    print(f"  罪名 (charges):")
    print(f"    Precision: {metrics['charges_precision']:.4f}")
    print(f"    Recall:    {metrics['charges_recall']:.4f}")
    print(f"    F1:        {metrics['charges_f1']:.4f}")

    print(f"\n  解析失败率: {parse_fail_count}/{len(predictions)}")
    print("=" * 60)

    # ---- 保存结果 ----
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = Path(args.model_path).name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_prefix = f"{model_name}_{args.split}_{timestamp}"

    # 保存指标
    metrics_file = os.path.join(args.output_dir, f"{result_prefix}_metrics.json")
    metrics_output = {
        "model": args.model_path,
        "split": args.split,
        "num_samples": len(dataset),
        "elapsed_seconds": round(elapsed, 1),
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
                "accuracy": round(metrics["decision_accuracy"], 4),
            },
            "charges": {
                "precision": round(metrics["charges_precision"], 4),
                "recall": round(metrics["charges_recall"], 4),
                "f1": round(metrics["charges_f1"], 4),
            },
        },
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {metrics_file}")

    # 保存详细预测
    details_file = os.path.join(args.output_dir, f"{result_prefix}_details.jsonl")
    with open(details_file, "w", encoding="utf-8") as f:
        for i, (pred, ref, raw) in enumerate(
            zip(predictions, references, raw_outputs)
        ):
            record = {
                "index": i,
                "id": dataset[i]["id"],
                "prediction": pred,
                "reference": ref,
                "raw_output": raw,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"详细结果已保存: {details_file}")


if __name__ == "__main__":
    main()
