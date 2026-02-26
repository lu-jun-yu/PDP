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
    python eval/evaluate.py --batch-size 500          # 分批推理，支持断点续推
    python eval/evaluate.py --no-resume               # 忽略断点，从头评估
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

def _strip_think_block(text: str) -> str:
    """去除 <think>...</think> 块，返回剩余内容。"""
    think_end = text.find("</think>")
    if think_end != -1:
        return text[think_end + len("</think>"):].strip()
    return text.strip()


def parse_answer(text: str) -> dict:
    """
    从模型输出中解析结构化字段（去除 <think> 块后解析）。

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

    answer_block = _strip_think_block(text)

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
        r"【最终结论】(.*?)$", answer_block, re.DOTALL
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="分批推理的批次大小，0 表示一次性推理所有样本 (default: 0)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用断点续推，从头开始评估",
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

    # ---- 断点续推 ----
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = Path(args.model_path).name
    checkpoint_file = os.path.join(
        args.output_dir, f"{model_name}_{args.split}_checkpoint.jsonl"
    )

    completed = {}  # index -> record
    if not args.no_resume and os.path.exists(checkpoint_file):
        logger.info(f"检测到断点文件: {checkpoint_file}")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed[record["index"]] = record
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"断点文件第 {line_no} 行解析失败，跳过")
        logger.info(f"从断点恢复: 已完成 {len(completed)}/{len(dataset)} 个样本")
    elif args.no_resume and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("已删除旧断点文件，从头开始评估")

    # 找出未完成的样本索引
    remaining_indices = [i for i in range(len(dataset)) if i not in completed]

    if not remaining_indices:
        logger.info("所有样本已完成推理，直接计算指标")
    else:
        logger.info(f"待推理样本数: {len(remaining_indices)}")

        # ---- 构建提示 ----
        logger.info("构建提示词...")
        conversations = []
        for i in remaining_indices:
            sample = dataset[i]
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

        # ---- 分批推理 + 断点保存 ----
        batch_size = args.batch_size if args.batch_size > 0 else len(conversations)
        num_batches = (len(conversations) + batch_size - 1) // batch_size
        logger.info(
            f"开始推理... (共 {num_batches} 个批次, "
            f"batch_size={'all' if args.batch_size == 0 else batch_size})"
        )
        start_time = time.time()
        new_count = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(conversations))
            batch_conversations = conversations[batch_start:batch_end]
            batch_indices = remaining_indices[batch_start:batch_end]

            if num_batches > 1:
                logger.info(
                    f"批次 {batch_idx + 1}/{num_batches}: "
                    f"样本 {batch_start}~{batch_end - 1} ({len(batch_conversations)} 个)"
                )

            outputs = llm.chat(
                messages=batch_conversations,
                sampling_params=sampling_params,
            )

            # 解析并追加保存到断点文件
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                for j, output in enumerate(outputs):
                    idx = batch_indices[j]
                    generated_text = output.outputs[0].text
                    parsed = parse_answer(generated_text)
                    sample = dataset[idx]
                    ref = {
                        "relevant_articles_cl": list(sample["relevant_articles_cl"]),
                        "relevant_articles_cpl": list(sample["relevant_articles_cpl"]),
                        "relevant_articles_cpr": list(sample["relevant_articles_cpr"]),
                        "decision": sample["decision"],
                        "charges": list(sample["charges"]),
                    }
                    record = {
                        "index": idx,
                        "id": sample["id"],
                        "prediction": parsed,
                        "reference": ref,
                        "raw_output": generated_text,
                    }
                    completed[idx] = record
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            new_count += len(batch_conversations)
            elapsed = time.time() - start_time
            logger.info(
                f"进度: {len(completed)}/{len(dataset)} "
                f"(本次新增 {new_count}, 耗时 {elapsed:.1f}s, "
                f"{new_count / elapsed:.1f} samples/s)"
            )

    # ---- 汇总结果 ----
    logger.info("汇总预测结果...")
    predictions = []
    references = []
    parse_fail_count = 0

    for i in range(len(dataset)):
        if i not in completed:
            logger.error(f"样本 {i} 缺失，请使用 --no-resume 重新评估")
            sys.exit(1)
        record = completed[i]
        predictions.append(record["prediction"])
        references.append(record["reference"])
        if not record["prediction"]["decision"]:
            parse_fail_count += 1

    if parse_fail_count > 0:
        logger.warning(
            f"有 {parse_fail_count}/{len(predictions)} 个样本未成功解析出决定字段"
        )

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

    # ---- 保存指标 ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_prefix = f"{model_name}_{args.split}_{timestamp}"
    metrics_file = os.path.join(args.output_dir, f"{result_prefix}_metrics.json")
    metrics_output = {
        "model": args.model_path,
        "split": args.split,
        "num_samples": len(dataset),
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

    # 保存详细预测（格式化 JSON）
    details_file = os.path.join(args.output_dir, f"{result_prefix}_details.json")
    details = []
    for i in range(len(dataset)):
        record = completed[i]
        details.append({
            "index": record["index"],
            "id": record["id"],
            "prediction": record["prediction"],
            "reference": record["reference"],
            "raw_output": record["raw_output"],
        })
    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存: {details_file}")

    # 清理断点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("断点文件已清理")


if __name__ == "__main__":
    main()
