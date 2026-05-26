#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/evaluate_vllm.py

使用 vLLM 对 PDP 数据集进行评估。

模型预测两个目标：
  - relevant_articles (法条，带前缀 cl:/cpl:/cpr:) — 过程评估
  - decision           (起诉决定)                       — 结果评估

评估分两级：
  - 第一级（二分类）：起诉 / 不起诉
  - 第二级（四分类）：起诉、相对不起诉、法定不起诉、存疑不起诉

支持 RQ2 风格的多次重复评估（每条样本重复 N 次，用于稳定性 / 方差分析），
通过 --num-repeats 控制；num_repeats > 1 时自动切换为 (sample_index, run_id)
断点键，并在 metrics.md 中追加 run-level 稳定性表。

Usage:
    python eval/evaluate_vllm.py
    python eval/evaluate_vllm.py --model-path models/Qwen3-0.6B
    python eval/evaluate_vllm.py --batch-size 500          # 分批推理，支持断点续推
    python eval/evaluate_vllm.py --no-resume               # 忽略断点，从头评估
    python eval/evaluate_vllm.py --prompt-variant definitions
    python eval/evaluate_vllm.py --prompt-variant one_shot
    # RQ2.2: 全 test split + 8 次重复
    python eval/evaluate_vllm.py --model-path models/Qwen2.5-7B-Instruct \
        --split test --num-repeats 8 --max-model-len 8192 --max-tokens 4096
    # Sampling/chat-template defaults are left to the model/vLLM backend.
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

from prompt_template import PROMPT_VARIANTS, build_messages
from eval.metrics import (
    build_metrics_markdown,
    build_rq2_stability_markdown,
    compute_metrics,
    normalize_decision_label,
)

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
            "relevant_articles": list[str],   # 带前缀，如 "cl:第XXX条"
            "decision": str,
        }
    """
    result = {
        "relevant_articles": [],
        "decision": "",
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
        articles = []

        cl_match = re.search(r"刑法[:：](.*?)(?:\n|$)", section_text)
        if cl_match:
            articles += [f"cl:{a}" for a in _split_articles(cl_match.group(1).strip())]

        cpl_match = re.search(r"刑事诉讼法[:：](.*?)(?:\n|$)", section_text)
        if cpl_match:
            articles += [f"cpl:{a}" for a in _split_articles(cpl_match.group(1).strip())]

        cpr_match = re.search(r"刑事诉讼规则[:：](.*?)(?:\n|$)", section_text)
        if cpr_match:
            articles += [f"cpr:{a}" for a in _split_articles(cpr_match.group(1).strip())]

        result["relevant_articles"] = articles

    # --- 最终结论 ---
    conclusion_section = re.search(
        r"【最终结论】(.*?)$", answer_block, re.DOTALL
    )
    if conclusion_section:
        section_text = conclusion_section.group(1)

        # 决定
        dec_match = re.search(r"决定[:：]\s*(.*?)(?:\n|$)", section_text)
        if dec_match:
            result["decision"] = dec_match.group(1).strip()

    return result


def _split_articles(raw: str) -> list[str]:
    """将 '第XXX条、第YYY条第Z款' 拆分为列表。"""
    items = re.split(r"[、，,;；]", raw)
    return [a.strip() for a in items if a.strip()]


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
        default="data/pdp4k",
        help="HuggingFace 数据集路径 (default: data/pdp4k)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="数据集 split 名 (default: test)",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="每条样本重复推理次数 (default: 1)。num_repeats>1 时启用 RQ2 风格的 (sample_index, run_id) 断点与稳定性表。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="基础随机种子 (default: None，即不显式设种子，与旧行为一致)。设为整数时，每次重复使用 seed=base_seed+run_id，确保独立性与可复现性。",
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
        "--enable-prefix-caching",
        action="store_true",
        default=True,
        help="启用 vLLM prefix caching (default: True)。多次重复评估同一 prompt 时显著加速。",
    )
    parser.add_argument(
        "--no-prefix-caching",
        dest="enable_prefix_caching",
        action="store_false",
        help="关闭 prefix caching",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="vLLM dtype (default: auto)。可选 auto / bfloat16 / float16 / float32。",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="结果输出目录 (default: results)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="分批推理的批次大小 (default: 100)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用断点续推，从头开始评估",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="original",
        help=(
            "消融实验 prompt 变体: original=原始prompt, "
            "definitions=原始prompt+definitions(prompt engineering), "
            "one_shot=原始prompt+one-shot learning(in-context learning) "
            "(default: original)"
        ),
    )
    args = parser.parse_args()

    if args.num_repeats < 1:
        logger.error("--num-repeats 至少为 1")
        sys.exit(1)

    multi_run = args.num_repeats > 1

    # ---- 加载数据 ----
    logger.info(f"加载数据集: {args.data_path} [{args.split}]")
    try:
        dataset_dict = load_from_disk(args.data_path)
    except FileNotFoundError:
        from datasets import load_dataset
        dataset_dict = load_dataset(args.data_path)

    if args.split not in dataset_dict:
        logger.error(
            "split '%s' 不存在。可用 splits: %s",
            args.split,
            list(dataset_dict.keys()),
        )
        sys.exit(1)

    dataset = dataset_dict[args.split]
    n = len(dataset)
    total_evals = n * args.num_repeats

    if multi_run:
        logger.info(
            "评估样本数: %d, 每样本重复: %d, 总推理次数: %d",
            n, args.num_repeats, total_evals,
        )
    else:
        logger.info(f"评估样本数: {n}")

    # ---- 断点续推 ----
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = Path(args.model_path).name
    if multi_run:
        variant_tag = f"{args.prompt_variant}_{args.split}_r{args.num_repeats}"
    else:
        variant_tag = args.prompt_variant
    checkpoint_file = os.path.join(
        args.output_dir, f"{model_name}_{variant_tag}_checkpoint.jsonl"
    )

    # 断点键统一为 (sample_index, run_id)；num_repeats=1 时 run_id 恒为 0
    completed: dict = {}  # (sample_index, run_id) -> record
    if not args.no_resume and os.path.exists(checkpoint_file):
        logger.info(f"检测到断点文件: {checkpoint_file}")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    si = record.get("sample_index", record.get("index"))
                    ri = record.get("run_id", 0)
                    if si is None:
                        raise KeyError("missing sample_index/index")
                    completed[(int(si), int(ri))] = record
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"断点文件第 {line_no} 行解析失败 ({e})，跳过")
        logger.info(
            "从断点恢复: 已完成 %d/%d 次推理",
            len(completed), total_evals,
        )
    elif args.no_resume and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("已删除旧断点文件，从头开始评估")

    # 找出未完成的 (sample_index, run_id)
    pending: list[tuple[int, int]] = []
    for si in range(n):
        for ri in range(args.num_repeats):
            if (si, ri) not in completed:
                pending.append((si, ri))

    if not pending:
        logger.info("所有样本已完成推理，直接计算指标")
    else:
        logger.info(f"待推理任务数: {len(pending)}")

        # ---- 构建提示（每条样本只渲染一次，跨 run 复用） ----
        logger.info("构建提示词...")
        variant_desc = {
            "original": "原始prompt",
            "definitions": "原始prompt+definitions (prompt engineering)",
            "one_shot": "原始prompt+one-shot learning (in-context learning)",
        }[args.prompt_variant]
        logger.info(f"消融实验模式：{variant_desc}")

        msg_cache: dict[int, list[dict]] = {}

        def messages_for(sample_index: int) -> list[dict]:
            if sample_index not in msg_cache:
                sample = dataset[sample_index]
                msg_cache[sample_index] = build_messages(
                    person_info=sample["person_info"],
                    procedure=sample["procedure"],
                    fact=sample["fact"],
                    prompt_variant=args.prompt_variant,
                )
            return msg_cache[sample_index]

        # ---- 加载模型 ----
        logger.info(f"加载模型: {args.model_path}")
        llm_kwargs = {
            "model": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": args.enable_prefix_caching,
            "dtype": args.dtype,
            "trust_remote_code": True,
        }
        if args.seed is not None:
            llm_kwargs["seed"] = args.seed
        llm = LLM(**llm_kwargs)

        # ---- 分批推理 + 断点保存 ----
        batch_size = args.batch_size if args.batch_size > 0 else len(pending)
        num_batches = (len(pending) + batch_size - 1) // batch_size
        logger.info(
            "开始推理... (共 %d 个批次, batch_size=%s, prefix_caching=%s)",
            num_batches,
            "all" if args.batch_size == 0 else batch_size,
            args.enable_prefix_caching,
        )
        start_time = time.time()
        new_count = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(pending))
            batch_tasks = pending[batch_start:batch_end]

            batch_conversations = [messages_for(si) for si, _ in batch_tasks]
            # 每个 (si, ri) 用 seed=base_seed+ri 的独立 SamplingParams：
            # 同一 ri 跨样本共享 seed 不会引入相关性（vLLM 每请求独立采样），
            # 不同 ri 之间生成轨迹独立但可复现，便于稳定性分析。
            # 当 --seed 未显式指定时（默认），不设置 seed，vLLM 走默认随机采样。
            if args.seed is not None:
                batch_sampling_params: list | SamplingParams = [
                    SamplingParams(
                        max_tokens=args.max_tokens,
                        seed=args.seed + ri,
                    )
                    for _, ri in batch_tasks
                ]
            else:
                batch_sampling_params = SamplingParams(max_tokens=args.max_tokens)

            if num_batches > 1:
                logger.info(
                    "批次 %d/%d: 任务 %d~%d (%d 个)",
                    batch_idx + 1, num_batches,
                    batch_start, batch_end - 1, len(batch_tasks),
                )

            outputs = llm.chat(
                messages=batch_conversations,
                sampling_params=batch_sampling_params,
            )

            # 解析并追加保存到断点文件
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                for j, output in enumerate(outputs):
                    si, ri = batch_tasks[j]
                    generated_text = output.outputs[0].text
                    parsed = parse_answer(generated_text)
                    sample = dataset[si]
                    ref = {
                        "relevant_articles": list(sample["relevant_articles"]),
                        "decision": sample["decision"],
                    }
                    record: dict = {
                        "sample_index": si,
                        "run_id": ri,
                        "id": sample["id"],
                        "person_info": sample["person_info"],
                        "procedure": sample["procedure"],
                        "fact": sample["fact"],
                        "prediction": parsed,
                        "reference": ref,
                        "raw_reasoning_and_decision": sample["raw_reasoning_and_decision"],
                        "raw_reasoning": "",
                        "raw_output": generated_text,
                    }
                    if not multi_run:
                        # 单次评估额外写入 index 字段以兼容旧下游脚本
                        record["index"] = si
                    completed[(si, ri)] = record
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            new_count += len(batch_tasks)
            elapsed = time.time() - start_time
            rate = new_count / elapsed if elapsed > 0 else 0.0
            remaining = len(pending) - new_count
            eta_sec = remaining / rate if rate > 0 else 0.0
            if eta_sec >= 3600:
                eta_str = f"{eta_sec / 3600:.1f}h"
            elif eta_sec >= 60:
                eta_str = f"{eta_sec / 60:.1f}min"
            else:
                eta_str = f"{eta_sec:.0f}s"
            logger.info(
                "进度: %d/%d (本次新增 %d, 耗时 %.1fs, %.2f infer/s, ETA %s)",
                len(completed), total_evals, new_count, elapsed, rate, eta_str,
            )

    # ---- 汇总结果 ----
    logger.info("汇总预测结果...")
    predictions = []
    references = []
    parse_fail_count = 0

    for si in range(n):
        for ri in range(args.num_repeats):
            key = (si, ri)
            if key not in completed:
                logger.error(
                    "缺失 sample_index=%d run_id=%d，请使用 --no-resume 重新评估",
                    si, ri,
                )
                sys.exit(1)
            record = completed[key]
            predictions.append(record["prediction"])
            references.append(record["reference"])
            if normalize_decision_label(record["prediction"].get("decision", "")) is None:
                parse_fail_count += 1

    if parse_fail_count > 0:
        logger.warning(
            f"有 {parse_fail_count}/{len(predictions)} 个样本的决定字段无法归一化为有效标签"
        )

    # ---- 计算指标 ----
    logger.info("计算评估指标...")
    metrics = compute_metrics(predictions, references)

    # ---- 输出结果 ----
    print("\n" + "=" * 60)
    if multi_run:
        print(
            f"  PDP 评估 — split={args.split}, "
            f"unique={n}, repeats={args.num_repeats}, total={total_evals}"
        )
    else:
        print(f"  PDP 评估结果 — {args.split} split ({n} samples)")
    print(f"  模型: {args.model_path}")
    print(f"  变体: {variant_tag}")
    print("=" * 60)

    print("\n【过程评估】")
    print(f"  法条 (articles):")
    print(f"    Precision: {metrics['articles_precision']:.4f}")
    print(f"    Recall:    {metrics['articles_recall']:.4f}")
    print(f"    F1:        {metrics['articles_f1']:.4f}")

    print("\n【结果评估】")
    print(f"  决定 (decision):")
    print(
        "    第一级（起诉/不起诉）"
        f"Macro-F1: {metrics['decision_level1_macro_f1']:.4f}, "
        f"Micro-F1: {metrics['decision_level1_micro']['f1']:.4f}"
    )
    for cls, info in metrics["level1_per_class"].items():
        print(
            f"      {cls}: F1={info['f1']:.4f} "
            f"(P={info['precision']:.4f}, R={info['recall']:.4f}, {info['count']} 样本)"
        )
    print(
        "    第二级（四分类）"
        f"Macro-F1: {metrics['decision_level2_macro_f1']:.4f}, "
        f"Micro-F1: {metrics['decision_level2_micro']['f1']:.4f}"
    )
    for cls, info in metrics["decision_per_class"].items():
        print(
            f"      {cls}: F1={info['f1']:.4f} "
            f"(P={info['precision']:.4f}, R={info['recall']:.4f}, {info['count']} 样本)"
        )

    print(f"\n  决定解析失败: {metrics['decision_parse_fail_count']}/{metrics['decision_eval_count']}")
    print("=" * 60)

    # ---- 保存指标 ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_subdir = os.path.join(args.output_dir, f"{model_name}_{variant_tag}_{timestamp}")
    os.makedirs(result_subdir, exist_ok=True)

    metrics_file = os.path.join(result_subdir, "metrics.md")
    metrics_output = build_metrics_markdown(
        metrics,
        model=args.model_path,
        variant=variant_tag,
        num_samples=total_evals,
    )

    if multi_run:
        metrics_output += (
            "\n## RQ2 重复测试\n\n"
            f"- split: `{args.split}`\n"
            f"- unique_cases: `{n}`\n"
            f"- repeats_per_case: `{args.num_repeats}`\n"
            f"- total_inferences: `{total_evals}`\n"
            f"- base_seed: `{args.seed}` "
            f"{'(每个 run_id 使用 seed=base_seed+run_id)' if args.seed is not None else '(未显式设置)'}\n"
        )

    # 构建详细预测条目（以备稳定性表与按混淆类别落盘）
    detail_entries: list[dict] = []
    for si in range(n):
        for ri in range(args.num_repeats):
            record = completed[(si, ri)]
            entry = {
                "sample_index": record.get("sample_index", record.get("index", si)),
                "run_id": record.get("run_id", ri),
                "id": record["id"],
                "person_info": record["person_info"],
                "procedure": record["procedure"],
                "fact": record["fact"],
                "prediction": record["prediction"],
                "reference": record["reference"],
                "raw_reasoning_and_decision": record["raw_reasoning_and_decision"],
                "raw_reasoning": record.get("raw_reasoning", ""),
                "raw_output": record["raw_output"],
            }
            if not multi_run:
                entry["index"] = entry["sample_index"]
            detail_entries.append(entry)

    if multi_run:
        stability_md = build_rq2_stability_markdown(detail_entries)
        if stability_md:
            metrics_output += stability_md

    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(metrics_output)
    logger.info(f"指标已保存: {metrics_file}")

    # 按 (参考decision, 预测decision) 分组保存详细预测
    details_groups: dict[str, list] = defaultdict(list)
    for entry in detail_entries:
        ref_dec = normalize_decision_label(entry["reference"].get("decision", ""))
        pred_dec = normalize_decision_label(entry["prediction"].get("decision", ""))
        if ref_dec is None:
            details_groups["参考标签解析错误"].append(entry)
        elif pred_dec is None:
            details_groups["解析错误"].append(entry)
        elif pred_dec == ref_dec:
            details_groups["正确"].append(entry)
        else:
            key = f"{ref_dec}_{pred_dec}"
            details_groups[key].append(entry)

    for group_name, data in details_groups.items():
        # 清理文件名中的非法字符（Windows 不允许 \ / : * ? " < > |）
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', group_name)[:80]
        filepath = os.path.join(result_subdir, f"details_{safe_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存: {filepath} ({len(data)} 条)")

    # 清理断点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("断点文件已清理")


if __name__ == "__main__":
    main()
