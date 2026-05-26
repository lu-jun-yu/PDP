#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/evaluate_openrouter_rq2.py

对 pdp4k 的 test_rq2 split 进行评估：每条样本重复调用模型 num_repeats 次（默认 5），
用于 RQ2 稳定性 / 方差分析。指标在「展开后的总次数」上计算（如 100×5=500）。

OpenRouter 与 RQ1 共用 eval.evaluate_openrouter.call_openrouter；raw_reasoning 仅保存 API
返回的推理字段（与 raw_output 无关，不从正文抄录）。

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python eval/evaluate_openrouter_rq2.py --model qwen/qwen3-8b
    python eval/evaluate_openrouter_rq2.py --model openai/gpt-5.4 --reasoning-effort high
    # 推理预算二选一：reasoning.effort（默认 auto）或 reasoning.max_tokens（与 effort 互斥）
    python eval/evaluate_openrouter_rq2.py --model anthropic/claude-opus-4.6 --reasoning-max-tokens 8192
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk

from eval.evaluate_openrouter import (
    OPENROUTER_BASE_URL,
    call_openrouter,
    parse_answer,
)
from eval.metrics import (
    build_metrics_markdown,
    build_rq2_result_f1_mean_se,
    build_rq2_stability_markdown,
    compute_metrics,
    normalize_decision_label,
)
from prompt_template import PROMPT_VARIANTS, build_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_repeat_batch(
    client,
    model: str,
    tasks: list[tuple[int, int, list[dict]]],
    dataset,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    reasoning_effort: str,
    reasoning_max_tokens: int | None,
    concurrency: int,
    checkpoint_file: str,
    completed: dict,
) -> int:
    """tasks: (sample_index, run_id, messages)。断点键为 (sample_index, run_id)。"""
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    async def process_one(
        sample_index: int,
        run_id: int,
        msgs: list[dict],
    ) -> None:
        generated = await call_openrouter(
            client,
            model,
            msgs,
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            reasoning_effort,
            semaphore,
            reasoning_max_tokens=reasoning_max_tokens,
        )
        generated_text = generated["content"]
        parsed = parse_answer(generated_text)
        sample = dataset[sample_index]
        ref = {
            "relevant_articles": list(sample["relevant_articles"]),
            "decision": sample["decision"],
        }
        record = {
            "sample_index": sample_index,
            "run_id": run_id,
            "id": sample["id"],
            "person_info": sample["person_info"],
            "procedure": sample["procedure"],
            "fact": sample["fact"],
            "prediction": parsed,
            "reference": ref,
            "raw_reasoning_and_decision": sample["raw_reasoning_and_decision"],
            "raw_reasoning": generated["reasoning"],
            "raw_output": generated_text,
        }
        key = (sample_index, run_id)
        completed[key] = record

        async with file_lock:
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    coros = [
        process_one(si, ri, msgs) for si, ri, msgs in tasks
    ]
    done_count = 0
    for coro in asyncio.as_completed(coros):
        await coro
        done_count += 1
        if done_count % 50 == 0 or done_count == len(coros):
            logger.info(f"  当前批次进度: {done_count}/{len(coros)}")

    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDP test_rq2 重复评估 (OpenRouter API)"
    )
    parser.add_argument("--model", required=True, help="OpenRouter 模型 id")
    parser.add_argument(
        "--data-path",
        default="data/pdp4k",
        help="HuggingFace 数据集目录 (default: data/pdp4k)",
    )
    parser.add_argument(
        "--split",
        default="test_rq2",
        help="数据集 split 名 (default: test_rq2)",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=5,
        help="每条样本重复推理次数 (default: 5)",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="original",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["auto", "none", "low", "medium", "high", "xhigh"],
        default="auto",
        help="OpenRouter reasoning.effort；与 --reasoning-max-tokens 互斥（指定 max_tokens 时请保持默认 auto）",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="OpenRouter reasoning.max_tokens（预算式）；Anthropic 系文档下限通常为 1024；与 --reasoning-effort 互斥",
    )
    args = parser.parse_args()

    if (
        args.reasoning_max_tokens is not None
        and args.reasoning_effort != "auto"
    ):
        parser.error(
            "--reasoning-max-tokens 与 --reasoning-effort 不能同时使用；"
            "使用 max_tokens 时请省略 --reasoning-effort（默认 auto）"
        )
    if args.reasoning_max_tokens is not None and args.reasoning_max_tokens < 1:
        parser.error("--reasoning-max-tokens 须为正整数")
    if args.reasoning_max_tokens is not None and args.reasoning_max_tokens < 1024:
        logger.warning(
            "reasoning.max_tokens=%s 低于 OpenRouter 文档对 Anthropic 等模型的常见下限 1024；"
            "可能被抬升或忽略，导致预算档位区分不明显。",
            args.reasoning_max_tokens,
        )

    if args.num_repeats < 1:
        logger.error("--num-repeats 至少为 1")
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("请设置环境变量 OPENROUTER_API_KEY")
        sys.exit(1)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("请先安装 openai 库: pip install openai")
        sys.exit(1)

    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=120.0,
        max_retries=0,
    )

    logger.info("加载数据集: %s [%s]", args.data_path, args.split)
    try:
        dataset_dict = load_from_disk(args.data_path)
    except FileNotFoundError:
        from datasets import load_dataset
        dataset_dict = load_dataset(args.data_path)

    if args.split not in dataset_dict:
        logger.error(
            "split '%s' 不存在。请先运行 sample_test_rq2.py 与 convert_to_hf_dataset.py。",
            args.split,
        )
        sys.exit(1)

    dataset = dataset_dict[args.split]
    n = len(dataset)
    total_evals = n * args.num_repeats
    logger.info("样本数: %d, 每样本重复: %d, 总推理次数: %d", n, args.num_repeats, total_evals)

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.replace("/", "_")
    if args.reasoning_max_tokens is not None:
        variant_tag = (
            f"{args.prompt_variant}_reasoning_maxtok_{args.reasoning_max_tokens}_"
            f"rq2_{args.split}_r{args.num_repeats}"
        )
    else:
        variant_tag = (
            f"{args.prompt_variant}_reasoning_{args.reasoning_effort}_"
            f"rq2_{args.split}_r{args.num_repeats}"
        )
    checkpoint_file = os.path.join(
        args.output_dir, f"{model_name}_{variant_tag}_checkpoint.jsonl"
    )

    completed: dict = {}
    if not args.no_resume and os.path.exists(checkpoint_file):
        logger.info("检测到断点文件: %s", checkpoint_file)
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed[(record["sample_index"], record["run_id"])] = record
                except (json.JSONDecodeError, KeyError):
                    logger.warning("断点文件第 %d 行解析失败，跳过", line_no)
        logger.info(
            "从断点恢复: 已完成 %d/%d 次推理",
            len(completed),
            total_evals,
        )
    elif args.no_resume and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("已删除旧断点文件")

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

    pending: list[tuple[int, int, list[dict]]] = []
    for si in range(n):
        for ri in range(args.num_repeats):
            key = (si, ri)
            if key not in completed:
                pending.append((si, ri, messages_for(si)))

    if pending:
        logger.info("待推理: %d 次", len(pending))
        batch_size = args.batch_size if args.batch_size > 0 else len(pending)
        num_batches = (len(pending) + batch_size - 1) // batch_size
        start_time = time.time()
        new_count = 0

        for batch_idx in range(num_batches):
            bs = batch_idx * batch_size
            be = min(bs + batch_size, len(pending))
            batch_tasks = pending[bs:be]
            logger.info(
                "批次 %d/%d: %d 次推理",
                batch_idx + 1,
                num_batches,
                len(batch_tasks),
            )
            new_count += asyncio.run(
                run_repeat_batch(
                    client=client,
                    model=args.model,
                    tasks=batch_tasks,
                    dataset=dataset,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_max_tokens=args.reasoning_max_tokens,
                    concurrency=args.concurrency,
                    checkpoint_file=checkpoint_file,
                    completed=completed,
                )
            )
            elapsed = time.time() - start_time
            logger.info(
                "总进度: %d/%d (本轮后新增累计 %d, %.1fs)",
                len(completed),
                total_evals,
                new_count,
                elapsed,
            )
    else:
        logger.info("全部推理已完成，直接汇总指标")

    predictions = []
    references = []
    parse_fail_count = 0

    for si in range(n):
        for ri in range(args.num_repeats):
            key = (si, ri)
            if key not in completed:
                logger.error("缺失 sample_index=%d run_id=%d，请检查断点或使用 --no-resume", si, ri)
                sys.exit(1)
            record = completed[key]
            predictions.append(record["prediction"])
            references.append(record["reference"])
            if normalize_decision_label(record["prediction"].get("decision", "")) is None:
                parse_fail_count += 1

    if parse_fail_count:
        logger.warning(
            "有 %d/%d 次决定字段无法归一化为有效标签",
            parse_fail_count,
            len(predictions),
        )

    metrics = compute_metrics(predictions, references)

    print("\n" + "=" * 60)
    print(f"  PDP RQ2 评估 — split={args.split}")
    print(f"  唯一样本: {n}, 每样本重复: {args.num_repeats}, 总次数: {total_evals}")
    print(f"  模型: {args.model}")
    print(f"  变体: {variant_tag}")
    print("=" * 60)

    print("\n【过程评估】")
    print(f"  法条 F1: {metrics['articles_f1']:.4f}")
    print("\n【结果评估】")
    print(
        f"  L1 Macro-F1: {metrics['decision_level1_macro_f1']:.4f}, "
        f"Micro-F1: {metrics['decision_level1_micro']['f1']:.4f}"
    )
    print(
        f"  L2 Macro-F1: {metrics['decision_level2_macro_f1']:.4f}, "
        f"Micro-F1: {metrics['decision_level2_micro']['f1']:.4f}"
    )
    print("=" * 60)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_subdir = os.path.join(
        args.output_dir, f"{model_name}_{variant_tag}_{timestamp}"
    )
    os.makedirs(result_subdir, exist_ok=True)

    detail_entries: list[dict] = []
    for si in range(n):
        for ri in range(args.num_repeats):
            record = completed[(si, ri)]
            detail_entries.append({
                "sample_index": record["sample_index"],
                "run_id": record["run_id"],
                "id": record["id"],
                "person_info": record["person_info"],
                "procedure": record["procedure"],
                "fact": record["fact"],
                "prediction": record["prediction"],
                "reference": record["reference"],
                "raw_reasoning_and_decision": record["raw_reasoning_and_decision"],
                "raw_reasoning": record.get("raw_reasoning", ""),
                "raw_output": record["raw_output"],
            })

    metrics_file = os.path.join(result_subdir, "metrics.md")
    md = build_metrics_markdown(
        metrics,
        model=args.model,
        variant=variant_tag,
        num_samples=total_evals,
        result_f1_mean_se=build_rq2_result_f1_mean_se(detail_entries),
    )
    reasoning_note = (
        f"`reasoning.max_tokens={args.reasoning_max_tokens}`"
        if args.reasoning_max_tokens is not None
        else f"`reasoning.effort={args.reasoning_effort}`"
    )
    md += (
        "\n## RQ2 重复测试\n\n"
        f"- split: `{args.split}`\n"
        f"- unique_cases: `{n}`\n"
        f"- repeats_per_case: `{args.num_repeats}`\n"
        f"- total_api_calls: `{total_evals}`\n"
        f"- reasoning_budget: {reasoning_note}\n"
    )
    stability_md = build_rq2_stability_markdown(detail_entries)
    if stability_md:
        md += stability_md
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("指标已保存: %s", metrics_file)

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
            key_name = f"{ref_dec}_{pred_dec}"
            details_groups[key_name].append(entry)

    for group_name, data in details_groups.items():
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", group_name)[:80]
        filepath = os.path.join(result_subdir, f"details_{safe_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("详细结果: %s (%d 条)", filepath, len(data))

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("断点文件已清理")


if __name__ == "__main__":
    main()
