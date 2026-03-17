#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/evaluate_openrouter.py

通过 OpenRouter API 对 PDP 数据集进行评估。

OpenRouter 提供 OpenAI 兼容接口，可调用多种模型（如 Qwen、DeepSeek、Gemini 等）。
API Key 通过环境变量 OPENROUTER_API_KEY 传入。

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python eval/evaluate_openrouter.py --model qwen/qwen3-8b
    python eval/evaluate_openrouter.py --model deepseek/deepseek-chat-v3-0324 --concurrency 10
    python eval/evaluate_openrouter.py --model google/gemini-2.0-flash-001 --no-resume
    python eval/evaluate_openrouter.py --model qwen/qwen3-8b --with-definitions
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

# 将项目根目录加入 sys.path，以便导入 prompt_template
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk

from prompt_template import build_messages
from eval.metrics import compute_metrics, build_metrics_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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

        cl_match = re.search(r"刑法[：:](.*?)(?:\n|$)", section_text)
        if cl_match:
            articles += [f"cl:{a}" for a in _split_articles(cl_match.group(1).strip())]

        cpl_match = re.search(r"刑事诉讼法[：:](.*?)(?:\n|$)", section_text)
        if cpl_match:
            articles += [f"cpl:{a}" for a in _split_articles(cpl_match.group(1).strip())]

        cpr_match = re.search(r"刑事诉讼规则[：:](.*?)(?:\n|$)", section_text)
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
        dec_match = re.search(r"决定[：:]\s*(.*?)(?:\n|$)", section_text)
        if dec_match:
            result["decision"] = dec_match.group(1).strip()

    return result


def _split_articles(raw: str) -> list[str]:
    """将 '第XXX条、第YYY条第Z款' 拆分为列表。"""
    items = re.split(r"[、，,;；]", raw)
    return [a.strip() for a in items if a.strip()]


# ============================================================
#  OpenRouter API 调用
# ============================================================

async def call_openrouter(
    client,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> str:
    """调用 OpenRouter API，带信号量控制并发和指数退避重试。"""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                # 只传入用户显式指定的参数，其余由 API/模型使用默认值
                kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if top_p is not None:
                    kwargs["top_p"] = top_p
                extra = {}
                if top_k is not None:
                    extra["top_k"] = top_k
                if min_p is not None:
                    extra["min_p"] = min_p
                if extra:
                    kwargs["extra_body"] = extra

                response = await client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e:
                err_msg = str(e)
                is_last = attempt == max_retries - 1
                # 速率限制、服务器错误、超时、连接错误 → 可重试
                retryable = (
                    "429" in err_msg
                    or "502" in err_msg
                    or "503" in err_msg
                    or "timed out" in err_msg.lower()
                    or "timeout" in err_msg.lower()
                    or "connection" in err_msg.lower()
                )
                if retryable and not is_last:
                    wait = min(2 ** attempt * 2, 60)
                    logger.warning(
                        f"API 请求失败 (attempt {attempt + 1}/{max_retries}): "
                        f"{err_msg[:100]}... 等待 {wait}s 后重试"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"API 请求失败（不可重试）: {err_msg[:200]}")
                    raise
        raise RuntimeError(f"API 请求在 {max_retries} 次重试后仍失败")


async def run_batch(
    client,
    model: str,
    conversations: list[list[dict]],
    indices: list[int],
    dataset,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    concurrency: int,
    checkpoint_file: str,
    completed: dict,
) -> int:
    """异步并发执行一批 API 请求，实时写入断点文件。返回新完成的数量。"""
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(pos: int) -> None:
        idx = indices[pos]
        msgs = conversations[pos]
        generated_text = await call_openrouter(
            client, model, msgs, max_tokens, temperature,
            top_p, top_k, min_p, semaphore,
        )

        parsed = parse_answer(generated_text)
        sample = dataset[idx]
        ref = {
            "relevant_articles": list(sample["relevant_articles"]),
            "decision": sample["decision"],
        }
        record = {
            "index": idx,
            "id": sample["id"],
            "person_info": sample["person_info"],
            "procedure": sample["procedure"],
            "fact": sample["fact"],
            "prediction": parsed,
            "reference": ref,
            "raw_reasoning_and_decision": sample["raw_reasoning_and_decision"],
            "raw_output": generated_text,
        }
        completed[idx] = record

        # 追加写入断点文件（加锁保证写入顺序）
        async with file_lock:
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    file_lock = asyncio.Lock()
    tasks = [process_one(i) for i in range(len(conversations))]

    done_count = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        done_count += 1
        if done_count % 50 == 0 or done_count == len(tasks):
            logger.info(f"  当前批次进度: {done_count}/{len(tasks)}")

    return len(conversations)


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PDP 数据集评估脚本 (OpenRouter API)")
    parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter 模型标识符 (如 qwen/qwen3-8b, deepseek/deepseek-chat-v3-0324)",
    )
    parser.add_argument(
        "--data-path",
        default="data/pdp10k",
        help="HuggingFace 数据集路径 (default: data/pdp10k)",
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
        default=None,
        help="生成温度 (不指定则使用模型/API 默认值)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-P 采样 (不指定则使用模型/API 默认值)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K 采样 (不指定则使用模型/API 默认值)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-P 采样 (不指定则使用模型/API 默认值)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="并发请求数 (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="分批推理的批次大小，0 表示一次性推理所有样本 (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="结果输出目录 (default: results)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用断点续推，从头开始评估",
    )
    parser.add_argument(
        "--with-definitions",
        action="store_true",
        help="消融实验：在系统提示中加入不起诉类型的定义与适用条件",
    )
    args = parser.parse_args()

    # ---- 检查 API Key ----
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("请设置环境变量 OPENROUTER_API_KEY")
        sys.exit(1)

    # ---- 延迟导入 openai ----
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("请先安装 openai 库: pip install openai")
        sys.exit(1)

    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=120.0,      # 连接+读取超时 120 秒（默认太短）
        max_retries=0,       # 关闭 SDK 内置重试，由 call_openrouter 自行控制
    )

    # ---- 加载数据 ----
    logger.info(f"加载数据集: {args.data_path} [test]")
    try:
        dataset_dict = load_from_disk(args.data_path)
    except FileNotFoundError:
        from datasets import load_dataset
        dataset_dict = load_dataset(args.data_path)
    dataset = dataset_dict["test"]

    logger.info(f"评估样本数: {len(dataset)}")

    # ---- 断点续推 ----
    os.makedirs(args.output_dir, exist_ok=True)
    # 模型名中的 "/" 替换为 "_"，用作文件名
    model_name = args.model.replace("/", "_")
    variant_tag = "with_def" if args.with_definitions else "baseline"
    checkpoint_file = os.path.join(
        args.output_dir, f"{model_name}_test_{variant_tag}_checkpoint.jsonl"
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
        if args.with_definitions:
            logger.info("消融实验模式：系统提示中包含不起诉类型定义")
        conversations = []
        for i in remaining_indices:
            sample = dataset[i]
            messages = build_messages(
                person_info=sample["person_info"],
                procedure=sample["procedure"],
                fact=sample["fact"],
                with_definitions=args.with_definitions,
            )
            conversations.append(messages)

        # ---- 分批推理 + 断点保存 ----
        batch_size = args.batch_size if args.batch_size > 0 else len(conversations)
        num_batches = (len(conversations) + batch_size - 1) // batch_size
        logger.info(
            f"开始推理... (共 {num_batches} 个批次, "
            f"batch_size={'all' if args.batch_size == 0 else batch_size}, "
            f"concurrency={args.concurrency})"
        )
        start_time = time.time()
        new_count = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(conversations))
            batch_conversations = conversations[batch_start:batch_end]
            batch_indices = remaining_indices[batch_start:batch_end]

            logger.info(
                f"批次 {batch_idx + 1}/{num_batches}: "
                f"样本 {batch_start}~{batch_end - 1} ({len(batch_conversations)} 个)"
            )

            batch_new = asyncio.run(
                run_batch(
                    client=client,
                    model=args.model,
                    conversations=batch_conversations,
                    indices=batch_indices,
                    dataset=dataset,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    concurrency=args.concurrency,
                    checkpoint_file=checkpoint_file,
                    completed=completed,
                )
            )

            new_count += batch_new
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
    print(f"  PDP 评估结果 — test split ({len(dataset)} samples)")
    print(f"  模型: {args.model}")
    print(f"  变体: {variant_tag}")
    print("=" * 60)

    print("\n【过程评估】")
    print(f"  法条 (articles):")
    print(f"    Precision: {metrics['articles_precision']:.4f}")
    print(f"    Recall:    {metrics['articles_recall']:.4f}")
    print(f"    F1:        {metrics['articles_f1']:.4f}")

    print("\n【结果评估】")
    print(f"  决定 (decision):")
    print(f"    第一级（起诉/不起诉）Accuracy: {metrics['decision_level1_accuracy']:.4f}")
    for cls, info in metrics["level1_per_class"].items():
        print(f"      {cls}: {info['accuracy']:.4f} ({info['count']} 样本)")
    print(f"    第二级（四分类）Accuracy:       {metrics['decision_level2_accuracy']:.4f}")
    for cls, info in metrics["decision_per_class"].items():
        print(f"      {cls}: {info['accuracy']:.4f} ({info['count']} 样本)")

    print(f"\n  解析失败率: {parse_fail_count}/{len(predictions)}")
    print("=" * 60)

    # ---- 保存指标 ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_subdir = os.path.join(
        args.output_dir, f"{model_name}_test_{variant_tag}_{timestamp}"
    )
    os.makedirs(result_subdir, exist_ok=True)

    metrics_file = os.path.join(result_subdir, "metrics.json")
    metrics_output = build_metrics_json(
        metrics,
        model=args.model,
        variant=variant_tag,
        num_samples=len(dataset),
        parse_fail_count=parse_fail_count,
    )
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {metrics_file}")

    # 构建详细预测并按 参考decision-预测decision 分组保存
    details_groups = defaultdict(list)
    for i in range(len(dataset)):
        record = completed[i]
        entry = {
            "index": record["index"],
            "id": record["id"],
            "person_info": record["person_info"],
            "procedure": record["procedure"],
            "fact": record["fact"],
            "prediction": record["prediction"],
            "reference": record["reference"],
            "raw_reasoning_and_decision": record["raw_reasoning_and_decision"],
            "raw_output": record["raw_output"],
        }
        ref_dec = record["reference"]["decision"]
        pred_dec = record["prediction"]["decision"]
        if pred_dec == ref_dec:
            details_groups["正确"].append(entry)
        else:
            key = f"{ref_dec}_{pred_dec}"
            details_groups[key].append(entry)

    for group_name, data in details_groups.items():
        filepath = os.path.join(result_subdir, f"details_{group_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存: {filepath} ({len(data)} 条)")

    # 清理断点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("断点文件已清理")


if __name__ == "__main__":
    main()
