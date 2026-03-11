#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_process/llm_judge.py

使用 LLM 作为评判者对 PDP 数据集进行多维度质量评估。

评估维度：
  1. label_leakage       — person_info/procedure/fact 是否泄露 decision 信息
  2. info_insufficient     — 输入字段是否包含足够信息推导 decision
  3. label_error    — decision 标签是否与 raw_reasoning_and_decision 吻合
  4. data_corrupted         — 各字段是否有 OCR 乱码/无用信息
  5. articles_inconsistent — relevant_articles 与 reasoning 中的法条是否一致

拒绝类型：
  - 标签泄露（直接泄露/间接泄露）
  - 信息不足
  - 标签错误
  - 数据污损
  - 法条不一致

输出（保存至 data/PDP_dataset_filtered/{split}/）：
  - judge_scores.json         — 每条数据的评分结果
  - dataset.json              — 所有维度通过的数据
  - rejected/{拒绝类型}.json  — 按拒绝类型分组的问题数据（供人工审查）

Usage:
    python data_process/llm_judge.py --split train
    python data_process/llm_judge.py --split test --concurrency 10
    python data_process/llm_judge.py --split both --max-samples 100
    python data_process/llm_judge.py --split train --no-resume
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIMENSIONS = [
    "label_leakage",
    "info_insufficient",
    "label_error",
    "data_corrupted",
    "articles_inconsistent",
]

DIMENSION_NAMES = {
    "label_leakage": "标签泄露",
    "info_insufficient": "信息不足",
    "label_error": "标签错误",
    "data_corrupted": "数据污损",
    "articles_inconsistent": "法条不一致",
}


# ============================================================
#  Prompt 构建
# ============================================================

SYSTEM_PROMPT = "你是一个数据质量评估专家，负责评估法律文书数据集的质量。请严格按要求对数据进行多维度0/1打分。"

USER_PROMPT_TEMPLATE = """请对以下法律文书数据进行质量评估。

【待评估数据】
ID: {id}

[person_info]
{person_info}

[procedure]
{procedure}

[fact]
{fact}

[relevant_articles]
{articles_str}

[decision]
{decision}

[raw_reasoning_and_decision]
{raw_reasoning_and_decision}

【评估维度与评分标准】

1. label_leakage（标签泄露）: 检查person_info、procedure、fact三个字段中是否泄露了decision的结果。分为两种情况：
   A. 直接泄露：这三个字段中出现了"不起诉决定""相对不起诉""存疑不起诉""法定不起诉""提起公诉""决定起诉""决定不起诉"等直接表明最终处理结果的表述
   B. 间接泄露：这三个字段中使用了"被不起诉人"（暗示不起诉结论）、"被告人"（暗示已起诉）等称谓，或出现"免予刑事处罚""免于起诉"等强暗示表述
   - 打0分：存在直接泄露或间接泄露（请在issues中注明是"直接泄露"还是"间接泄露"）
   - 打1分：这三个字段仅包含案件信息（"犯罪嫌疑人""涉嫌XX罪""移送审查起诉""取保候审"等属于正常案件描述，不算泄露）

2. info_insufficient（信息不足）: 检查person_info、procedure、fact三个字段合在一起是否包含足够信息来辅助推导decision。
   - 打0分：缺少关键案件事实（fact中无具体犯罪行为描述），或关键字段过于简短（fact不足30字），或缺少基本程序信息
   - 打1分：包含犯罪嫌疑人基本信息、案件事实描述（何时何地做了什么）、基本程序信息

3. label_error（标签错误）: 检查decision字段是否与raw_reasoning_and_decision的内容吻合。
   - 打0分：raw_reasoning_and_decision中明确的处理决定与decision标签不一致（如reasoning说"决定不起诉"但decision标为"起诉"，或reasoning说的不起诉类型与decision标签不同）
   - 打1分：decision与reasoning中的结论一致

4. data_corrupted（数据污损）: 检查person_info、procedure、fact、raw_reasoning_and_decision四个字段的文本质量。
   - 打0分：存在大量OCR乱码（不可读字符）、HTML标签、页眉页脚等无关内容、明显的正则提取错误导致的无用信息、大段不可理解的文字
   - 打1分：文本基本可读，内容与法律文书相关（少量脱敏符号如***、**不算质量问题）

5. articles_inconsistent（法条不一致）: 检查relevant_articles中的法条是否与raw_reasoning_and_decision中引用的法条基本一致。
   - 打0分：relevant_articles中有多条在reasoning中完全未提及的法条，或reasoning中明确引用的重要法条在relevant_articles中大量缺失
   - 打1分：两者基本一致（允许小幅偏差，如条款号表述差异）

请直接以JSON格式返回评估结果，不要包含任何其他文字：
{{"label_leakage": 0或1, "info_insufficient": 0或1, "label_error": 0或1, "data_corrupted": 0或1, "articles_inconsistent": 0或1, "issues": "如有0分维度请简述原因，否则填'无'"}}"""


def build_judge_prompt(item: dict) -> list[dict]:
    """构建 LLM 评判 prompt。"""
    articles_str = (
        "、".join(item["relevant_articles"])
        if item["relevant_articles"]
        else "（无）"
    )

    user_content = USER_PROMPT_TEMPLATE.format(
        id=item["id"],
        person_info=item["person_info"],
        procedure=item["procedure"],
        fact=item["fact"],
        articles_str=articles_str,
        decision=item["decision"],
        raw_reasoning_and_decision=item["raw_reasoning_and_decision"],
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ============================================================
#  响应解析
# ============================================================

def _strip_think_block(text: str) -> str:
    """去除 <think>...</think> 块。"""
    think_end = text.find("</think>")
    if think_end != -1:
        return text[think_end + len("</think>"):].strip()
    return text.strip()


def _validate_scores(result: dict) -> dict | None:
    """验证并规范化评分结果。"""
    scores = {}
    for dim in DIMENSIONS:
        if dim not in result:
            return None
        val = result[dim]
        if isinstance(val, (int, float)):
            scores[dim] = 1 if val >= 0.5 else 0
        elif isinstance(val, str):
            scores[dim] = 1 if val.strip() in ("1", "true", "True") else 0
        else:
            return None
    scores["issues"] = str(result.get("issues", ""))
    return scores


def parse_judge_response(text: str) -> dict | None:
    """解析 LLM 评判响应，提取 JSON 评分。"""
    text = _strip_think_block(text)

    # 尝试直接解析
    try:
        return _validate_scores(json.loads(text.strip()))
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块（含嵌套 issues 字符串）
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return _validate_scores(json.loads(json_match.group()))
        except json.JSONDecodeError:
            pass

    # 尝试去除 markdown 代码块标记后解析
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
    try:
        return _validate_scores(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    return None


# ============================================================
#  异步 API 调用
# ============================================================

async def judge_single(
    client: AsyncOpenAI,
    item: dict,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    """评估单条数据，含重试逻辑。"""
    messages = build_judge_prompt(item)

    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                    extra_body={"enable_thinking": False},
                )
                text = response.choices[0].message.content
                scores = parse_judge_response(text)
                if scores is not None:
                    return {
                        "id": item["id"],
                        "scores": scores,
                        "raw_response": text,
                    }
                logger.warning(
                    f"[{item['id']}] 第{attempt+1}次解析失败，原始响应: {text[:100]}..."
                )
            except Exception as e:
                logger.warning(f"[{item['id']}] API错误 (第{attempt+1}次): {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    logger.error(f"[{item['id']}] {max_retries}次重试均失败")
    return {"id": item["id"], "scores": None, "raw_response": ""}


async def process_dataset(
    data: list[dict],
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    checkpoint_path: str,
    resume: bool,
) -> dict:
    """异步处理整个数据集，支持断点续推。"""
    # 加载断点
    completed = {}
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed[record["id"]] = record
                except (json.JSONDecodeError, KeyError):
                    pass
        logger.info(f"从断点恢复: 已完成 {len(completed)}/{len(data)}")

    remaining = [item for item in data if item["id"] not in completed]

    if not remaining:
        logger.info("所有样本已完成评估")
        return completed

    logger.info(f"待评估样本数: {len(remaining)}")

    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.time()
    done_count = 0
    total = len(remaining)
    lock = asyncio.Lock()

    # 断点文件用追加模式
    checkpoint_file = open(checkpoint_path, "a", encoding="utf-8")

    async def process_one(item):
        nonlocal done_count
        result = await judge_single(client, item, model, semaphore)

        async with lock:
            completed[result["id"]] = result
            checkpoint_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            checkpoint_file.flush()
            done_count += 1

            if done_count % 50 == 0 or done_count == total:
                elapsed = time.time() - start_time
                speed = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / speed if speed > 0 else 0
                logger.info(
                    f"进度: {done_count}/{total} "
                    f"({done_count/total:.1%}, "
                    f"{speed:.1f} samples/s, "
                    f"ETA: {eta/60:.0f}min)"
                )

    # 分批提交任务，避免一次性创建过多协程
    batch_size = concurrency * 10
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        tasks = [process_one(item) for item in batch]
        await asyncio.gather(*tasks)

    checkpoint_file.close()
    return completed


# ============================================================
#  汇总与过滤
# ============================================================

def aggregate_and_filter(data: list[dict], scores: dict, output_dir: str):
    """汇总评分，生成过滤后数据集和问题数据文件。"""
    os.makedirs(output_dir, exist_ok=True)

    total = len(data)
    dim_fail_counts = {dim: 0 for dim in DIMENSIONS}
    parse_fail_count = 0
    all_pass_count = 0

    filtered = []
    rejected_by_dim = {dim: [] for dim in DIMENSIONS}
    failed_parse = []
    score_records = []

    for item in data:
        item_id = item["id"]
        result = scores.get(item_id)

        if result is None or result.get("scores") is None:
            parse_fail_count += 1
            failed_parse.append(item)
            score_records.append({
                "id": item_id,
                "scores": None,
                "parse_failed": True,
            })
            continue

        item_scores = result["scores"]
        score_records.append({
            "id": item_id,
            "scores": {dim: item_scores[dim] for dim in DIMENSIONS},
            "issues": item_scores.get("issues", ""),
        })

        all_pass = True
        for dim in DIMENSIONS:
            if item_scores[dim] == 0:
                dim_fail_counts[dim] += 1
                rejected_by_dim[dim].append(item)
                all_pass = False

        if all_pass:
            all_pass_count += 1
            filtered.append(item)

    # ---- 保存结果 ----

    # 1. 评分记录
    scores_path = os.path.join(output_dir, "judge_scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(score_records, f, ensure_ascii=False, indent=2)
    logger.info(f"评分结果: {scores_path}")

    # 2. 过滤后数据集
    filtered_path = os.path.join(output_dir, "dataset.json")
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    logger.info(f"过滤后数据集: {filtered_path} ({len(filtered)} 条)")

    # 3. 按拒绝类型分组的问题数据
    rejected_dir = os.path.join(output_dir, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)
    for dim in DIMENSIONS:
        if rejected_by_dim[dim]:
            name = DIMENSION_NAMES[dim]
            path = os.path.join(rejected_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rejected_by_dim[dim], f, ensure_ascii=False, indent=2)
            logger.info(f"  {name}: {path} ({len(rejected_by_dim[dim])} 条)")

    # 4. 解析失败
    if failed_parse:
        path = os.path.join(rejected_dir, "解析失败.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(failed_parse, f, ensure_ascii=False, indent=2)
        logger.info(f"  解析失败: {path} ({len(failed_parse)} 条)")

    # ---- 打印汇总 ----
    print("\n" + "=" * 60)
    print("  LLM Judge 评估汇总")
    print("=" * 60)
    print(f"  总样本数:   {total}")
    print(f"  全部通过:   {all_pass_count} ({all_pass_count/total:.1%})")
    print(f"  解析失败:   {parse_fail_count}")
    print()
    print("  各拒绝类型统计:")
    for dim in DIMENSIONS:
        name = DIMENSION_NAMES[dim]
        count = dim_fail_counts[dim]
        print(f"    {name:<6s}: {count} ({count/total:.1%})")
    print("=" * 60)


# ============================================================
#  入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM Judge — PDP 数据集多维度质量评估"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="评估哪个数据集 (default: both)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-235b-a22b-instruct-2507",
        help="评判模型 (default: qwen3-235b-a22b-instruct-2507)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="并发请求数 (default: 5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最大评估样本数，0=全部 (default: 0)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用断点续推，从头开始",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="数据集根目录 (default: data/PDP_dataset)",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    # 初始化 API 客户端
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    splits = ["train", "test"] if args.split == "both" else [args.split]
    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data" / "PDP_dataset"
    output_root = PROJECT_ROOT / "data" / "PDP_dataset_filtered"

    for split in splits:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"开始处理 {split} 数据集")
        logger.info(f"{'=' * 50}")

        data_path = data_root / split / "dataset.json"
        if not data_path.exists():
            logger.error(f"数据文件不存在: {data_path}")
            continue

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if args.max_samples > 0:
            data = data[: args.max_samples]
            logger.info(f"限制评估样本数: {args.max_samples}")

        logger.info(f"样本总数: {len(data)}")
        logger.info(f"评判模型: {args.model}")
        logger.info(f"并发数:   {args.concurrency}")

        output_dir = str(output_root / split)
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "judge_checkpoint.jsonl")

        if args.no_resume and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("已删除旧断点文件")

        # 异步评估
        scores = asyncio.run(
            process_dataset(
                data,
                client,
                args.model,
                args.concurrency,
                checkpoint_path,
                resume=not args.no_resume,
            )
        )

        # 汇总与过滤
        aggregate_and_filter(data, scores, output_dir)

        # 清理断点文件
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("断点文件已清理")

    logger.info("\n全部完成！")


if __name__ == "__main__":
    main()
