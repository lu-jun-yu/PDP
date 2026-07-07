#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate non-overlapping balanced CoT-style SFT data for RQ3 cold start.

The script samples from data/PDP_dataset/train/dataset.json, excludes every
sample ID already used by the RQ3/RL DatasetDict, keeps class balance, uses
every field in each sampled record as generation context, adds one text field
containing the generated native-style <think>...</think> + structured answer target, and
saves a new DatasetDict.

Example:
    python data_process/RQ3/generate_pdp2k_rq3_sft.py 
      --source-file data\PDP_dataset\train\dataset.json
      --model qwen/qwen3-max-thinking
      --rl-data-path data/pdp2k_rq3
      --output-dir data/pdp2k_rq3_sft
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from datasets import Dataset, DatasetDict, load_from_disk
except ImportError as exc:
    raise SystemExit("请先安装依赖: pip install datasets") from exc

from build_pdp2k_rq3_dataset import FEATURES, LABEL_ORDER, load_json
from prompt_template import (
    PROMPT_VARIANT_ORIGINAL,
    PROMPT_VARIANTS,
    build_messages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FIELD = "cot_data"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3-max-thinking"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_RL_MAX_COMPLETION_LENGTH = 1536


@dataclass(frozen=True)
class LLMOptions:
    max_retries: int

SYSTEM_PROMPT = """\
你是中国刑事检察文书的监督微调数据标注助手。你的任务不是重新预测结论，而是在已给定参考结果的条件下，只补全 SFT 目标输出中缺失的两个文本字段。

你必须严格输出 JSON 对象，且只能包含两个键：
{{
  "CoT": "...",
  "审查分析": "..."
}}

不要输出 Markdown 代码块、解释、前后缀或完整 <think> / 最终答案文本。完整 SFT target 将由程序根据你的 JSON、适用法条和最终决定自行拼接。

标注原则：
1. SFT 训练时，模型输入只包含案件可见信息；你会在用户消息中看到完整 SFT 输入。
2. SFT 训练时，模型输出是完整 assistant target；你会在用户消息中看到不完整的 SFT 输出模板。
3. 本次额外提供的 relevant_articles、decision、raw_reasoning_and_decision 是标注辅助信息，仅用于帮助你补全 CoT 和审查分析，不是未来模型推理时可见的输入。
4. "CoT" 应先从案件事实、证据状态、程序经过和法律适用切入，再自然收束到给定 decision；不要第一句就宣布最终决定。
5. "审查分析" 应像检察文书审查意见一样，用案件事实和法律依据说明为什么得到该结论。
6. 不得写“因为 decision/参考标注/给定结果是某类，所以应当某类”等元话语；不得把参考标注当作法律理由。
7. 不得暴露 decision、raw_reasoning_and_decision 等字段名，也不得暗示你看到了标注辅助字段。
8. 不要编造样本中没有的关键事实。若辅助字段之间存在差异，应在给定 relevant_articles 和 decision 约束下选择能够自洽解释该结果的事实与法律路径。
9. JSON 中两段文本合计不得超过 {max_answer_tokens} tokens。
"""

USER_PROMPT_TEMPLATE = """\
请为下面这条 SFT 样本补全缺失的 "CoT" 和 "审查分析"。

【最大回答长度】
JSON 中 "CoT" 与 "审查分析" 两段文本合计不超过 {max_answer_tokens} tokens。

【完整 SFT 输入（未来训练时模型实际看到的 prompt）】
{sft_input_messages}

【不完整 SFT 输出（程序稍后会自行拼成完整 assistant target）】
{partial_sft_output}

【本次标注辅助字段（未来模型不可见，仅用于补全上面的缺失部分）】
relevant_articles: {relevant_articles}
decision: {decision}
raw_reasoning_and_decision: {raw_reasoning_and_decision}

【输出 JSON 格式】
{{
  "CoT": "先分析事实、证据、程序、法条和责任边界，再自然推出给定决定的推理链",
  "审查分析": "面向最终输出中【审查分析】小节的正式审查意见"
}}

只输出上述 JSON 对象本身。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建与 RQ3/RL 训练集不重叠的 balanced CoT SFT 数据集"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENROUTER_MODEL,
        help=f"OpenRouter 模型 id (default: {DEFAULT_OPENROUTER_MODEL})",
    )
    parser.add_argument(
        "--source-file",
        default="data/PDP_dataset/train/dataset.json",
        help="原始训练 JSON，用于重采样 SFT 数据",
    )
    parser.add_argument(
        "--rl-data-path",
        default="data/pdp2k_rq3",
        help="需要排除的 RQ3/RL DatasetDict 路径",
    )
    parser.add_argument("--output-dir", default="data/pdp2k_rq3_sft")
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="断点目录；默认使用 <output-dir>_checkpoints",
    )
    parser.add_argument(
        "--output-field",
        default=DEFAULT_OUTPUT_FIELD,
        help=f"新增字段名 (default: {DEFAULT_OUTPUT_FIELD})",
    )
    parser.add_argument(
        "--prompt-variant",
        default=PROMPT_VARIANT_ORIGINAL,
        choices=PROMPT_VARIANTS,
        help="构造完整 SFT 输入时使用的 prompt 变体，应与 train/sft.py 保持一致",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="只处理指定 split；默认处理全部 split。默认 SFT 数据只有 train split。",
    )
    parser.add_argument(
        "--sft-split",
        default="train",
        help="输出 SFT split 名 (default: train)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1000,
        help="每类 SFT 样本数；默认 1000，即总计 4000 条。0 表示取剩余样本的最小类数量。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="每个 split 最多处理 N 条；0 表示全部。用于 smoke test。",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=DEFAULT_RL_MAX_COMPLETION_LENGTH,
        help=(
            "仅写入标注提示词的回答长度约束，不作为 OpenRouter API max_tokens；"
            f"默认 {DEFAULT_RL_MAX_COMPLETION_LENGTH}，与 RQ3 DAPO "
            "MAX_COMPLETION_LENGTH 对齐"
        ),
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许删除已存在的 output-dir 后重新保存",
    )
    parser.add_argument(
        "--replace-field",
        action="store_true",
        help="当 output-field 已存在时先删除旧字段再写入",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="保存成功后保留断点 JSONL",
    )
    return parser.parse_args()


def load_dataset_dict(data_path: Path) -> DatasetDict:
    data = load_from_disk(str(data_path))
    if isinstance(data, DatasetDict):
        return data
    if isinstance(data, Dataset):
        return DatasetDict({"train": data})
    raise TypeError(f"不支持的数据集类型: {type(data)!r}")


def resolve_splits(dataset_dict: DatasetDict, requested: list[str] | None) -> list[str]:
    available = list(dataset_dict.keys())
    if not requested:
        return available

    resolved: list[str] = []
    for name in requested:
        candidates = [
            name,
            name.replace("-", "_"),
            name.replace("_", "-"),
            name.lower(),
            name.lower().replace("-", "_"),
            name.upper(),
            name.upper().replace("-", "_"),
        ]
        for candidate in candidates:
            if candidate in dataset_dict and candidate not in resolved:
                resolved.append(candidate)
                break
        else:
            raise ValueError(f"split 不存在: {name!r}; 可用 split: {available}")
    return resolved


def collect_dataset_ids(dataset_dict: DatasetDict) -> set[str]:
    ids: set[str] = set()
    for dataset in dataset_dict.values():
        ids.update(str(item_id) for item_id in dataset["id"])
    return ids


def build_nonoverlap_sft_dataset(
    *,
    source_file: Path,
    rl_data_path: Path,
    split_name: str,
    samples_per_class: int,
    seed: int,
) -> DatasetDict:
    if not source_file.is_file():
        raise FileNotFoundError(f"原始训练文件不存在: {source_file}")
    if not rl_data_path.exists():
        raise FileNotFoundError(f"RQ3/RL 数据集不存在: {rl_data_path}")

    source_records = load_json(source_file)
    rl_dataset = load_dataset_dict(rl_data_path)
    excluded_ids = collect_dataset_ids(rl_dataset)

    buckets: dict[str, list[dict]] = {label: [] for label in LABEL_ORDER}
    unknown: list[str] = []
    duplicate_ids: set[str] = set()
    seen_ids: set[str] = set()

    for item in source_records:
        item_id = str(item.get("id", ""))
        if item_id in excluded_ids:
            continue
        if item_id in seen_ids:
            duplicate_ids.add(item_id)
            continue
        seen_ids.add(item_id)

        decision = item.get("decision", "")
        if decision in buckets:
            buckets[decision].append(item)
        else:
            unknown.append(decision)

    if unknown:
        raise ValueError(f"存在未识别的 decision 标签: {dict(Counter(unknown))}")

    counts = {label: len(rows) for label, rows in buckets.items()}
    min_count = min(counts.values())
    per_class = min_count if samples_per_class == 0 else samples_per_class
    if per_class <= 0:
        raise ValueError("--samples-per-class 必须为非负数；0 表示自动取最小类数量")
    if min_count < per_class:
        raise ValueError(
            f"非重叠剩余样本不足以每类采 {per_class} 条；"
            f"各类剩余数量: {counts}"
        )

    rng = random.Random(seed)
    rows: list[dict] = []
    selected_counts: dict[str, int] = {}
    for label in LABEL_ORDER:
        rng.shuffle(buckets[label])
        selected = buckets[label][:per_class]
        rows.extend(selected)
        selected_counts[label] = len(selected)
    rng.shuffle(rows)

    logger.info("原始样本数: %d", len(source_records))
    logger.info("排除 RQ3/RL 已用 ID: %d", len(excluded_ids))
    if duplicate_ids:
        logger.warning("非重叠候选中存在重复 id，已跳过重复项: %d", len(duplicate_ids))
    logger.info("非重叠剩余各类数量: %s", counts)
    logger.info("SFT balanced 采样: %s，总计 %d", selected_counts, len(rows))

    return DatasetDict({split_name: Dataset.from_list(rows, features=FEATURES)})


def format_articles(articles: Any) -> str:
    if articles is None:
        return "无"
    if not isinstance(articles, list):
        articles = list(articles)
    if not articles:
        return "无"

    groups: dict[str, list[str]] = {
        "刑法": [],
        "刑事诉讼法": [],
        "刑事诉讼规则": [],
        "其他": [],
    }
    prefix_map = {
        "cl:": "刑法",
        "cpl:": "刑事诉讼法",
        "cpr:": "刑事诉讼规则",
    }
    for item in articles:
        raw = str(item).strip()
        target = "其他"
        value = raw
        for prefix, label in prefix_map.items():
            if raw.startswith(prefix):
                target = label
                value = raw[len(prefix):].strip()
                break
        groups[target].append(value)

    lines = []
    for label, values in groups.items():
        if values:
            lines.append(f"{label}：" + "、".join(values))
    return "\n".join(lines)


def build_sft_input_messages(sample: dict[str, Any], prompt_variant: str) -> list[dict[str, str]]:
    return build_messages(
        person_info=sample.get("person_info", ""),
        procedure=sample.get("procedure", ""),
        fact=sample.get("fact", ""),
        prompt_variant=prompt_variant,
    )


def build_partial_sft_output(sample: dict[str, Any]) -> str:
    article_block = format_articles(sample.get("relevant_articles") or [])
    decision = sample.get("decision", "")
    return f"""\
<think>
{{待补全 CoT}}
</think>

【适用法条】
{article_block}

【审查分析】
{{待补全审查分析}}

【最终结论】
决定：{decision}
"""


def sample_to_messages(
    sample: dict[str, Any],
    *,
    max_answer_tokens: int,
    prompt_variant: str,
) -> list[dict[str, str]]:
    raw_articles = json.dumps(
        list(sample.get("relevant_articles") or []),
        ensure_ascii=False,
    )
    article_hint = format_articles(sample.get("relevant_articles") or [])
    relevant_articles = f"{raw_articles}\n按法域整理：\n{article_hint}"
    sft_input_messages = json.dumps(
        build_sft_input_messages(sample, prompt_variant),
        ensure_ascii=False,
        indent=2,
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        max_answer_tokens=max_answer_tokens,
        sft_input_messages=sft_input_messages,
        partial_sft_output=build_partial_sft_output(sample),
        relevant_articles=relevant_articles,
        decision=sample.get("decision", ""),
        raw_reasoning_and_decision=sample.get("raw_reasoning_and_decision", ""),
    )
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(max_answer_tokens=max_answer_tokens),
        },
        {"role": "user", "content": user_prompt},
    ]


def normalize_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = normalize_output(text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def validate_annotation_json(obj: dict[str, Any]) -> dict[str, str]:
    if not isinstance(obj, dict):
        raise ValueError("annotation output is not a JSON object")
    cot = str(obj.get("CoT") or "").strip()
    analysis = str(obj.get("审查分析") or "").strip()
    if not cot:
        raise ValueError("annotation JSON missing non-empty 'CoT'")
    if not analysis:
        raise ValueError("annotation JSON missing non-empty '审查分析'")
    return {"CoT": cot, "审查分析": analysis}


def build_complete_cot_text(sample: dict[str, Any], annotation: dict[str, str]) -> str:
    article_block = format_articles(sample.get("relevant_articles") or [])
    decision = str(sample.get("decision") or "").strip()
    if decision not in LABEL_ORDER:
        raise ValueError(f"未知 decision 标签: {decision!r}")
    return f"""\
<think>
{annotation["CoT"]}
</think>

【适用法条】
{article_block}

【审查分析】
{annotation["审查分析"]}

【最终结论】
决定：{decision}
"""


def load_completed(checkpoint_file: Path, output_field: str) -> dict[int, str]:
    completed: dict[int, str] = {}
    if not checkpoint_file.is_file():
        return completed

    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                idx = int(record["index"])
                value = str(record[output_field])
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning(
                    "跳过断点文件 %s 第 %d 行: %s",
                    checkpoint_file,
                    line_no,
                    exc,
                )
                continue
            completed[idx] = value
    return completed


def remove_tree_checked(path: Path, *, forbidden: set[Path]) -> None:
    resolved = path.resolve()
    if resolved in forbidden:
        raise ValueError(f"拒绝删除受保护目录: {resolved}")
    if resolved == PROJECT_ROOT or PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"拒绝删除项目目录外或项目根目录: {resolved}")
    if len(resolved.parts) <= len(PROJECT_ROOT.parts):
        raise ValueError(f"拒绝删除过短路径: {resolved}")
    shutil.rmtree(resolved)


async def call_llm_json(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    top_p: float | None,
    semaphore: asyncio.Semaphore,
    llm_options: LLMOptions,
    tag: str,
) -> tuple[dict[str, str] | None, str, str]:
    last_error = ""
    last_raw = ""
    for attempt in range(llm_options.max_retries):
        async with semaphore:
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "extra_body": {"reasoning": {"enabled": True}},
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if top_p is not None:
                    kwargs["top_p"] = top_p

                response = await client.chat.completions.create(**kwargs)
                choices = getattr(response, "choices", None) or []
                if not choices:
                    raise ValueError("OpenRouter returned empty choices")
                message = getattr(choices[0], "message", None)
                if message is None:
                    raise ValueError("OpenRouter returned empty message")
                last_raw = (message.content or "").strip()
                if not last_raw:
                    raise ValueError("OpenRouter returned empty content")

                obj = parse_json_object(last_raw)
                if obj is None:
                    last_error = "JSON parse failed"
                    logger.warning(
                        "[%s] JSON parse failed at attempt %d/%d: %s...",
                        tag,
                        attempt + 1,
                        llm_options.max_retries,
                        last_raw[:180],
                    )
                else:
                    try:
                        return validate_annotation_json(obj), last_raw, ""
                    except ValueError as exc:
                        last_error = str(exc)
                        logger.warning(
                            "[%s] JSON schema failed at attempt %d/%d: %s",
                            tag,
                            attempt + 1,
                            llm_options.max_retries,
                            exc,
                        )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "[%s] API error at attempt %d/%d: %s",
                    tag,
                    attempt + 1,
                    llm_options.max_retries,
                    exc,
                )
        if attempt < llm_options.max_retries - 1:
            await asyncio.sleep(min(60, 2**attempt))
    return None, last_raw, last_error


async def generate_one(
    *,
    client: Any,
    semaphore: asyncio.Semaphore,
    dataset: Dataset,
    split_name: str,
    idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample = dataset[idx]
    messages = sample_to_messages(
        sample,
        max_answer_tokens=args.max_answer_tokens,
        prompt_variant=args.prompt_variant,
    )
    annotation, raw_output, error = await call_llm_json(
        client=client,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        semaphore=semaphore,
        llm_options=LLMOptions(
            max_retries=args.max_retries,
        ),
        tag=f"{split_name}:{idx}",
    )
    if annotation is None:
        return {
            "split": split_name,
            "index": idx,
            "id": sample.get("id", ""),
            "ok": False,
            "error": error or "OpenRouter call or JSON parse failed",
            "raw_annotation_output": raw_output,
        }
    text = build_complete_cot_text(sample, annotation)
    return {
        "split": split_name,
        "index": idx,
        "id": sample.get("id", ""),
        "ok": True,
        args.output_field: text,
        "annotation": annotation,
        "raw_annotation_output": raw_output,
    }


async def generate_split_async(
    *,
    client: Any,
    dataset: Dataset,
    split_name: str,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    n = len(dataset)
    if args.max_samples > 0:
        n = min(n, args.max_samples)
        dataset = dataset.select(range(n))

    checkpoint_file = checkpoint_dir / f"{split_name}_json_annotation.jsonl"
    completed = load_completed(checkpoint_file, args.output_field)
    pending = [idx for idx in range(n) if idx not in completed]
    logger.info(
        "[%s] samples=%d completed=%d pending=%d",
        split_name,
        n,
        len(completed),
        len(pending),
    )

    if pending:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(args.concurrency)
        start = time.time()
        generated_count = 0

        with checkpoint_file.open("a", encoding="utf-8") as f:
            for offset in range(0, len(pending), args.concurrency):
                chunk = pending[offset: offset + args.concurrency]
                tasks = [
                    generate_one(
                        client=client,
                        semaphore=semaphore,
                        dataset=dataset,
                        split_name=split_name,
                        idx=idx,
                        args=args,
                    )
                    for idx in chunk
                ]
                success_count = 0
                for record in await asyncio.gather(*tasks):
                    idx = int(record["index"])
                    if not record.get("ok"):
                        logger.warning(
                            "[%s:%d] annotation failed: %s",
                            split_name,
                            idx,
                            record.get("error", ""),
                        )
                        continue
                    completed[idx] = str(record[args.output_field])
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    success_count += 1
                f.flush()

                generated_count += success_count
                elapsed = time.time() - start
                rate = generated_count / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[%s] progress %d/%d, chunk_success=%d/%d, %.2f samples/s",
                    split_name,
                    len(completed),
                    n,
                    success_count,
                    len(chunk),
                    rate,
                )

    missing = [idx for idx in range(n) if idx not in completed]
    if missing:
        raise RuntimeError(f"[{split_name}] 仍有未完成样本: {missing[:10]}")
    return [completed[idx] for idx in range(n)]


async def async_main() -> None:
    args = parse_args()

    source_file = Path(args.source_file)
    rl_data_path = Path(args.rl_data_path)
    output_dir = Path(args.output_dir)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else output_dir.with_name(output_dir.name + "_checkpoints")
    )

    if args.concurrency < 1:
        raise ValueError("--concurrency 必须为正整数")
    if args.max_samples < 0:
        raise ValueError("--max-samples 不能为负数")
    if args.max_answer_tokens < 1:
        raise ValueError("--max-answer-tokens 必须为正整数")
    if rl_data_path.resolve() == output_dir.resolve():
        raise ValueError("--output-dir 不能与 --rl-data-path 相同")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"输出目录已存在: {output_dir}。如需覆盖，请添加 --overwrite"
            )
        remove_tree_checked(output_dir, forbidden={rl_data_path.resolve()})

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("请设置环境变量 OPENROUTER_API_KEY")
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("请先安装依赖: pip install openai") from exc
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=180.0,
        max_retries=0,
    )

    logger.info("构建非重叠 balanced SFT 数据集")
    dataset_dict = build_nonoverlap_sft_dataset(
        source_file=source_file,
        rl_data_path=rl_data_path,
        split_name=args.sft_split,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    split_names = resolve_splits(dataset_dict, args.splits)
    logger.info("处理 splits: %s", ", ".join(split_names))

    for split_name in split_names:
        columns = dataset_dict[split_name].column_names
        if args.output_field in columns and not args.replace_field:
            raise ValueError(
                f"split {split_name!r} 已存在字段 {args.output_field!r}；"
                "如需替换，请添加 --replace-field"
            )

    logger.info(
        "OpenRouter 标注模型: %s  并发: %d  prompt_answer_token_limit: %d",
        args.model,
        args.concurrency,
        args.max_answer_tokens,
    )

    output_splits: dict[str, Dataset] = {}
    for split_name in split_names:
        dataset = dataset_dict[split_name]
        cot_values = await generate_split_async(
            client=client,
            dataset=dataset,
            split_name=split_name,
            checkpoint_dir=checkpoint_dir,
            args=args,
        )

        if args.max_samples > 0:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        if args.output_field in dataset.column_names:
            dataset = dataset.remove_columns(args.output_field)
        output_splits[split_name] = dataset.add_column(args.output_field, cot_values)

    output_dataset = DatasetDict(output_splits)
    logger.info("保存数据集: %s", output_dir)
    output_dataset.save_to_disk(str(output_dir))

    manifest = {
        "source_file": str(source_file),
        "rl_data_path": str(rl_data_path),
        "model": args.model,
        "output_field": args.output_field,
        "sft_sampling": {
            "excluded_dataset": str(rl_data_path),
            "split": args.sft_split,
            "samples_per_class": args.samples_per_class,
            "seed": args.seed,
            "label_order": list(LABEL_ORDER),
        },
        "splits": {name: len(output_dataset[name]) for name in output_dataset.keys()},
        "max_samples": args.max_samples,
        "generation": {
            "max_answer_tokens_prompt_only": args.max_answer_tokens,
            "api_max_tokens": None,
            "api_reasoning": {"enabled": True},
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "provider": "openrouter",
            "base_url": OPENROUTER_BASE_URL,
            "prompt_variant": args.prompt_variant,
            "annotation_output_schema": {"CoT": "string", "审查分析": "string"},
        },
    }
    with (output_dir / "cot_generation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not args.keep_checkpoints and checkpoint_dir.exists():
        remove_tree_checked(
            checkpoint_dir,
            forbidden={rl_data_path.resolve(), output_dir.resolve()},
        )

    logger.info("完成: %s", output_dir)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
