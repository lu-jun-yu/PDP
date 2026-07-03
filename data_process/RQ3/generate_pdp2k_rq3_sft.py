#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate non-overlapping balanced CoT-style SFT data for RQ3 cold start.

The script samples from data/PDP_dataset/train/dataset.json, excludes every
sample ID already used by the RQ3/RL DatasetDict, keeps class balance, uses
every field in each sampled record as generation context, adds one text field
containing the generated <think>...</think><answer>...</answer> target, and
saves a new DatasetDict.

Example:
    python data_process/RQ3/generate_pdp2k_rq3_sft.py ^
      --model-path models/Qwen3-8B ^
      --rl-data-path data/pdp2k_rq3 ^
      --output-dir data/pdp2k_rq3_sft
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from datasets import Dataset, DatasetDict, load_from_disk
except ImportError as exc:
    raise SystemExit("请先安装依赖: pip install datasets") from exc

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:
    raise SystemExit("请先安装依赖: pip install vllm") from exc

from build_pdp2k_rq3_dataset import FEATURES, LABEL_ORDER, load_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FIELD = "cot_data"

SYSTEM_PROMPT = """\
你是中国刑事检察文书的监督微调数据生成助手。你的任务是根据给定样本的全部字段，生成一条可用于 SFT 的 CoT 目标文本。

硬性要求：
1. 必须综合使用样本中的 id、meta、person_info、procedure、fact、relevant_articles、decision、raw_reasoning_and_decision、source_url 等全部字段信息。
2. relevant_articles 与 decision 是参考标注，生成内容必须与它们保持一致；不要把最终决定改成其他类别。
3. 输出必须且只能包含一个 <think> 块和一个 <answer> 块，不要输出 Markdown 代码块、解释或额外前后缀。
4. <think> 中写出面向训练数据的审查推理链，覆盖案件事实、证据状态、程序情况、法条依据与决定类型判断。
5. <answer> 中必须严格使用指定小标题：【适用法条】、【审查分析】、【最终结论】。
6. 【最终结论】中的“决定”只能是：起诉、相对不起诉、法定不起诉、存疑不起诉。
"""

USER_PROMPT_TEMPLATE = """\
请根据以下样本全部字段生成 CoT 数据。

【样本字段】
id: {id}
meta: {meta}
person_info: {person_info}
procedure: {procedure}
fact: {fact}
relevant_articles: {relevant_articles}
decision: {decision}
raw_reasoning_and_decision: {raw_reasoning_and_decision}
source_url: {source_url}

【输出格式】
<think>
{{CoT}}
</think>

<answer>
【适用法条】
刑法：第xxx条第xx款
刑事诉讼法：第xxx条第xx项、第xxx条第xx款
刑事诉讼规则：第xxx条第xx款

【审查分析】
{{审查分析}}

【最终结论】
决定：{{决定}}
</answer>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建与 RQ3/RL 训练集不重叠的 balanced CoT SFT 数据集"
    )
    parser.add_argument("--model-path", default="models/Qwen3-8B")
    parser.add_argument(
        "--source-file",
        default="data/PDP_dataset/train/dataset.json",
        help="原始训练 JSON，用于重采样 SFT 数据",
    )
    parser.add_argument(
        "--rl-data-path",
        "--data-path",
        dest="rl_data_path",
        default="data/pdp2k_rq3",
        help="需要排除的 RQ3/RL DatasetDict 路径；--data-path 为兼容别名",
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--dtype",
        default="auto",
        help="vLLM dtype: auto / bfloat16 / float16 / float32",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
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


def sample_to_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    meta = json.dumps(sample.get("meta") or {}, ensure_ascii=False)
    raw_articles = json.dumps(
        list(sample.get("relevant_articles") or []),
        ensure_ascii=False,
    )
    article_hint = format_articles(sample.get("relevant_articles") or [])
    relevant_articles = f"{raw_articles}\n按法域整理：\n{article_hint}"
    user_prompt = USER_PROMPT_TEMPLATE.format(
        id=sample.get("id", ""),
        meta=meta,
        person_info=sample.get("person_info", ""),
        procedure=sample.get("procedure", ""),
        fact=sample.get("fact", ""),
        relevant_articles=relevant_articles,
        decision=sample.get("decision", ""),
        raw_reasoning_and_decision=sample.get("raw_reasoning_and_decision", ""),
        source_url=sample.get("source_url", ""),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


def remove_tree_checked(path: Path, *, forbidden: set[Path]) -> None:
    resolved = path.resolve()
    if resolved in forbidden:
        raise ValueError(f"拒绝删除受保护目录: {resolved}")
    if resolved == PROJECT_ROOT or PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"拒绝删除项目目录外或项目根目录: {resolved}")
    if len(resolved.parts) <= len(PROJECT_ROOT.parts):
        raise ValueError(f"拒绝删除过短路径: {resolved}")
    shutil.rmtree(resolved)


def generate_split(
    *,
    llm: LLM,
    dataset: Dataset,
    split_name: str,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    n = len(dataset)
    if args.max_samples > 0:
        n = min(n, args.max_samples)
        dataset = dataset.select(range(n))

    checkpoint_file = checkpoint_dir / f"{split_name}.jsonl"
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
        batch_size = args.batch_size if args.batch_size > 0 else len(pending)
        sampling_params = build_sampling_params(args)
        start = time.time()
        generated_count = 0

        with checkpoint_file.open("a", encoding="utf-8") as f:
            for batch_start in range(0, len(pending), batch_size):
                batch_indices = pending[batch_start: batch_start + batch_size]
                messages = [sample_to_messages(dataset[idx]) for idx in batch_indices]
                outputs = llm.chat(messages=messages, sampling_params=sampling_params)

                for idx, output in zip(batch_indices, outputs, strict=True):
                    text = normalize_output(output.outputs[0].text)
                    completed[idx] = text
                    f.write(
                        json.dumps(
                            {
                                "split": split_name,
                                "index": idx,
                                "id": dataset[idx].get("id", ""),
                                args.output_field: text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                f.flush()

                generated_count += len(batch_indices)
                elapsed = time.time() - start
                rate = generated_count / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[%s] progress %d/%d, %.2f samples/s",
                    split_name,
                    len(completed),
                    n,
                    rate,
                )

    missing = [idx for idx in range(n) if idx not in completed]
    if missing:
        raise RuntimeError(f"[{split_name}] 仍有未完成样本: {missing[:10]}")
    return [completed[idx] for idx in range(n)]


def main() -> None:
    args = parse_args()

    source_file = Path(args.source_file)
    rl_data_path = Path(args.rl_data_path)
    output_dir = Path(args.output_dir)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else output_dir.with_name(output_dir.name + "_checkpoints")
    )

    if args.batch_size < 0:
        raise ValueError("--batch-size 不能为负数")
    if args.max_samples < 0:
        raise ValueError("--max-samples 不能为负数")
    if rl_data_path.resolve() == output_dir.resolve():
        raise ValueError("--output-dir 不能与 --rl-data-path 相同")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"输出目录已存在: {output_dir}。如需覆盖，请添加 --overwrite"
            )
        remove_tree_checked(output_dir, forbidden={rl_data_path.resolve()})

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

    logger.info("加载模型: %s", args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
    )

    output_splits: dict[str, Dataset] = {}
    for split_name in split_names:
        dataset = dataset_dict[split_name]
        cot_values = generate_split(
            llm=llm,
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
        "model_path": args.model_path,
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
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
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


if __name__ == "__main__":
    main()
