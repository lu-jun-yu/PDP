#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_process/Stage3/llm_filter_data_refinement.py

论文 Stage 3（数据精炼 / Data refinement）的 LLM 自动筛查脚本。

由于 PDP-Bench 共 4,630 条，逐条人工筛查不现实，故先用中文 SOTA 模型 GLM-5.1
对每条样本逐一判定是否违反三条数据质量标准，自动标记（flag）所有可能违反任一标准的样本；
**只输出被标记的“有问题子集”**，随后由法律专家仅对该子集做二次复核与精炼，
再由 data_process/Stage3/apply_refinement.py 把专家修订结果覆盖回全集。

本脚本只做“判定与标记”，不修改任何原始字段。

三条数据质量标准（任一不满足即标记）：
  1. 事实-证据格式化 (fact_evidence_formatting)
     每条案件的证据信息应以可识别的“证据清单”格式呈现；清单可以使用序号、分号、顿号或分行。
  2. 事实-标签一致性 (fact_label_consistency)
     结合嫌疑人信息、程序信息和事实，判断是否能推出与标签一致的公诉决定，或是否明显推出相反决定。
  3. 无数据泄露 (no_data_leakage)
     事实部分不得含有可据以推断公诉决定的冗余信息；例如“上述证据经依法收集、客观真实、
     足以认定指控的犯罪事实”一类表述会让人直接排除“存疑不起诉（证据不足）”类别。

失败处理：LLM 调用失败或 JSON 解析失败的样本，一律当作“没跑成功”，在本次运行内自动
重试若干轮；仍失败的留待下次 --resume 继续跑，**不会**被写入 checkpoint，也不会进入标记子集。

输入：
- data/public_sources/Stage2/dataset_reviewed_merged.json （Stage2 合并后的全集）

输出：
- data/public_sources/Stage3/dataset_refinement_flagged.json
    JSON list，仅含被标记（有问题）的样本。每条 = 原始记录全字段 + `refinement` 注记
    （含 violations 与各标准 compliant/reason/leakage_spans），供法律专家二次复核与精炼。
- data/public_sources/Stage3/dataset_refinement_flagged_checkpoint.jsonl
    JSONL，每条**已成功判定**的样本一行（紧凑判定），用于断点续跑与审计；非交付物。

Usage:
    set OPENROUTER_API_KEY=sk-or-...
    python data_process/Stage3/llm_filter_data_refinement.py
    python data_process/Stage3/llm_filter_data_refinement.py --concurrency 5 --max-samples 20
    python data_process/Stage3/llm_filter_data_refinement.py --fresh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage2"
STAGE3_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage3"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-5.1"
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_REASONING_MAX_TOKENS = 6144
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_ROUNDS = 2
DEFAULT_API_TIMEOUT = 180.0
DEFAULT_SAVE_EVERY = 20
DEFAULT_ATOMIC_WRITE_RETRIES = 8
DEFAULT_ATOMIC_WRITE_RETRY_SLEEP = 0.25

DEFAULT_INPUT = STAGE2_DIR / "dataset_reviewed_merged.json"
DEFAULT_FLAGGED_OUTPUT = STAGE3_DIR / "dataset_refinement_flagged.json"

# 三条数据质量标准的字段名（与 LLM 输出 JSON 的键一致）
CRITERIA = (
    "fact_evidence_formatting",
    "fact_label_consistency",
    "no_data_leakage",
)
CRITERIA_LABELS = {
    "fact_evidence_formatting": "事实-证据格式化",
    "fact_label_consistency": "事实-标签一致性",
    "no_data_leakage": "无数据泄露",
}


SYSTEM_PROMPT = """
你是中文刑事检察文书数据质量审核专家，精通《中华人民共和国刑法》《中华人民共和国刑事诉讼法》
和《人民检察院刑事诉讼规则》，熟悉检察机关审查起诉阶段的事实认定与证据规则。

任务：对一条结构化检察文书记录，逐一判定它是否满足三条数据质量标准，并标记任一不满足的样本。

判定对象主要是 `fact`（案件事实与证据部分）；判定“事实-标签一致性”时需结合 `person_info`、`procedure`、
`fact`、`decision`
（四分类标签：起诉 / 相对不起诉 / 法定不起诉 / 存疑不起诉）与 `raw_reasoning_and_decision`
（检察机关原始说理与处理结论）。

你可以先进行内部思考与自检；如果输出思考过程，必须放在最终 JSON 之前。
最终答案的最后一段必须是一个单独、完整、可被 json.loads 解析的 JSON 对象；JSON 之后不得再输出任何文字。
不要使用 markdown 代码块包裹 JSON。
""".strip()

USER_TEMPLATE = """
请审核下面这条检察文书记录是否满足三条数据质量标准，并对任一不满足的标准做出标记。
对每条标准：
- `compliant=true` 表示满足该标准（不需要人工处理）；
- `compliant=false` 表示违反该标准（需要标记并交由人工复核），并在 `reason` 中给出简短依据。

【三条数据质量标准】

1. 事实-证据格式化 (fact_evidence_formatting)
   - 要求：案件的证据信息应以可识别的“证据清单”格式呈现。清单不强制要求“1. ... 2. ... 3. ...”序号；
     只要能看出不同证据项被分隔列举即可，例如序号列表、分行列表、分号分隔、顿号分隔的证据名称列表。
   - 违反（compliant=false）：`fact` 中的证据信息以非结构化的散文/段落形式描述，
     无法识别出相对独立的证据项；或根本未列出证据。
   - 满足（compliant=true）：`fact` 已包含可识别的证据清单。
   - 注意：只看证据“是否以清单方式列举”这一形式问题，不评价证据本身是否充分。

2. 事实-标签一致性 (fact_label_consistency)
   - 要求：结合 `person_info`、`procedure` 和 `fact`，判断这些输入事实是否可以推出与 `decision` 一致的处理结论。
   - 违反（compliant=false）：输入事实更合理地推出与 `decision` 不同的结论，或 `fact` 与 `raw_reasoning_and_decision`
     在关键事实、证据状态、处理方向上明显矛盾。
   - 满足（compliant=true）：输入事实可以推出与 `decision` 一致的结论。
   - 注意：不能仅因为 `fact` 较短、没有直接体现全部说理，就判为违反。

3. 无数据泄露 (no_data_leakage)
   - 要求：`fact` 部分不得含有可据以直接推断公诉决定的冗余信息。
   - 违反（compliant=false）：`fact` 中出现对证据/事实的“结论性评价”或预示处理方向的表述，
     使人无需分析即可推断标签。典型例子：
       * “证据确实、充分”“事实清楚”“应当追究刑事责任”等结论性断言；
       * “犯罪情节轻微，依法不需要判处刑罚”“不构成犯罪”等直接指向当前决定的措辞。
   - 满足（compliant=true）：`fact` 仅客观陈述行为、情节、证据目录，不含上述结论性/导向性表述。
   - 注意：“认定上述事实的证据如下”等证据清单引导语不算泄露，仅罗列证据名称（如“1. 书证；2. 证人证言”）不算泄露；
     对证据做“确实充分/足以认定”这类评价才算泄露。

【输出格式】
最终 JSON 必须且只能包含以下结构（键名与类型完全一致）：
{{
  "fact_evidence_formatting": {{"compliant": true 或 false, "reason": "简短中文依据"}},
  "fact_label_consistency":   {{"compliant": true 或 false, "reason": "简短中文依据"}},
  "no_data_leakage":          {{"compliant": true 或 false, "reason": "简短中文依据", "leakage_spans": ["原文片段", ...]}}
}}
说明：
- `leakage_spans` 仅在 `no_data_leakage.compliant=false` 时列出导致泄露的 `fact` 原文片段；
  满足时给出空列表 []。
- `reason` 用一两句话说明判定理由，不要复述整段原文。
- 只输出三条标准的判定 JSON；脚本会在原记录基础上附加 `refinement` 注记形成最终文件。
- 如果你输出了思考过程，JSON 必须出现在最后，且 JSON 后面不得有任何文字。

【待审核记录】
{record_json}
""".strip()


@dataclass(frozen=True)
class LLMOptions:
    max_output_tokens: int
    reasoning_max_tokens: int
    max_retries: int
    enable_thinking: bool


@dataclass
class IncrementalWriter:
    """增量落盘：仅标记子集（flagged）+ 续跑/审计 checkpoint。

    flagged 子集是交付物，从 verdicts_by_id（仅含 flagged=true）与输入全集字段拼接而成；
    checkpoint 仅记录**成功判定**的样本，用于断点续跑与审计。
    失败样本不落盘，视为“没跑”，由本次重试或下次 --resume 重新处理。
    """

    flagged_path: Path
    checkpoint_path: Path
    records_by_id: dict[str, dict[str, Any]]
    input_order: list[str]
    verdicts_by_id: dict[str, dict[str, Any]]
    save_every: int = DEFAULT_SAVE_EVERY
    dirty_count: int = 0
    handled_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    flagged_count: int = 0

    def on_result(self, result: dict[str, Any]) -> None:
        self.handled_count += 1
        verdict = result.get("verdict")
        if not result.get("ok") or not isinstance(verdict, dict):
            # 失败样本：不落盘，视为没跑
            self.fail_count += 1
            return

        self.success_count += 1
        key = normalize_case_id(result.get("id"))
        if key:
            self.verdicts_by_id[key] = verdict
            if verdict.get("flagged"):
                self.flagged_count += 1
            append_checkpoint_line(self.checkpoint_path, build_checkpoint_row(result))

        self.dirty_count += 1
        if self.save_every <= 1 or self.dirty_count >= self.save_every:
            self.flush()

    def build_flagged_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in self.input_order:
            verdict = self.verdicts_by_id.get(key)
            if not verdict or not verdict.get("flagged"):
                continue
            record = self.records_by_id.get(key)
            if record is None:
                continue
            entry = dict(record)
            entry["refinement"] = {
                "violations": verdict.get("violations", []),
                "criteria": verdict.get("criteria", {}),
            }
            rows.append(entry)
        return rows

    def flush(self, force: bool = False) -> None:
        if self.dirty_count <= 0 and not force:
            return
        try:
            write_json(self.flagged_path, self.build_flagged_rows())
            self.dirty_count = 0
        except PermissionError as exc:
            logger.warning("增量落盘失败，文件可能被占用；稍后会继续重试: %s", exc)


def normalize_case_id(value: Any) -> str:
    return str(value or "").replace("〔", "[").replace("〕", "]").strip()


def _strip_think_block(text: str) -> str:
    think_end = text.find("</think>")
    if think_end != -1:
        return text[think_end + len("</think>") :].strip()
    return text.strip()


def _extract_first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _extract_last_balanced_json_object(text: str) -> str | None:
    candidates: list[str] = []
    search_start = 0
    while search_start < len(text):
        start = text.find("{", search_start)
        if start == -1:
            break
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    search_start = idx + 1
                    break
        else:
            break
    return candidates[-1] if candidates else None


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = _strip_think_block(text).replace("\ufeff", "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    balanced = _extract_last_balanced_json_object(text)
    if balanced:
        try:
            obj = json.loads(balanced)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    balanced = _extract_first_balanced_json_object(text)
    if balanced:
        try:
            obj = json.loads(balanced)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    cleaned = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_text_block(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\xa0", " ").replace("\u3000", " ")
    text = re.sub(r"[\u2000-\u200f\u2028\u2029\u202f\u205f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_record_schema(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    return {
        "id": str(record.get("id") or ""),
        "meta": {
            "date": str(meta.get("date") or ""),
            "province": str(meta.get("province") or ""),
        },
        "person_info": str(record.get("person_info") or ""),
        "procedure": str(record.get("procedure") or ""),
        "fact": str(record.get("fact") or ""),
        "relevant_articles": list(record.get("relevant_articles") or []),
        "decision": str(record.get("decision") or ""),
        "raw_reasoning_and_decision": str(record.get("raw_reasoning_and_decision") or ""),
        "source_url": str(record.get("source_url") or ""),
    }


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    # 仅提供判定三条标准所需的字段，聚焦判定、节省 token。
    prompt_record = {
        "id": str(record.get("id") or ""),
        "decision": str(record.get("decision") or ""),
        "relevant_articles": list(record.get("relevant_articles") or []),
        "person_info": normalize_text_block(record.get("person_info", "")),
        "procedure": normalize_text_block(record.get("procedure", "")),
        "fact": normalize_text_block(record.get("fact", "")),
        "raw_reasoning_and_decision": normalize_text_block(
            record.get("raw_reasoning_and_decision", "")
        ),
    }
    record_json = json.dumps(prompt_record, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(record_json=record_json)},
    ]


async def call_llm_json(
    client: AsyncOpenAI,
    messages: list[dict[str, str]],
    model: str,
    semaphore: asyncio.Semaphore,
    llm_options: LLMOptions,
    tag: str,
) -> tuple[dict[str, Any] | None, str, str]:
    last_raw = ""
    last_error = ""
    for attempt in range(llm_options.max_retries):
        async with semaphore:
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": llm_options.max_output_tokens,
                }
                if llm_options.enable_thinking and llm_options.reasoning_max_tokens > 0:
                    kwargs["extra_body"] = {
                        "reasoning": {"max_tokens": llm_options.reasoning_max_tokens}
                    }
                response = await client.chat.completions.create(**kwargs)
                last_raw = response.choices[0].message.content or ""
                obj = parse_json_object(last_raw)
                if obj is not None:
                    return obj, last_raw, ""
                logger.warning("[%s] 第%s次 JSON 解析失败: %s...", tag, attempt + 1, last_raw[:180])
            except Exception as exc:
                last_error = str(exc)
                logger.warning("[%s] API 错误 (第%s次): %s", tag, attempt + 1, exc)
        if attempt < llm_options.max_retries - 1:
            await asyncio.sleep(2**attempt)
    return None, last_raw, last_error


def _coerce_compliant(value: Any) -> bool | None:
    """把 LLM 返回的 compliant 值归一化为 bool；无法识别返回 None。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1", "是", "满足", "合规", "compliant"):
            return True
        if v in ("false", "no", "0", "否", "不满足", "违反", "不合规", "non-compliant"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def build_verdict(llm_obj: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """根据 LLM 输出构造判定结果 verdict，并返回 (verdict, issues)。

    单条标准缺失或无法解析时，按“违反”（需人工复核）处理，并记录 issue；
    注意这是“模型给了结果但某条标准字段缺失”的局部从严，不同于整体调用失败（那会判为没跑）。
    verdict = {"flagged": bool, "violations": [...], "criteria": {...}}
    """
    issues: list[str] = []
    criteria_out: dict[str, Any] = {}
    violations: list[str] = []

    for key in CRITERIA:
        raw = llm_obj.get(key)
        compliant: bool | None = None
        reason = ""
        leakage_spans: list[str] = []

        if isinstance(raw, dict):
            compliant = _coerce_compliant(raw.get("compliant"))
            reason = str(raw.get("reason") or "").strip()
            if key == "no_data_leakage":
                spans = raw.get("leakage_spans")
                if isinstance(spans, list):
                    leakage_spans = [str(s).strip() for s in spans if str(s).strip()]
        elif raw is not None:
            # 模型可能直接给出 bool/字符串而非对象
            compliant = _coerce_compliant(raw)

        if compliant is None:
            # 缺失或无法解析 -> 从严标记为违反，交由人工复核
            issues.append(f"missing_or_bad_{key}")
            compliant = False
            if not reason:
                reason = "模型未给出有效判定，从严标记为待人工复核"

        entry: dict[str, Any] = {"compliant": compliant, "reason": reason}
        if key == "no_data_leakage":
            entry["leakage_spans"] = leakage_spans
        criteria_out[key] = entry
        if not compliant:
            violations.append(key)

    verdict = {
        "flagged": bool(violations),
        "violations": violations,
        "criteria": criteria_out,
    }
    return verdict, sorted(set(issues))


async def process_one_record(
    index: int,
    record: dict[str, Any],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    llm_options: LLMOptions,
) -> dict[str, Any]:
    normalized = normalize_record_schema(record)
    case_id = normalized.get("id", "")
    result: dict[str, Any] = {
        "index": index,
        "id": case_id,
        "ok": False,
        "error": None,
        "issues": [],
    }

    obj, raw_response, api_error = await call_llm_json(
        client=client,
        messages=build_messages(normalized),
        model=model,
        semaphore=semaphore,
        llm_options=llm_options,
        tag=f"{index + 1}/{case_id or 'no-id'}",
    )
    if api_error:
        result["api_error"] = api_error

    if obj is None:
        # 整体失败：当作没跑，留待重试 / --resume
        result["error"] = "LLM 调用失败" if api_error and not raw_response else "LLM 返回 JSON 解析失败"
        return result

    verdict, issues = build_verdict(obj)
    result.update({"ok": True, "verdict": verdict, "issues": issues})
    return result


async def run_all(
    pending: list[tuple[int, dict[str, Any]]],
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    llm_options: LLMOptions,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    start = time.time()

    async def _one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, rec = item
        return await process_one_record(
            index=idx,
            record=rec,
            client=client,
            model=model,
            semaphore=semaphore,
            llm_options=llm_options,
        )

    tasks = [asyncio.create_task(_one(item)) for item in pending]
    results: list[dict[str, Any]] = []
    completed = 0
    try:
        for finished in asyncio.as_completed(tasks):
            result = await finished
            results.append(result)
            completed += 1
            if on_result is not None:
                on_result(result)
            if completed % max(1, min(20, len(pending))) == 0 or completed == len(pending):
                logger.info("运行进度: %s/%s", completed, len(pending))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("本批完成: %s 条，用时 %.1fs", completed, time.time() - start)
    return results


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    last_error: PermissionError | None = None
    for attempt in range(DEFAULT_ATOMIC_WRITE_RETRIES):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < DEFAULT_ATOMIC_WRITE_RETRIES - 1:
                time.sleep(DEFAULT_ATOMIC_WRITE_RETRY_SLEEP * (attempt + 1))
    if last_error is not None:
        raise last_error


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_text(path, json.dumps(records, ensure_ascii=False, indent=2))


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"输入文件应为 JSON list: {path}")
    bad_count = sum(1 for item in data if not isinstance(item, dict))
    if bad_count:
        logger.warning("跳过非对象记录: %s", bad_count)
    return [normalize_record_schema(item) for item in data if isinstance(item, dict)]


def build_checkpoint_row(result: dict[str, Any]) -> dict[str, Any]:
    """构造 checkpoint 行：紧凑判定（不含原始记录文本）。仅对成功样本调用。"""
    verdict = result.get("verdict") or {}
    row: dict[str, Any] = {
        "index": result.get("index"),
        "id": result.get("id", ""),
        "flagged": bool(verdict.get("flagged")),
        "violations": list(verdict.get("violations") or []),
        "criteria": verdict.get("criteria", {}),
    }
    if result.get("issues"):
        row["issues"] = list(result["issues"])
    return row


def append_checkpoint_line(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """读取 checkpoint，返回 id -> verdict（仅成功判定样本），用于续跑跳过。"""
    verdicts_by_id: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return verdicts_by_id
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "criteria" not in obj:
            continue
        key = normalize_case_id(obj.get("id"))
        if not key:
            continue
        verdicts_by_id[key] = {
            "flagged": bool(obj.get("flagged")),
            "violations": list(obj.get("violations") or []),
            "criteria": obj.get("criteria", {}),
        }
    return verdicts_by_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="论文 Stage 3 数据精炼：用 GLM-5.1 标记违反三条数据质量标准的“有问题子集”"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="输入 JSON 文件（默认 data/public_sources/Stage2/dataset_reviewed_merged.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_FLAGGED_OUTPUT),
        help="被标记子集输出 JSON（默认 data/public_sources/Stage3/dataset_refinement_flagged.json）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="续跑/审计 checkpoint JSONL；为空时由 --output 推导（*_checkpoint.jsonl）",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter 模型 id (default: {DEFAULT_MODEL})")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="并发数")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=DEFAULT_REASONING_MAX_TOKENS,
        help=f"reasoning token budget (default: {DEFAULT_REASONING_MAX_TOKENS})",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="disable reasoning token budget (default: enabled)",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="单条样本 API 内部重试次数")
    parser.add_argument(
        "--retry-rounds",
        type=int,
        default=DEFAULT_RETRY_ROUNDS,
        help="本次运行内，对失败样本整体重试的轮数（默认 2）",
    )
    parser.add_argument("--api-timeout", type=float, default=DEFAULT_API_TIMEOUT)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="每处理 N 条增量落盘")
    parser.add_argument("--max-samples", type=int, default=0, help="最多处理记录数，0=全部")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="断点续跑，跳过 checkpoint 中已成功判定的案号（默认启用）",
    )
    resume_group.add_argument(
        "--fresh",
        dest="resume",
        action="store_false",
        help="从头重跑，不读取旧 checkpoint；若 checkpoint 已存在则拒绝运行",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = output_path.with_name(output_path.stem + "_checkpoint.jsonl")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("请设置环境变量 OPENROUTER_API_KEY")
        sys.exit(1)

    records = load_json_list(input_path)
    input_order = [normalize_case_id(record.get("id")) for record in records]
    records_by_id = {
        normalize_case_id(record.get("id")): record
        for record in records
        if normalize_case_id(record.get("id"))
    }

    if args.resume:
        verdicts_by_id = load_checkpoint(checkpoint_path)
    else:
        if checkpoint_path.exists():
            raise SystemExit(
                f"--fresh 不会自动删除已有 checkpoint：{checkpoint_path}\n"
                "请先手动移动/删除该文件，或通过 --checkpoint 指定一个新的 checkpoint 路径。"
            )
        verdicts_by_id = {}

    done_ids = set(verdicts_by_id)
    pending = [
        (idx, record)
        for idx, record in enumerate(records)
        if normalize_case_id(record.get("id")) not in done_ids
    ]
    pre_skipped = len(records) - len(pending)
    if args.max_samples > 0:
        pending = pending[: args.max_samples]

    writer = IncrementalWriter(
        flagged_path=output_path,
        checkpoint_path=checkpoint_path,
        records_by_id=records_by_id,
        input_order=input_order,
        verdicts_by_id=verdicts_by_id,
        save_every=max(1, args.save_every),
    )

    logger.info("输入: %s (%s 条)", input_path, len(records))
    logger.info("标记子集输出: %s", output_path)
    logger.info("checkpoint: %s", checkpoint_path)
    logger.info(
        "模型: %s  并发: %s  temperature: API default  thinking: %s  reasoning_tokens: %s  retry_rounds: %s",
        args.model,
        args.concurrency,
        "on" if not args.disable_thinking else "off",
        max(0, args.reasoning_max_tokens),
        max(0, args.retry_rounds),
    )
    if args.resume:
        logger.info("断点续跑已启用，预跳过: %s 条", pre_skipped)

    if not pending:
        writer.flush(force=True)
        logger.info("没有待处理记录，直接结束。当前标记子集: %s 条", len(writer.build_flagged_rows()))
        return

    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=args.api_timeout,
        max_retries=0,
    )
    llm_options = LLMOptions(
        max_output_tokens=max(1, args.max_output_tokens),
        reasoning_max_tokens=max(0, args.reasoning_max_tokens),
        max_retries=max(1, args.max_retries),
        enable_thinking=not args.disable_thinking,
    )

    interrupted = False
    remaining = pending
    retry_rounds = max(0, args.retry_rounds)
    try:
        for attempt in range(retry_rounds + 1):
            if not remaining:
                break
            if attempt > 0:
                backoff = min(30, 5 * attempt)
                logger.info(
                    "失败样本重试 第%s/%s 轮：%s 条（等待 %ss）",
                    attempt,
                    retry_rounds,
                    len(remaining),
                    backoff,
                )
                time.sleep(backoff)
            results = asyncio.run(
                run_all(
                    pending=remaining,
                    client=client,
                    model=args.model,
                    concurrency=max(1, args.concurrency),
                    llm_options=llm_options,
                    on_result=writer.on_result,
                )
            )
            failed_idx = [r["index"] for r in results if not r.get("ok")]
            remaining = [(idx, records[idx]) for idx in failed_idx]
    except KeyboardInterrupt:
        interrupted = True
        writer.flush(force=True)
        logger.warning("检测到中断，已保存当前进度；可使用 --resume 继续。")

    writer.flush(force=True)

    # 标记统计
    flagged_rows = writer.build_flagged_rows()
    per_criterion = {key: 0 for key in CRITERIA}
    for verdict in writer.verdicts_by_id.values():
        for key in verdict.get("violations") or []:
            if key in per_criterion:
                per_criterion[key] += 1

    still_failed = len(remaining)
    logger.info("已成功判定: %s/%s（预跳过 %s）", len(writer.verdicts_by_id), len(records), pre_skipped)
    if still_failed:
        logger.warning(
            "仍有 %s 条在 %s 轮重试后失败，未跑成功；请稍后用 --resume 重新运行以补齐。",
            still_failed,
            retry_rounds,
        )
        for idx, rec in remaining[:20]:
            logger.warning("  未完成: %s", normalize_case_id(rec.get("id")) or f"index={idx}")
    logger.info("被标记（有问题）子集: %s 条", len(flagged_rows))
    for key in CRITERIA:
        logger.info("  违反「%s」: %s", CRITERIA_LABELS[key], per_criterion[key])
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
