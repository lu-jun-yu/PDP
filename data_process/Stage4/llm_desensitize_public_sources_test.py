#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_process/llm_desensitize_public_sources_test.py

使用 GLM-5.1 对 data/public_sources/Stage3/dataset_reviewed.json 做敏感信息脱敏，
输出 data/public_sources/Stage4/dataset_final.json。

处理范围：
- 人名：犯罪嫌疑人、被害人、证人、同案人、收件人等自然人姓名；
- 地址：户籍地、现住址、收件地址、住家/门牌/楼栋/单元/室号等精确地址；
- 身份证号：完整或半脱敏身份证号统一进一步遮蔽；
- 手机号：手机号、联系电话、收件电话等；
- 其他：出生日期、银行卡/账号、微信/QQ/邮箱、快递/物流号、车牌号等可识别个人的信息。

输出 schema 与输入保持一致，只改文本字段中的敏感信息。

Usage:
    set OPENROUTER_API_KEY=sk-or-...
    python data_process/llm_desensitize_public_sources_test.py
    python data_process/llm_desensitize_public_sources_test.py --concurrency 5 --max-samples 20
    python data_process/llm_desensitize_public_sources_test.py --resume
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
STAGE3_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage3"
STAGE4_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage4"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-5.1"
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_OUTPUT_TOKENS = 8192 * 4
DEFAULT_REASONING_MAX_TOKENS = 6144
DEFAULT_MAX_RETRIES = 4
DEFAULT_API_TIMEOUT = 180.0
DEFAULT_SAVE_EVERY = 20
DEFAULT_ATOMIC_WRITE_RETRIES = 8
DEFAULT_ATOMIC_WRITE_RETRY_SLEEP = 0.25
MASK_ADDRESS_NUMBERS = False

TEXT_FIELDS = (
    "person_info",
    "procedure",
    "fact",
    "raw_reasoning_and_decision",
)

SCHEMA_KEYS = (
    "id",
    "meta",
    "person_info",
    "procedure",
    "fact",
    "relevant_articles",
    "decision",
    "raw_reasoning_and_decision",
    "source_url",
)

SYSTEM_PROMPT = """
你是中文刑事检察文书数据脱敏助手，专门处理公开法律文书中的个人敏感信息。

任务：对结构化检察文书记录做最小必要脱敏。你必须尽量保留法律事实、叙事顺序、证据结构、
罪名、金额、数量、案发时间、程序时间、法条和处理结论，只替换能够识别自然人或私人主体的敏感信息。

你可以先进行内部思考、列出映射和自检；如果输出思考过程，必须放在最终 JSON 之前。
最终答案的最后一段必须是一个单独、完整、可被 json.loads 解析的 JSON 对象；JSON 之后不得再输出任何文字。
不要使用 markdown 代码块包裹 JSON。

核心原则：
1. 忠实原文，不新增事实，不删除影响案件事实判断的行为、金额、时间、证据名称或法条。
2. 同一条记录内，同一自然人必须使用一致的脱敏名称；已有“某”“某某”“某甲/某乙”等脱敏名称可以保留。
3. 犯罪嫌疑人、被害人、证人、同案人、亲属、车主、收件人、银行卡户名等自然人真实姓名都要脱敏。
4. 公安机关、检察院、法院、司法鉴定中心、行政机关、道路/高速/收费站等公共机构或公共地点一般不要脱敏。
5. 私人住址、户籍地精确门牌、收件地址、私人公司/店铺/培训学校/住宅小区等可识别个人或私人主体的信息要脱敏。
6. 只返回四个键：person_info、procedure、fact、raw_reasoning_and_decision；不得新增、删除或改名。
""".strip()

USER_TEMPLATE = """
请对下面 JSON 记录中的文本字段做敏感信息脱敏。你可以先思考，但最终输出的最后一段必须是一个单独的 JSON 对象：

{{"person_info":"","procedure":"","fact":"","raw_reasoning_and_decision":""}}

【输入字段说明】
- `person_info`：人员身份、出生日期、身份证号、户籍/现住址、职业单位、强制措施、前科劣迹等。
- `procedure`：侦查、移送审查起诉、退补、延期、权利告知、讯问、听取意见等程序事实。
- `fact`：案件事实与证据目录，常含犯罪嫌疑人、被害人、证人、车辆、银行卡、手机号、收件地址等。
- `raw_reasoning_and_decision`：本院认为、法条适用和处理决定，也可能重复出现自然人姓名。

【必须处理的敏感信息】
1. 人名：犯罪嫌疑人、被害人、证人、同案人、收件人、亲属、车主、银行卡户名等自然人真实姓名。
   - 两字姓名通常脱敏为“姓某”，三字及以上通常脱敏为“姓某某”。
   - 同姓多人需要区分时，可使用“王某甲”“王某乙”等，且全文保持一致。
   - 已经是“李某某”“王某甲”“肖某某（另案处理）”这类形式的，不要过度改写。
   - 少数民族姓名、外文名、绰号/网名如果能识别个人，也要改为“某某”或保留角色关系后遮蔽。
2. 地址：户籍所在地、现住址、家住、收件地址、私人住宅、门牌、楼栋、单元、室号、村组等精确地址。
   - 可保留省、市、县/区、乡镇/街道等粗粒度地点；门牌、楼栋、单元、房号、村组、私宅名称等要遮蔽。
   - 案发地点、道路、收费站、公共场所等如是案件事实必要信息，可以保留到合理粗粒度，但私人住址和具体房号应遮蔽。
3. 身份证号：完整或半脱敏身份证号都统一改为“******************”。
4. 手机号：手机号、联系电话、收件电话等统一改为“1**********”或“***********”。
5. 其他：出生年月日、银行卡号/账户号、支付宝/微信/QQ/邮箱、快递/物流单号、完整车牌号、私人公司/店铺/学校具体名称等可识别信息。
   - 自然人的出生日期改为“19**年**月**日出生”或“20**年**月**日出生”；不要改案发时间、程序时间、鉴定时间。
   - 银行卡号、账号、快递单号、社交账号等保留字段含义，遮蔽具体号码。
   - 完整车牌号可保留省份简称和首字母，例如“贵A****”。
   - 公司、店铺、培训学校、个体工商户等私人主体名称，如果会识别到具体个人经营主体，可用“**公司”“**店”“**培训学校”等遮蔽。

【不要改动】
1. 不要改案号、meta、source_url、decision、relevant_articles；这些字段不在输出中。
2. 不要改罪名、金额、数量、案发日期、程序日期、法条、证据种类、诉讼程序和处理结论。
3. 不要把“犯罪嫌疑人”改成“被告人”或“被不起诉人”。
4. 不要为了脱敏而概括、删减或重写事实；能局部替换就局部替换。
5. 不要脱敏法律机关、司法鉴定中心、公安局、检察院、法院、行政区划、省市县名称。

【一致性要求】
1. 同一自然人在四个字段内反复出现时，必须保持同一个脱敏名称。
2. 不同自然人不要合并成同一个称谓；必要时用“某甲/某乙/某丙”区分。
3. 如果输入已经部分脱敏，只补足仍可识别的部分，不要把已经稳定的“某某”称谓改乱。

【示例】
输入片段：
犯罪嫌疑人韩冬，男，1978年11月26日出生，公民身份号码420684199707124517，现住贵州省赫章县城关镇和平路12号3栋2单元201室，联系电话13812345678。韩冬驾驶贵FO1234号小型普通客车搭载文煜韬前往李长明家。
输出片段：
犯罪嫌疑人韩某某，男，19**年**月**日出生，公民身份号码******************，现住贵州省赫章县城关镇**路**号**栋**单元**室，联系电话1**********。韩某某驾驶贵F****号小型普通客车搭载文某某前往李某某家。

【最终输出要求】
最终 JSON 必须只有以下四个键，键名和类型必须完全一致：
{{"person_info":"...","procedure":"...","fact":"...","raw_reasoning_and_decision":"..."}}
如果你输出了思考过程，JSON 必须出现在最后，且 JSON 后面不能有任何文字。

【待脱敏 JSON】
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
    output_path: Path
    issues_path: Path
    records_by_id: dict[str, dict[str, Any]]
    order: list[str]
    issue_records: dict[str, dict[str, Any]]
    save_every: int = DEFAULT_SAVE_EVERY
    dirty_count: int = 0
    handled_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    warn_count: int = 0
    skipped_count: int = 0

    def on_result(self, result: dict[str, Any]) -> None:
        self.handled_count += 1
        if result.get("skipped"):
            self.skipped_count += 1
            return

        if result.get("ok"):
            self.success_count += 1
            record = result.get("record")
            if isinstance(record, dict):
                key = normalize_case_id(record.get("id"))
                if key and key not in self.order:
                    self.order.append(key)
                if key:
                    self.records_by_id[key] = record
            if result.get("issues"):
                self.warn_count += 1
        else:
            self.fail_count += 1

        update_issue_records(self.issue_records, result)
        self.dirty_count += 1
        if self.save_every <= 1 or self.dirty_count >= self.save_every or not result.get("ok"):
            self.flush()

    def flush(self, force: bool = False) -> None:
        if self.dirty_count <= 0 and not force:
            return
        rows = [self.records_by_id[key] for key in self.order if key in self.records_by_id]
        try:
            write_json(self.output_path, rows)
            write_issue_log_records(self.issues_path, self.issue_records)
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


FULL_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]"
    r"(?![A-Za-z0-9])"
)
PARTIAL_ID_RE = re.compile(
    r"(?<![A-Za-z0-9*])"
    r"(?:[1-9]\d{5}|\*{6})(?:(?:18|19|20)\d{2}|\*{4})[\d*]{8}"
    r"(?![A-Za-z0-9*])"
)
MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
PARTIAL_MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{2}\*{3,5}\d{4}(?!\d)")
EMAIL_RE = re.compile(r"(?<![\w.%-])([A-Za-z0-9._%+-]{2})[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![\w.%-])")
LICENSE_PLATE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z])"
    r"[A-Z0-9]{5,6}"
    r"(?![A-Za-z0-9])"
)
ADDRESS_NUMBER_RE = re.compile(r"\d{1,5}(?=号楼|栋|幢|单元|室|房|号|组)")
ACCOUNT_CONTEXT_RE = re.compile(
    r"(?P<label>(?:银行卡(?:号)?|银行账号|卡号|账号|账户号|支付宝(?:账号|账户)?|微信(?:号|账号)?|QQ号|快递单号|物流号|运单号|收件电话|联系电话|联系方式|手机号码|手机号|电话)\s*[为是:：]?\s*)"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9_@.\-*]{4,})"
)
BIRTH_DATE_RE = re.compile(
    r"(?:(?P<prefix>生于|出生日期[:：]?|出生年月日[:：]?|出生时间[:：]?)\s*"
    r"(?P<prefix_year>(?:18|19|20)\d{2})年\s*\d{1,2}月\s*\d{1,2}日"
    r"|(?P<suffix_year>(?:18|19|20)\d{2})年\s*\d{1,2}月\s*\d{1,2}日\s*(?P<suffix>出生|生))"
)


def mask_account_value(value: str) -> str:
    if "@" in value and "." in value:
        return EMAIL_RE.sub(r"\1***@\2", value)
    if re.fullmatch(r"1\*{10}", value):
        return "1**********"
    if re.fullmatch(r"1[3-9][\d*]{9}", value):
        return "1**********"
    if re.fullmatch(r"1[3-9]\d{9}", value):
        return "1**********"
    if re.fullmatch(r"\d{6,}", value):
        return "*" * min(max(len(value), 6), 18)
    if len(value) <= 2:
        return "*" * len(value)
    return value[0] + "***" + value[-1]


def mask_birth_date(match: re.Match[str]) -> str:
    prefix = match.group("prefix") or ""
    suffix = match.group("suffix") or ""
    year = match.group("prefix_year") or match.group("suffix_year") or "19"
    masked = f"{year[:2]}**年**月**日"
    return f"{prefix}{masked}{suffix}"


def regex_desensitize_text(text: Any, *, mask_address_numbers: bool = False) -> str:
    out = normalize_text_block(text)
    if not out:
        return ""
    out = BIRTH_DATE_RE.sub(mask_birth_date, out)
    out = FULL_ID_RE.sub("******************", out)
    out = PARTIAL_ID_RE.sub("******************", out)
    out = PARTIAL_MOBILE_RE.sub("1**********", out)
    out = MOBILE_RE.sub("1**********", out)
    out = EMAIL_RE.sub(r"\1***@\2", out)
    out = LICENSE_PLATE_RE.sub(r"\1****", out)
    out = ACCOUNT_CONTEXT_RE.sub(lambda m: m.group("label") + mask_account_value(m.group("value")), out)
    if mask_address_numbers:
        out = ADDRESS_NUMBER_RE.sub("**", out)
    return out


def regex_desensitize_record(
    record: dict[str, Any], *, mask_address_numbers: bool = False
) -> dict[str, Any]:
    out = normalize_record_schema(record)
    for field in TEXT_FIELDS:
        out[field] = regex_desensitize_text(
            out.get(field, ""), mask_address_numbers=mask_address_numbers
        )
    return out


def detect_regex_residue(record: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    combined = "\n".join(str(record.get(field) or "") for field in TEXT_FIELDS)
    partial_id_matches = [
        match.group()
        for match in PARTIAL_ID_RE.finditer(combined)
        if any(ch.isdigit() for ch in match.group())
    ]
    if FULL_ID_RE.search(combined) or partial_id_matches:
        issues.append("possible_id_card_residue")
    if MOBILE_RE.search(combined) or PARTIAL_MOBILE_RE.search(combined):
        issues.append("possible_mobile_residue")
    if BIRTH_DATE_RE.search(combined):
        issues.append("possible_birth_date_residue")
    return issues


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
    prompt_record = {field: normalize_text_block(record.get(field, "")) for field in TEXT_FIELDS}
    prompt_record["id"] = str(record.get("id") or "")
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


def merge_llm_fields(original: dict[str, Any], llm_obj: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out = normalize_record_schema(original)
    issues: list[str] = []
    for field in TEXT_FIELDS:
        value = llm_obj.get(field)
        if isinstance(value, str):
            out[field] = value
        else:
            issues.append(f"missing_or_bad_{field}")
    out = regex_desensitize_record(out, mask_address_numbers=MASK_ADDRESS_NUMBERS)
    issues.extend(detect_regex_residue(out))
    return out, sorted(set(issues))


async def process_one_record(
    index: int,
    record: dict[str, Any],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    llm_options: LLMOptions,
    fallback_on_error: bool,
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
    if raw_response:
        result["raw_response"] = raw_response[:4000]
    if api_error:
        result["api_error"] = api_error

    if obj is None:
        result["error"] = "LLM 调用失败" if api_error and not raw_response else "LLM 返回 JSON 解析失败"
        if fallback_on_error:
            fallback_record = regex_desensitize_record(
                normalized, mask_address_numbers=MASK_ADDRESS_NUMBERS
            )
            issues = ["llm_failed_regex_fallback", *detect_regex_residue(fallback_record)]
            result.update({"ok": True, "record": fallback_record, "issues": sorted(set(issues))})
        return result

    final_record, issues = merge_llm_fields(normalized, obj)
    result.update({"ok": True, "record": final_record, "issues": issues})
    return result


async def run_all(
    pending: list[tuple[int, dict[str, Any]]],
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    llm_options: LLMOptions,
    fallback_on_error: bool,
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
            fallback_on_error=fallback_on_error,
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
    try:
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
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


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


def build_issue_log_row(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "index": result.get("index"),
        "id": result.get("id", ""),
        "ok": bool(result.get("ok")),
        "skipped": bool(result.get("skipped")),
        "error": result.get("error"),
        "issues": list(result.get("issues") or []),
    }
    if result.get("api_error"):
        row["api_error"] = str(result["api_error"])[:1000]
    if result.get("raw_response"):
        row["raw_response"] = str(result["raw_response"])[:4000]
    return row


def update_issue_records(issue_records: dict[str, dict[str, Any]], result: dict[str, Any]) -> None:
    key = normalize_case_id(result.get("id")) or str(result.get("index"))
    if not key:
        return
    has_issue = (not result.get("ok")) or bool(result.get("issues"))
    if has_issue:
        issue_records[key] = build_issue_log_row(result)
    else:
        issue_records.pop(key, None)


def write_issue_log_records(path: Path, issue_records: dict[str, dict[str, Any]]) -> None:
    rows = [json.dumps(issue_records[key], ensure_ascii=False) for key in sorted(issue_records)]
    atomic_write_text(path, "\n".join(rows))


def load_issue_log_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        key = normalize_case_id(obj.get("id")) or str(obj.get("index"))
        if key:
            records[key] = obj
    return records


def load_existing_output(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not path.is_file():
        return {}, []
    records = load_json_list(path)
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        key = normalize_case_id(record.get("id"))
        if not key:
            continue
        if key not in order:
            order.append(key)
        by_id[key] = record
    return by_id, order


def issue_log_path(output_path: Path, explicit_path: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return output_path.with_name(output_path.stem + "_issues.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 GLM-5.1 对 dataset_reviewed.json 做敏感信息脱敏，输出 dataset_final.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(STAGE3_DIR / "dataset_reviewed.json"),
        help="输入 JSON 文件",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(STAGE4_DIR / "dataset_final.json"),
        help="输出 JSON 文件",
    )
    parser.add_argument("--issues-log", type=str, default="", help="失败/残留问题日志 JSONL")
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
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--api-timeout", type=float, default=DEFAULT_API_TIMEOUT)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="每处理 N 条增量落盘")
    parser.add_argument("--max-samples", type=int, default=0, help="最多处理记录数，0=全部")
    parser.add_argument("--resume", action="store_true", help="断点续跑，跳过输出中已有案号")
    parser.add_argument("--skip-existing", action="store_true", help="同 --resume，只跳过已有输出记录")
    parser.add_argument(
        "--fallback-on-error",
        action="store_true",
        help="LLM 失败时仍写入规则脱敏兜底结果，并在 issues log 标记",
    )
    parser.add_argument(
        "--mask-address-numbers",
        action="store_true",
        help="规则兜底时遮蔽“号/栋/单元/室/组”等前面的数字；默认关闭，避免误伤案发事实",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = Path(args.input)
    output_path = Path(args.output)
    issues_path = issue_log_path(output_path, args.issues_log)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("请设置环境变量 OPENROUTER_API_KEY")
        sys.exit(1)

    global MASK_ADDRESS_NUMBERS
    MASK_ADDRESS_NUMBERS = args.mask_address_numbers

    records = load_json_list(input_path)
    input_order = [normalize_case_id(record.get("id")) for record in records]

    use_existing = args.resume or args.skip_existing
    if use_existing:
        existing_by_id, existing_order = load_existing_output(output_path)
        records_by_id = existing_by_id
        order = [key for key in input_order if key in existing_by_id]
        for key in existing_order:
            if key not in order:
                order.append(key)
        issue_records = load_issue_log_records(issues_path) if args.resume else {}
    else:
        records_by_id, order, issue_records = {}, [], {}

    done_ids = set(records_by_id) if use_existing else set()
    pending = [(idx, record) for idx, record in enumerate(records) if normalize_case_id(record.get("id")) not in done_ids]
    pre_skipped = len(records) - len(pending)
    if args.max_samples > 0:
        pending = pending[: args.max_samples]

    writer = IncrementalWriter(
        output_path=output_path,
        issues_path=issues_path,
        records_by_id=records_by_id,
        order=order,
        issue_records=issue_records,
        save_every=max(1, args.save_every),
    )

    logger.info("输入: %s (%s 条)", input_path, len(records))
    logger.info("输出: %s", output_path)
    logger.info("issues: %s", issues_path)
    logger.info(
        "模型: %s  并发: %s  temperature: API default  thinking: %s  reasoning_tokens: %s",
        args.model,
        args.concurrency,
        "on" if not args.disable_thinking else "off",
        max(0, args.reasoning_max_tokens),
    )
    if use_existing:
        logger.info("断点/跳过已有输出已启用，预跳过: %s 条", pre_skipped)

    if not pending:
        writer.flush(force=True)
        logger.info("没有待处理记录，直接结束。")
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
    results: list[dict[str, Any]] = []
    try:
        results = asyncio.run(
            run_all(
                pending=pending,
                client=client,
                model=args.model,
                concurrency=max(1, args.concurrency),
                llm_options=llm_options,
                fallback_on_error=args.fallback_on_error,
                on_result=writer.on_result,
            )
        )
    except KeyboardInterrupt:
        interrupted = True
        writer.flush(force=True)
        logger.warning("检测到中断，已保存当前进度；可使用 --resume 继续。")

    for result in results:
        if not result.get("ok"):
            logger.error("失败 %s: %s", result.get("id"), result.get("error"))

    writer.flush(force=True)
    logger.info(
        "本批: 成功 %s, 失败 %s, 警告 %s, 预跳过 %s",
        writer.success_count,
        writer.fail_count,
        writer.warn_count,
        pre_skipped,
    )
    logger.info("当前输出记录数: %s/%s", len(writer.records_by_id), len(records))
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()

