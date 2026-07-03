#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_process/llm_extract_public_sources_test.py

从 data/public_sources/Stage1/ 下「起诉」「不起诉」子目录中的文书 .txt 原文出发，
采用“LLM 抽取 + 程序化校验/归一化 + 必要时二次修订”的混合流程，输出：
- dataset_llm_processed.json：通过 LLM 抽取并产出结构化结果的样本；
- dataset_llm_manual_review.json：抽取成功但仍有残留校验问题、需要人工审核的样本；
- dataset_llm_issues.jsonl：失败样本与仍有残留问题的样本日志。

输出 schema 与 public_sources/*/dataset.json 一致。

Usage:
    set OPENROUTER_API_KEY=sk-or-...
    python data_process/llm_extract_public_sources_test.py
    python data_process/llm_extract_public_sources_test.py --concurrency 2 --max-files 5
    python data_process/llm_extract_public_sources_test.py --skip-existing --review-passes 1
    python data_process/llm_extract_public_sources_test.py --resume
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
from typing import Callable

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_process.extract_documents import (
    PROVINCE_SHORT,
    PROVINCES_FULL,
    detect_decision_type,
    extract_articles,
    parse_document,
    remove_label_leakage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage1"
STAGE2_DIR = PROJECT_ROOT / "data" / "public_sources" / "Stage2"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-5.1"
DEFAULT_CONCURRENCY = 20
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_REASONING_MAX_TOKENS = 6144
DEFAULT_MAX_RETRIES = 4
DEFAULT_REVIEW_PASSES = 1
DEFAULT_MIN_FACT_CHARS = 60
DEFAULT_API_TIMEOUT = 240.0
DEFAULT_SAVE_EVERY = 1
DEFAULT_ATOMIC_WRITE_RETRIES = 8
DEFAULT_ATOMIC_WRITE_RETRY_SLEEP = 0.25

ALLOWED_DECISIONS = frozenset({"起诉", "相对不起诉", "法定不起诉", "存疑不起诉"})

FATAL_ISSUES = frozenset(
    {
        "missing_person",
        "missing_procedure",
        "missing_fact",
        "missing_reasoning",
        "short_fact",
        "fact_evidence_only",
        "decision_empty",
        "decision_mismatch",
    }
)

ISSUE_DESCRIPTIONS = {
    "missing_person": "person_info 为空",
    "missing_procedure": "procedure 为空",
    "missing_fact": "fact 为空",
    "missing_reasoning": "raw_reasoning_and_decision 为空",
    "short_fact": "fact 过短，案件事实不充分",
    "fact_evidence_only": "fact 只剩证据目录或证据评价，缺少核心案件事实",
    "decision_empty": "decision 为空或不合法",
    "decision_mismatch": "decision 与文书类型/说理不一致",
    "missing_date": "meta.date 缺失",
    "missing_province": "meta.province 缺失",
    "missing_articles": "relevant_articles 缺失",
    "fact_contaminated": "fact 混入说理/法条/决定性表述",
    "label_leakage_in_person_info": "person_info 含标签泄露称谓",
    "label_leakage_in_procedure": "procedure 含标签泄露称谓",
    "label_leakage_in_fact": "fact 含标签泄露称谓",
    "bad_source_url": "source_url 不是合法 http/https URL",
}

LEAKAGE_PATTERNS = (
    "被不起诉人",
    "被告人",
    "不起诉决定书",
    "决定不起诉",
    "提起公诉",
    "决定起诉",
    "相对不起诉",
    "法定不起诉",
    "存疑不起诉",
)

FACT_CONTAMINATION_RE = re.compile(
    r"(本院认为|依照《|根据《|决定对|决定不起诉|提起公诉|不符合起诉条件|证据不足|情节显著轻微)"
)

FACT_LEAD_RE = re.compile(
    r"^.*?(?:"
    r"经本院依法审查(?:查)?明"
    r"|本院依法审查查明"
    r"|(?:经|现)依法(?:审查|侦查)查明"
    r"|经本院审查(?:查明|认定(?:的事实)?)"
    r"|经我院(?:依法)?审查查明"
    r"|经审查(?:查明|认定(?:的事实)?)"
    r"|经依法审查认定的犯罪事实(?:如下)?"
    r"|依法(?:审查|侦查)查明"
    r"|(?:检察|侦查|公安)机关认定的?(?:犯罪|案件)?事实"
    r"|(?:公安局|公安分局|分局).{0,15}(?:移送(?:审查)?起诉)?认定"
    r"|审查认定事实"
    r"|移送(?:审查)?起诉(?:认定|事实)"
    r"|移送机关认定"
    r"|起诉意见书认定"
    r"|现查明"
    r"|认定.{0,20}的?犯罪行为如下"
    r")[，,：:\s]*"
)
FACT_TITLE_RE = re.compile(r"^(?:犯罪事实|指控事实|案件事实)[，,：:\s]*")

SYSTEM_PROMPT = """
你是中文刑事检察文书结构化抽取与质检助手。
你的任务是把原始检察文书清洗为高质量训练数据。

要求：
1. 忠实原文，不得编造文书中没有的事实、法条、时间、地点、人物关系或处理结论。
2. 允许先在脑中充分思考、比对、定位段落并自检，但最终只能输出一个合法 JSON 对象。
3. 不得输出思考过程、解释、markdown、前后缀说明或任何额外文字在最终回答里。
4. 无法确定时留空字符串或空数组，不要猜测。
""".strip()

ONE_SHOT_EXAMPLE = """【one-shot示例】

【示例输入目录类型】
起诉

【示例输入案号】
七新检刑诉〔2024〕74号

【示例输入全文】
起诉书（芦某某危险驾驶案）来源： 作者： 编辑： 时间：2024-12-04 字体：【 大 中 小 】
黑龙江省七台河市新兴区人民检察院
起 诉 书
七新检刑诉〔2024〕74号
被告人芦某某，男，1981年**月**日出生，公民身份号码******************，初中文化，无固定职业，现住黑龙江省七台河市新兴区**乡**村平房，户籍所在地黑龙江省七台河市新兴区。2024年2月2日，因涉嫌危险驾驶罪被黑龙江省七台河市公安局直属分局取保候审。
本案由黑龙江省七台河市公安局直属分局侦查终结，以被告人芦某某危险驾驶案，于2024年9月2日向本院移送起诉。本院受理后，于2024年9月2日已告知被告人有权委托辩护人和认罪认罚可能导致的法律后果，听取了被告人及其值班律师的意见，审查了全部案件材料。
经依法审查查明：
2024年1月10日9时30分许，被告人芦某某无证、醉酒后驾驶三轮摩托车，沿新兴区**乡行驶至**路13公里处被七台河市公安局直属公路巡逻民警大队查获。经鉴定，芦某某血液中乙醇含量为160.9mg/100ml。
经侦查，被告人芦某某于2024年2月2日经公安机关传唤后到案。
认定上述事实的证据如下：
1.书证受案登记表、立案决定书、案件侦破经过、人口、车辆、驾驶证信息查询结果、行政处罚决定书等；
2.被告人芦某某的供述与辩解；
3.鉴定意见。
本院认为，被告人芦某某醉酒后驾驶机动车在道路上行驶，其行为触犯了《中华人民共和国刑法》第一百三十三条之一，犯罪事实清楚，证据确实、充分，应当以危险驾驶罪追究其刑事责任。根据《中华人民共和国刑事诉讼法》第一百七十六条的规定，提起公诉，请依法判处。
此致
黑龙江省七台河市新兴区人民法院
检 察 官 吴迪
检察官助理 贾博
2024年9月10日
附件：1.被告人芦某某现已被取保候审；
2.案卷材料和证据一册；
3.《量刑建议书》二份。

【示例输出】
{{"id":"七新检刑诉〔2024〕74号","meta":{{"date":"2024-09-10","province":"黑龙江省"}},"person_info":"犯罪嫌疑人芦某某，男，1981年**月**日出生，公民身份号码******************，初中文化，无固定职业，现住黑龙江省七台河市新兴区**乡**村平房，户籍所在地黑龙江省七台河市新兴区。2024年2月2日，因涉嫌危险驾驶罪被黑龙江省七台河市公安局直属分局取保候审。","procedure":"本案由黑龙江省七台河市公安局直属分局侦查终结，以犯罪嫌疑人芦某某危险驾驶案，于2024年9月2日向本院移送起诉。本院受理后，于2024年9月2日已告知犯罪嫌疑人有权委托辩护人和认罪认罚可能导致的法律后果，听取了犯罪嫌疑人及其值班律师的意见，审查了全部案件材料。","fact":"2024年1月10日9时30分许，犯罪嫌疑人芦某某无证、醉酒后驾驶三轮摩托车，沿新兴区**乡行驶至**路13公里处被七台河市公安局直属公路巡逻民警大队查获。经鉴定，芦某某血液中乙醇含量为160.9mg/100ml。\n经侦查，犯罪嫌疑人芦某某于2024年2月2日经公安机关传唤后到案。\n认定上述事实的证据如下：\n1.书证受案登记表、立案决定书、案件侦破经过、人口、车辆、驾驶证信息查询结果、行政处罚决定书等；\n2.犯罪嫌疑人芦某某的供述与辩解；\n3.鉴定意见。","relevant_articles":["cl:第一百三十三条之一","cpl:第一百七十六条"],"decision":"起诉","raw_reasoning_and_decision":"本院认为，犯罪嫌疑人芦某某醉酒后驾驶机动车在道路上行驶，其行为触犯了《中华人民共和国刑法》第一百三十三条之一，犯罪事实清楚，证据确实、充分，应当以危险驾驶罪追究其刑事责任。根据《中华人民共和国刑事诉讼法》第一百七十六条的规定，提起公诉，请依法判处。","source_url":""}}

【示例要点】
1. `person_info / procedure / fact` 统一改成“犯罪嫌疑人”。
2. `fact` 必须保留具体行为事实；可以保留紧随其后的“经侦查”补充叙述和“认定上述事实的证据如下”证据目录，但不能只剩证据目录。
3. 如果文书里同时出现“公安机关认定”“起诉意见书认定”“经依法审查查明”等多版事实，优先保留最后一版、也就是本院审查确认的事实版本。
4. `fact` 不能混入“本院认为”“依照”“决定如下”“提起公诉”“决定不起诉”等说理或结论。
5. `raw_reasoning_and_decision` 只保留“本院认为”后的说理与决定主文，不含“此致”、附件、署名、日期。
6. `relevant_articles` 只保留 reasoning 中真正出现的法条。
"""

USER_TEMPLATE = ONE_SHOT_EXAMPLE + """

请处理下面的检察文书，并只输出一个合法 JSON 对象，键必须完整且与下方结构完全一致：

{{"id":"","meta":{{"date":"","province":""}},"person_info":"","procedure":"","fact":"","relevant_articles":[],"decision":"","raw_reasoning_and_decision":"","source_url":""}}

【当前任务】
1. 若实际文书是起诉书，则 `decision` 必须是 `"起诉"`。
2. 若实际文书是不起诉决定书，则必须结合说理与决定主文，将 `decision` 判为 `"相对不起诉"`、`"法定不起诉"` 或 `"存疑不起诉"`。
3. `person_info / procedure / fact` 统一使用“犯罪嫌疑人”，不得保留“被告人”“被不起诉人”。
4. `person_info` 只保留人员身份、住址、强制措施等信息。
5. `procedure` 只保留侦查、移送审查起诉、退补、延期、告知权利、讯问、听取意见等程序信息。
6. `fact` 必须保留具体行为事实；如果文书在事实段后面紧跟“经侦查……”补充叙述或“认定上述事实的证据如下：”证据目录，可以一并保留。
7. 如果文书里同时出现“公安机关认定”“起诉意见书认定”“经依法审查查明”等多版事实，优先保留最后一版，也就是本院审查确认的事实版本。
8. `fact` 不能只剩证据目录、证据评价或“上述证据收集程序合法”之类总结句，也不要混入“本院认为”“依照”“决定如下”“提起公诉”“决定不起诉”等说理或结论。
9. `raw_reasoning_and_decision` 只保留“本院认为”后的说理与决定主文，不包含标题、来源行、此致、附件、署名、日期。
10. `relevant_articles` 只保留 `raw_reasoning_and_decision` 中实际出现的法条，格式只能是 `cl:/cpl:/cpr:`。
11. `meta.date` 统一为 `YYYY-MM-DD`，`meta.province` 统一为省级行政区全称。
12. 删除网页噪声，如“来源：/作者：/编辑：/时间：/字体：”和 HTML 残片。
13. 忠实原文，不确定就留空，不要编造。

【文件名案号】
{stem_id}

【原始文书全文】
{raw_text}
"""

REVIEW_SYSTEM_PROMPT = """
你是中文刑事检察文书数据质检与修订助手。
你会看到原始文书、当前 JSON 和发现的问题清单。
请根据原始文书修正当前 JSON，只输出一个合法 JSON 对象，不要输出解释。
修正必须忠实原文，不得编造。
""".strip()

REVIEW_USER_TEMPLATE = """请修正下面的抽取结果，并只输出修正后的 JSON 对象。

【目录类型提示】
{doc_type_cn}

【文件名案号】
{stem_id}

【当前问题】
{issues_text}

【当前 JSON】
{current_json}

【规则抽取参考】
{fallback_json}

【修订要求】
1. 必须保留原 schema，不得新增或删除键。
2. 优先修正：空字段、decision 错误、fact 与 reasoning 混写、标签泄露称谓、法条缺失或不一致。
3. person_info / procedure / fact 必须使用中性称谓“犯罪嫌疑人”。
4. `fact` 必须保留具体行为事实；若有“经侦查”补充叙述和“认定上述事实的证据如下”证据目录，可以保留，但不能只剩证据目录或证据评价。
5. 若事实部分有多版表述，优先保留最后一版、即本院审查确认的事实版本；不要混入“本院认为”“依照”“决定如下”“提起公诉”“决定不起诉”等说理结论。
6. raw_reasoning_and_decision 只保留说理和决定主文，不包含此致、附件、署名、日期、来源/作者/时间/字体等网页噪声。
7. relevant_articles 只保留 reasoning 中真正出现的法条。
8. 忠实原文，无法确定时留空。

【原始文书全文】
{raw_text}
"""


@dataclass(frozen=True)
class LLMOptions:
    max_output_tokens: int
    reasoning_max_tokens: int
    max_retries: int
    review_passes: int
    min_fact_chars: int
    enable_thinking: bool


@dataclass
class IncrementalWriter:
    out_path: Path
    manual_review_path: Path
    issues_path: Path
    by_processed: dict[str, dict]
    by_manual_review: dict[str, dict]
    issue_records: dict[str, dict]
    save_every: int = DEFAULT_SAVE_EVERY
    dirty_count: int = 0
    handled_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    warn_count: int = 0
    skipped_count: int = 0

    def on_result(self, result: dict) -> None:
        self.handled_count += 1
        if result.get("skipped"):
            self.skipped_count += 1
            return

        if result.get("ok"):
            self.success_count += 1
            if result.get("issues"):
                self.warn_count += 1
        else:
            self.fail_count += 1

        apply_result_to_buckets(self.by_processed, self.by_manual_review, result)
        update_issue_records(self.issue_records, result)
        self.dirty_count += 1

        if (
            self.save_every <= 1
            or self.dirty_count >= self.save_every
            or not result.get("ok")
        ):
            self.flush()

    def flush(self, force: bool = False) -> None:
        if self.dirty_count <= 0 and not force:
            return

        list_processed = sorted(
            self.by_processed.values(),
            key=lambda row: normalize_case_num(str(row.get("id") or "")),
        )
        list_manual_review = sorted(
            self.by_manual_review.values(),
            key=lambda row: normalize_case_num(str(row.get("id") or "")),
        )
        try:
            write_sorted_json(self.out_path, list_processed)
            write_sorted_json(self.manual_review_path, list_manual_review)
            write_issue_log_records(self.issues_path, self.issue_records)
            self.dirty_count = 0
        except PermissionError as exc:
            logger.warning(
                "增量落盘失败，文件可能被编辑器、OneDrive 或其他进程占用；"
                "将保留当前进度并在后续继续重试: %s",
                exc,
            )

    @property
    def touched_count(self) -> int:
        return self.success_count + self.fail_count


def normalize_case_num(s: str) -> str:
    return s.replace("〔", "[").replace("〕", "]")


def id_for_output(stem: str) -> str:
    return stem.replace("[", "〔").replace("]", "〕")


def doc_type_from_text_or_path(txt_path: Path, clean_text: str) -> str:
    header = "\n".join(clean_text.splitlines()[:6])
    if "不起诉决定书" in header:
        return "non_prosecution"
    if "起诉书" in header and "不起诉" not in header:
        return "prosecution"
    return "prosecution" if txt_path.parent.name == "起诉" else "non_prosecution"


def doc_type_cn(doc_type: str) -> str:
    return "起诉" if doc_type == "prosecution" else "不起诉"


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


def parse_json_object(text: str) -> dict | None:
    text = _strip_think_block(text).replace("\ufeff", "").strip()
    try:
        obj = json.loads(text)
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


def normalize_text_block(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\xa0", " ").replace("\u3000", " ")
    text = re.sub(r"[\u2000-\u200f\u2028\u2029\u202f\u205f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_metadata_lines(raw_text: str) -> str:
    lines = normalize_text_block(raw_text).split("\n")
    body_lines: list[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i <= 2 and re.search(r"(来源：|作者：|编辑：|时间：|字体：)", s):
            continue
        body_lines.append(line)
    return "\n".join(body_lines).strip()


def preclean_raw_text(raw_text: str) -> str:
    text = strip_metadata_lines(raw_text)
    lines: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            lines.append("")
            continue
        s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
        s = s.replace(" ", "").replace("　", " ")
        s = re.sub(r"[ \t]+", " ", s)
        lines.append(s)
    return normalize_text_block("\n".join(lines))


def compact_len(text: str) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


def normalize_meta_date(value: str) -> str:
    s = normalize_text_block(value)
    if not s:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    m = re.search(r"(\d{4})[年/\-.](\d{1,2})[月/\-.](\d{1,2})日?", s)
    if m:
        y, mm, dd = m.groups()
        return f"{y}-{int(mm):02d}-{int(dd):02d}"
    return ""


def extract_last_date(text: str) -> str:
    matches = re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not matches:
        return ""
    y, mm, dd = matches[-1]
    return f"{y}-{int(mm):02d}-{int(dd):02d}"


def normalize_province(value: str, fallback: str, clean_text: str) -> str:
    s = normalize_text_block(value)
    if s in PROVINCES_FULL:
        return s
    if s in PROVINCE_SHORT:
        return PROVINCE_SHORT[s]
    for short, full in PROVINCE_SHORT.items():
        if short and short in s:
            return full
        if full and full in s:
            return full

    fb = normalize_text_block(fallback)
    if fb in PROVINCES_FULL:
        return fb

    header = "\n".join(clean_text.splitlines()[:8])
    for full in PROVINCES_FULL:
        if full in header:
            return full
    for short, full in PROVINCE_SHORT.items():
        if short in header:
            return full
    return ""


def normalize_source_url(url: str) -> str:
    s = normalize_text_block(url)
    s = s.rstrip("，。；,;）)]】》")
    return s if re.match(r"^https?://", s) else ""


def extract_source_url(text: str) -> str:
    m = re.search(r"https?://[^\s<>\"]+", text)
    return normalize_source_url(m.group(0)) if m else ""


def normalize_articles_list(articles: object) -> list[str]:
    if isinstance(articles, str):
        items = re.split(r"[、，,\n;；]+", articles)
    elif isinstance(articles, list):
        items = [str(x) for x in articles]
    else:
        items = []

    out: list[str] = []
    for item in items:
        s = normalize_text_block(item)
        if not s:
            continue
        if s.startswith(("cl:", "cpl:", "cpr:")):
            out.append(s)
            continue
        if "刑事诉讼规则" in s:
            s = re.sub(r"^.*?刑事诉讼规则[》]?", "", s).lstrip("：:")
            out.append(f"cpr:{s}")
        elif "刑事诉讼法" in s:
            s = re.sub(r"^.*?刑事诉讼法[》]?", "", s).lstrip("：:")
            out.append(f"cpl:{s}")
        elif "刑法" in s:
            s = re.sub(r"^.*?刑法[》]?", "", s).lstrip("：:")
            out.append(f"cl:{s}")
    return dedupe_preserve_order(out)


def extract_articles_from_reasoning(reasoning: str) -> list[str]:
    if not reasoning:
        return []
    cl, cpl, cpr = extract_articles(reasoning)
    return (
        [f"cl:{a}" for a in cl]
        + [f"cpl:{a}" for a in cpl]
        + [f"cpr:{a}" for a in cpr]
    )


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def contains_label_leakage(text: str) -> bool:
    s = str(text or "")
    return any(marker in s for marker in LEAKAGE_PATTERNS)


def fact_is_contaminated(text: str) -> bool:
    return bool(FACT_CONTAMINATION_RE.search(str(text or "")))


def fact_is_evidence_only(text: str) -> bool:
    s = normalize_text_block(text)
    if not s:
        return False
    if s.startswith("。"):
        s = s.lstrip("。；;，, ")
    if s.startswith(("认定上述事实的证据如下", "上述证据收集程序合法")):
        return True
    marker = "认定上述事实的证据如下"
    if marker in s:
        prefix = normalize_text_block(s.split(marker, 1)[0]).lstrip("。；;，, ")
        if compact_len(prefix) < 20:
            return True
    return False


def clean_fact_block(text: str) -> str:
    text = normalize_text_block(text)
    if FACT_TITLE_RE.match(text):
        text = FACT_TITLE_RE.sub("", text, count=1)
    text = FACT_LEAD_RE.sub("", text, count=1)
    return normalize_text_block(text)


def clean_reasoning_block(text: str) -> str:
    text = normalize_text_block(text)
    if not text:
        return ""

    if "本院认为" in text:
        text = text[text.find("本院认为") :]
    else:
        m = re.search(r"(经(?:本院|依法)?审查并(?:经)?(?:[一二三四五六七八九两\d]+次)?(?:退回)?补充(?:侦查|调查).*)", text)
        if m:
            text = m.group(1)

    cut_points = [
        text.find(marker)
        for marker in (
            "\n此致",
            "\n附件：",
            "\n附：",
            "\n检察官",
            "\n检 察 官",
            "\n检  察  官",
            "\n书记员",
            "\n被不起诉人如",
            "\n被害人如",
        )
        if text.find(marker) != -1
    ]
    if cut_points:
        text = text[: min(cut_points)]

    text = re.sub(r"\n[\u4e00-\u9fa5]{2,30}人民检察院\s*$", "", text).strip()
    text = re.sub(r"\n[（(]院印[）)]\s*$", "", text).strip()
    text = re.sub(r"\n\d{4}年\d{1,2}月\d{1,2}日\s*$", "", text).strip()
    return normalize_text_block(text)


def prefer_text(primary: str, fallback: str, *, min_chars: int) -> str:
    primary = normalize_text_block(primary)
    fallback = normalize_text_block(fallback)
    if not fallback:
        return primary
    if not primary:
        return fallback
    if compact_len(primary) < min_chars and compact_len(fallback) > compact_len(primary):
        return fallback
    return primary


def build_empty_record(stem: str, merged_source_url: str) -> dict:
    return {
        "id": id_for_output(stem),
        "meta": {"date": "", "province": ""},
        "person_info": "",
        "procedure": "",
        "fact": "",
        "relevant_articles": [],
        "decision": "",
        "raw_reasoning_and_decision": "",
        "source_url": normalize_source_url(merged_source_url),
    }


def coerce_record(obj: dict | None, stem: str, merged_source_url: str) -> dict:
    out = build_empty_record(stem, merged_source_url)
    if not isinstance(obj, dict):
        return out

    meta_in = obj.get("meta")
    if isinstance(meta_in, dict):
        out["meta"]["date"] = normalize_text_block(str(meta_in.get("date") or ""))
        out["meta"]["province"] = normalize_text_block(
            str(meta_in.get("province") or "")
        )

    out["person_info"] = normalize_text_block(obj.get("person_info") or "")
    out["procedure"] = normalize_text_block(obj.get("procedure") or "")
    out["fact"] = normalize_text_block(obj.get("fact") or "")
    out["relevant_articles"] = normalize_articles_list(obj.get("relevant_articles"))
    decision = normalize_text_block(obj.get("decision") or "")
    out["decision"] = decision if decision in ALLOWED_DECISIONS else ""
    out["raw_reasoning_and_decision"] = normalize_text_block(
        obj.get("raw_reasoning_and_decision") or ""
    )
    out["source_url"] = (
        normalize_source_url(str(obj.get("source_url") or ""))
        or out["source_url"]
    )
    return out


def infer_expected_decision(
    doc_type: str,
    reasoning: str,
    fallback_decision: str,
    llm_decision: str,
) -> str:
    if doc_type == "prosecution":
        return "起诉"
    detected = detect_decision_type(reasoning) if reasoning else None
    if detected in ALLOWED_DECISIONS:
        return detected
    if fallback_decision in ALLOWED_DECISIONS and fallback_decision != "起诉":
        return fallback_decision
    if llm_decision in ALLOWED_DECISIONS and llm_decision != "起诉":
        return llm_decision
    return ""


def normalize_record(
    obj: dict | None,
    stem: str,
    merged_source_url: str,
    clean_text: str,
    fallback: dict | None,
    doc_type: str,
    min_fact_chars: int,
) -> dict:
    rec = coerce_record(obj, stem, merged_source_url)
    fallback = fallback if isinstance(fallback, dict) else {}
    fallback_meta = fallback.get("meta") if isinstance(fallback.get("meta"), dict) else {}

    rec["person_info"] = prefer_text(
        remove_label_leakage(rec["person_info"]),
        str(fallback.get("person_info") or ""),
        min_chars=20,
    )
    rec["procedure"] = prefer_text(
        remove_label_leakage(rec["procedure"]),
        str(fallback.get("procedure") or ""),
        min_chars=20,
    )
    rec["fact"] = clean_fact_block(
        prefer_text(
            remove_label_leakage(rec["fact"]),
            str(fallback.get("fact") or ""),
            min_chars=min_fact_chars,
        )
    )
    if fact_is_evidence_only(rec["fact"]):
        fallback_fact = clean_fact_block(str(fallback.get("fact") or ""))
        if fallback_fact and not fact_is_evidence_only(fallback_fact):
            rec["fact"] = fallback_fact
    rec["raw_reasoning_and_decision"] = clean_reasoning_block(
        prefer_text(
            rec["raw_reasoning_and_decision"],
            str(fallback.get("raw_reasoning_and_decision") or ""),
            min_chars=30,
        )
    )

    if fact_is_contaminated(rec["fact"]):
        fallback_fact = clean_fact_block(str(fallback.get("fact") or ""))
        if fallback_fact and not fact_is_contaminated(fallback_fact):
            rec["fact"] = fallback_fact

    rec["meta"]["date"] = (
        normalize_meta_date(rec["meta"]["date"])
        or normalize_meta_date(str(fallback_meta.get("date") or ""))
        or extract_last_date(clean_text)
    )
    rec["meta"]["province"] = normalize_province(
        rec["meta"]["province"],
        str(fallback_meta.get("province") or ""),
        clean_text,
    )

    reasoning = rec["raw_reasoning_and_decision"]
    rule_articles = extract_articles_from_reasoning(reasoning)
    fallback_articles = normalize_articles_list(fallback.get("relevant_articles"))
    llm_articles = normalize_articles_list(rec["relevant_articles"])
    rec["relevant_articles"] = dedupe_preserve_order(
        rule_articles or llm_articles or fallback_articles
    )

    rec["decision"] = infer_expected_decision(
        doc_type,
        reasoning,
        str(fallback.get("decision") or ""),
        rec["decision"],
    )

    rec["source_url"] = (
        normalize_source_url(rec["source_url"])
        or normalize_source_url(merged_source_url)
        or extract_source_url(clean_text)
    )

    if rec["id"] != id_for_output(stem):
        rec["id"] = id_for_output(stem)
    return rec


def detect_issues(
    rec: dict,
    doc_type: str,
    min_fact_chars: int,
) -> list[str]:
    issues: list[str] = []

    if not rec.get("person_info"):
        issues.append("missing_person")
    if not rec.get("procedure"):
        issues.append("missing_procedure")
    if not rec.get("fact"):
        issues.append("missing_fact")
    if not rec.get("raw_reasoning_and_decision"):
        issues.append("missing_reasoning")
    if rec.get("fact") and compact_len(rec["fact"]) < min_fact_chars:
        issues.append("short_fact")
    if rec.get("fact") and fact_is_evidence_only(rec["fact"]):
        issues.append("fact_evidence_only")
    if not rec.get("decision"):
        issues.append("decision_empty")

    expected = infer_expected_decision(
        doc_type,
        str(rec.get("raw_reasoning_and_decision") or ""),
        str(rec.get("decision") or ""),
        str(rec.get("decision") or ""),
    )
    if expected and rec.get("decision") != expected:
        issues.append("decision_mismatch")

    if not rec.get("meta", {}).get("date"):
        issues.append("missing_date")
    if not rec.get("meta", {}).get("province"):
        issues.append("missing_province")
    if (
        rec.get("raw_reasoning_and_decision")
        and any(
            kw in rec["raw_reasoning_and_decision"]
            for kw in ("刑法", "刑事诉讼法", "刑事诉讼规则")
        )
        and not rec.get("relevant_articles")
    ):
        issues.append("missing_articles")
    if rec.get("fact") and fact_is_contaminated(rec["fact"]):
        issues.append("fact_contaminated")
    if rec.get("person_info") and contains_label_leakage(rec["person_info"]):
        issues.append("label_leakage_in_person_info")
    if rec.get("procedure") and contains_label_leakage(rec["procedure"]):
        issues.append("label_leakage_in_procedure")
    if rec.get("fact") and contains_label_leakage(rec["fact"]):
        issues.append("label_leakage_in_fact")
    if rec.get("source_url") and not normalize_source_url(rec["source_url"]):
        issues.append("bad_source_url")
    return dedupe_preserve_order(issues)


def issue_score(issues: list[str]) -> int:
    return sum(10 if issue in FATAL_ISSUES else 1 for issue in issues)


def record_richness(rec: dict) -> int:
    return (
        compact_len(rec.get("person_info", ""))
        + compact_len(rec.get("procedure", ""))
        + compact_len(rec.get("fact", ""))
        + compact_len(rec.get("raw_reasoning_and_decision", ""))
        + 12 * len(rec.get("relevant_articles", []))
        + (20 if rec.get("meta", {}).get("date") else 0)
        + (20 if rec.get("meta", {}).get("province") else 0)
        + (10 if rec.get("decision") else 0)
    )


def choose_better_candidate(
    current_rec: dict,
    current_issues: list[str],
    candidate_rec: dict,
    candidate_issues: list[str],
) -> tuple[dict, list[str]]:
    cur_key = (issue_score(current_issues), -record_richness(current_rec))
    cand_key = (issue_score(candidate_issues), -record_richness(candidate_rec))
    if cand_key < cur_key:
        return candidate_rec, candidate_issues
    return current_rec, current_issues


def list_txt_files(test_dir: Path) -> list[Path]:
    files: list[Path] = []
    for sub in ("起诉", "不起诉"):
        d = test_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.txt")):
            if p.is_file():
                files.append(p)
    return files


def manual_review_output_path(processed_path: Path) -> Path:
    return processed_path.with_name("dataset_llm_manual_review.json")


def issue_log_path(processed_path: Path, override: str) -> Path:
    if override:
        return Path(override)
    return processed_path.with_name("dataset_llm_issues.jsonl")


def load_existing_split_outputs(
    processed_path: Path,
    manual_review_path: Path,
) -> tuple[dict[str, dict], dict[str, dict]]:
    by_processed: dict[str, dict] = {}
    by_manual_review: dict[str, dict] = {}

    for path, bucket in (
        (processed_path, by_processed),
        (manual_review_path, by_manual_review),
    ):
        if not path.is_file():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("id"):
                bucket[normalize_case_num(str(row["id"]).strip())] = row

    for key, row in by_manual_review.items():
        by_processed.setdefault(key, row)
    return by_processed, by_manual_review


def apply_result_to_buckets(
    by_processed: dict[str, dict],
    by_manual_review: dict[str, dict],
    result: dict,
) -> None:
    if not result.get("ok") or result.get("skipped"):
        return
    rec = result.get("record")
    if not isinstance(rec, dict) or not rec.get("id"):
        return
    key = normalize_case_num(str(rec["id"]).strip())
    by_processed[key] = rec
    if result.get("issues"):
        by_manual_review[key] = rec
    else:
        by_manual_review.pop(key, None)


def apply_results_to_buckets(
    by_processed: dict[str, dict],
    by_manual_review: dict[str, dict],
    results: list[dict],
) -> None:
    for res in results:
        apply_result_to_buckets(by_processed, by_manual_review, res)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    )
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


def write_sorted_json(path: Path, records: list[dict]) -> None:
    atomic_write_text(
        path,
        json.dumps(records, ensure_ascii=False, indent=2),
    )


def build_issue_log_row(result: dict) -> dict:
    row = {
        "id": result.get("id", ""),
        "path": result.get("path", ""),
        "ok": bool(result.get("ok")),
        "skipped": bool(result.get("skipped")),
        "error": result.get("error"),
        "issues": list(result.get("issues") or []),
    }
    if result.get("api_error"):
        row["api_error"] = str(result["api_error"])[:1000]
    if result.get("raw_response"):
        row["raw_response"] = str(result["raw_response"])[:4000]
    if result.get("record"):
        row["record"] = result["record"]
    return row


def load_issue_log_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.is_file():
        return records

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("读取 issues log 失败，将从空状态开始: %s", exc)
        return records

    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or "").strip()
        if not row_id and isinstance(row.get("record"), dict):
            row_id = str(row["record"].get("id") or "").strip()
        if not row_id:
            continue
        records[normalize_case_num(row_id)] = row
    return records


def update_issue_records(issue_records: dict[str, dict], result: dict) -> None:
    result_id = str(result.get("id") or "").strip()
    if not result_id and isinstance(result.get("record"), dict):
        result_id = str(result["record"].get("id") or "").strip()
    if not result_id:
        return

    key = normalize_case_num(result_id)
    has_issue = (not result.get("ok")) or bool(result.get("issues"))
    if has_issue:
        issue_records[key] = build_issue_log_row(result)
    else:
        issue_records.pop(key, None)


def write_issue_log_records(path: Path, issue_records: dict[str, dict]) -> None:
    rows = [
        json.dumps(issue_records[key], ensure_ascii=False)
        for key in sorted(issue_records)
    ]
    atomic_write_text(path, "\n".join(rows))


def write_issue_log(path: Path, results: list[dict]) -> None:
    issue_records: dict[str, dict] = {}
    for result in results:
        update_issue_records(issue_records, result)
    write_issue_log_records(path, issue_records)


def load_source_url_index(dataset_path: Path) -> dict[str, str]:
    if not dataset_path.is_file():
        return {}
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 dataset.json 失败，将不合并 source_url: %s", exc)
        return {}
    if not isinstance(raw, list):
        return {}

    out: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        rid = normalize_case_num(str(item.get("id") or "").strip())
        url = normalize_source_url(str(item.get("source_url") or ""))
        if rid and url:
            out[rid] = url
    return out


def filter_paths_by_ids(paths: list[Path], excluded_ids: set[str] | None) -> tuple[list[Path], int]:
    if not excluded_ids:
        return paths, 0
    pending = [path for path in paths if normalize_case_num(path.stem) not in excluded_ids]
    return pending, len(paths) - len(pending)


def build_extract_messages(stem_id: str, doc_type: str, raw_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                stem_id=stem_id,
                doc_type_cn=doc_type_cn(doc_type),
                raw_text=raw_text,
            ),
        },
    ]


def render_issues(issues: list[str]) -> str:
    if not issues:
        return "- 无"
    return "\n".join(
        f"- {ISSUE_DESCRIPTIONS.get(issue, issue)}" for issue in issues
    )


def build_review_messages(
    stem_id: str,
    doc_type: str,
    raw_text: str,
    current_record: dict,
    fallback_record: dict | None,
    issues: list[str],
) -> list[dict]:
    fallback_json = json.dumps(
        fallback_record or build_empty_record(stem_id, ""),
        ensure_ascii=False,
        indent=2,
    )
    current_json = json.dumps(current_record, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REVIEW_USER_TEMPLATE.format(
                stem_id=stem_id,
                doc_type_cn=doc_type_cn(doc_type),
                raw_text=raw_text,
                current_json=current_json,
                fallback_json=fallback_json,
                issues_text=render_issues(issues),
            ),
        },
    ]


async def call_llm_json(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
    llm_options: LLMOptions,
    tag: str,
) -> tuple[dict | None, str, str]:
    last_raw = ""
    last_error = ""
    for attempt in range(llm_options.max_retries):
        async with semaphore:
            try:
                kwargs = {
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
                logger.warning(
                    "[%s] 第%s次 JSON 解析失败: %s...",
                    tag,
                    attempt + 1,
                    last_raw[:180],
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "[%s] API 错误 (第%s次): %s",
                    tag,
                    attempt + 1,
                    exc,
                )
        if attempt < llm_options.max_retries - 1:
            await asyncio.sleep(2**attempt)
    return None, last_raw, last_error


async def process_one_file(
    txt_path: Path,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    url_index: dict[str, str],
    skip_existing_ids: set[str] | None,
    llm_options: LLMOptions,
) -> dict:
    stem = txt_path.stem
    norm = normalize_case_num(stem)
    result: dict = {
        "path": str(txt_path),
        "id": id_for_output(stem),
        "ok": False,
        "error": None,
        "issues": [],
    }

    if skip_existing_ids is not None and norm in skip_existing_ids:
        result["ok"] = True
        result["skipped"] = True
        return result

    try:
        raw = txt_path.read_text(encoding="utf-8")
    except OSError as exc:
        result["error"] = f"读取失败: {exc}"
        return result

    clean_text = preclean_raw_text(raw)
    if not clean_text:
        result["error"] = "正文为空"
        return result

    doc_type = doc_type_from_text_or_path(txt_path, clean_text)
    merged_url = url_index.get(norm, "")
    fallback = parse_document(str(txt_path), doc_type)

    obj, raw_resp, api_error = await call_llm_json(
        client=client,
        messages=build_extract_messages(id_for_output(stem), doc_type, clean_text),
        model=model,
        semaphore=semaphore,
        llm_options=llm_options,
        tag=f"{stem}/extract",
    )
    result["raw_response"] = raw_resp
    if api_error:
        result["api_error"] = api_error
    if obj is None:
        result["error"] = "LLM 调用失败" if api_error and not raw_resp else "LLM 返回解析失败"
        return result

    best_record = normalize_record(
        obj=obj,
        stem=stem,
        merged_source_url=merged_url,
        clean_text=clean_text,
        fallback=fallback,
        doc_type=doc_type,
        min_fact_chars=llm_options.min_fact_chars,
    )
    best_issues = detect_issues(best_record, doc_type, llm_options.min_fact_chars)

    for review_idx in range(llm_options.review_passes):
        if not best_issues:
            break
        review_obj, review_raw, review_api_error = await call_llm_json(
            client=client,
            messages=build_review_messages(
                stem_id=id_for_output(stem),
                doc_type=doc_type,
                raw_text=clean_text,
                current_record=best_record,
                fallback_record=fallback,
                issues=best_issues,
            ),
            model=model,
            semaphore=semaphore,
            llm_options=llm_options,
            tag=f"{stem}/review{review_idx + 1}",
        )
        if review_obj is None:
            if review_api_error:
                result["api_error"] = review_api_error
            continue
        candidate = normalize_record(
            obj=review_obj,
            stem=stem,
            merged_source_url=merged_url,
            clean_text=clean_text,
            fallback=fallback,
            doc_type=doc_type,
            min_fact_chars=llm_options.min_fact_chars,
        )
        candidate_issues = detect_issues(
            candidate, doc_type, llm_options.min_fact_chars
        )
        best_record, best_issues = choose_better_candidate(
            best_record,
            best_issues,
            candidate,
            candidate_issues,
        )
        if review_raw:
            result["raw_response"] = review_raw

    fatal = [issue for issue in best_issues if issue in FATAL_ISSUES]
    if fatal:
        result["error"] = "仍存在关键质量问题: " + ", ".join(fatal)
        result["issues"] = best_issues
        result["record"] = best_record
        return result

    result["record"] = best_record
    result["issues"] = best_issues
    result["ok"] = True
    return result


async def run_all(
    paths: list[Path],
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    url_index: dict[str, str],
    skip_existing_ids: set[str] | None,
    llm_options: LLMOptions,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    start = time.time()

    async def _one(path: Path) -> dict:
        return await process_one_file(
            txt_path=path,
            client=client,
            model=model,
            semaphore=semaphore,
            url_index=url_index,
            skip_existing_ids=skip_existing_ids,
            llm_options=llm_options,
        )

    tasks = [asyncio.create_task(_one(path)) for path in paths]
    results: list[dict] = []
    completed = 0

    try:
        for finished in asyncio.as_completed(tasks):
            result = await finished
            results.append(result)
            completed += 1
            if on_result is not None:
                on_result(result)
            if completed % max(1, min(20, len(paths))) == 0 or completed == len(paths):
                logger.info("运行进度: %s/%s", completed, len(paths))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("全部完成: %s 个文件, 用时 %.1fs", completed, time.time() - start)
    return list(results)


def summarize_issue_counts(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for res in results:
        for issue in res.get("issues") or []:
            counts[issue] = counts.get(issue, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage1 文书 .txt -> 高质量 LLM 抽取 -> dataset_llm_processed.json + dataset_llm_manual_review.json"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(STAGE1_DIR),
        help="Stage1 input directory containing 起诉/ and 不起诉/ txt files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Processed output path; default is data/public_sources/Stage2/dataset_llm_processed.json",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default="",
        help="Optional rules dataset; default is data/public_sources/Stage1/dataset.json",
    )
    parser.add_argument(
        "--issues-log",
        type=str,
        default="",
        help="Issue log JSONL; default is data/public_sources/Stage2/dataset_llm_issues.jsonl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：跳过已成功写入输出 JSON 的案号，并复用已有 issues log",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter 模型 id")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"并发数，默认 {DEFAULT_CONCURRENCY}",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"最大输出 token，默认 {DEFAULT_MAX_OUTPUT_TOKENS}",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=DEFAULT_REASONING_MAX_TOKENS,
        help=f"隐藏推理 token 预算，默认 {DEFAULT_REASONING_MAX_TOKENS}",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="关闭 reasoning token 预算（默认开启）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"单次调用最大重试次数，默认 {DEFAULT_MAX_RETRIES}",
    )
    parser.add_argument(
        "--review-passes",
        type=int,
        default=DEFAULT_REVIEW_PASSES,
        help=f"发现问题后的二次修订轮数，默认 {DEFAULT_REVIEW_PASSES}",
    )
    parser.add_argument(
        "--min-fact-chars",
        type=int,
        default=DEFAULT_MIN_FACT_CHARS,
        help=f"fact 最小有效字符数阈值，默认 {DEFAULT_MIN_FACT_CHARS}",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=DEFAULT_API_TIMEOUT,
        help=f"API 超时秒数，默认 {DEFAULT_API_TIMEOUT}",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="最多处理文件数，0=全部",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help=f"每处理 N 个结果就增量落盘一次，默认 {DEFAULT_SAVE_EVERY}",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="dataset_llm_processed.json 与 dataset_llm_manual_review.json 中已有同案号则跳过",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_dir = Path(args.input_dir)
    out_path = (
        Path(args.output) if args.output else STAGE2_DIR / "dataset_llm_processed.json"
    )
    manual_review_path = manual_review_output_path(out_path)
    ds_path = Path(args.dataset_json) if args.dataset_json else STAGE1_DIR / "dataset.json"
    issues_path = issue_log_path(out_path, args.issues_log)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("请设置环境变量 OPENROUTER_API_KEY")
        sys.exit(1)

    paths = list_txt_files(test_dir)
    if not paths:
        logger.error("未找到任何 .txt: %s/起诉 或 %s/不起诉", test_dir, test_dir)
        sys.exit(1)

    url_index = load_source_url_index(ds_path)
    use_existing_outputs = args.skip_existing or args.resume
    skip_ids: set[str] | None = None
    if use_existing_outputs:
        by_processed, by_manual_review = load_existing_split_outputs(
            out_path, manual_review_path
        )
        skip_ids = set(by_processed) | set(by_manual_review)
    else:
        by_processed, by_manual_review = {}, {}
    issue_records = load_issue_log_records(issues_path) if args.resume else {}
    pending_paths, pre_skipped = filter_paths_by_ids(paths, skip_ids)
    if args.max_files > 0:
        pending_paths = pending_paths[: args.max_files]
    writer = IncrementalWriter(
        out_path=out_path,
        manual_review_path=manual_review_path,
        issues_path=issues_path,
        by_processed=by_processed,
        by_manual_review=by_manual_review,
        issue_records=issue_records,
        save_every=max(1, args.save_every),
    )

    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=args.api_timeout,
        max_retries=0,
    )
    llm_options = LLMOptions(
        max_output_tokens=args.max_output_tokens,
        reasoning_max_tokens=max(0, args.reasoning_max_tokens),
        max_retries=max(1, args.max_retries),
        review_passes=max(0, args.review_passes),
        min_fact_chars=max(1, args.min_fact_chars),
        enable_thinking=not args.disable_thinking,
    )

    logger.info(
        "输入目录: %s  文件数: %s  待处理: %s",
        test_dir,
        len(paths),
        len(pending_paths),
    )
    logger.info(
        "输出: %s 与 %s  issues: %s",
        out_path,
        manual_review_path,
        issues_path,
    )
    logger.info(
        "模型: %s  并发: %s  temperature: API default  thinking: %s  reasoning_tokens: %s  review_passes: %s",
        args.model,
        args.concurrency,
        "on" if llm_options.enable_thinking else "off",
        llm_options.reasoning_max_tokens,
        llm_options.review_passes,
    )
    if args.resume:
        logger.info(
            "resume: 已启用，将预跳过已成功写入的 %s 个案号",
            pre_skipped,
        )
    elif args.skip_existing:
        logger.info("skip-existing: 已启用，将预跳过 %s 个案号", pre_skipped)

    if not pending_paths:
        writer.flush(force=True)
        logger.info("没有待处理文件，直接结束。")
        sys.exit(0)

    interrupted = False
    try:
        results = asyncio.run(
            run_all(
                paths=pending_paths,
                client=client,
                model=args.model,
                concurrency=args.concurrency,
                url_index=url_index,
                skip_existing_ids=None,
                llm_options=llm_options,
                on_result=writer.on_result,
            )
        )
    except KeyboardInterrupt:
        interrupted = True
        results = []
        writer.flush(force=True)
        logger.warning("检测到中断，已增量保存当前进度。可使用 --resume 继续运行。")

    for res in results:
        if not res.get("ok"):
            logger.error("失败 %s: %s", res.get("path"), res.get("error"))

    writer.flush(force=True)
    if writer.dirty_count > 0:
        logger.warning(
            "程序结束时仍有 %s 批未成功落盘；请关闭占用输出文件的程序后重新运行 --resume。",
            writer.dirty_count,
        )
    list_processed = sorted(
        by_processed.values(), key=lambda row: normalize_case_num(str(row.get("id") or ""))
    )
    list_manual_review = sorted(
        by_manual_review.values(), key=lambda row: normalize_case_num(str(row.get("id") or ""))
    )

    logger.info(
        "本批: 成功写入/更新 %s, 预跳过 %s, 失败 %s, 需关注 %s",
        writer.success_count,
        pre_skipped,
        writer.fail_count,
        writer.warn_count,
    )
    logger.info(
        "输出: LLM 成功 %s 条 -> %s；需人工审核 %s 条 -> %s",
        len(list_processed),
        out_path,
        len(list_manual_review),
        manual_review_path,
    )

    issue_counts = summarize_issue_counts(results)
    if issue_counts:
        logger.info("问题统计: %s", issue_counts)

    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()


