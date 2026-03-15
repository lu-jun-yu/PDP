#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_documents.py

从北大法宝不起诉决定书 / 起诉书原文 (.txt) 中，基于规则提取结构化字段，输出 JSON。
用于构建 PDP (Prosecution Decision Prediction) 数据集。

输出 schema:
{
    "id": "FBM-CLI.P.XXXXXXXX",
    "meta": { "date": "2023-07-07", "province": "江西省" },
    "person_info": "某人XXX...",
    "procedure": "本案由XXX...",
    "fact": "...",
    "relevant_articles": ["cl:第XXX条", "cpl:第XXX条第X款", "cpr:第XXX条"],
    "decision": "起诉" | "相对不起诉" | "法定不起诉" | "存疑不起诉",
    "raw_reasoning_and_decision": "本院认为，..."
}

Usage:
    python extract_documents.py
    python extract_documents.py --input-dir data/pku_fabao --output-dir data/PDP_dataset
    python extract_documents.py --limit 10   # 每个 split 只处理 10 条（调试用）
    python extract_documents.py --split-date 2022-03-01  # 按日期划分 train/test
"""

import os
import re
import json
import shutil
import logging
import argparse
from collections import Counter, defaultdict

import cpca

# ============================================================================
#  配置 & 常量
# ============================================================================

PROVINCE_SHORT = {
    "北京": "北京市", "天津": "天津市", "上海": "上海市", "重庆": "重庆市",
    "河北": "河北省", "山西": "山西省", "辽宁": "辽宁省", "吉林": "吉林省",
    "黑龙江": "黑龙江省",
    "江苏": "江苏省", "浙江": "浙江省", "安徽": "安徽省", "福建": "福建省",
    "江西": "江西省", "山东": "山东省", "河南": "河南省", "湖北": "湖北省",
    "湖南": "湖南省", "广东": "广东省", "海南": "海南省",
    "四川": "四川省", "贵州": "贵州省", "云南": "云南省", "陕西": "陕西省",
    "甘肃": "甘肃省", "青海": "青海省",
    "广西": "广西壮族自治区", "内蒙古": "内蒙古自治区", "西藏": "西藏自治区",
    "宁夏": "宁夏回族自治区", "新疆": "新疆维吾尔自治区",
}
PROVINCES_FULL = list(PROVINCE_SHORT.values())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 法条匹配模式：(第)?X条(之X)?(第X款)?(第(X)项)?
# 支持中文数字和阿拉伯数字，"第"可缺失
ART_PAT = (
    r"第?[零一二三四五六七八九十百千\d]+"
    r"条(?:之[一二三四五六七八九十\d]+)?"
    r"(?:第[一二三四五六七八九十\d]+款)?"
    r"(?:第[（(]?[一二三四五六七八九十\d]+[）)]?项)?"
)


def digit_to_chinese(num: int) -> str:
    """阿拉伯数字转中文数字。仅支持 1-9999（法条号码范围）。

    超出范围的数字返回原数字字符串。
    """
    if num == 0:
        return "零"
    if num > 9999:
        # 法条号码通常不超过 9999，超出范围保留原样
        return str(num)

    digits = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    result = ""

    # 千位
    if num >= 1000:
        result += digits[num // 1000] + "千"
        num %= 1000

    # 百位
    if num >= 100:
        result += digits[num // 100] + "百"
        num %= 100
    elif result and num > 0:
        result += "零"

    # 十位（十位为1且无更高位时省略"一"，如 10→"十"，16→"十六"）
    if num >= 10:
        if num // 10 == 1 and not result:
            result += "十"
        else:
            result += digits[num // 10] + "十"
        num %= 10
    elif result and num > 0:
        result += "零"

    # 个位
    if num > 0:
        result += digits[num]

    return result


def normalize_article_number(article: str) -> str:
    """将法条字符串规范化：阿拉伯数字转中文、补全缺失的"第"。

    仅用于法条提取后的规范化，不用于全文转换。
    例如：
        "第133条" → "第一百三十三条"
        "第2款" → "第二款"
        "一百七十七条" → "第一百七十七条"
    """
    def replace_digit(m):
        digit_str = m.group(0)
        try:
            num = int(digit_str)
            return digit_to_chinese(num)
        except (ValueError, TypeError):
            return digit_str

    result = re.sub(r"\d+", replace_digit, article)
    # 补全条号前缺失的"第"
    if not result.startswith("第"):
        result = "第" + result
    return result


# ============================================================================
#  元数据提取
# ============================================================================

def extract_id(text: str) -> str:
    """提取 ID，如 FBM-CLI.P.13998518"""
    m = re.search(r"CLI\.P\.(\d+)", text)
    return f"FBM-CLI.P.{m.group(1)}" if m else ""


def extract_date_from_end(end_text: str) -> str | None:
    """从 end section 中提取完整日期（年月日）。

    Args:
        end_text: classify_paragraphs 得到的 "end" 段文本

    Returns:
        日期字符串如 "2023-07-07"，提取失败返回 None
    """
    dates = re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日", end_text)
    if dates:
        y, m, d = dates[-1]
        return f"{y}-{int(m):02d}-{int(d):02d}"
    return None


def extract_procuratorate(body: str) -> str:
    """提取检察院全称"""
    m = re.search(r"([\u4e00-\u9fa5]{2,30}人民检察院)", body)
    return m.group(1) if m else ""


def extract_province_from_header(header_text: str) -> str:
    """从 header 中提取省份。

    优先使用 cpca 库从 header 中的地址信息（检察院全称等）提取省份，
    回退到关键词匹配。
    """
    df = cpca.transform([header_text])
    province = df.iloc[0]["省"]
    if province and isinstance(province, str) and province.strip():
        return province.strip()
    # 回退：关键词匹配
    for full_name in PROVINCES_FULL:
        if full_name in header_text:
            return full_name
    for short, full in PROVINCE_SHORT.items():
        if f"{short}省" in header_text or f"{short}市" in header_text:
            return full
    return ""


# ============================================================================
#  文本预处理
# ============================================================================

ADDRESS_SUFFIXES = ["号", "弄", "村", "社", "屯", "队", "街", "路"]


def anonymize_address(text: str) -> str:
    """将地址中的门牌号数字脱敏，如 '123号' → '**号'。"""
    for suffix in ADDRESS_SUFFIXES:
        text = re.sub(rf'\d{{1,4}}{suffix}', f'**{suffix}', text)
    return text


def remove_label_leakage(text: str) -> str:
    """删除文本中的标签泄露词汇，避免模型直接从输入中推断决定类型。

    替换规则：
        - "被不起诉人" → "犯罪嫌疑人"（审查起诉阶段的中性称呼）
        - "被告人" → "犯罪嫌疑人"（起诉书中的称呼，同样泄露决定类型）
    注意：先替换"被不起诉人"再替换"被告人"，因为"被不起诉人"不包含"被告人"子串，
    两者互不干扰。
    """
    text = text.replace("被不起诉人", "犯罪嫌疑人")
    text = text.replace("被告人", "犯罪嫌疑人")
    return text


def strip_metadata_lines(raw_text: str) -> str:
    """去掉文件头两行（法宝引证码 & 原文链接），返回正文"""
    lines = raw_text.strip().split("\n")
    body_lines = []
    skipped = 0
    for line in lines:
        s = line.strip()
        if skipped < 2 and ("【法宝引证码】" in s or "原文链接" in s):
            skipped += 1
            continue
        body_lines.append(line)
    return "\n".join(body_lines).strip()


# ============================================================================
#  段落级分类
# ============================================================================

SECTION_ORDER = ["header", "person", "procedure", "fact",
                 "reasoning", "end", "tail"]


def _detect_paragraph_section(para: str, current: str, doc_type: str):
    """判断单个段落所属 section，返回 section 名或 None（保持当前 section）。

    仅在检测到比 current 更靠后的 section 时才返回非 None 值（前向约束）。
    """
    stripped = para.strip()
    if not stripped:
        return None

    cur_idx = SECTION_ORDER.index(current)

    # ---- header（仅当当前仍在 header 时）----
    if current == "header":
        # 以"被不起诉人"/"被告人"/"被不起诉单位"开头 → 进入 person
        # 支持编号前缀，如 "1.被告人xxx"、"1、被告人xxx"
        if re.match(r"(?:\d+[.、．]\s*)?(?:被不起诉人|被不起诉单位|被告人|犯罪嫌疑人)", stripped):
            return "person"
        # 仍属 header 的情况
        if any(kw in stripped for kw in ["不起诉决定书", "起诉书", "人民检察院", "人民法院"]):
            return None  # 保持 header
        if re.search(r"[〔\[]\d{4}[〕\]]", stripped):
            return None  # 案号，保持 header
        if len(stripped) < 50:
            return None  # 短段落，保持 header
        # 长段落：不 return，fall through 到后续 section 检测

    # ---- person → procedure ----
    if cur_idx == SECTION_ORDER.index("person"):
        if stripped.startswith("本案由"):
            return "procedure"

    # ---- procedure / fact → fact ----
    if cur_idx == SECTION_ORDER.index("procedure") or cur_idx == SECTION_ORDER.index("fact"):
        # 标准事实标记
        fact_pats = [
            r"经本院依法审查查明", r"经依法审查查明", r"现依法审查查明",
            r"经本院审查查明", r"经本院审查认定", r"经审查认定的事实",
            r"经审查认定", r"检察机关认定", r"侦查机关认定的犯罪事实",
            r"公安机关认定的事实", r"审查认定事实", r"移送机关认定",
            r"起诉意见书认定[：:]",
            r"本院依法审查查明",
            r"经审查查明",
            r"经本院依法审查明",
            r"现查明[：:]",
            r"经依法审查认定的犯罪事实",
            r"移送审查起诉事实[：:]",
            r"认定.{0,20}的?犯罪行为如下",
            r"经依法侦查查明",
            r"依法审查查明",
            r"依法侦查查明",
            r"经我院(?:依法)?审查查明",
            r".{0,30}移送(?:审查)?起诉认定",
            r".{0,30}(?:公安局|公安分局|分局).{0,15}(?:移送(?:审查)?起诉)?认定[：:]",
        ]
        for pat in fact_pats:
            if re.match(pat, stripped):
                return "fact"
        # 起诉书结构化标题：独立行"犯罪事实"/"指控事实"/"案件事实"
        if stripped in ("犯罪事实", "指控事实", "案件事实"):
            return "fact"

    # ---- → reasoning ----
    # 严格按序：只有已进入 fact 后才允许检测 reasoning，
    # 防止从 person/procedure 直接跳到 reasoning 导致 fact 被跳过。
    # 注意：所有模式使用 re.match（段落开头），避免在长事实段中间误触发。
    if cur_idx == SECTION_ORDER.index("fact"):
        if re.match(r"本院(?:审查|仍然?)?认为", stripped):
            return "reasoning"
        if re.match(r"经(?:本院|依法)审查(?:，)?认为|本院依据", stripped):
            return "reasoning"
        # 存疑不起诉：经（本院）审查并（经X次）退回补充侦查/调查
        if re.match(r"经(?:本院|依法)?审查并(?:经)?(?:[一二三四五六七八九两\d]+次)?(?:退回)?补充(?:侦查|调查)", stripped):
            return "reasoning"

    # ---- → end ----
    if cur_idx == SECTION_ORDER.index("reasoning"):
        if doc_type == "prosecution":
            if stripped.startswith("此致"):
                return "end"
        else:
            if re.match(r"被不起诉人如(?:果)?不服", stripped):
                return "end"
            if re.match(r"被害人(?:的?(?:近亲属|法定代理人|家属))?如(?:果)?不服", stripped):
                return "end"

        # 通用：签署机关（xxx检察院）或单独日期行
        if re.match(r".{0,10}(?:人民检察院|检察院)\s*$", stripped):
            return "end"
        if re.match(r"\d{4}年\d{1,2}月\d{1,2}日\s*$", stripped):
            return "end"

    return None


def classify_paragraphs(body: str, doc_type: str = "non_prosecution") -> dict:
    """将文书正文按段落顺序分类为各 section。

    Args:
        body: 文书正文（已去掉引证码和链接两行）
        doc_type: "non_prosecution" 或 "prosecution"

    Returns:
        dict[str, str]，key 为 section 名（person/procedure/fact/...），value 为文本
    """
    paragraphs = body.split("\n")
    current = "header"
    sections = defaultdict(list)

    for para in paragraphs:
        detected = _detect_paragraph_section(para, current, doc_type)
        if detected is not None:
            new_idx = SECTION_ORDER.index(detected)
            cur_idx = SECTION_ORDER.index(current)
            if new_idx > cur_idx:
                current = detected
            elif new_idx == cur_idx:
                if detected == "fact" and current == "fact":
                    sections["fact"] = []
        sections[current].append(para)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


# ============================================================================
#  标签提取
# ============================================================================

_NEGATION_PAT = re.compile(r"不(?:属于|符合|构成|适用|是)")


def extract_articles(reasoning: str):
    """从推理文本中提取刑法、刑事诉讼法、刑事诉讼规则条文列表。"""
    cleaned = reasoning
    cl, cpl, cpr = [], [], []

    for law_name, target in [
        ("《刑法》", cl),
        ("《中华人民共和国刑法》", cl),
        ("《刑事诉讼法》", cpl),
        ("《中华人民共和国刑事诉讼法》", cpl),
        ("《人民检察院刑事诉讼规则》", cpr),
    ]:
        for m in re.finditer(re.escape(law_name), cleaned):
            # 检查法律名称前方是否存在否定词，跳过"不属于/不符合/不构成..."等引用
            # 但如果否定词与法律名称之间有句号/分号等句子终结符，
            # 说明否定词修饰的是前一句（如"不构成犯罪。依照《刑事诉讼法》..."），不应跳过
            before = cleaned[max(0, m.start() - 10) : m.start()]
            neg_m = _NEGATION_PAT.search(before)
            if neg_m and not re.search(r"[，。；！？]", before[neg_m.end():]):
                continue
            after = cleaned[m.end() :]
            # 去除法律名称后面的空白字符，使匹配更稳健
            after = after.lstrip()
            # 匹配紧随其后的一组法条（可能用 、和 等连接）
            arts_m = re.match(
                ART_PAT + r"(?:[、，和及]" + ART_PAT + r")*", after
            )
            if arts_m:
                for a in re.findall(ART_PAT, arts_m.group(0)):
                    # 将阿拉伯数字转换为中文数字
                    a = normalize_article_number(a)
                    if a not in target:
                        target.append(a)
    return cl, cpl, cpr


def detect_decision_type(reasoning: str) -> str | None:
    """判定不起诉类型。

    分类逻辑参考法律规定和实务文书特征，按优先级判断：
        1. 法定不起诉：不构成犯罪、没有犯罪事实等（刑诉法第16条、第177条第1款）
        2. 存疑不起诉：证据不足、事实不清（刑诉法第175条第4款）
        3. 相对不起诉：犯罪情节轻微、不需要判处刑罚（刑诉法第177条第2款）

    Returns:
        '法定不起诉' | '存疑不起诉' | '相对不起诉' | None（无法分类）
    """
    c = reasoning

    # === 第一类：法定不起诉（绝对不构成犯罪）===
    # 法条依据：刑诉法第16条、第177条第1款
    # 放宽法条匹配：不再要求法律名称与条号相邻（实务文书常在两者之间引用其他法条）
    # 支持格式变体：缺字（一百七七条）、"之二"替代"第二款"、"条二款"省略"第"等
    # 注意：177条和175条号码足够独特，无需法律名称约束；16条较短，保留法律名称要求
    _m177 = r"(?:一百七十七|一百七七|177)\s*条?"
    # has_177_1 保留刑事诉讼法前缀（放宽至50字间距），避免匹配刑法第177条（伪造金融票证罪）
    has_177_1 = bool(re.search(r"刑事诉讼法[》]?.{0,50}" + _m177 + r"\s*(?:之规定\s*)?第?(?:一|1)\s*[款项]", c))
    has_177_2 = bool(re.search(_m177 + r"\s*(?:之规定\s*)?(?:第?(?:二|2)\s*款|之二)", c))
    has_16 = bool(re.search(r"刑事诉讼法[》]?.{0,50}第(?:十六|16)条", c))

    # 同时引用第一款和第二款，文书异常，归入 unknown
    if has_177_1 and has_177_2:
        return None
    if has_16 or (has_177_1 and not has_177_2):
        return "法定不起诉"
    if any(kw in c for kw in [
        "不构成犯罪",
        "不认定为犯罪",
        "没有犯罪事实",
        "不认为是犯罪",
        "不具有犯罪事实",
        "情节显著轻微、危害不大",
        "情节显著轻微，危害不大",
        "不负刑事责任",
    ]):
        return "法定不起诉"
    # "不构成XXX罪"（如"不构成盗窃罪"）也属法定不起诉
    if re.search(r"不构成[\u4e00-\u9fa5、]{2,20}罪", c):
        return "法定不起诉"

    # === 第二类：存疑不起诉（证据不足）===
    # 法条依据：刑诉法第175条第4款
    has_175_4 = bool(re.search(r"(?:一百七十五|175)\s*条\s*(?:之规定\s*)?第?(?:四|4)\s*款", c))
    if has_175_4:
        return "存疑不起诉"
    if any(kw in c for kw in [
        "存疑不起诉适用",
        "证据不足",
        "事实不清",
        "现有证据难以证实",
        "犯罪事实不清",
        "不符合起诉条件",
    ]):
        return "存疑不起诉"

    # === 第三类：相对不起诉（有罪但免予起诉）===
    # 法条依据：刑诉法第177条第2款
    if has_177_2:
        return "相对不起诉"
    if any(kw in c for kw in [
        "相对不起诉",
        "酌定不起诉",
        "犯罪情节轻微",
        "犯罪情节较轻",
        "不需要判处刑罚",
        "免除刑罚",
        "免予刑事处罚",
        "免除处罚",
    ]):
        return "相对不起诉"
    if re.search(r"情节(?:较为)?(?:轻微|较轻)", c):
        return "相对不起诉"

    # === 无法分类 ===
    return None


# ============================================================================
#  核心：单文档解析
# ============================================================================

def parse_document(filepath: str, doc_type: str = "non_prosecution") -> dict | None:
    """解析单个文书，返回结构化字典；解析失败返回 None。

    Args:
        filepath: 文件路径
        doc_type: "non_prosecution"（不起诉决定书）或 "prosecution"（起诉书）
    """
    filename = os.path.basename(filepath)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        logger.debug(f"读取文件失败: {filename} -> {e}")
        return None

    if not raw_text.strip():
        return None

    # ----- 元数据 -----
    doc_id = extract_id(raw_text)

    # ----- 正文 -----
    body = strip_metadata_lines(raw_text)

    # ----- 分段（先分段，再从特定段提取元数据）-----
    sections = classify_paragraphs(body, doc_type)

    # ----- 从 end 段提取日期 -----
    date_str = extract_date_from_end(sections.get("end", ""))

    # ----- 从 header 段提取省份 -----
    province = extract_province_from_header(sections.get("header", ""))

    input_person = sections.get("person", "")
    # 过滤无关角色信息（如辩护人、指定辩护人）
    input_person = "\n".join(
        line for line in input_person.split("\n")
        if not re.match(r"\s*(?:指定)?辩护人", line)
    ).strip()
    input_procedure = sections.get("procedure", "")
    input_fact = sections.get("fact", "")
    reasoning = sections.get("reasoning", "")
    # 去除推理段末尾的签署信息（检察院名、院印、日期，可能出现任意组合）
    reasoning = re.sub(r"\s*\d{4}年\d{1,2}月\d{1,2}日\s*$", "", reasoning).strip()
    reasoning = re.sub(r"\s*[（(]院印[）)]\s*$", "", reasoning).strip()
    reasoning = re.sub(r"\s*[\u4e00-\u9fa5]{2,30}人民检察院\s*$", "", reasoning).strip()

    # ----- 去掉事实段落的引导语 -----
    input_fact = re.sub(
        r"^.*?(?:"
        r"经本院依法审查(?:查)?明"        # 经本院依法审查查明 / 经本院依法审查明(笔误)
        r"|本院依法审查查明"               # 缺"经"
        r"|(?:经|现)依法(?:审查|侦查)查明" # 经依法审查查明 / 现依法审查查明 / 经依法侦查查明
        r"|经本院审查(?:查明|认定(?:的事实)?)"
        r"|经我院(?:依法)?审查查明"        # "我院"代替"本院"
        r"|经审查(?:查明|认定(?:的事实)?)" # 经审查查明 / 经审查认定
        r"|经依法审查认定的犯罪事实(?:如下)?"
        r"|依法(?:审查|侦查)查明"          # 缺"经"和"本院"
        r"|(?:检察|侦查|公安)机关认定的?(?:犯罪|案件)?事实"
        r"|(?:公安局|公安分局|分局).{0,15}(?:移送(?:审查)?起诉)?认定"
        r"|审查认定事实"
        r"|移送(?:审查)?起诉(?:认定|事实)"
        r"|移送机关认定"
        r"|起诉意见书认定"
        r"|现查明"
        r"|认定.{0,20}的?犯罪行为如下"
        r"|犯罪事实|指控事实|案件事实"
        r")[，,：:\s]*",
        "", input_fact
    )

    # ----- 脱敏 -----
    input_person = anonymize_address(input_person)

    # ----- 防止标签泄露 -----
    input_person = remove_label_leakage(input_person)
    input_procedure = remove_label_leakage(input_procedure)
    input_fact = remove_label_leakage(input_fact)

    # ----- 标签 -----
    cl_articles, cpl_articles, cpr_articles = extract_articles(reasoning)

    if doc_type == "prosecution":
        decision = "起诉"
    else:
        decision = detect_decision_type(reasoning)

    # ----- 质量标记 -----
    warnings = []

    # ----- 文书类型校验：检查实际内容是否与声称的 doc_type 一致 -----
    # 防止源目录中放错的文书被错误提取（如起诉书放入不起诉目录）
    header_text = sections.get("header", "")
    header_first_lines = "\n".join(header_text.split("\n")[:6])
    if doc_type == "non_prosecution":
        # 标题含"起诉书"但不含"不起诉" → 实际是起诉书
        if "起诉书" in header_first_lines and "不起诉" not in header_first_lines:
            warnings.append("doc_type_mismatch")
    elif doc_type == "prosecution":
        # 标题以"不起诉决定书"开头 → 实际是不起诉决定书
        for line in header_first_lines.split("\n"):
            if line.strip().startswith("不起诉决定书"):
                warnings.append("doc_type_mismatch")
                break

    # 未识别出任何实质内容段落 → 文件内容为空
    has_content = any(k in sections for k in ("person", "procedure", "fact", "reasoning"))
    if not has_content:
        warnings.append("empty")
    # 文书类型不匹配时，无需再检查其他质量问题
    elif "doc_type_mismatch" in warnings:
        pass
    else:
        if not province:
            warnings.append("missing_province")
        if not input_person:
            warnings.append("missing_person")
        elif not input_procedure:
            warnings.append("missing_procedure")
        elif not input_fact:
            warnings.append("missing_fact")
        elif not reasoning:
            warnings.append("missing_reasoning")
        else:
            if decision is None:
                warnings.append("unclassified_decision")
            if input_fact and re.search(
                r"《(?:中华人民共和国)?(?:刑法|刑事诉讼法)》|《人民检察院刑事诉讼规则》",
                input_fact,
            ):
                warnings.append("fact_contaminated")

            # 法条检查
            # - 刑诉法/刑诉规则：至少包含一个
            # - 刑法：仅起诉 / 相对不起诉需要
            if not cpl_articles and not cpr_articles:
                warnings.append("missing_procedural_law")
            if doc_type == "prosecution" or decision == "相对不起诉":
                if not cl_articles:
                    warnings.append("missing_cl")
            
            # 元数据检查
            if not date_str:
                warnings.append("missing_date")

    result = {
        "id": doc_id,
        "meta": {
            "date": date_str,
            "province": province,
        },
        "person_info": input_person,
        "procedure": input_procedure,
        "fact": input_fact,
        "relevant_articles": (
            [f"cl:{a}" for a in cl_articles]
            + [f"cpl:{a}" for a in cpl_articles]
            + [f"cpr:{a}" for a in cpr_articles]
        ),
        "decision": decision,
        "raw_reasoning_and_decision": reasoning,
    }
    if warnings:
        result["_warnings"] = warnings

    return result


# ============================================================================
#  批量处理
# ============================================================================

def _save_to_unknown(fpath: str, unknown_base: str, reason: str):
    """将文件复制到 unknown/<reason>/ 子文件夹。"""
    dest_dir = os.path.join(unknown_base, reason)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(fpath, os.path.join(dest_dir, os.path.basename(fpath)))


# 有任何一项 warning 就归入 unknown 的标记集合
UNKNOWN_WARNINGS = {
    "empty",
    "doc_type_mismatch",
    "unclassified_decision",
    "missing_person",
    "missing_procedure",
    "missing_fact",
    "missing_reasoning",
    "missing_province",
    "missing_date",
    "missing_cl",
    "missing_procedural_law",
    "fact_contaminated",
}


def process_split(input_dir: str, output_path: str, split_name: str, limit: int = None):
    """处理一个 split（train / test）下的所有文书（不起诉决定书 + 起诉书）。

    对于解析失败或存在质量问题的文件，按原因分类保存到 unknown/ 子文件夹：
        unknown/empty/                  — 文件内容为空（核心段落全部缺失）
        unknown/doc_type_mismatch/      — 文书实际类型与源目录不一致
        unknown/unclassified_decision/  — 无法判定不起诉类型
        unknown/missing_person/         — 缺少当事人信息
        unknown/missing_procedure/     — 缺少程序段
        unknown/missing_fact/           — 缺少事实段（死亡案件豁免）
        unknown/missing_reasoning/      — 缺少推理段
        unknown/missing_province/       — 缺少省份信息
        unknown/missing_date/           — 缺少日期信息
        unknown/missing_cl/             — 缺少刑法法条
        unknown/missing_procedural_law/ — 缺少刑事诉讼法和刑事诉讼规则法条
        unknown/fact_contaminated/      — 事实段混入推理内容（含法律引用）
        unknown/parse_error/            — 解析过程中抛出异常
    """
    output_dir = os.path.dirname(output_path) or "."
    unknown_dir = os.path.join(output_dir, "unknown")

    # 清空上一次运行残留的 unknown 目录，避免累积
    if os.path.isdir(unknown_dir):
        shutil.rmtree(unknown_dir, ignore_errors=True)

    # 定义文书来源：(子目录名, 文书类型)
    sources = [
        ("不起诉", "non_prosecution"),
        ("起诉", "prosecution"),
    ]

    results = []
    stats = Counter()
    unknown_stats = Counter()

    for subdir, doc_type in sources:
        dir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(dir_path):
            continue

        files = sorted(f for f in os.listdir(dir_path) if f.endswith(".txt"))
        if limit:
            files = files[:limit]
        total = len(files)
        logger.info(f"[{split_name}/{subdir}] 待处理: {total} 个文件")

        for i, fname in enumerate(files, 1):
            fpath = os.path.join(dir_path, fname)
            try:
                record = parse_document(fpath, doc_type)
                if record:
                    warnings = record.get("_warnings", [])
                    hit_warnings = [w for w in warnings if w in UNKNOWN_WARNINGS]

                    if hit_warnings:
                        for w in hit_warnings:
                            _save_to_unknown(fpath, unknown_dir, w)
                            unknown_stats[w] += 1
                    else:
                        record.pop("_warnings", None)
                        results.append(record)
                        stats[record["decision"]] += 1
                else:
                    _save_to_unknown(fpath, unknown_dir, "empty_or_unparseable")
                    unknown_stats["empty_or_unparseable"] += 1
            except Exception as e:
                _save_to_unknown(fpath, unknown_dir, "parse_error")
                unknown_stats["parse_error"] += 1
                logger.debug(f"  解析异常: {fname} -> {e}")

            if i % 500 == 0 or i == total:
                logger.info(f"  [{split_name}/{subdir}] {i}/{total}")

    if not results and not unknown_stats:
        logger.warning(f"[{split_name}] 未找到任何文书文件")
        return [], {}

    # 保存 JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总统计
    total_unknown = sum(unknown_stats.values())
    stats["_unknown"] = total_unknown
    ok = sum(v for k, v in stats.items() if not k.startswith("_"))

    # 日志
    logger.info(f"[{split_name}] 完成: 成功 {ok}, 未分类 {total_unknown}")
    for dt in ["起诉", "相对不起诉", "法定不起诉", "存疑不起诉"]:
        logger.info(f"  {dt}: {stats.get(dt, 0)}")
    if total_unknown > 0:
        logger.info(f"  未分类文件已保存至: {unknown_dir}")
        for reason, cnt in sorted(unknown_stats.items()):
            logger.info(f"    {reason}: {cnt}")

    return results, dict(stats)


# ============================================================================
#  入口
# ============================================================================

def _collect_all_files(input_dir: str) -> list[tuple[str, str]]:
    """递归收集 input_dir 下所有文书文件路径及其类型。

    遍历目录树，将 "不起诉" 目录下的 .txt 视为 non_prosecution，
    将 "起诉" 目录下的 .txt 视为 prosecution。

    Returns:
        [(文件路径, 文书类型), ...]
    """
    files = []
    for root, _dirs, filenames in os.walk(input_dir):
        dirname = os.path.basename(root)
        if dirname == "不起诉":
            doc_type = "non_prosecution"
        elif dirname == "起诉":
            doc_type = "prosecution"
        else:
            continue
        for f in sorted(filenames):
            if f.endswith(".txt"):
                files.append((os.path.join(root, f), doc_type))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="从北大法宝文书（不起诉决定书 + 起诉书）中提取结构化 JSON"
    )
    parser.add_argument(
        "--input-dir", default="data/pku_fabao",
        help="输入根目录，包含 train/test 子目录 (default: data/pku_fabao)",
    )
    parser.add_argument(
        "--output-dir", default="data/PDP_dataset",
        help="输出根目录 (default: data/PDP_dataset)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="每个 split 每种文书最多处理条数（调试用）",
    )
    parser.add_argument(
        "--split-date", type=str, default=None,
        help="按日期划分 train/test（格式：YYYY-MM-DD），早于此日期为 train，其余为 test；"
             "不指定则按输入目录结构划分",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="输出 DEBUG 级日志",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.split_date:
        # ---- 按日期划分模式 ----
        all_files = _collect_all_files(args.input_dir)
        if args.limit:
            all_files = all_files[:args.limit]
        logger.info(f"按日期划分模式: split_date={args.split_date}, 共 {len(all_files)} 个文件")

        train_results, test_results = [], []
        stats = {"train": Counter(), "test": Counter()}
        unknown_dir = os.path.join(args.output_dir, "unknown")
        unknown_stats = Counter()

        if os.path.isdir(unknown_dir):
            shutil.rmtree(unknown_dir, ignore_errors=True)

        total = len(all_files)
        for i, (fpath, doc_type) in enumerate(all_files, 1):
            try:
                record = parse_document(fpath, doc_type)
                if record:
                    warnings = record.get("_warnings", [])
                    hit_warnings = [w for w in warnings if w in UNKNOWN_WARNINGS]
                    if hit_warnings:
                        for w in hit_warnings:
                            _save_to_unknown(fpath, unknown_dir, w)
                            unknown_stats[w] += 1
                    else:
                        record.pop("_warnings", None)
                        date_str = record["meta"].get("date")
                        if date_str and date_str < args.split_date:
                            train_results.append(record)
                            stats["train"][record["decision"]] += 1
                        else:
                            test_results.append(record)
                            stats["test"][record["decision"]] += 1
                else:
                    _save_to_unknown(fpath, unknown_dir, "empty_or_unparseable")
                    unknown_stats["empty_or_unparseable"] += 1
            except Exception as e:
                _save_to_unknown(fpath, unknown_dir, "parse_error")
                unknown_stats["parse_error"] += 1
                logger.debug(f"  解析异常: {os.path.basename(fpath)} -> {e}")

            if i % 500 == 0 or i == total:
                logger.info(f"  进度: {i}/{total}")

        # 保存 JSON
        for split, results in [("train", train_results), ("test", test_results)]:
            out_path = os.path.join(args.output_dir, split, "dataset.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        # 总结
        print("\n" + "=" * 65)
        print(f"  提取完成！（按日期划分: {args.split_date}）")
        print("=" * 65)
        for split in ["train", "test"]:
            results = train_results if split == "train" else test_results
            print(f"\n  [{split}]  成功 {len(results)}")
            for dt in ["起诉", "相对不起诉", "法定不起诉", "存疑不起诉"]:
                print(f"      {dt}: {stats[split].get(dt, 0)}")

        total_unknown = sum(unknown_stats.values())
        if total_unknown > 0:
            print(f"\n  未分类: {total_unknown}")
            for reason, cnt in sorted(unknown_stats.items()):
                print(f"      {reason}: {cnt}")
        print("=" * 65)

    else:
        # ---- 默认：按目录结构划分 ----
        all_stats = {}
        for split in ["train", "test"]:
            in_dir = os.path.join(args.input_dir, split)
            out_path = os.path.join(args.output_dir, split, "dataset.json")
            _, stats = process_split(in_dir, out_path, split, limit=args.limit)
            all_stats[split] = stats

        print("\n" + "=" * 65)
        print("  提取完成！")
        print("=" * 65)
        for split, stats in all_stats.items():
            ok = sum(v for k, v in stats.items() if not k.startswith("_"))
            unknown = stats.get("_unknown", 0)
            print(f"\n  [{split}]  成功 {ok}  |  未分类 {unknown}")
            for dt in ["起诉", "相对不起诉", "法定不起诉", "存疑不起诉"]:
                print(f"      {dt}: {stats.get(dt, 0)}")
        print("=" * 65)


if __name__ == "__main__":
    main()
