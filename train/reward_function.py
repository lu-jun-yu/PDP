#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train/reward_function.py

DAPO 训练的奖励函数，共四项：
  1. format_reward   — 严格格式匹配奖励 (0 / 2)，完全匹配 prompt_template 模板
  2. decision_reward  — 结果奖励: decision 正确与否 (0 / 2)
  3. process_reward   — 过程奖励: 法条 F1 (0 / 1)
  4. citation_reward  — 法条引用一致性奖励: 审查分析引用 vs 适用法条声明 的 F1
"""

import re


# ============================================================
#  基础工具
# ============================================================

def _split_articles(raw: str) -> list[str]:
    items = re.split(r"[、，,;；]", raw)
    return [a.strip() for a in items if a.strip()]


def _set_prf1(pred: set, gold: set) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _get_content(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return last.get("content", str(last)) if isinstance(last, dict) else str(last)
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _strip_think_block(text: str) -> str:
    """去除 <think>...</think> 块，返回剩余内容。"""
    think_end = text.find("</think>")
    if think_end != -1:
        return text[think_end + len("</think>"):].strip()
    return text.strip()


def _parse_answer(text: str) -> dict:
    """从模型输出中解析结构化字段（去除 <think> 块后解析）。"""
    result = {
        "relevant_articles": [],
        "decision": "",
    }

    answer_block = _strip_think_block(text)

    # 适用法条
    sec = re.search(r"【适用法条】(.*?)(?=【审查分析】|【最终结论】|$)", answer_block, re.DOTALL)
    if sec:
        articles = []
        cl = re.search(r"刑法[：:](.*?)(?:\n|$)", sec.group(1))
        if cl:
            articles += [f"cl:{a}" for a in _split_articles(cl.group(1).strip())]
        cpl = re.search(r"刑事诉讼法[：:](.*?)(?:\n|$)", sec.group(1))
        if cpl:
            articles += [f"cpl:{a}" for a in _split_articles(cpl.group(1).strip())]
        cpr = re.search(r"刑事诉讼规则[：:](.*?)(?:\n|$)", sec.group(1))
        if cpr:
            articles += [f"cpr:{a}" for a in _split_articles(cpr.group(1).strip())]
        result["relevant_articles"] = articles

    # 最终结论
    sec = re.search(r"【最终结论】(.*?)$", answer_block, re.DOTALL)
    if sec:
        dec = re.search(r"决定[：:]\s*(.*?)(?:\n|$)", sec.group(1))
        if dec:
            result["decision"] = dec.group(1).strip()

    return result


# ============================================================
#  辅助：审查分析中引用的法条提取
# ============================================================

_ARTICLE_PAT = r"(第[零一二三四五六七八九十百千\d]+条(?:第[零一二三四五六七八九十百千\d]+款)?)"


def _extract_cited_articles(text: str) -> set:
    """从【审查分析】段落正则提取引用的法条，返回带前缀的混合集合。"""
    answer_block = _strip_think_block(text)

    analysis = re.search(
        r"【审查分析】(.*?)(?=【最终结论】|$)", answer_block, re.DOTALL
    )
    if not analysis:
        return set()

    body = analysis.group(1)
    mixed = set()
    for a in re.findall(r"刑法》?" + _ARTICLE_PAT, body):
        mixed.add(f"cl:{a}")
    for a in re.findall(r"刑事诉讼法》?" + _ARTICLE_PAT, body):
        mixed.add(f"cpl:{a}")
    for a in re.findall(r"刑事诉讼规则》?" + _ARTICLE_PAT, body):
        mixed.add(f"cpr:{a}")
    return mixed


# ============================================================
#  辅助：严格格式检查
# ============================================================

# 合法的决定类型
_VALID_DECISIONS = {"起诉", "相对不起诉", "法定不起诉", "存疑不起诉"}

# 法条条目：第X条（第X款）格式，多条用"、"分隔
_ART_ITEM = r"第[零一二三四五六七八九十百千\d]+条(?:第[零一二三四五六七八九十百千\d]+款)?"
_ART_LIST = _ART_ITEM + r"(?:、" + _ART_ITEM + r")*"

# 单行法律条目：刑法/刑事诉讼法/刑事诉讼规则：第X条、第X条
_LAW_LINE = r"(?:刑法|刑事诉讼法|刑事诉讼规则)[：:]" + _ART_LIST

# 完整格式正则，严格匹配 prompt_template 中的示例结构：
#   【适用法条】\n
#   刑法：第X条、第X条\n
#   (可选更多法律行)\n
#   \n
#   【审查分析】\n
#   ...分析文本...\n
#   \n
#   【最终结论】\n
#   决定：四选一
_STRICT_FORMAT_RE = re.compile(
    r"【适用法条】\n"
    r"(?:" + _LAW_LINE + r"\n)+"           # 至少一行法条，每行以换行结尾
    r"\n"                                   # 段落间空行
    r"【审查分析】\n"
    r"(?:(?!【).)+\n"                       # 分析正文（至少一个字符），不含【标签
    r"\n"                                   # 段落间空行
    r"【最终结论】\n"
    r"决定[：:](?:起诉|相对不起诉|法定不起诉|存疑不起诉)"  # 决定行
    r"\s*$",                                # 末尾允许空白
    re.DOTALL,
)


def _check_format(text: str) -> bool:
    """
    严格格式检查，完全匹配 prompt_template 中给出的输出模板。

    要求：
    1. 去除 <think>...</think> 后的正文必须严格以【适用法条】开头
    2. 三个段落标签各出现恰好一次，顺序固定
    3. 【适用法条】后每行格式为 "法律名：第X条、第X条"
    4. 段落之间有且仅有一个空行
    5. 【最终结论】后紧跟 "决定：" + 四种合法决定之一
    6. 不允许出现任何额外的【...】标签
    """
    answer_block = _strip_think_block(text)

    # 三个标签各出现恰好一次
    for tag in ("【适用法条】", "【审查分析】", "【最终结论】"):
        if answer_block.count(tag) != 1:
            return False

    # 不允许出现除三个合法标签外的其他【...】标签
    all_brackets = re.findall(r"【[^】]*】", answer_block)
    if len(all_brackets) != 3:
        return False

    # 用完整正则匹配
    return _STRICT_FORMAT_RE.match(answer_block) is not None


# ============================================================
#  四项奖励函数（供 GRPOTrainer 调用）
# ============================================================

def format_reward_func(completions, **kwargs) -> list[float]:
    """格式匹配奖励：严格匹配得 2，否则 0。"""
    return [2.0 if _check_format(_get_content(c)) else 0.0 for c in completions]


def decision_reward_func(completions, **kwargs) -> list[float]:
    """结果奖励：decision 正确得 2，错误得 0。"""
    ref_decisions = kwargs["decision"]
    rewards = []
    for c, ref in zip(completions, ref_decisions):
        parsed = _parse_answer(_get_content(c))
        rewards.append(2.0 if parsed["decision"] == ref else 0.0)
    return rewards


def process_reward_func(completions, **kwargs) -> list[float]:
    """
    过程奖励（0~1）：法条 F1。
    """
    ref_articles = kwargs["relevant_articles"]

    rewards = []
    for c, g_arts in zip(completions, ref_articles):
        parsed = _parse_answer(_get_content(c))

        # --- 法条 F1 ---
        pred_arts = set(parsed["relevant_articles"])
        gold_arts = set(g_arts)
        _, _, f1_arts = _set_prf1(pred_arts, gold_arts)

        rewards.append(f1_arts)

    return rewards


def citation_reward_func(completions, **kwargs) -> list[float]:
    """
    法条引用一致性奖励（0~1）：
    审查分析中引用的法条 vs 输出自身【适用法条】中声明的法条，
    混合 F1（带法律前缀）。
    """
    rewards = []
    for c in completions:
        text = _get_content(c)
        parsed = _parse_answer(text)

        cited = _extract_cited_articles(text)
        declared = set(parsed["relevant_articles"])
        _, _, f1 = _set_prf1(cited, declared)
        rewards.append(f1)

    return rewards
