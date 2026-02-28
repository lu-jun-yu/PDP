#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train/reward_function.py

GRPO 训练的奖励函数，共四项：
  1. format_reward   — 格式匹配奖励 (0 / 2)
  2. decision_reward  — 结果奖励: decision 正确与否 (0 / 2)
  3. process_reward   — 过程奖励: charges_score + 混合法条 F1
  4. citation_reward  — 法条引用一致性奖励: 审查分析引用 vs 适用法条声明 的混合 F1
"""

import re


# ============================================================
#  基础工具（与 eval/evaluate.py 保持一致）
# ============================================================

def _split_articles(raw: str) -> list[str]:
    items = re.split(r"[、，,;；]", raw)
    return [a.strip() for a in items if a.strip()]


def _split_charges(raw: str) -> list[str]:
    """将罪名列表拆分，只在"罪"字后的分隔符处切分，避免误拆含"、"的罪名。"""
    raw = raw.strip().rstrip("。.，,；;、")
    parts = re.split(r"(?<=罪)[、，,;；]\s*", raw)
    return [p.strip() for p in parts if p.strip()]


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
        "relevant_articles_cl": [],
        "relevant_articles_cpl": [],
        "relevant_articles_cpr": [],
        "decision": "",
        "charges": [],
    }

    answer_block = _strip_think_block(text)

    # 适用法条
    sec = re.search(r"【适用法条】(.*?)(?=【审查分析】|【最终结论】|$)", answer_block, re.DOTALL)
    if sec:
        cl = re.search(r"刑法[：:](.*?)(?:\n|$)", sec.group(1))
        if cl:
            result["relevant_articles_cl"] = _split_articles(cl.group(1).strip())
        cpl = re.search(r"刑事诉讼法[：:](.*?)(?:\n|$)", sec.group(1))
        if cpl:
            result["relevant_articles_cpl"] = _split_articles(cpl.group(1).strip())
        cpr = re.search(r"刑事诉讼规则[：:](.*?)(?:\n|$)", sec.group(1))
        if cpr:
            result["relevant_articles_cpr"] = _split_articles(cpr.group(1).strip())

    # 最终结论
    sec = re.search(r"【最终结论】(.*?)$", answer_block, re.DOTALL)
    if sec:
        dec = re.search(r"决定[：:]\s*(.*?)(?:\n|$)", sec.group(1))
        if dec:
            result["decision"] = dec.group(1).strip()
        chg = re.search(r"罪名[：:]\s*(.*?)(?:\n|$)", sec.group(1))
        if chg:
            raw = chg.group(1).strip()
            if raw and raw != "无":
                result["charges"] = _split_charges(raw)

    return result


def _build_mixed_articles(cl: list, cpl: list, cpr: list) -> set:
    """将各法律的条文加前缀后合并为一个集合，避免不同法律的相同条号冲突。"""
    mixed = set()
    for a in cl:
        mixed.add(f"cl:{a}")
    for a in cpl:
        mixed.add(f"cpl:{a}")
    for a in cpr:
        mixed.add(f"cpr:{a}")
    return mixed


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
#  辅助：格式检查
# ============================================================

def _check_format(text: str) -> bool:
    """检查输出是否符合预定格式：<think> 推理 + 三个结构化段落。"""
    answer_block = _strip_think_block(text)

    # 三个段落标题
    for tag in ("【适用法条】", "【审查分析】", "【最终结论】"):
        if tag not in answer_block:
            return False

    # 适用法条须至少包含一种法律的法条
    art_sec = re.search(r"【适用法条】(.*?)【审查分析】", answer_block, re.DOTALL)
    if not art_sec:
        return False
    has_any_law = (
        re.search(r"刑法[：:]", art_sec.group(1))
        or re.search(r"刑事诉讼法[：:]", art_sec.group(1))
        or re.search(r"刑事诉讼规则[：:]", art_sec.group(1))
    )
    if not has_any_law:
        return False

    # 最终结论须包含决定和罪名
    con_sec = re.search(r"【最终结论】(.*)", answer_block, re.DOTALL)
    if not con_sec:
        return False
    if not re.search(r"决定[：:]", con_sec.group(1)):
        return False
    if not re.search(r"罪名[：:]", con_sec.group(1)):
        return False

    return True


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
    过程奖励（0~2）：charges_score + 混合法条 F1。

    将三类法条（刑法 / 刑诉法 / 刑事诉讼规则）加前缀后合并为一个集合，
    计算一次 F1，无需分法律平均或动态分母。

    charges：起诉/相对不起诉 → F1，法定/存疑不起诉 → 空=1 非空=0。
    """
    ref_decisions = kwargs["decision"]
    ref_charges = kwargs["charges"]
    ref_cl = kwargs["relevant_articles_cl"]
    ref_cpl = kwargs["relevant_articles_cpl"]
    ref_cpr = kwargs["relevant_articles_cpr"]

    rewards = []
    for c, dec, g_chg, g_cl, g_cpl, g_cpr in zip(
        completions, ref_decisions, ref_charges, ref_cl, ref_cpl, ref_cpr
    ):
        parsed = _parse_answer(_get_content(c))

        # --- charges ---
        if dec in ("起诉", "相对不起诉"):
            _, _, s_chg = _set_prf1(set(parsed["charges"]), set(g_chg))
        else:
            s_chg = 1.0 if len(parsed["charges"]) == 0 else 0.0

        # --- 混合法条 F1 ---
        pred_arts = _build_mixed_articles(
            parsed["relevant_articles_cl"],
            parsed["relevant_articles_cpl"],
            parsed["relevant_articles_cpr"],
        )
        gold_arts = _build_mixed_articles(g_cl, g_cpl, g_cpr)
        _, _, f1_arts = _set_prf1(pred_arts, gold_arts)

        rewards.append(s_chg + f1_arts)

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
        declared = _build_mixed_articles(
            parsed["relevant_articles_cl"],
            parsed["relevant_articles_cpl"],
            parsed["relevant_articles_cpr"],
        )
        _, _, f1 = _set_prf1(cited, declared)
        rewards.append(f1)

    return rewards
