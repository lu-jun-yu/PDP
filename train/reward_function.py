#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train/reward_function.py

DAPO 训练的主奖励函数采用克制的结果监督：
  - 格式正确且 decision 与 gold 完全一致: 1.0
  - 格式正确但 decision 错误: 0.1
  - 格式错误 / decision 无法解析: 0.0

该设置便于 RQ3 诊断不同公诉决定边界在相同 RL 结果反馈下的可学习性差异。
"""

import re


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
    """从模型输出中解析最终 decision（去除 <think> 块后解析）。"""
    result = {"decision": ""}
    answer_block = _strip_think_block(text)

    sec = re.search(r"【最终结论】(.*?)$", answer_block, re.DOTALL)
    if sec:
        dec = re.search(r"决定[：:]\s*(.*?)(?:\n|$)", sec.group(1))
        if dec:
            result["decision"] = dec.group(1).strip()

    return result


# ============================================================
#  辅助：格式检查
# ============================================================

# 合法的决定类型
_VALID_DECISIONS = {"起诉", "相对不起诉", "法定不起诉", "存疑不起诉"}


def _normalize_decision_label(label: str) -> str | None:
    """将输出中的 decision 归一化为四类标签；无法归一化则返回 None。"""
    if not isinstance(label, str):
        return None
    normalized = label.strip().replace("**", "")
    normalized = re.sub(r"[。；;，,、\s]+$", "", normalized)
    return normalized if normalized in _VALID_DECISIONS else None


def _check_basic_format(text: str) -> bool:
    """
    RQ3/DAPO 使用的轻量格式检查。

    仅要求三个结构标签存在且顺序正确、审查分析非空，并且最终 decision
    能被解析为四个合法标签之一；不奖励法条完整性或法条匹配程度。
    """
    answer_block = _strip_think_block(text)
    tags = ("【适用法条】", "【审查分析】", "【最终结论】")

    positions = []
    for tag in tags:
        if answer_block.count(tag) != 1:
            return False
        positions.append(answer_block.find(tag))
    if positions != sorted(positions):
        return False

    all_brackets = re.findall(r"【[^】]*】", answer_block)
    if len(all_brackets) != 3:
        return False

    analysis_start = positions[1] + len("【审查分析】")
    analysis_end = positions[2]
    if not answer_block[analysis_start:analysis_end].strip():
        return False

    parsed = _parse_answer(text)
    return _normalize_decision_label(parsed["decision"]) is not None


# ============================================================
#  奖励函数（供 GRPOTrainer 调用）
# ============================================================

def decision_format_reward_func(completions, **kwargs) -> list[float]:
    """
    RQ3 主实验奖励：
      - 格式正确且四分类 decision 完全正确: 1.0
      - 格式正确但四分类 decision 错误: 0.1
      - 格式错误或 decision 无法解析: 0.0
    """
    ref_decisions = kwargs["decision"]
    rewards = []
    for completion, ref in zip(completions, ref_decisions):
        content = _get_content(completion)
        if not _check_basic_format(content):
            rewards.append(0.0)
            continue

        parsed = _parse_answer(content)
        pred_decision = _normalize_decision_label(parsed["decision"])
        ref_decision = _normalize_decision_label(ref)
        rewards.append(1.0 if pred_decision == ref_decision else 0.1)
    return rewards
