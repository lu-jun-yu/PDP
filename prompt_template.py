#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_template.py

检察机关审查起诉决定预测任务的提示词模板。
输入：person_info, procedure, fact
输出：<think> 推理 + 结构化答案（适用法条、审查分析、最终结论）

模型输出格式：
    <think>
    {思维链推理}
    </think>

    【适用法条】...
    【审查分析】...（须引用适用法条）
    【最终结论】...

决定分四类：起诉、相对不起诉、法定不起诉、存疑不起诉。
法条使用"、"号分隔，便于提取。

消融实验变体：
  1. original：原始 prompt（无定义、无示例）。
  2. definitions：原始 prompt + 四类决定定义与适用条件（prompt engineering）。
  3. one_shot：原始 prompt + 一组输入输出示例（in-context learning）。
"""

PROMPT_VARIANT_ORIGINAL = "original"
PROMPT_VARIANT_DEFINITIONS = "definitions"
PROMPT_VARIANT_ONE_SHOT = "one_shot"

PROMPT_VARIANTS = (
    PROMPT_VARIANT_ORIGINAL,
    PROMPT_VARIANT_DEFINITIONS,
    PROMPT_VARIANT_ONE_SHOT,
)


# ============================================================
#  definitions 变体：决定类型定义
# ============================================================

_DEFINITIONS_BLOCK = """\

各类决定的定义与适用条件：

1. 起诉
依据：《刑事诉讼法》第一百七十六条。
适用条件：人民检察院认为犯罪嫌疑人的犯罪事实已经查清，证据确实、充分，依法应当追究刑事责任的，应当作出起诉决定，按照审判管辖的规定，向人民法院提起公诉。
具体而言，同时满足以下条件时应当起诉：
（一）犯罪事实清楚，有明确的犯罪行为和结果；
（二）证据确实、充分，足以认定犯罪事实；
（三）依法应当追究刑事责任，不存在法定不起诉、存疑不起诉或相对不起诉的适用情形。

2. 法定不起诉
依据：《刑事诉讼法》第一百七十七条第一款。
适用条件：犯罪嫌疑人没有犯罪事实，或者具有《刑事诉讼法》第十六条规定的情形之一：
（一）情节显著轻微、危害不大，不认为是犯罪的；
（二）犯罪已过追诉时效期限的；
（三）经特赦令免除刑罚的；
（四）依照刑法告诉才处理的犯罪，没有告诉或者撤回告诉的；
（五）犯罪嫌疑人、被告人死亡的；
（六）其他法律规定免予追究刑事责任的。

3. 存疑不起诉
依据：《刑事诉讼法》第一百七十五条第四款。
适用条件：经过补充侦查的案件，人民检察院仍然认为证据不足，不符合起诉条件的，应当作出不起诉决定。证据不足包括：
（一）据以定罪的证据存在疑问，无法查证属实的；
（二）犯罪构成要件事实缺乏必要的证据予以证明的；
（三）据以定罪的证据之间的矛盾不能合理排除的；
（四）根据证据得出的结论具有其他可能性，不能排除合理怀疑的。

4. 相对不起诉
依据：《刑事诉讼法》第一百七十七条第二款；《人民检察院刑事诉讼规则》第三百七十条。
适用条件：案件事实清楚、证据确实充分，行为已经构成犯罪，本应符合起诉条件；但犯罪情节轻微，依照刑法规定不需要判处刑罚或者免除刑罚的，人民检察院经检察长批准可以作出不起诉决定。
判断时应重点区分：
（一）不是没有犯罪事实，也不是《刑事诉讼法》第十六条规定的不追究刑事责任情形，不能归入法定不起诉；
（二）不是证据不足或者不能排除合理怀疑，不能归入存疑不起诉；
（三）通常结合自首、坦白、认罪认罚、积极退赃退赔、赔偿谅解、初犯偶犯、危害后果较轻等从宽情节综合判断，但这些情节本身不能替代“犯罪情节轻微”和“依法不需要判处刑罚或者免除刑罚”的要求。"""


# ============================================================
#  系统提示词
# ============================================================

SYSTEM_PROMPT = """\
你是检察机关公诉人的智能助手，精通《中华人民共和国刑法》、《中华人民共和国刑事诉讼法》和《人民检察院刑事诉讼规则》。\
请根据案件信息（犯罪嫌疑人信息、案件程序、案件事实），识别适用法条，分析案件事实，给出公诉决定建议。

规则：
- 决定仅限四类：起诉、相对不起诉、法定不起诉、存疑不起诉。
- 多个法条之间用"、"号分隔。

审查分析须引用适用法条。请按以下格式输出结构化回答：

【适用法条】（仅使用中文数字列出引用的法条，未引用的可省略）
刑法：第XXX条、第XXX条第XXX款
刑事诉讼法：第XXX条、第XXX条第XXX款
刑事诉讼规则：第XXX条、第XXX条

【审查分析】
（引用适用法条分析案件事实）

【最终结论】
决定：起诉/相对不起诉/法定不起诉/存疑不起诉（四选一）"""

# 消融实验：在基线提示末尾插入四类决定的定义
SYSTEM_PROMPT_WITH_DEFINITIONS = SYSTEM_PROMPT + _DEFINITIONS_BLOCK

USER_PROMPT_TEMPLATE = """\
请对以下案件进行审查分析，并给出公诉决定。

【犯罪嫌疑人信息】
{person_info}

【案件程序】
{procedure}

【案件事实】
{fact}"""


# ============================================================
#  one_shot 变体：一组完整输入输出示例
# ============================================================

# 选自 data/public_sources/test/dataset_reviewed_true_only_llm.json 中
# source_url 为空的样本：吴堡检刑不诉〔2023〕9号。
_ONE_SHOT_USER_CONTENT = USER_PROMPT_TEMPLATE.format(
    person_info=(
        "犯罪嫌疑人李某某，男，1981年10月23日生，公民身份号码"
        "******19811023****，汉族，小学文化，户籍所在地吴堡县**镇**村**区**号，"
        "现住吴堡县**镇**村**区**号。"
    ),
    procedure=(
        "因涉嫌危险驾驶罪，2023年8月21日被吴堡县公安局刑事拘留。"
        "2023年8月24日被吴堡县人民检察院取保候审。"
    ),
    fact=(
        "2023年08月17日18时55分，犯罪嫌疑人李某某驾驶无牌号“速卡迪牌”"
        "普通二轮摩托车，沿沿黄观光路由西向东行驶至沿黄观光路吴堡横沟温泉附近时，"
        "被我大队执勤民警查获，经榆林市公安交通司法鉴定中心检验，"
        "李某某血液中乙醇浓度为107.84mg／100ml。"
    ),
)

_ONE_SHOT_ASSISTANT_CONTENT = """\
【适用法条】
刑法：第一百三十三条之一、第三十七条
刑事诉讼法：第一百七十七条第二款

【审查分析】
李某某实施了《中华人民共和国刑法》第一百三十三条之一规定的行为，但犯罪情节轻微，未造成严重社会影响，且认罪认罚。根据《中华人民共和国刑法》第三十七条的规定，不需要判处刑罚。建议依据《中华人民共和国刑事诉讼法》第一百七十七条第二款的规定，对李某某作相对不起诉处理。

【最终结论】
决定：相对不起诉"""


def _resolve_prompt_variant(prompt_variant: str) -> str:
    """解析 prompt 变体。"""
    if prompt_variant not in PROMPT_VARIANTS:
        choices = "、".join(PROMPT_VARIANTS)
        raise ValueError(f"未知 prompt_variant: {prompt_variant!r}，可选值：{choices}")
    return prompt_variant


def _select_system_prompt(prompt_variant: str) -> str:
    """根据消融实验变体选择系统提示词。"""
    if prompt_variant == PROMPT_VARIANT_DEFINITIONS:
        return SYSTEM_PROMPT_WITH_DEFINITIONS
    return SYSTEM_PROMPT


def _build_user_content(person_info: str, procedure: str, fact: str) -> str:
    """构建用户侧案件输入。"""
    return USER_PROMPT_TEMPLATE.format(
        person_info=person_info,
        procedure=procedure,
        fact=fact,
    )


def build_messages(
    person_info: str,
    procedure: str,
    fact: str,
    *,
    prompt_variant: str = PROMPT_VARIANT_ORIGINAL,
) -> list[dict]:
    """构建 chat 格式的消息列表。

    Args:
        prompt_variant: 消融实验变体，可选 original / definitions / one_shot。
    """
    prompt_variant = _resolve_prompt_variant(prompt_variant)
    messages = [{"role": "system", "content": _select_system_prompt(prompt_variant)}]

    if prompt_variant == PROMPT_VARIANT_ONE_SHOT:
        messages.extend(
            [
                {"role": "user", "content": _ONE_SHOT_USER_CONTENT},
                {"role": "assistant", "content": _ONE_SHOT_ASSISTANT_CONTENT},
            ]
        )

    messages.append(
        {
            "role": "user",
            "content": _build_user_content(person_info, procedure, fact),
        }
    )
    return messages


def build_prompt_text(
    person_info: str,
    procedure: str,
    fact: str,
    *,
    prompt_variant: str = PROMPT_VARIANT_ORIGINAL,
) -> str:
    """构建纯文本提示（不使用 chat 模板时的备选方案）。"""
    prompt_variant = _resolve_prompt_variant(prompt_variant)
    parts = [_select_system_prompt(prompt_variant)]

    if prompt_variant == PROMPT_VARIANT_ONE_SHOT:
        parts.extend(
            [
                "【示例输入】\n" + _ONE_SHOT_USER_CONTENT,
                "【示例输出】\n" + _ONE_SHOT_ASSISTANT_CONTENT,
            ]
        )

    parts.append(_build_user_content(person_info, procedure, fact))
    return "\n\n".join(parts)
