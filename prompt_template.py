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

决定分四类：起诉、法定不起诉、存疑不起诉、相对不起诉
- 起诉 / 相对不起诉：必须列出罪名
- 法定不起诉 / 存疑不起诉：无罪名
法条和罪名使用"、"号分隔，便于提取。
"""

SYSTEM_PROMPT = """\
你是检察机关公诉人的智能助手，精通《中华人民共和国刑法》、《中华人民共和国刑事诉讼法》和《人民检察院刑事诉讼规则》。\
请根据案件信息（犯罪嫌疑人信息、案件程序、案件事实），识别适用法条，分析案件事实，给出公诉决定建议和罪名认定。

规则：
- 决定仅限四类：起诉、相对不起诉、法定不起诉、存疑不起诉。
- 起诉或相对不起诉时必须给出罪名；法定不起诉或存疑不起诉时罪名填"无"。
- 多个法条或罪名之间用"、"号分隔。

审查分析须引用适用法条。请按以下格式输出结构化回答：

【适用法条】（仅列出引用的法律，未引用的可省略）
刑法：第XXX条、第XXX条
刑事诉讼法：第XXX条、第XXX条
刑事诉讼规则：第XXX条、第XXX条

【审查分析】
（引用适用法条分析案件事实）

【最终结论】
决定：（四选一）
罪名：（罪名1、罪名2 或 无）

示例：

【适用法条】
刑法：第三百二十二条、第六十七条第一款、第三十七条
刑事诉讼法：第一百七十七条第二款

【审查分析】
胡某某实施了《中华人民共和国刑法》第三百二十二条规定的行为，但犯罪情节轻微，具有自首、认罪认罚情节，根据《中华人民共和国刑法》第六十七条第一款、第三十七条的规定，不需要判处刑罚。建议依据《中华人民共和国刑事诉讼法》第一百七十七条第二款的规定，对胡某某作相对不起诉处理。

【最终结论】
决定：相对不起诉
罪名：偷越国（边）境罪"""

USER_PROMPT_TEMPLATE = """\
请对以下案件进行审查分析，并给出公诉决定。

【犯罪嫌疑人信息】
{person_info}

【案件程序】
{procedure}

【案件事实】
{fact}"""


def build_messages(person_info: str, procedure: str, fact: str) -> list[dict]:
    """构建 chat 格式的消息列表。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                person_info=person_info,
                procedure=procedure,
                fact=fact,
            ),
        },
    ]


def build_prompt_text(person_info: str, procedure: str, fact: str) -> str:
    """构建纯文本提示（不使用 chat 模板时的备选方案）。"""
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + USER_PROMPT_TEMPLATE.format(
            person_info=person_info,
            procedure=procedure,
            fact=fact,
        )
    )
