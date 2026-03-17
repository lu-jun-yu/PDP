#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/failure_analysis.py

PDP 任务挑战性深度分析脚本。
重点分析：
  - 起诉（情节严重/轻微）误判为三类不起诉
  - 三类不起诉之间的互相混淆

输出：
  1. challenge_analysis_report.md  — 挑战分析报告
  2. pdp_paper_introduction.md     — 学术论文引言草稿

Usage:
    python eval/failure_analysis.py
    python eval/failure_analysis.py --result-dir results/Qwen3-4B_test_20260305_185907
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

DECISION_TYPES = [
    "起诉",
    "相对不起诉", "法定不起诉", "存疑不起诉",
]
PROSECUTION = ["起诉"]
NON_PROSECUTION = ["相对不起诉", "法定不起诉", "存疑不起诉"]

MITIGATING_KW = [
    "认罪认罚", "自首", "坦白", "如实供述", "赔偿", "谅解",
    "初犯", "偶犯", "退赃", "退缴", "悔罪", "自愿认罪",
    "主动投案", "立功",
]
EVIDENCE_KW = [
    "证据不足", "事实不清", "无法认定", "不能证实", "无法证实",
    "不能排除", "无法排除", "存疑", "补充侦查", "退回", "退补",
]
STATUTORY_KW = [
    "不构成犯罪", "情节显著轻微", "危害不大", "未达到",
    "追诉时效", "特赦", "告诉才处理", "撤回告诉",
]


def input_text(r):
    return " ".join([r.get("person_info", ""), r.get("procedure", ""), r.get("fact", "")])


def think_block(raw):
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    return m.group(1).strip() if m else ""


def kw_found(text, kws):
    return [k for k in kws if k in text]


def load_all_records(result_dir):
    correct, errors = [], []
    for fname in os.listdir(result_dir):
        if not fname.startswith("details_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(result_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        if fname == "details_正确.json":
            correct = data
        else:
            errors.extend(data)
    metrics = {}
    mp = os.path.join(result_dir, "metrics.json")
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return correct, errors, metrics


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return {d["id"]: d for d in json.load(f)}


class Analyzer:
    def __init__(self, correct, errors, metrics, ds_map):
        self.correct = correct
        self.errors = errors
        self.metrics = metrics
        self.ds = ds_map
        self.all = correct + errors

        self.by_ref = defaultdict(list)
        for r in self.all:
            self.by_ref[r["reference"]["decision"]].append(r)

        self.confusion = Counter()
        for r in self.all:
            ref = r["reference"]["decision"]
            pred = r["prediction"]["decision"] or "（解析失败）"
            self.confusion[(ref, pred)] += 1

    def _error_group(self, ref_dec, pred_dec):
        return [r for r in self.errors
                if r["reference"]["decision"] == ref_dec
                and r["prediction"]["decision"] == pred_dec]

    def _analyze_case(self, r):
        inp = input_text(r)
        reasoning = r.get("raw_reasoning_and_decision", "")
        raw = r.get("raw_output", "")
        think = think_block(raw)
        ref = r["reference"]["decision"]
        pred = r["prediction"]["decision"]

        info = {"id": r["id"], "ref": ref, "pred": pred,
                "charges": r["reference"].get("charges", []),
                "root_cause": "", "detail": ""}

        inp_mit = kw_found(inp, MITIGATING_KW)
        reas_mit_only = [k for k in MITIGATING_KW if k in reasoning and k not in inp]
        think_mit = kw_found(think, MITIGATING_KW + ["从轻", "减轻", "情节轻微"])
        inp_stat = kw_found(inp, STATUTORY_KW)
        think_stat = kw_found(think, ["不构成犯罪", "情节显著轻微", "追诉时效", "第十六条", "第16条"])
        think_evi = kw_found(think, ["证据不足", "事实不清", "无法认定", "存疑"])
        has_supplement = any(k in r.get("procedure", "") for k in ["退回", "补充侦查", "退补", "补查重报"])

        # 起诉 -> 不起诉
        if ref in PROSECUTION and pred in NON_PROSECUTION:
            if pred == "相对不起诉":
                if inp_mit:
                    info["root_cause"] = "MITIGATING_OVERWEIGHT"
                    mit_str = ",".join(inp_mit)
                    info["detail"] = "输入含从轻线索(" + mit_str + ")，模型过度权重"
                else:
                    info["root_cause"] = "LLM_OVER_LENIENT"
                    info["detail"] = "无明显从轻线索仍判不起诉"
            elif pred == "法定不起诉":
                info["root_cause"] = "CRIME_THRESHOLD_MISJUDGE"
                info["detail"] = "错误认为不构成犯罪/达不到入罪门槛"
            elif pred == "存疑不起诉":
                info["root_cause"] = "EVIDENCE_OVER_SKEPTICISM"
                info["detail"] = "对充分证据产生不当怀疑"

        # 不起诉之间互相混淆
        elif ref in NON_PROSECUTION and pred in NON_PROSECUTION and ref != pred:
            pair = (ref, pred)
            pair_map = {
                ("相对不起诉", "法定不起诉"): ("GUILT_BOUNDARY", "混淆[有罪免刑](相对)与[不构成犯罪](法定)"),
                ("法定不起诉", "相对不起诉"): ("GUILT_BOUNDARY", "将[不构成犯罪]误判为[有罪但情节轻微]"),
                ("相对不起诉", "存疑不起诉"): ("EVIDENCE_VS_MERCY", "混淆[有罪免诉]与[证据不足]"),
                ("存疑不起诉", "相对不起诉"): ("EVIDENCE_VS_MERCY", "将[证据不足]误判为[有罪但情节轻微]"),
                ("存疑不起诉", "法定不起诉"): ("EVIDENCE_VS_STATUTORY", "混淆[证据不足]与[不构成犯罪]"),
                ("法定不起诉", "存疑不起诉"): ("EVIDENCE_VS_STATUTORY", "将[不构成犯罪]误判为[证据不足]"),
            }
            if pair in pair_map:
                info["root_cause"], info["detail"] = pair_map[pair]

        # 不起诉 -> 起诉
        elif ref in NON_PROSECUTION and pred in PROSECUTION:
            if ref == "相对不起诉":
                if not inp_mit and reas_mit_only:
                    info["root_cause"] = "INPUT_INFO_INSUFFICIENT"
                    mit_str = ",".join(reas_mit_only)
                    info["detail"] = "输入缺失关键从轻情节(" + mit_str + ")"
                elif inp_mit and think_mit:
                    info["root_cause"] = "REASONING_DECISION_DISCONNECT"
                    info["detail"] = "推理中识别了从轻情节但仍判起诉"
                elif inp_mit and not think_mit:
                    info["root_cause"] = "LLM_COMPREHENSION_FAILURE"
                    info["detail"] = "输入有从轻线索但模型未捕获"
                else:
                    info["root_cause"] = "INPUT_INFO_INSUFFICIENT"
                    info["detail"] = "输入中无从轻情节信息"
            elif ref == "存疑不起诉":
                if has_supplement and think_evi:
                    info["root_cause"] = "REASONING_DECISION_DISCONNECT"
                    info["detail"] = "识别了证据问题但仍判起诉"
                elif has_supplement:
                    info["root_cause"] = "LLM_SIGNAL_MISSED"
                    info["detail"] = "退补线索存在但模型未利用"
                else:
                    info["root_cause"] = "EVIDENCE_ASSESSMENT_FAILURE"
                    info["detail"] = "缺乏自主评估证据链完整性的能力"
            elif ref == "法定不起诉":
                if inp_stat and think_stat:
                    info["root_cause"] = "REASONING_DECISION_DISCONNECT"
                    info["detail"] = "识别了法定免责事由但仍判起诉"
                elif inp_stat:
                    info["root_cause"] = "LLM_COMPREHENSION_FAILURE"
                    info["detail"] = "输入有法定线索但模型未捕获"
                else:
                    info["root_cause"] = "STATUTORY_ASSESSMENT_FAILURE"
                    info["detail"] = "无法判断行为是否达到犯罪门槛"
        return info

    # =========================================================
    #  深层能力缺陷诊断（法律专业视角）
    # =========================================================
    def _generate_deep_deficit_section(self):
        """生成深层能力缺陷诊断章节 (Section 7-8)。

        从法律专业人士（检察官/律师）的视角诊断 LLM 在审查起诉
        各项核心法律推理能力上的缺失，每项均附定量证据。
        """
        L = []
        def p(t=""):
            L.append(t)

        gs = self.metrics
        total = gs.get("num_samples", len(self.all))
        decision_info = gs.get("result_metrics", {}).get("decision", {})
        per_class = decision_info.get("per_class", {})

        p("---")
        p("## 7. 深层能力缺陷诊断：从法律专业视角审视 LLM 的失败")
        p()
        p("上述章节从数据层面描述了错误的分布。本节从"
          "**检察官/律师**的专业视角追问更根本的问题："
          "**LLM 在哪些核心法律推理能力上存在缺陷，"
          "导致它系统性地无法胜任审查起诉工作？**")
        p()
        p("一个合格的检察官在审查起诉时，需要运用以下六项核心能力。"
          "以下逐项分析 LLM 在每项能力上的表现。")
        p()

        # ==========================================================
        #  7.1 犯罪构成要件的结构化判断
        # ==========================================================
        p("### 7.1 犯罪构成要件的结构化判断能力")
        p()
        p("**检察官如何做**: "
          "审查起诉的第一步是判断犯罪是否成立。检察官按照"
          "**犯罪构成四要件**逐一审查：犯罪主体（年龄、责任能力）、"
          "主观方面（故意/过失）、犯罪客体（侵犯的法益）、"
          "客观方面（行为、结果、因果关系）。"
          "四个要件**全部满足**才构成犯罪（→起诉或相对不起诉）；"
          "**任一不满足**即不构成犯罪（→法定不起诉）。"
          "这是一个**结构化的逻辑检验**，而非模糊的整体判断。")
        p()
        p("**LLM 的缺陷**: "
          "模型对案件事实进行**整体性的文本理解**，"
          "而非按要件逐一拆解检验。"
          "这导致它无法准确区分"
          "「犯罪成立但情节轻微」(相对不起诉) 与"
          "「犯罪根本不成立」(法定不起诉)。")
        p()

        # 数据：法定不起诉案件中，模型 think 中是否做了要件分析
        fd_cases = self.by_ref.get("法定不起诉", [])
        fd_total = len(fd_cases)
        ELEMENT_KW = [
            "犯罪构成", "构成要件", "犯罪主体", "主观方面", "客观方面",
            "犯罪客体", "主观恶性", "主观故意", "犯罪过失",
            "刑事责任年龄", "刑事责任能力",
        ]
        fd_elem_think = sum(
            1 for r in fd_cases
            if kw_found(think_block(r.get("raw_output", "")), ELEMENT_KW))
        fd_correct = sum(
            1 for r in fd_cases
            if r["prediction"]["decision"] == "法定不起诉")
        fd_to_xd = self.confusion.get(("法定不起诉", "相对不起诉"), 0)
        fd_to_p = self.confusion.get(("法定不起诉", "起诉"), 0)

        p("**定量证据:**")
        p()
        p("| 指标 | 数值 |")
        p("|------|------|")
        p("| 法定不起诉总量 | {} |".format(fd_total))
        p("| 法定不起诉准确率 | {:.1%} |".format(
            fd_correct / fd_total if fd_total else 0))
        p("| 误判为相对不起诉（有罪但轻微） | {} ({:.1%}) |".format(
            fd_to_xd, fd_to_xd / fd_total if fd_total else 0))
        p("| 误判为起诉 | {} ({:.1%}) |".format(
            fd_to_p, fd_to_p / fd_total if fd_total else 0))
        p("| 推理中进行要件分析的案例 | {}/{} ({:.1%}) |".format(
            fd_elem_think, fd_total,
            fd_elem_think / fd_total if fd_total else 0))
        p()

        # 法定不起诉案件中的具体原因：刑法第16条各项
        STAT_REASONS = {
            "情节显著轻微": "情节显著轻微危害不大",
            "不构成犯罪": "依法不构成犯罪",
            "追诉时效": "超过追诉时效",
            "未达到刑事责任年龄": "未达刑事责任年龄",
            "精神病": "不具备刑事责任能力",
        }
        fd_reason_cnt = Counter()
        for r in fd_cases:
            reas = r.get("raw_reasoning_and_decision", "")
            for k, label in STAT_REASONS.items():
                if k in reas:
                    fd_reason_cnt[label] += 1
        if fd_reason_cnt:
            p("**法定不起诉的具体法律原因（据检察官审查意见）:**")
            p()
            for label, n in fd_reason_cnt.most_common():
                p("- {}: {} 例".format(label, n))
            p()

        p("> **诊断**: 法定不起诉准确率极低，"
          "且 {:.1%} 被错误判为「相对不起诉」。"
          "这意味着模型在面对一个**根本不构成犯罪**的案件时，"
          "仍然倾向于认定犯罪成立——"
          "它缺乏按犯罪构成要件逐一检验的结构化判断能力，"
          "无法识别「要件不满足→犯罪不成立」的逻辑。"
          "仅 {:.1%} 的案例中模型的推理涉及了构成要件分析。".format(
              fd_to_xd / fd_total if fd_total else 0,
              fd_elem_think / fd_total if fd_total else 0))
        p()

        # ==========================================================
        #  7.2 证据裁判能力
        # ==========================================================
        p("### 7.2 证据裁判能力")
        p()
        p("**检察官如何做**: "
          "审查起诉的核心环节是**证据审查**。检察官需评估：(1) "
          "每项证据是否符合「三性」标准（合法性、关联性、真实性）；"
          "(2) 全案证据能否形成完整的**证据链**；"
          "(3) 证据链是否达到「排除合理怀疑」的证明标准。"
          "当证据链存在缺口、关键证据矛盾或不足以排除合理怀疑时，"
          "应作出存疑不起诉。这需要检察官**逐一审查每项证据**"
          "并评估它们之间的逻辑关系。")
        p()
        p("**LLM 的缺陷**: "
          "模型处理的是案件**事实叙述**文本，而非原始证据材料。"
          "事实叙述本身就是对证据的综合整理——"
          "读起来犯罪事实清楚、证据充分。"
          "模型无法穿透文本表面，去质疑「这段叙述背后的证据链完整吗？」"
          "「证人证言与物证一致吗？」「口供有没有被印证？」")
        p()

        doubt_cases = self.by_ref.get("存疑不起诉", [])
        d_total = len(doubt_cases)

        # 程序性退补信号 vs 准确率
        SUPP_KW = ["退回补充侦查", "退补", "补查重报", "退回", "补充侦查"]
        d_has_supp = [r for r in doubt_cases
                      if any(k in r.get("procedure", "") for k in SUPP_KW)]
        d_no_supp = [r for r in doubt_cases if r not in d_has_supp]
        supp_ok = sum(1 for r in d_has_supp
                      if r["prediction"]["decision"] == "存疑不起诉")
        no_supp_ok = sum(1 for r in d_no_supp
                         if r["prediction"]["decision"] == "存疑不起诉")

        # 输入中是否含有明确的证据问题描述
        EVI_INPUT_KW = ["证据不足", "事实不清", "无法认定", "不能证实",
                        "无法证实", "无法排除", "不能排除"]
        d_has_evi_inp = [r for r in doubt_cases
                         if kw_found(input_text(r), EVI_INPUT_KW)]
        d_no_evi_inp = [r for r in doubt_cases if r not in d_has_evi_inp]
        evi_inp_ok = sum(1 for r in d_has_evi_inp
                         if r["prediction"]["decision"] == "存疑不起诉")
        no_evi_inp_ok = sum(1 for r in d_no_evi_inp
                            if r["prediction"]["decision"] == "存疑不起诉")

        p("**定量证据 — 存疑不起诉案件的证据信号与准确率:**")
        p()
        p("| 条件 | 案例数 | 正确预测 | 准确率 |")
        p("|------|--------|---------|--------|")
        if d_has_supp:
            p("| 程序含退回补充侦查记录 | {} | {} | {:.1%} |".format(
                len(d_has_supp), supp_ok,
                supp_ok / len(d_has_supp)))
        if d_no_supp:
            p("| 程序无退补记录 | {} | {} | {:.1%} |".format(
                len(d_no_supp), no_supp_ok,
                no_supp_ok / len(d_no_supp) if d_no_supp else 0))
        if d_has_evi_inp:
            p("| 输入含明确证据不足描述 | {} | {} | {:.1%} |".format(
                len(d_has_evi_inp), evi_inp_ok,
                evi_inp_ok / len(d_has_evi_inp)))
        if d_no_evi_inp:
            p("| 输入无明确证据问题描述 | {} | {} | {:.1%} |".format(
                len(d_no_evi_inp), no_evi_inp_ok,
                no_evi_inp_ok / len(d_no_evi_inp) if d_no_evi_inp else 0))
        p("| **合计** | **{}** | **{}** | **{:.1%}** |".format(
            d_total,
            sum(1 for r in doubt_cases
                if r["prediction"]["decision"] == "存疑不起诉"),
            per_class.get("存疑不起诉", {}).get("accuracy", 0)))
        p()

        # 存疑不起诉的预测去向
        doubt_pred = Counter(
            r["prediction"]["decision"] or "（解析失败）"
            for r in doubt_cases)
        p("**存疑不起诉案件的预测去向:**")
        p()
        for dec, n in doubt_pred.most_common():
            mark = " (正确)" if dec == "存疑不起诉" else ""
            p("- -> {}: {} 例 ({:.1%}){}".format(
                dec, n, n / d_total, mark))
        p()

        p("> **诊断**: {:.1%} 的存疑不起诉案件被误判为起诉。"
          "即使程序中存在「退回补充侦查」这样的强信号"
          "（在刑事实务中，退补几乎等同于证据不足的明确标志），"
          "模型的准确率仍然极低。"
          "根本原因在于：**模型读到的是案件事实叙述，"
          "而非原始证据材料**。"
          "事实叙述经过了侦查机关的整理，读起来逻辑通顺、"
          "事实清楚——模型看到的是一个「表面完整的故事」，"
          "无法识别故事背后证据链的隐含漏洞。"
          "这是检察官最核心的专业能力之一，"
          "也是 LLM 最难以习得的能力。".format(
              doubt_pred.get("起诉", 0) / d_total))
        p()

        # ==========================================================
        #  7.3 犯罪情节的综合评价能力
        # ==========================================================
        p("### 7.3 犯罪情节的综合评价能力")
        p()
        p("**检察官如何做**: "
          "确认犯罪成立后，检察官需要对**犯罪情节**进行综合评价，"
          "以决定是起诉还是相对不起诉。评价维度包括："
          "犯罪性质（暴力犯罪还是过失犯罪）、"
          "社会危害性（实际造成的损害程度）、"
          "犯罪情节（手段是否恶劣、金额大小）、"
          "人身危险性（有无前科、是否可能再犯）。"
          "检察官需要将这些**相互对抗**的因素放在一起"
          "**综合权衡**——从轻情节不是「免死金牌」，"
          "认罪认罚也不是「不起诉通行证」——"
          "最终决定取决于犯罪严重程度是否压过从轻因素。")
        p()
        p("**LLM 的缺陷**: "
          "模型将从轻情节当作「不起诉特征」进行检测——"
          "看到认罪认罚、自首等关键词就倾向于不起诉。"
          "它无法理解：同样有认罪认罚情节，"
          "盗窃500元可以不起诉，盗窃50万则必须起诉。"
          "**情节的法律意义取决于与犯罪严重程度的相对比较**，"
          "而非情节本身的存在。")
        p()

        # 从轻线索 × 实际决定 × 准确率
        p_total = sum(len(self.by_ref[d]) for d in PROSECUTION)
        np_total = sum(len(self.by_ref[d]) for d in NON_PROSECUTION)

        weigh_table = {}
        for r in self.all:
            ref = r["reference"]["decision"]
            pred = r["prediction"]["decision"]
            ref_l1 = "起诉" if ref in PROSECUTION else "不起诉"
            pred_l1 = "起诉" if pred in PROSECUTION else "不起诉"
            n_mit = len(kw_found(input_text(r), MITIGATING_KW))
            g = str(n_mit) if n_mit <= 3 else "4+"
            key = (ref_l1, g)
            if key not in weigh_table:
                weigh_table[key] = [0, 0]
            weigh_table[key][0] += 1
            if ref_l1 == pred_l1:
                weigh_table[key][1] += 1

        p("**定量证据 — 从轻情节数量与二分类准确率:**")
        p()
        p("| 从轻情节数 | 起诉案例(样本/准确率) | 不起诉案例(样本/准确率) |")
        p("|-----------|---------------------|----------------------|")
        for g in ["0", "1", "2", "3", "4+"]:
            p_d = weigh_table.get(("起诉", g), [0, 0])
            np_d = weigh_table.get(("不起诉", g), [0, 0])
            p_s = "{} / {:.1%}".format(
                p_d[0], p_d[1] / p_d[0] if p_d[0] else 0)
            np_s = "{} / {:.1%}".format(
                np_d[0], np_d[1] / np_d[0] if np_d[0] else 0)
            p("| {} | {} | {} |".format(g, p_s, np_s))
        p()

        # 核心发现：认罪认罚在起诉/不起诉中的分布
        RRRF_KW = "认罪认罚"
        p_rrrf = sum(1 for r in self.all
                     if r["reference"]["decision"] in PROSECUTION
                     and RRRF_KW in input_text(r))
        np_rrrf = sum(1 for r in self.all
                      if r["reference"]["decision"] in NON_PROSECUTION
                      and RRRF_KW in input_text(r))
        # 起诉且有认罪认罚 → 模型是否还能判对
        p_rrrf_cases = [r for r in self.all
                        if r["reference"]["decision"] in PROSECUTION
                        and RRRF_KW in input_text(r)]
        p_rrrf_correct = sum(
            1 for r in p_rrrf_cases
            if r["prediction"]["decision"] in PROSECUTION)

        p("**「认罪认罚」悖论:**")
        p()
        p("| 条件 | 案例数 | 模型二分类准确率 |")
        p("|------|--------|----------------|")
        p("| 起诉 + 含认罪认罚 | {} ({:.1%}) | {:.1%} |".format(
            p_rrrf, p_rrrf / p_total if p_total else 0,
            p_rrrf_correct / len(p_rrrf_cases) if p_rrrf_cases else 0))
        np_rrrf_cases = [r for r in self.all
                         if r["reference"]["decision"] in NON_PROSECUTION
                         and RRRF_KW in input_text(r)]
        np_rrrf_correct = sum(
            1 for r in np_rrrf_cases
            if r["prediction"]["decision"] in NON_PROSECUTION)
        p("| 不起诉 + 含认罪认罚 | {} ({:.1%}) | {:.1%} |".format(
            np_rrrf, np_rrrf / np_total if np_total else 0,
            np_rrrf_correct / len(np_rrrf_cases)
            if np_rrrf_cases else 0))
        p_no_rrrf_cases = [r for r in self.all
                           if r["reference"]["decision"] in PROSECUTION
                           and RRRF_KW not in input_text(r)]
        p_no_rrrf_ok = sum(
            1 for r in p_no_rrrf_cases
            if r["prediction"]["decision"] in PROSECUTION)
        if p_no_rrrf_cases:
            p("| 起诉 + 无认罪认罚 | {} | {:.1%} |".format(
                len(p_no_rrrf_cases),
                p_no_rrrf_ok / len(p_no_rrrf_cases)))
        p()

        p("> **诊断**: 认罪认罚在起诉案例中的出现率 ({:.1%}) "
          "远高于不起诉案例 ({:.1%})——"
          "这与直觉相悖，却完全符合司法实践："
          "大量认罪认罚案件仍需起诉，"
          "因为犯罪性质严重或社会危害性大。"
          "但模型看到「认罪认罚」就大幅倾向于不起诉，"
          "导致含认罪认罚的起诉案例准确率仅 {:.1%}。"
          "**模型将「认罪认罚」当作「不起诉标签」**，"
          "而非检察官那样将其作为综合评价中的一个因素。".format(
              p_rrrf / p_total if p_total else 0,
              np_rrrf / np_total if np_total else 0,
              p_rrrf_correct / len(p_rrrf_cases) if p_rrrf_cases else 0))
        p()

        # ==========================================================
        #  7.4 程序性事实的法律意义解读
        # ==========================================================
        p("### 7.4 程序性事实的法律意义解读能力")
        p()
        p("**检察官如何做**: "
          "案件的程序信息蕴含丰富的法律意义。"
          "「退回补充侦查」意味着证据不足以支持起诉，"
          "是存疑不起诉的强信号；"
          "「取保候审」暗示人身危险性较低，利于不起诉方向；"
          "「变更强制措施」（由逮捕变更为取保）提示案件存在变化；"
          "「延长审查起诉期限」暗示案件复杂或证据有疑问。"
          "检察官能从程序信息中**解码出法律含义**。")
        p()
        p("**LLM 的缺陷**: "
          "模型将程序信息作为普通文本处理，"
          "未能解码其背后的法律含义——"
          "看到「退回补充侦查」只是读到了一段程序描述，"
          "而非理解「这个案件的证据有问题」。")
        p()

        # 程序信号对各类别的影响分析
        PROC_SIGNALS = {
            "退回补充侦查": ["退回补充侦查", "退回", "退补", "补查重报"],
            "取保候审": ["取保候审"],
            "逮捕": ["经.*批准逮捕", "执行逮捕"],
        }

        p("**定量证据 — 程序信号的法律预测价值:**")
        p()
        p("| 程序信号 | 存疑不起诉中出现率 | 起诉中出现率 | "
          "含信号时存疑不诉准确率 |")
        p("|----------|------------------|------------|---------------------|")

        for sig_name, sig_kws in PROC_SIGNALS.items():
            if sig_name == "逮捕":
                # 逮捕用正则
                d_sig = sum(
                    1 for r in doubt_cases
                    if re.search(r"批准逮捕|执行逮捕|逮捕",
                                 r.get("procedure", "")))
                p_sig = sum(
                    1 for r in self.by_ref.get("起诉", [])
                    if re.search(r"批准逮捕|执行逮捕|逮捕",
                                 r.get("procedure", "")))
            else:
                d_sig = sum(
                    1 for r in doubt_cases
                    if any(k in r.get("procedure", "") for k in sig_kws))
                p_sig = sum(
                    1 for r in self.by_ref.get("起诉", [])
                    if any(k in r.get("procedure", "") for k in sig_kws))

            d_sig_ok = 0
            if sig_name != "逮捕":
                d_sig_ok = sum(
                    1 for r in doubt_cases
                    if any(k in r.get("procedure", "") for k in sig_kws)
                    and r["prediction"]["decision"] == "存疑不起诉")
            else:
                d_sig_ok = sum(
                    1 for r in doubt_cases
                    if re.search(r"批准逮捕|执行逮捕|逮捕",
                                 r.get("procedure", ""))
                    and r["prediction"]["decision"] == "存疑不起诉")

            d_rate = d_sig / d_total if d_total else 0
            p_rate = p_sig / p_total if p_total else 0
            sig_acc = d_sig_ok / d_sig if d_sig else 0
            p("| {} | {:.1%} ({}/{}) | {:.1%} | {:.1%} |".format(
                sig_name, d_rate, d_sig, d_total, p_rate, sig_acc))
        p()

        p("> **诊断**: 「退回补充侦查」在存疑不起诉中的出现率"
          "远高于起诉案例——这在刑事实务中是最明确的证据不足信号。"
          "但即使存在这个强信号，模型的存疑不起诉预测准确率仍极低。"
          "模型将程序信息当作**背景噪声**而非**法律信号**。")
        p()

        # ==========================================================
        #  7.5 主观恶性与人身危险性评估
        # ==========================================================
        p("### 7.5 主观恶性与人身危险性评估能力")
        p()
        p("**检察官如何做**: "
          "起诉决定不仅取决于客观行为，还取决于行为人的"
          "**主观恶性**和**人身危险性**。主观恶性包括："
          "故意还是过失？预谋还是临时起意？"
          "人身危险性包括：是否有前科或累犯？"
          "是否初犯偶犯？案后是否真诚悔罪？"
          "检察官需要从案卷材料中**推断**这些主观状态——"
          "它们通常不会被直白地写出来，"
          "而是隐含在犯罪嫌疑人的行为模式中。")
        p()
        p("**LLM 的缺陷**: "
          "模型可以提取明确写出的信息（如「有前科」「系初犯」），"
          "但难以从客观行为中**推断**主观状态。"
          "例如，「案发后主动留在现场等待处理」"
          "隐含着自首意愿和较低的人身危险性；"
          "「多次实施同类犯罪」隐含着较高的人身危险性"
          "和主观恶性。")
        p()

        # person_info 中的主体信息分析
        SUBJ_EXPLICIT = {
            "初犯": ["初犯"],
            "偶犯": ["偶犯"],
            "前科/累犯": ["前科", "累犯", "曾因", "曾被"],
            "未成年": ["未成年", "未满十八"],
        }

        p("**定量证据 — 主体信息的出现率及其与预测准确率的关系:**")
        p()
        p("| 主体信息 | 不起诉中出现率 | 起诉中出现率 | "
          "含该信息时四分类准确率 |")
        p("|----------|-------------|------------|---------------------|")

        for label, kws in SUBJ_EXPLICIT.items():
            np_has = sum(
                1 for r in self.all
                if r["reference"]["decision"] in NON_PROSECUTION
                and any(k in input_text(r) for k in kws))
            p_has = sum(
                1 for r in self.all
                if r["reference"]["decision"] in PROSECUTION
                and any(k in input_text(r) for k in kws))
            all_has = [r for r in self.all
                       if any(k in input_text(r) for k in kws)]
            has_ok = sum(1 for r in all_has
                         if r["prediction"]["decision"]
                         == r["reference"]["decision"])
            np_rate = np_has / np_total if np_total else 0
            p_rate = p_has / p_total if p_total else 0
            has_acc = has_ok / len(all_has) if all_has else 0
            if np_has > 0 or p_has > 0:
                p("| {} | {:.1%} ({}) | {:.1%} ({}) | {:.1%} ({}/{}) |"
                  .format(label, np_rate, np_has, p_rate, p_has,
                          has_acc, has_ok, len(all_has)))
        p()

        # 行为推断分析：案后行为与准确率
        POSTACT_KW = {
            "主动投案/现场等候": ["主动投案", "在现场等候", "等候处理",
                              "自动投案"],
            "逃逸/潜逃": ["逃逸", "潜逃", "逃离", "在逃"],
            "赔偿谅解": ["赔偿", "谅解"],
            "拒不认罪": ["拒不认罪", "拒不供述", "翻供", "拒不交代"],
        }

        p("**案后行为（隐含主观恶性指标）与预测准确率:**")
        p()
        p("| 案后行为 | 出现数 | 四分类准确率 |")
        p("|----------|--------|------------|")
        for label, kws in POSTACT_KW.items():
            has = [r for r in self.all
                   if any(k in r.get("fact", "") for k in kws)]
            ok = sum(1 for r in has
                     if r["prediction"]["decision"]
                     == r["reference"]["decision"])
            if has:
                p("| {} | {} | {:.1%} |".format(
                    label, len(has), ok / len(has)))
        p()

        p("> **诊断**: 主观恶性和人身危险性的评估"
          "是检察官「定性判断」的核心。"
          "案后行为（如主动投案 vs 逃逸）"
          "和犯罪嫌疑人背景（如初犯 vs 累犯）"
          "包含了丰富的主观恶性信息，"
          "但模型无法将这些客观事实"
          "**转化为对主观状态的推断**。"
          "检察官看到「案发后在现场等候处理」，"
          "会自然推断嫌疑人有自首意愿和悔罪态度；"
          "模型只是将其视为一段普通的事实描述。")
        p()

        # ==========================================================
        #  7.6 刑事政策的理解与适用
        # ==========================================================
        p("### 7.6 刑事政策的理解与适用能力")
        p()
        p("**检察官如何做**: "
          "起诉决定不仅是法律判断，也是**政策判断**。"
          "当前中国刑事司法的核心政策包括：(1) "
          "「宽严相济」——根据犯罪性质和情节灵活把握起诉标准；"
          "(2) 「少捕慎诉慎押」——对轻微犯罪倾向于不起诉；"
          "(3) 「认罪认罚从宽」——对认罪认罚的嫌疑人从宽处理"
          "**但从宽不等于不诉**。"
          "检察官需要在具体案件中把握政策的适用边界。")
        p()
        p("**LLM 的缺陷**: "
          "模型缺乏对刑事政策的理解——"
          "它不知道「认罪认罚从宽」的边界在哪里，"
          "不知道哪些犯罪属于「当严则严」的范围。"
          "这导致它对政策性关键词做出机械反应"
          "而非灵活适用。")
        p()

        # 分罪名分析：看看不同犯罪类型的准确率差异
        charge_acc = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in self.all:
            charges = r["reference"].get("charges", [])
            correct = (r["prediction"]["decision"]
                       == r["reference"]["decision"])
            for ch in charges:
                charge_acc[ch]["total"] += 1
                if correct:
                    charge_acc[ch]["correct"] += 1

        # 只展示样本数 >= 50 的罪名
        sig_charges = [(ch, d) for ch, d in charge_acc.items()
                       if d["total"] >= 50]
        sig_charges.sort(key=lambda x: x[1]["correct"] / x[1]["total"])

        if sig_charges:
            p("**定量证据 — 不同罪名的四分类准确率"
              "（样本>=50）:**")
            p()
            p("| 罪名 | 样本数 | 准确率 |")
            p("|------|--------|--------|")
            for ch, d in sig_charges[:8]:
                p("| {} | {} | {:.1%} |".format(
                    ch, d["total"],
                    d["correct"] / d["total"]))
            p("| ... | | |")
            for ch, d in sig_charges[-3:]:
                p("| {} | {} | {:.1%} |".format(
                    ch, d["total"],
                    d["correct"] / d["total"]))
            p()

        # 预测分布偏差
        actual_dist = Counter()
        pred_dist = Counter()
        for r in self.all:
            actual_dist[r["reference"]["decision"]] += 1
            pred_dec = r["prediction"]["decision"] or "（解析失败）"
            pred_dist[pred_dec] += 1

        p("**预测分布偏差（模型的「政策偏好」）:**")
        p()
        p("| 决定类型 | 真实占比 | 预测占比 | 偏差 |")
        p("|----------|---------|---------|------|")
        for dt in DECISION_TYPES + ["（解析失败）"]:
            a = actual_dist.get(dt, 0)
            pr = pred_dist.get(dt, 0)
            p("| {} | {:.1%} ({}) | {:.1%} ({}) | {:+.1%} |".format(
                dt, a / total, a, pr / total, pr, (pr - a) / total))
        p()

        p("> **诊断**: 不同罪名间的准确率差异巨大，"
          "反映出模型缺乏对「宽严相济」政策的理解。"
          "在实务中，危险驾驶等轻罪的不起诉率较高，"
          "故意伤害等严重犯罪的起诉率较高——"
          "检察官根据犯罪性质灵活调整起诉标准。"
          "模型缺乏这种政策敏感性，"
          "对所有犯罪类型适用相同的决策逻辑。")
        p()

        # ==========================================================
        #  8. 核心结论
        # ==========================================================
        p("---")
        p("## 8. 核心结论：PDP 任务暴露的六项法律推理能力缺陷")
        p()
        p("| # | 法律能力 | 检察官的做法 | LLM 的缺陷 | 核心证据 |")
        p("|---|---------|-----------|----------|---------|")
        p("| 1 | 犯罪构成要件判断 "
          "| 按四要件逐一检验犯罪是否成立 "
          "| 整体性文本理解，无法按要件拆解 "
          "| 法定不诉准确率 {:.1%}，{:.1%} 被错判为相对不诉 |".format(
              fd_correct / fd_total if fd_total else 0,
              fd_to_xd / fd_total if fd_total else 0))
        p("| 2 | 证据裁判 "
          "| 审查每项证据的三性，评估证据链完整性 "
          "| 只能处理事实叙述文本，无法穿透到证据层 "
          "| 存疑不诉准确率 {:.1%} |".format(
              per_class.get("存疑不起诉", {}).get("accuracy", 0)))
        p("| 3 | 犯罪情节综合评价 "
          "| 权衡犯罪性质、社会危害性、情节、人身危险性 "
          "| 将从轻情节当作分类特征，无法进行相对权衡 "
          "| 含认罪认罚的起诉案例准确率 {:.1%} |".format(
              p_rrrf_correct / len(p_rrrf_cases) if p_rrrf_cases else 0))
        p("| 4 | 程序信息法律解读 "
          "| 从程序事实中解码法律含义 "
          "| 将程序信息当作背景文本 "
          "| 有退补信号仍误判为起诉 |")
        p("| 5 | 主观恶性评估 "
          "| 从客观行为推断主观状态 "
          "| 只能提取显式信息，无法推断隐含状态 "
          "| 案后行为信息未有效利用 |")
        p("| 6 | 刑事政策适用 "
          "| 根据犯罪类型灵活把握起诉标准 "
          "| 对所有案件适用同一决策逻辑 "
          "| 不同罪名间准确率差异巨大 |")
        p()

        p("### 为何这些能力对 LLM 构成根本挑战？")
        p()
        p("这六项能力缺陷指向一个共同的根源：**PDP 要求的是"
          "司法判断力（judicial judgment），而非文本理解力"
          "（text comprehension）**。")
        p()
        p("- **犯罪构成要件判断**要求的是"
          "**结构化的逻辑检验**——逐要件审查并得出是/否结论")
        p("- **证据裁判**要求的是"
          "**对信息可靠性的质疑能力**——"
          "不是理解文本说了什么，而是判断文本所述是否有充分证据支撑")
        p("- **情节综合评价**要求的是"
          "**多因素相对权衡能力**——"
          "在相互对抗的因素间做出比较判断")
        p("- **程序信息解读**要求的是"
          "**法律符号解码能力**——"
          "将程序事实转化为法律判断依据")
        p("- **主观恶性评估**要求的是"
          "**从客观到主观的推断能力**——"
          "从外在行为推断内心状态")
        p("- **刑事政策适用**要求的是"
          "**政策敏感性**——"
          "根据社会治理目标灵活调整法律适用标准")
        p()
        p("LLM 擅长的是从文本中提取信息、匹配模式、生成流畅的分析。"
          "但审查起诉所需的是**超越文本**的判断力——"
          "质疑证据、权衡情节、推断主观、把握政策。"
          "这些能力构成了检察官的专业核心，"
          "也正是当前 LLM 的能力边界。"
          "PDP 任务的价值在于，"
          "它精确地标定了这些能力边界的位置。")
        p()

        return L

    # =========================================================
    #  挑战分析报告
    # =========================================================
    def generate_challenge_report(self):
        L = []
        def p(t=""):
            L.append(t)

        gs = self.metrics
        total = gs.get("num_samples", len(self.all))
        decision_info = gs.get("result_metrics", {}).get("decision", {})
        per_class = decision_info.get("per_class", {})
        level1_acc = decision_info.get("level1_accuracy", 0)
        level1_per_class = decision_info.get("level1_per_class", {})
        art_f1 = gs.get("process_metrics", {}).get("articles", {}).get("f1", 0)
        parse_fail = gs.get("parse_fail_count", 0)
        n_correct = len(self.correct)
        n_loaded = len(self.all)

        shorts = {
            "起诉": "起诉",
            "相对不起诉": "相对不诉", "法定不起诉": "法定不诉",
            "存疑不起诉": "存疑不诉", "（解析失败）": "解析失败",
        }

        # ---- 1. 总览 ----
        p("# PDP 任务挑战性深度分析报告")
        p()
        model_name = gs.get("model", "N/A")
        p("模型: " + model_name + "  |  测试样本: " + str(total))
        p()
        p("## 1. 总体结果")
        p()
        p("| 指标 | 值 |")
        p("|------|-----|")
        acc_str = "{:.2%}".format(n_correct / n_loaded)
        p("| 决定准确率 | " + acc_str + " |")
        p("| 法条 F1 | {:.4f} |".format(art_f1))
        pf_str = "{} ({:.1%})".format(parse_fail, parse_fail / total)
        p("| 解析失败 | " + pf_str + " |")
        p()
        p("### 各类决定准确率")
        p()
        p("| 决定类型 | 样本数 | 准确率 |")
        p("|----------|--------|--------|")
        for dt in DECISION_TYPES:
            info = per_class.get(dt, {})
            p("| {} | {} | {:.2%} |".format(dt, info.get("count", 0), info.get("accuracy", 0)))
        p()

        # 二分类视角
        p("### 二分类视角（起诉 / 不起诉）")
        p()
        p("> 对被告人而言，哪种不起诉类型并不重要，重要的是**是否被起诉**。")
        p()
        p_info = level1_per_class.get("起诉", {})
        np_info = level1_per_class.get("不起诉", {})
        p("| 类别 | 样本数 | 准确率 |")
        p("|------|--------|--------|")
        p("| **整体** | **{}** | **{:.2%}** |".format(total, level1_acc))
        p("| 起诉 | {} | {:.2%} |".format(p_info.get("count", 0), p_info.get("accuracy", 0)))
        p("| 不起诉 | {} | {:.2%} |".format(np_info.get("count", 0), np_info.get("accuracy", 0)))
        p()
        if p_info.get("accuracy", 0) and np_info.get("accuracy", 0):
            gap = p_info["accuracy"] - np_info["accuracy"]
            if abs(gap) > 0.05:
                higher = "起诉" if gap > 0 else "不起诉"
                lower = "不起诉" if gap > 0 else "起诉"
                p("> 模型识别 **{}** 的能力（{:.1%}）显著优于 **{}**（{:.1%}），差距 {:.1f} 个百分点。".format(
                    higher, max(p_info["accuracy"], np_info["accuracy"]),
                    lower, min(p_info["accuracy"], np_info["accuracy"]),
                    abs(gap) * 100))
                p()
        p()

        # 混淆矩阵
        p("### 混淆矩阵")
        p()
        all_pred = DECISION_TYPES + ["（解析失败）"]
        header_parts = ["| 真实＼预测 |"]
        for d in all_pred:
            header_parts.append(" " + shorts.get(d, d) + " |")
        header_parts.append(" 合计 |")
        p("".join(header_parts))
        p("|" + "---|" * (len(all_pred) + 2))
        for ref_dt in DECISION_TYPES:
            vals = [self.confusion.get((ref_dt, pd), 0) for pd in all_pred]
            row_total = sum(vals)
            cells = " | ".join(str(v) if v else "-" for v in vals)
            p("| **" + shorts[ref_dt] + "** | " + cells + " | " + str(row_total) + " |")
        p()

        # ---- 2. 起诉->不起诉 ----
        p("---")
        p("## 2. 挑战一：起诉误判为不起诉")
        p()
        p("此类错误意味着模型对**应当追究刑责**的案件错误放纵。")
        p()

        p2np = {}
        for ref in PROSECUTION:
            for pred in NON_PROSECUTION:
                g = self._error_group(ref, pred)
                if g:
                    p2np[(ref, pred)] = g

        p("### 2.1 全景统计")
        p()
        p("| 真实决定 | 误判为 | 数量 | 占该类比例 |")
        p("|----------|--------|------|-----------|")
        total_p2np = 0
        for (ref, pred), g in sorted(p2np.items(), key=lambda x: -len(x[1])):
            ref_total = len(self.by_ref[ref])
            rate = "{:.1%}".format(len(g) / ref_total)
            p("| " + ref + " | " + pred + " | " + str(len(g)) + " | " + rate + " |")
            total_p2np += len(g)
        total_p = sum(len(self.by_ref[d]) for d in PROSECUTION)
        p("| **合计** | | **{}** | **{:.1%}** |".format(total_p2np, total_p2np / total_p))
        p()

        sub_idx = 1
        for (ref, pred), group in sorted(p2np.items(), key=lambda x: -len(x[1])):
            sub_idx += 1
            p("### 2.{} {} -> {} ({} 例)".format(sub_idx, shorts[ref], shorts[pred], len(group)))
            p()

            causes = Counter()
            for r in group:
                a = self._analyze_case(r)
                causes[a["root_cause"]] += 1

            cause_desc = {
                "MITIGATING_OVERWEIGHT": "从轻情节过度权重",
                "LLM_OVER_LENIENT": "无因宽大",
                "CRIME_THRESHOLD_MISJUDGE": "入罪门槛误判",
                "EVIDENCE_OVER_SKEPTICISM": "证据过度怀疑",
            }
            p("**根因分布:**")
            p()
            for c, n in causes.most_common():
                p("- {}: {} 例 -- {}".format(c, n, cause_desc.get(c, c)))
            p()

            if pred == "相对不起诉":
                mit_rates = Counter()
                for r in group:
                    for kw in kw_found(input_text(r), MITIGATING_KW):
                        mit_rates[kw] += 1
                if mit_rates:
                    p("**输入中迷惑性从轻线索:**")
                    p()
                    for kw, n in mit_rates.most_common(10):
                        p("- {}: {}/{} ({:.1%})".format(kw, n, len(group), n / len(group)))
                    p()
                    p("> 这些案例虽含从轻情节，但犯罪严重程度足以起诉。模型难以判断从轻情节够不够抵消犯罪严重程度。")
                    p()

            if pred == "法定不起诉":
                p("> 模型将已构成犯罪的行为误判为不构成犯罪，反映出对入罪门槛的掌握不足。")
                p()

            p("**典型案例:**")
            p()
            for i, r in enumerate(group[:2]):
                a = self._analyze_case(r)
                p("案例 {} [ID: {}]".format(i + 1, r["id"]))
                p("- 根因: " + a["root_cause"] + " -- " + a["detail"])
                p("- 事实: " + r.get("fact", "")[:250] + "...")
                p("- 检察官: " + r.get("raw_reasoning_and_decision", "")[:200] + "...")
                p()

        # ---- 3. 不起诉互混 ----
        p("---")
        p("## 3. 挑战二：三类不起诉之间的混淆")
        p()
        p("三种不起诉类型对应截然不同的法律含义：")
        p()
        p("| 类型 | 核心含义 | 前提 | 法律依据 |")
        p("|------|---------|------|---------|")
        p("| 相对不起诉 | 有罪但免诉 | 犯罪成立+情节轻微+可免刑 | 刑诉法第177条第2款 |")
        p("| 法定不起诉 | 无罪/免责 | 不构成犯罪或法定免责事由 | 刑诉法第16条 |")
        p("| 存疑不起诉 | 证据不足 | 证据不足以支持起诉 | 刑诉法第175条第4款 |")
        p()
        p("> 混淆这三者 = 无法区分「有罪免刑」「无罪」「证不够」三个完全不同的法律状态。")
        p()

        np_conf = {}
        for ref in NON_PROSECUTION:
            for pred in NON_PROSECUTION:
                if ref != pred:
                    g = self._error_group(ref, pred)
                    if g:
                        np_conf[(ref, pred)] = g

        p("### 3.1 全景统计")
        p()
        p("| 真实 | 误判为 | 数量 | 占该类比例 |")
        p("|------|--------|------|-----------|")
        total_np_conf = 0
        for (ref, pred), g in sorted(np_conf.items(), key=lambda x: -len(x[1])):
            ref_total = len(self.by_ref[ref])
            p("| {} | {} | {} | {:.1%} |".format(ref, pred, len(g), len(g) / ref_total))
            total_np_conf += len(g)
        p("| **合计** | | **{}** | |".format(total_np_conf))
        p()

        pair_meta = {
            ("相对不起诉", "法定不起诉"): (
                "[有罪免刑] -> [无罪]",
                "最核心的法律概念混淆。相对不起诉承认犯罪成立但情节轻微可免刑；法定不起诉认为根本不构成犯罪。边界在于**犯罪构成要件是否满足**。"
            ),
            ("法定不起诉", "相对不起诉"): (
                "[无罪] -> [有罪免刑]",
                "将不构成犯罪的行为错误认定为构成犯罪但情节轻微。模型倾向于「宁可认罪再宽大」而非正确判断「不构成犯罪」。"
            ),
            ("相对不起诉", "存疑不起诉"): (
                "[有罪免诉] -> [证据不足]",
                "错误地质疑了充分证据，将有罪案件判为证据不足。"
            ),
            ("存疑不起诉", "相对不起诉"): (
                "[证据不足] -> [有罪免诉]",
                "在证据不足时错误认定犯罪成立。缺乏评估证据链完整性的元认知能力——案件叙述表面看犯罪成立，但证据链有隐含漏洞。"
            ),
            ("存疑不起诉", "法定不起诉"): (
                "[证据不足] -> [不构成犯罪]",
                "混淆「证不够」与「法不罚」，前者是证据问题，后者是法律问题。"
            ),
            ("法定不起诉", "存疑不起诉"): (
                "[不构成犯罪] -> [证据不足]",
                "将法律层面的「不罚」混淆为事实层面的「证不够」。"
            ),
        }

        sub_idx = 1
        for (ref, pred), g in sorted(np_conf.items(), key=lambda x: -len(x[1])):
            sub_idx += 1
            title, desc = pair_meta.get((ref, pred), (ref + "->" + pred, ""))
            p("### 3.{} {} ({} 例)".format(sub_idx, title, len(g)))
            p()
            p(desc)
            p()

            causes = Counter()
            for r in g:
                a = self._analyze_case(r)
                causes[a["root_cause"]] += 1
            if causes:
                parts = ", ".join("{}({})".format(c, n) for c, n in causes.most_common())
                p("**根因:** " + parts)
                p()

            if g:
                ex = g[0]
                p("**典型案例** [ID: {}]".format(ex["id"]))
                p("- 事实: " + ex.get("fact", "")[:250] + "...")
                p("- 检察官: " + ex.get("raw_reasoning_and_decision", "")[:200] + "...")
                p()

        # ---- 4. 信息差距 ----
        p("---")
        p("## 4. 深层原因：输入信息差距")
        p()
        p("### 4.1 相对不起诉的信息鸿沟")
        p()
        p("| 从轻情节 | 输入中出现 | 审查意见中出现 | 仅在审查意见 | 缺失率 |")
        p("|----------|-----------|-------------|-------------|--------|")
        rnp = self.by_ref.get("相对不起诉", [])
        rnp_total = len(rnp)
        for kw in MITIGATING_KW:
            inp_n = sum(1 for r in rnp if kw in input_text(r))
            reas_n = sum(1 for r in rnp if kw in r.get("raw_reasoning_and_decision", ""))
            only_n = sum(1 for r in rnp if kw in r.get("raw_reasoning_and_decision", "") and kw not in input_text(r))
            if reas_n > 0 or inp_n > 0:
                gap = only_n / reas_n if reas_n else 0
                p("| {} | {} | {} | {} | {:.1%} |".format(kw, inp_n, reas_n, only_n, gap))
        p()

        p("### 4.2 线索数 vs 相对不起诉准确率")
        p()
        clue_acc = defaultdict(lambda: {"t": 0, "c": 0})
        for r in rnp:
            n = len(kw_found(input_text(r), MITIGATING_KW))
            g = str(n) if n <= 3 else "4+"
            clue_acc[g]["t"] += 1
            if r["prediction"]["decision"] == "相对不起诉":
                clue_acc[g]["c"] += 1
        p("| 线索数 | 样本量 | 正确 | 准确率 |")
        p("|--------|--------|------|--------|")
        for g in ["0", "1", "2", "3", "4+"]:
            v = clue_acc[g]
            if v["t"]:
                p("| {} | {} | {} | {:.1%} |".format(g, v["t"], v["c"], v["c"] / v["t"]))
        p()

        p("### 4.3 从轻线索悖论")
        p()
        p_mit = sum(1 for r in self.all if r["reference"]["decision"] in PROSECUTION and kw_found(input_text(r), MITIGATING_KW))
        np_mit = sum(1 for r in self.all if r["reference"]["decision"] in NON_PROSECUTION and kw_found(input_text(r), MITIGATING_KW))
        tp = sum(len(self.by_ref[d]) for d in PROSECUTION)
        tnp = sum(len(self.by_ref[d]) for d in NON_PROSECUTION)
        p("- 起诉案例含从轻线索: {}/{} ({:.1%})".format(p_mit, tp, p_mit / tp))
        p("- 不起诉案例含从轻线索: {}/{} ({:.1%})".format(np_mit, tnp, np_mit / tnp))
        p()
        p("> **核心发现**: 起诉案例中从轻线索出现率高于不起诉案例。PDP 的本质难度在于**相对权衡**而非特征检测。")
        p()

        # ---- 5. 存疑不起诉 ----
        p("---")
        p("## 5. 专题：存疑不起诉——最难的子任务")
        p()
        doubt = self.by_ref.get("存疑不起诉", [])
        d_total = len(doubt)
        d_correct = sum(1 for r in doubt if r["prediction"]["decision"] == "存疑不起诉")
        p("- 样本数: {}，正确率: {}/{} ({:.1%})".format(d_total, d_correct, d_total, d_correct / d_total))
        p()
        d_pred = Counter(r["prediction"]["decision"] or "（解析失败）" for r in doubt)
        p("**预测分布:**")
        p()
        for dec, n in d_pred.most_common():
            p("- {}: {} ({:.1%})".format(dec, n, n / d_total))
        p()
        d_supp = sum(1 for r in doubt if any(k in r.get("procedure", "") for k in ["退回", "补充侦查", "退补", "补查重报"]))
        d_evi_reas = sum(1 for r in doubt if kw_found(r.get("raw_reasoning_and_decision", ""), EVIDENCE_KW))
        d_evi_inp = sum(1 for r in doubt if kw_found(input_text(r), EVIDENCE_KW))
        p("- 程序含退补记录: {}/{} ({:.1%})".format(d_supp, d_total, d_supp / d_total))
        p("- 审查意见含证据不足关键词: {}/{}".format(d_evi_reas, d_total))
        p("- 输入含证据不足关键词: {}/{}".format(d_evi_inp, d_total))
        p()
        p("> 案件事实表面上犯罪成立，但证据链有**隐含漏洞**。模型需要**元认知能力**：评估事实叙述本身的可靠性。")
        p()
        wrong = [r for r in doubt if r["prediction"]["decision"] != "存疑不起诉"]
        for i, r in enumerate(wrong[:2]):
            p("**案例 {}** [ID: {}]".format(i + 1, r["id"]))
            p("- 预测: " + str(r["prediction"]["decision"]))
            p("- 事实: " + r.get("fact", "")[:300] + "...")
            p("- 检察官: " + r.get("raw_reasoning_and_decision", "")[:250] + "...")
            p()

        # ---- 6. 法条 ----
        p("---")
        p("## 6. 挑战三：三法联合法条引用")
        p()
        p("法条整体 F1: **{:.4f}**".format(art_f1))
        p()
        source_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        arabic_n = 0
        for r in self.all:
            pa = set(r["prediction"].get("relevant_articles", []))
            ra = set(r["reference"].get("relevant_articles", []))
            if any(re.search(r"第\d+条", a) for a in pa):
                arabic_n += 1
            for pfx in ["cl:", "cpl:", "cpr:"]:
                ps = {a for a in pa if a.startswith(pfx)}
                rs = {a for a in ra if a.startswith(pfx)}
                tp_n = len(ps & rs)
                source_stats[pfx]["tp"] += tp_n
                source_stats[pfx]["fp"] += len(ps) - tp_n
                source_stats[pfx]["fn"] += len(rs) - tp_n
        names = {"cl:": "刑法", "cpl:": "刑事诉讼法", "cpr:": "刑事诉讼规则"}
        p("| 法律来源 | Precision | Recall | F1 |")
        p("|----------|-----------|--------|----|")
        for pfx, s in source_stats.items():
            t, fp, fn = s["tp"], s["fp"], s["fn"]
            pr = t / (t + fp) if t + fp else 0
            rc = t / (t + fn) if t + fn else 0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
            p("| {} | {:.4f} | {:.4f} | {:.4f} |".format(names.get(pfx, pfx), pr, rc, f1))
        p()
        p("- 使用阿拉伯数字而非中文数字: {} 例".format(arabic_n))
        p()
        p("> 程序法条文的引用**依赖决定类型的正确判断**——必须先判对决定才能选对程序条文。")
        p()

        # ---- 7-8. 深层能力缺陷诊断 & 核心结论 ----
        L.extend(self._generate_deep_deficit_section())

        return "\n".join(L)

    # =========================================================
    #  论文引言
    # =========================================================
    def generate_paper_intro(self):
        L = []
        def p(t=""):
            L.append(t)

        gs = self.metrics
        total = gs.get("num_samples", len(self.all))
        decision_info = gs.get("result_metrics", {}).get("decision", {})
        per_class = decision_info.get("per_class", {})
        level1_acc = decision_info.get("level1_accuracy", 0)
        level1_per_class = decision_info.get("level1_per_class", {})
        art_f1 = gs.get("process_metrics", {}).get("articles", {}).get("f1", 0)
        n_correct = len(self.correct)
        n_loaded = len(self.all)
        acc = n_correct / n_loaded if n_loaded else 0
        p_l1_acc = level1_per_class.get("起诉", {}).get("accuracy", 0)
        np_l1_acc = level1_per_class.get("不起诉", {}).get("accuracy", 0)
        doubt_acc = per_class.get("存疑不起诉", {}).get("accuracy", 0)
        p_mit = sum(1 for r in self.all if r["reference"]["decision"] in PROSECUTION and kw_found(input_text(r), MITIGATING_KW))
        np_mit = sum(1 for r in self.all if r["reference"]["decision"] in NON_PROSECUTION and kw_found(input_text(r), MITIGATING_KW))
        tp = sum(len(self.by_ref[d]) for d in PROSECUTION)
        tnp = sum(len(self.by_ref[d]) for d in NON_PROSECUTION)
        total_p2np = sum(len(self._error_group(ref, pred)) for ref in PROSECUTION for pred in NON_PROSECUTION)
        total_np_conf = sum(len(self._error_group(ref, pred)) for ref in NON_PROSECUTION for pred in NON_PROSECUTION if ref != pred)

        p("# Prosecution Decision Prediction: A Novel Challenge for Legal NLP")
        p()
        p("---")
        p()
        p("## 1. Introduction")
        p()
        p("法律人工智能（Legal AI）是 NLP 的重要应用方向。以法律判决预测（Legal Judgment Prediction, LJP）为代表的任务受到广泛关注 [Zhong et al., 2018; Xu et al., 2020; Feng et al., 2022]。LJP 基于法院视角，给定案件事实预测法官裁判结果——核心子任务包括罪名预测和量刑预测。CAIL [Xiao et al., 2018] 等基准推动了该领域发展。")
        p()
        p("然而，从刑法理论的犯罪构成体系审视，LJP 本质上聚焦于**违法性**（Rechtswidrigkeit）层面：罪名预测判断行为违反了哪条法律，量刑预测评估违法的严重程度。语言模型在这一层面表现尚可——给定结构化的案件事实，通过模式匹配即可建立「行为→罪名」「情节→刑期」的映射。但犯罪认定的完整逻辑还包括另一关键维度——**有责性**（Schuld）：行为人是否具有主观恶意？是否超过追诉时效？是否构成正当防卫或紧急避险？是否具备刑事责任能力？LJP 任务的数据天然回避了这一维度——进入审判的案件，责任认定问题已在起诉阶段被检察官解决。")
        p()
        p("这意味着 LJP **并未触及语言模型在责任认定上的脆弱性**。现有研究几乎完全聚焦于审判阶段，忽视了一个同样关键的前置环节——**审查起诉阶段**。在中国刑事司法中，检察机关的**公诉决定**直接决定案件是否进入审判程序，每年涉及数十万件案件。正是在这一阶段，检察官必须综合评估违法性与有责性，作出起诉或不起诉的决定。")
        p()
        p("我们提出**公诉决定预测**（Prosecution Decision Prediction, PDP），"
          "一个全新的法律 NLP 任务。PDP 与 LJP 的本质区别在于：PDP 要求模型"
          "**同时评估违法性与有责性**，并在二者之间进行权衡。"
          "三种不起诉类型分别对应有责性判断的三种结论——"
          "有罪但责任轻微可免刑（相对不起诉）、"
          "不具备刑事责任（法定不起诉）、"
          "责任无法证实（存疑不起诉）。")
        p()
        p("然而，违法性与有责性的理论框架仍不足以完整揭示 PDP 的独特挑战。"
          "从检察官的实务视角审视，审查起诉的本质是"
          "**公诉裁量权的行使**（Prosecutorial Discretion）——"
          "一个包含三个递进层次的审查过程：")
        p()
        p("1. **能不能诉？**（规范层）——犯罪构成四要件是否全部满足？"
          "任一要件不满足即不构成犯罪 → **法定不起诉**（刑诉法第16条）")
        p("2. **够不够诉？**（证据层）——证据链是否完整，"
          "是否达到「事实清楚、证据确实充分」的起诉标准？"
          "不够 → **存疑不起诉**（刑诉法第175条）")
        p("3. **该不该诉？**（裁量层）——犯罪成立且证据充分后，"
          "起诉是否有必要？犯罪严重性与从轻情节的相对权衡——"
          "不必要 → **相对不起诉**（刑诉法第177条）；三层均通过 → **起诉**")
        p()
        p("LJP 仅涉及第一层的简化版本（行为→罪名的映射）；"
          "PDP 要求模型同时驾驭三个层次的判断——"
          "从规范推理（Legal Subsumption）到证据评价（Evidence Evaluation）"
          "再到价值裁量（Prosecutorial Discretion）——"
          "这构成对 LLM 法律推理能力的全方位考验。")
        p()

        p("## 2. Task Definition")
        p()
        p("**输入**: 犯罪嫌疑人信息 (`person_info`)、案件程序 (`procedure`)、案件事实 (`fact`)")
        p()
        p("**输出**:")
        p("1. `decision` -- 二级分类公诉决定:")
        p("   - 第一级：起诉 / 不起诉")
        p("   - 第二级：起诉、相对不起诉 (刑诉法第177条)、法定不起诉 (刑诉法第16条)、存疑不起诉 (刑诉法第175条)")
        p("2. `relevant_articles` -- 适用法条（刑法+刑诉法+诉讼规则）")
        p()
        p("| 维度 | LJP | PDP |")
        p("|------|-----|-----|")
        p("| 决策主体 | 法官 | 检察官 |")
        p("| 诉讼阶段 | 审判 | 审查起诉 |")
        p("| 数据来源 | 裁判文书 | 检察文书(12309) |")
        p("| 决定类型 | 有罪/无罪+量刑 | 二级四分类（起诉/3类不起诉） |")
        p("| 法条范围 | 刑法 | 刑法+刑诉法+诉讼规则 |")
        p("| 认定维度 | 违法性（罪名+量刑） | 违法性+有责性联合判断 |")
        p("| 审查层次 | 事实→罪名→量刑（法律适用） | 能不能诉→够不够诉→该不该诉（司法裁量） |")
        p()

        # --- Section 3 额外统计 ---
        SUPP_KW_INTRO = ["退回补充侦查", "退补", "补查重报", "退回", "补充侦查"]
        fd_acc = per_class.get("法定不起诉", {}).get("accuracy", 0)
        xd_acc = per_class.get("相对不起诉", {}).get("accuracy", 0)
        fd_total_n = per_class.get("法定不起诉", {}).get("count", 0)
        doubt_total_n = per_class.get("存疑不起诉", {}).get("count", 0)
        doubt_cases_intro = self.by_ref.get("存疑不起诉", [])
        doubt_to_p = self.confusion.get(("存疑不起诉", "起诉"), 0)
        doubt_with_supp = sum(1 for r in doubt_cases_intro
                              if kw_found(r.get("procedure", ""), SUPP_KW_INTRO))
        p_cases_intro = self.by_ref.get("起诉", [])
        p_with_supp = sum(1 for r in p_cases_intro
                          if kw_found(r.get("procedure", ""), SUPP_KW_INTRO))
        fd_to_xd = self.confusion.get(("法定不起诉", "相对不起诉"), 0)
        p_with_rr = sum(1 for r in p_cases_intro if "认罪认罚" in input_text(r))
        np_cases_intro = [r for r in self.all
                          if r["reference"]["decision"] in NON_PROSECUTION]
        np_with_rr = sum(1 for r in np_cases_intro if "认罪认罚" in input_text(r))

        p("## 3. Key Challenges: LLM 的三大法律能力缺失")
        p()
        p("对 Qwen3-4B 在 {} 个测试样本上的评估显示："
          "即使简化为起诉/不起诉二分类，准确率也仅 **{:.1%}**"
          "（起诉 {:.1%}，不起诉 {:.1%}）；"
          "细化到四分类准确率更降至 **{:.1%}**。".format(
            total, level1_acc, p_l1_acc, np_l1_acc, acc))
        p()
        p("PDP 的三层审查结构揭示了 LLM 三项根本性的法律能力缺失，"
          "每一项精确对应一种不起诉类型的预测失败：")
        p()
        p("| 审查层次 | 核心问题 | 所需法律能力 | 失败→对应决定 | 模型准确率 |")
        p("|---------|---------|------------|-------------|-----------|")
        p("| 规范层 | 能不能诉？ | 规范涵摄 | 法定不起诉 | {:.1%} |".format(fd_acc))
        p("| 证据层 | 够不够诉？ | 证据评价 | 存疑不起诉 | {:.1%} |".format(doubt_acc))
        p("| 裁量层 | 该不该诉？ | 起诉裁量 | 相对不起诉 | {:.1%} |".format(xd_acc))
        p()

        p("### 3.1 证据评价能力缺失——「够不够诉」判不准")
        p()
        p("**检察官如何做**: "
          "审查每一项证据的「三性」（合法性、关联性、真实性），"
          "评估证据链是否完整，判断是否达到"
          "「事实清楚、证据确实充分」的起诉标准。"
          "看到「退回补充侦查」两次仍无法补充的，"
          "即知证据存在重大缺陷。")
        p()
        p("**LLM 的缺陷**: "
          "模型读到的是案件事实的**文本叙述**，不是证据本身。"
          "它看到「张三于某日盗窃某物」，"
          "看不到这个事实背后的证据链有多脆弱。"
          "即使文本中明确出现「退回补充侦查」的程序信号，"
          "模型也无法将其解读为「证据不足」的法律含义。")
        p()
        p("**定量证据**: "
          "存疑不起诉准确率仅 **{:.1%}**，"
          "{} 例中 {} 例 ({:.1%}) 被误判为起诉。"
          "「退回补充侦查」在存疑不起诉案件中出现率 {:.1%}，"
          "而在起诉案件中仅 {:.1%}"
          "——如此强烈的程序信号几乎被模型完全忽视。".format(
            doubt_acc,
            doubt_total_n, doubt_to_p,
            doubt_to_p / doubt_total_n if doubt_total_n else 0,
            doubt_with_supp / len(doubt_cases_intro) if doubt_cases_intro else 0,
            p_with_supp / len(p_cases_intro) if p_cases_intro else 0))
        p()

        p("### 3.2 规范涵摄能力缺失——「能不能诉」判不准")
        p()
        p("**检察官如何做**: "
          "通过法律三段论——"
          "大前提（法律规范）+ 小前提（案件事实）→ 结论。"
          "对犯罪构成四要件逐一检验："
          "主体（刑事责任能力）、主观方面（故意/过失）、"
          "客体（侵犯的法益）、客观方面（行为与结果）。"
          "**任何一个要件不满足 = 不构成犯罪 = 法定不起诉。**")
        p()
        p("**LLM 的缺陷**: "
          "模型不做要件拆解，而是整体模式匹配。"
          "它看到案件有从轻情节（初犯、退赔、谅解）"
          "就倾向输出「相对不起诉」，"
          "而不会先问「犯罪是否成立」这个前提问题。"
          "它把「不构成犯罪」（法定不诉）与"
          "「构成犯罪但情节轻微」（相对不诉）混为一谈——"
          "因为它缺乏「先判断犯罪是否成立，"
          "再判断情节轻重」的递进式推理结构。")
        p()
        p("**定量证据**: "
          "法定不起诉准确率仅 **{:.1%}**，"
          "其中 {} 例 ({:.1%}) 被误判为相对不起诉——"
          "恰好说明模型跳过了「犯罪是否成立」的前置判断，"
          "直接落入「情节轻重」的判断。".format(
            fd_acc,
            fd_to_xd,
            fd_to_xd / fd_total_n if fd_total_n else 0))
        p()

        p("### 3.3 起诉裁量能力缺失——「该不该诉」判不准")
        p()
        p("**检察官如何做**: "
          "犯罪成立、证据充分之后，还要判断「起诉是否有必要」。"
          "这是一个**开放的价值判断**——"
          "权衡犯罪的严重程度（罪名性质、社会危害性、犯罪数额）"
          "与从轻因素（认罪认罚、初犯偶犯、退赔谅解、自首立功）"
          "的相对强度。"
          "同样是认罪认罚+退赔谅解，盗窃 2000 元可以不诉，"
          "盗窃 50 万元必须起诉——"
          "区别在于犯罪严重性的「底线」因罪名而异。")
        p()
        p("**LLM 的缺陷**: "
          "模型把从轻情节当作「不起诉的分类特征」，"
          "而非放在天平上与犯罪严重性做**相对权衡**。"
          "它无法理解「认罪认罚从宽 ≠ 免罚」、"
          "「情节轻微」在不同罪名下的标准完全不同。"
          "本质上，**裁量是一种价值判断，"
          "而 LLM 只能做模式匹配**。")
        p()
        p("**定量证据**: "
          "起诉案件中 {:.1%} 含认罪认罚，"
          "不起诉案件中仅 {:.1%}"
          "——认罪认罚在起诉案件中的出现率**远高于**不起诉，"
          "根本不是区分起诉与否的决定性特征，"
          "但模型显然将其当作了关键不起诉信号。"
          "在 {} 例起诉案件中，{} 例 ({:.1%}) "
          "被错误判为不起诉。".format(
            p_with_rr / len(p_cases_intro) if p_cases_intro else 0,
            np_with_rr / len(np_cases_intro) if np_cases_intro else 0,
            tp, total_p2np,
            total_p2np / tp if tp else 0))
        p()

        p("此外，PDP 要求同时引用刑法、刑事诉讼法和刑事诉讼规则三部法律"
          "（法条 F1 仅 {:.4f}），"
          "且关键从轻情节在输入文本中往往隐含而非显式呈现——"
          "模型需**推断**而非**提取**，"
          "进一步加剧了上述三项能力缺陷的影响。".format(art_f1))
        p()

        p("> **根本洞察**: LJP 测试的是 LLM 的**法律适用能力**"
          "（把事实涵摄到法条上）；"
          "PDP 测试的是 LLM 的**司法裁量能力**"
          "（像检察官一样做多层次的综合判断）。"
          "LLM 的三大能力缺失——证据评价、规范涵摄、起诉裁量——"
          "分别导致了存疑不起诉、法定不起诉、相对不起诉的预测失败，"
          "共同指向一个根本问题："
          "**PDP 要求的是司法判断力（Judicial Judgment），"
          "而非文本理解力（Text Comprehension）。**")
        p()

        p("## 4. Contributions")
        p()
        p("1. **新任务**: 首次定义 PDP，将法律 NLP 从审判扩展至审查起诉阶段")
        p("2. **新数据集**: 10,000+ 真实检察文书，两级四分类+三法联合法条标注")
        p("3. **新挑战**: 从检察官实务视角揭示 LLM 在证据评价、规范涵摄、起诉裁量三大法律能力上的系统性缺陷")
        p("4. **基线方法**: GRPO + 多维组合奖励函数")
        p()

        p("## 5. Dataset & Preliminary Results")
        p()
        p("| 统计项 | 值 |")
        p("|--------|-----|")
        p("| 测试集 | {} |".format(total))
        for dt in DECISION_TYPES:
            info = per_class.get(dt, {})
            p("| - {} | {} |".format(dt, info.get("count", 0)))
        p()
        p("| Metric | Qwen3-4B |")
        p("|--------|----------|")
        p("| Binary Accuracy (起诉/不起诉) | {:.2%} |".format(level1_acc))
        p("| - 起诉 | {:.2%} |".format(p_l1_acc))
        p("| - 不起诉 | {:.2%} |".format(np_l1_acc))
        p("| 4-Class Accuracy | {:.2%} |".format(acc))
        for dt in DECISION_TYPES:
            info = per_class.get(dt, {})
            p("| - {} | {:.2%} |".format(dt, info.get("accuracy", 0)))
        p("| Article F1 | {:.4f} |".format(art_f1))
        p()
        p("结果表明 PDP 远非当前 LLM 能轻松解决的任务，为法律 AI 社区提供了有意义的新基准。")
        p()
        return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description="PDP 挑战性深度分析 (v2)")
    parser.add_argument("--result-dir", default="results/Qwen3-4B_test_20260305_185907")
    parser.add_argument("--data-path", default="data/PDP_dataset/test/dataset.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = args.output_dir or args.result_dir

    sys.stdout.reconfigure(encoding="utf-8")

    print("加载评估结果: " + args.result_dir)
    correct, errors, metrics = load_all_records(args.result_dir)
    print("  正确: {}, 错误: {}".format(len(correct), len(errors)))

    print("加载原始数据集: " + args.data_path)
    ds = load_dataset(args.data_path)
    print("  原始记录: {}".format(len(ds)))

    ana = Analyzer(correct, errors, metrics, ds)

    print("\n生成挑战分析报告...")
    rpt = ana.generate_challenge_report()
    p1 = os.path.join(out, "challenge_analysis_report.md")
    with open(p1, "w", encoding="utf-8") as f:
        f.write(rpt)
    print("  -> " + p1)

    print("生成论文引言草稿...")
    intro = ana.generate_paper_intro()
    p2 = os.path.join(out, "pdp_paper_introduction.md")
    with open(p2, "w", encoding="utf-8") as f:
        f.write(intro)
    print("  -> " + p2)

    n = len(correct) + len(errors)
    print("\n" + "=" * 50)
    print("  决定准确率: {:.2%}".format(len(correct) / n))
    print("  法条 F1: {:.4f}".format(metrics.get("process_metrics", {}).get("articles", {}).get("f1", 0)))
    print("  已生成 2 份文档")
    print("=" * 50)


if __name__ == "__main__":
    main()
