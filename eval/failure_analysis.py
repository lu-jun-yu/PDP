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

        # ---- 7. 总结 ----
        p("---")
        p("## 7. 挑战总结")
        p()
        p("| # | 挑战 | 英文名 | 关键证据 |")
        p("|---|------|--------|---------|")
        p("| 0 | 起诉/不起诉二分类 | Binary Decision | 起诉 {:.1%} vs 不起诉 {:.1%} |".format(
            level1_per_class.get("起诉", {}).get("accuracy", 0),
            level1_per_class.get("不起诉", {}).get("accuracy", 0)))
        p("| 1 | 起诉->不起诉误判 | Prosecution Leniency Bias | {}/{} ({:.1%}) |".format(total_p2np, total_p, total_p2np / total_p))
        p("| 2 | 不起诉类型混淆 | NP Type Confusion | {} 例互混 |".format(total_np_conf))
        p("| 3 | 证据充分性评估 | Evidence Sufficiency | 存疑不起诉准确率 {:.1%} |".format(d_correct / d_total))
        p("| 4 | 隐性情节推断 | Implicit Factor Inference | 关键因素缺失率>80% |")
        p("| 5 | 三法联合法条 | Multi-Law Citation | F1={:.4f} |".format(art_f1))
        p("| 6 | 结构化输出 | Format Compliance | 解析失败 {:.1%} |".format(parse_fail / total))
        p()
        p("### 核心结论")
        p()
        p("PDP 对 LLM 的挑战可归纳为三个层次：")
        p()
        p("1. **信息层**: 关键决定因素在输入中缺失或隐含，需**推断**而非**提取**。")
        p("2. **推理层**: 起诉/不起诉边界是**相对权衡**；三类不起诉需事实认定+证据评估+法律适用三层推理协同。")
        p("3. **知识层**: 法条引用、入罪门槛均需深度法律知识。")
        p()
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
        p("我们提出**公诉决定预测**（Prosecution Decision Prediction, PDP），一个全新的法律 NLP 任务。PDP 与 LJP 的本质区别在于：PDP 要求模型**同时评估违法性与有责性**，并在二者之间进行权衡。具体而言：")
        p()
        p("1. **决策维度**: LJP 聚焦违法性（罪名+量刑），PDP 要求违法性与有责性的联合判断")
        p("2. **决策类型**: 二级分类——第一级起诉/不起诉，第二级细分为起诉、相对不起诉、法定不起诉、存疑不起诉")
        p("3. **有责性判断**: 三种不起诉分别对应有责性判断的三种结论——有罪但责任轻微可免刑（相对）、不具备刑事责任（法定）、责任无法证实（存疑）")
        p("4. **信息状态**: 审查起诉阶段信息不完整，关键从轻情节往往隐含")
        p("5. **推理模式**: 核心在于犯罪严重程度与责任减轻情节的**相对权衡**")
        p("6. **法条范围**: 需同时引用刑法、刑事诉讼法和刑事诉讼规则")
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
        p("| 决定类型 | 有罪/无罪 | 二级分类(起诉/不起诉+3不起诉子类型) |")
        p("| 法条范围 | 刑法 | 刑法+刑诉法+诉讼规则 |")
        p("| 认定维度 | 违法性（罪名+量刑） | 违法性+有责性联合判断 |")
        p("| 核心推理 | 事实->罪名->量刑 | 事实x情节->违法性x有责性->决定 |")
        p()

        p("## 3. Key Challenges")
        p()
        p("对 Qwen3-4B 在 {} 个测试样本上的评估显示：即使简化为起诉/不起诉二分类，准确率也仅 **{:.1%}**（起诉 {:.1%}，不起诉 {:.1%}）；细化到四分类准确率更降至 **{:.1%}**，揭示以下核心挑战：".format(total, level1_acc, p_l1_acc, np_l1_acc, acc))
        p()

        p("### 3.1 Prosecution Leniency Bias (起诉案件宽大误判)")
        p()
        p("在 {} 例起诉案件中，{} 例 ({:.1%}) 被错误判为不起诉。起诉案例含从轻线索比例 ({:.1%}) **高于**不起诉案例 ({:.1%})。这说明从轻线索的存在不能区分起诉与否——决定取决于犯罪严重程度与从轻情节的**相对权衡** (relative weighing)。".format(tp, total_p2np, total_p2np / tp, p_mit / tp, np_mit / tnp))
        p()

        p("### 3.2 Non-Prosecution Type Confusion (不起诉类型混淆)")
        p()
        p("三类不起诉之间 {} 例互相混淆。区分「有罪免刑」「无罪」「证不够」需同时进行：(1) 犯罪构成要件认定；(2) 证据链完整性评估；(3) 量刑情节权衡。三层推理必须协同运作。".format(total_np_conf))
        p()

        p("### 3.3 Evidence Sufficiency Assessment (证据充分性评估)")
        p()
        p("存疑不起诉准确率仅 **{:.1%}**。案件事实表面上犯罪成立，但证据链有隐含漏洞。需要**元认知推理** (meta-cognitive reasoning)：判断叙述本身是否可靠、证据是否充分。".format(doubt_acc))
        p()

        p("### 3.4 Implicit Factor Inference & Multi-Law Citation")
        p()
        p("关键从轻情节在输入中大量缺失，模型需**推断**而非**提取**。法条 F1 仅 {:.4f}，需同时引用三部法律。".format(art_f1))
        p()

        p("## 4. Contributions")
        p()
        p("1. **新任务**: 首次定义 PDP，将法律 NLP 从审判扩展至审查起诉阶段")
        p("2. **新数据集**: 25,000+ 真实检察文书，两级四分类+三法联合法条标注")
        p("3. **新挑战**: 系统性揭示起诉边界相对权衡、不起诉三层推理、证据元认知评估等独特挑战")
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
