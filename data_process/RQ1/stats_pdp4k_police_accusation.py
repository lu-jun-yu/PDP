#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 PDP-Bench test 中公安机关认定罪名分布。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "public_sources" / "Stage4" / "dataset_final.json"
DEFAULT_OUTPUT = ROOT / "data" / "PDP-Bench" / "police_accusation_distribution_test.json"

PAT_PROC_SHEAN = re.compile(r"涉嫌(?:犯有)?([^，,。；;\n]+?罪)")
# 先吞掉姓名，再匹配「涉嫌X罪」或「X案」
PAT_PROC_CASE = re.compile(
    r"侦查终结，以犯罪嫌疑人[^，,。；;\n]*?"
    r"(?:涉嫌([^，,。；;\n]+?罪)|([\u4e00-\u9fff、\[\]（）()]{2,40}?(?:案|罪)))[，,]"
)
PAT_PERSON = re.compile(r"因涉嫌(?:犯有)?([^，,。；;\n]+?罪)")
PAT_POLICE_OPINION = re.compile(
    r"(?:公安局|公安机关)[^。；;\n]{0,80}?"
    r"(?:认为|认定)[^。；;\n]{0,50}?"
    r"(?:构成|涉嫌)(?:犯有)?([^，,。；;\n]+?罪)"
)
# 文书脱敏姓名（须含「某」）：张某某、李某甲、王某 等
PAT_NAME_PREFIX = re.compile(
    r"^(?:"
    r"[\u4e00-\u9fff]{1,4}某[甲乙丙丁\d]?"
    r"|[\u4e00-\u9fff]{1,2}某"
    r")+"
)
PAT_CHARGE_TAIL = re.compile(
    r"([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff、\[\]（）()《》""]{2,50}?罪)$"
)
PAT_NOISE = re.compile(r"（另案处理）|等人|等\d+人|等人?涉嫌")

CHARGE_ALIASES = {
    "帮助信息网络犯罪": "帮助信息网络犯罪活动罪",
    "掩饰、隐瞒犯罪": "掩饰、隐瞒犯罪所得罪",
    "掩饰隐瞒犯罪": "掩饰、隐瞒犯罪所得罪",
    "隐瞒犯罪": "掩饰、隐瞒犯罪所得罪",
    "隐瞒犯罪所得": "掩饰、隐瞒犯罪所得罪",
}


def _normalize_charge_once(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^[、,，;；\s]+", "", text)
    text = PAT_NOISE.sub("", text)
    if "涉嫌" in text:
        text = text.split("涉嫌")[-1]
    text = re.sub(r"^犯有", "", text)
    while True:
        stripped = PAT_NAME_PREFIX.sub("", text)
        if stripped == text:
            break
        text = stripped
    m = PAT_CHARGE_TAIL.search(text)
    if m:
        text = m.group(1)
    if text.endswith("案"):
        text = text[:-1] + "罪"
    text = CHARGE_ALIASES.get(text, text)
    if text and not text.endswith("罪"):
        text = text + "罪"
    return text


def normalize_charge(text: str) -> str:
    return _normalize_charge_once(text)


def split_raw_charges(raw: str) -> list[str]:
    """从一段文本中提取各罪名片段（以「罪」结尾）。"""
    raw = raw.strip().lstrip("、,，;；")
    spans = re.findall(r"[^、,，;；]+?罪", raw)
    if spans:
        return [s.strip() for s in spans]
    return [raw] if raw else []


def is_valid_charge(charge: str) -> bool:
    return (
        bool(charge)
        and charge.endswith("罪")
        and len(charge) >= 3
        and "某" not in charge
        and not charge.startswith("、")
    )


def extract_police_charges(row: dict) -> list[str]:
    procedure = row.get("procedure") or ""
    person_info = row.get("person_info") or ""
    fact = row.get("fact") or ""

    found: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        for part in split_raw_charges(raw):
            charge = normalize_charge(part)
            if is_valid_charge(charge) and charge not in seen:
                seen.add(charge)
                found.append(charge)

    for m in PAT_PROC_SHEAN.findall(procedure):
        add(m)
    if not found:
        m = PAT_PROC_CASE.search(procedure)
        if m:
            raw = (m.group(1) or m.group(2) or "").strip()
            if raw:
                add(raw)
    if not found:
        for m in PAT_PERSON.findall(person_info):
            add(m)
    if not found:
        for m in PAT_POLICE_OPINION.findall(fact):
            add(m)

    return found


def sample_label(charges: list[str]) -> str:
    """每条样本唯一标签：单罪用罪名，多罪用排序后的组合。"""
    if not charges:
        return "（未提取）"
    if len(charges) == 1:
        return charges[0]
    return " + ".join(sorted(charges))


def build_report(rows: list[dict]) -> dict:
    total = len(rows)

    sample_counter: Counter[str] = Counter()
    single_charge_counter: Counter[str] = Counter()
    by_decision: dict[str, Counter[str]] = {}
    no_charge = 0
    multi_charge = 0
    source_counter: Counter[str] = Counter()

    for row in rows:
        charges = extract_police_charges(row)
        decision = row.get("decision") or "未知"
        by_decision.setdefault(decision, Counter())

        label = sample_label(charges)
        sample_counter[label] += 1
        by_decision[decision][label] += 1

        if not charges:
            no_charge += 1
        elif len(charges) == 1:
            single_charge_counter[charges[0]] += 1
        else:
            multi_charge += 1

        procedure = row.get("procedure") or ""
        if PAT_PROC_SHEAN.search(procedure):
            source_counter["procedure_涉嫌"] += 1
        elif PAT_PROC_CASE.search(procedure):
            source_counter["procedure_案名"] += 1
        elif PAT_PERSON.findall(row.get("person_info") or ""):
            source_counter["person_info_only"] += 1
        elif PAT_POLICE_OPINION.search(row.get("fact") or ""):
            source_counter["fact_公安认为"] += 1
        else:
            source_counter["none"] += 1

    dec_counts = Counter(r.get("decision") or "未知" for r in rows)
    single_total = sum(single_charge_counter.values())

    def dist(counter: Counter[str]) -> list[dict]:
        return [
            {
                "label": label,
                "count": cnt,
                "ratio": round(cnt / total, 6),
                "percent": round(100 * cnt / total, 4),
            }
            for label, cnt in counter.most_common()
        ]

    multi_combos = [
        {"label": label, "count": cnt, "ratio": round(cnt / total, 6)}
        for label, cnt in sample_counter.most_common()
        if " + " in label
    ]

    by_decision_out = {}
    for decision, ctr in sorted(by_decision.items()):
        n = sum(ctr.values())
        by_decision_out[decision] = {
            "total": n,
            "distribution": [
                {
                    "label": label,
                    "count": cnt,
                    "ratio": round(cnt / n, 6),
                    "percent": round(100 * cnt / n, 4),
                }
                for label, cnt in ctr.most_common()
            ],
        }

    return {
        "meta": {
            "dataset": "PDP-Bench.test",
            "source_file": str(DATA.relative_to(ROOT)),
            "total_samples": total,
            "counting_unit": "sample",
            "extraction_priority": [
                "procedure: 涉嫌X罪 / 案名",
                "person_info: 因涉嫌X罪",
                "fact: 公安机关认为构成X罪",
            ],
            "label_rule": "单罪→罪名；多罪→「A + B」组合；每条样本计 1 次",
        },
        "summary": {
            "no_charge_samples": no_charge,
            "single_charge_samples": single_total,
            "multi_charge_samples": multi_charge,
            "unique_sample_labels": len(sample_counter),
            "unique_single_charges": len(single_charge_counter),
            "extraction_source": dict(source_counter),
            "decision_counts": dict(dec_counts),
        },
        "sample_distribution": dist(sample_counter),
        "single_charge_distribution": [
            {
                "charge": charge,
                "count": cnt,
                "ratio_of_single_charge_samples": round(cnt / single_total, 6),
                "percent_of_single_charge_samples": round(100 * cnt / single_total, 4),
                "ratio_of_all_samples": round(cnt / total, 6),
                "percent_of_all_samples": round(100 * cnt / total, 4),
            }
            for charge, cnt in single_charge_counter.most_common()
        ],
        "multi_charge_distribution": multi_combos,
        "by_decision": by_decision_out,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="统计 PDP-Bench test 公安机关认定罪名（样本级）")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="输出 JSON 路径",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印摘要到 stdout")
    args = parser.parse_args()

    rows = json.loads(DATA.read_text(encoding="utf-8"))
    report = build_report(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.quiet:
        total = report["meta"]["total_samples"]
        print(f"已保存: {output_path}")
        print(f"样本数 N={total}")
        print("样本分布 Top 10:")
        for item in report["sample_distribution"][:10]:
            print(f"  {item['label']}: {item['count']} ({item['percent']}%)")


if __name__ == "__main__":
    main()
