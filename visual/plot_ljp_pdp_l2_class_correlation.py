#!/usr/bin/env python
"""Plot LJP-vs-PDP-L2 class-wise metric correlations with matplotlib.

Usage:
    python visual/plot_ljp_pdp_l2_class_correlation.py

Outputs:
    paper/figures/ljp_pdp_l2_metric_correlation.png
    paper/figures/ljp_pdp_l2_metric_correlation.pdf
    paper/figures/ljp_pdp_l2_metric_correlation.svg

Each point denotes one model. The x-axis is CAIL2018 charge prediction
Macro-F1. The three rows show PDP Level-2 class-wise F1, Precision, and
Recall; the four columns show IENP, SNP, DNP, and P.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"


@dataclass(frozen=True)
class ModelResult:
    model: str
    short: str
    family: str
    cail_macro_f1: float
    l2_macro_f1: float
    ienp_precision: float
    ienp_recall: float
    ienp_f1: float
    snp_precision: float
    snp_recall: float
    snp_f1: float
    dnp_precision: float
    dnp_recall: float
    dnp_f1: float
    p_precision: float
    p_recall: float
    p_f1_value: float

    @property
    def p_f1(self) -> float:
        return self.p_f1_value


RESULTS: tuple[ModelResult, ...] = (
    ModelResult("GPT-5.4", "GPT", "International closed", 0.6806, 0.5396, 0.5161, 0.7921, 0.6250, 0.2462, 0.3902, 0.3019, 0.3380, 0.3021, 0.3190, 0.9182, 0.9069, 0.9125),
    ModelResult("Gemini-3.1-Pro", "Gem", "International closed", 0.8374, 0.7216, 0.8235, 0.8317, 0.8276, 0.4478, 0.7317, 0.5556, 0.5015, 0.7000, 0.5843, 0.9646, 0.8775, 0.9190),
    ModelResult("Claude-Opus-4.6", "Cl", "International closed", 0.7844, 0.6057, 0.7396, 0.7030, 0.7208, 0.6250, 0.2439, 0.3509, 0.4077, 0.4417, 0.4240, 0.9291, 0.9249, 0.9270),
    ModelResult("DeepSeek-V4-Pro", "DS", "Chinese closed", 0.8097, 0.6412, 0.8852, 0.5347, 0.6667, 0.4474, 0.8293, 0.5812, 0.3282, 0.6667, 0.4399, 0.9486, 0.8154, 0.8770),
    ModelResult("Qwen3.6-Max", "QwM", "Chinese closed", 0.8238, 0.6728, 0.7579, 0.7129, 0.7347, 0.6098, 0.6098, 0.6098, 0.3419, 0.6667, 0.4520, 0.9519, 0.8443, 0.8949),
    ModelResult("GPT-OSS-20B", "OSS", "Open weight", 0.3042, 0.3010, 1.0000, 0.0099, 0.0196, 0.1048, 0.3171, 0.1576, 0.2982, 0.0708, 0.1145, 0.8814, 0.9454, 0.9122),
    ModelResult("Qwen3.5-35B-A3B", "QwO", "Open weight", 0.7362, 0.5540, 0.5955, 0.5248, 0.5579, 0.5652, 0.3171, 0.4062, 0.2836, 0.5854, 0.3821, 0.9321, 0.8156, 0.8700),
)

METRIC_ROWS: tuple[tuple[str, tuple[tuple[str, Callable[[ModelResult], float]], ...]], ...] = (
    (
        "F1",
        (
            ("IENP", lambda row: row.ienp_f1),
            ("SNP", lambda row: row.snp_f1),
            ("DNP", lambda row: row.dnp_f1),
            ("P", lambda row: row.p_f1),
        ),
    ),
    (
        "Precision",
        (
            ("IENP", lambda row: row.ienp_precision),
            ("SNP", lambda row: row.snp_precision),
            ("DNP", lambda row: row.dnp_precision),
            ("P", lambda row: row.p_precision),
        ),
    ),
    (
        "Recall",
        (
            ("IENP", lambda row: row.ienp_recall),
            ("SNP", lambda row: row.snp_recall),
            ("DNP", lambda row: row.dnp_recall),
            ("P", lambda row: row.p_recall),
        ),
    ),
)

MODEL_STYLE = {
    "GPT-5.4": {"color": "#0072B2", "marker": "o", "legend": "GPT-5.4"},
    "Gemini-3.1-Pro": {"color": "#D55E00", "marker": "o", "legend": "Gemini"},
    "Claude-Opus-4.6": {"color": "#009E73", "marker": "o", "legend": "Claude"},
    "DeepSeek-V4-Pro": {"color": "#CC79A7", "marker": "o", "legend": "DeepSeek"},
    "Qwen3.6-Max": {"color": "#E69F00", "marker": "o", "legend": "Qwen-Max"},
    "GPT-OSS-20B": {"color": "#56B4E9", "marker": "o", "legend": "GPT-OSS"},
    "Qwen3.5-35B-A3B": {"color": "#000000", "marker": "o", "legend": "Qwen-35B"},
}

LABEL_OFFSETS = {
    "GPT": (-0.018, -0.045),
    "Gem": (0.012, -0.045),
    "Cl": (-0.026, 0.045),
    "DS": (0.012, 0.04),
    "QwM": (0.012, -0.05),
    "OSS": (0.014, 0.04),
    "QwO": (-0.034, 0.04),
}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = sum((x - x_mean) ** 2 for x in xs)
    y_den = sum((y - y_mean) ** 2 for y in ys)
    if x_den == 0 or y_den == 0:
        return float("nan")
    return numerator / math.sqrt(x_den * y_den)


def ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def regression_line(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0, y_mean
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def filter_results(rows: Iterable[ModelResult], subset: str) -> list[ModelResult]:
    rows = list(rows)
    if subset == "all":
        return rows
    if subset == "closed":
        return [row for row in rows if row.family != "Open weight"]
    if subset == "exclude-gpt-oss":
        return [row for row in rows if row.model != "GPT-OSS-20B"]
    raise ValueError(f"Unknown subset: {subset}")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#1F2933",
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
        }
    )


def plot(
    rows: Sequence[ModelResult],
    output: Path,
    write_pdf: bool,
    write_svg: bool,
    show_labels: bool,
    show_legend: bool,
) -> None:
    configure_matplotlib()

    fig, axes = plt.subplots(3, 4, figsize=(7.05, 4.65), sharex=True, sharey=True)
    x_values = [row.cail_macro_f1 for row in rows]
    x_min, x_max = 0.25, 0.90
    y_min, y_max = 0.0, 1.0
    line_x = [x_min, x_max]

    for row_idx, (metric_name, class_getters) in enumerate(METRIC_ROWS):
        for col_idx, (class_name, getter) in enumerate(class_getters):
            ax = axes[row_idx][col_idx]
            y_values = [getter(row) for row in rows]
            r = pearson(x_values, y_values)
            rho = spearman(x_values, y_values)
            slope, intercept = regression_line(x_values, y_values)
            line_y = [slope * x + intercept for x in line_x]

            if metric_name == "F1":
                ax.plot(line_x, line_x, linestyle="--", linewidth=0.75, color="#A7B1BC", alpha=0.68, zorder=1)
            ax.plot(line_x, line_y, linewidth=1.05, color="#222831", alpha=0.82, zorder=2)

            for result in rows:
                style = MODEL_STYLE[result.model]
                x = result.cail_macro_f1
                y = getter(result)
                ax.scatter(
                    x,
                    y,
                    s=20,
                    marker=style["marker"],
                    color=style["color"],
                    alpha=0.94,
                    edgecolor="#FFFFFF",
                    linewidth=0.5,
                    zorder=3,
                )
                if show_labels:
                    dx, dy = LABEL_OFFSETS.get(result.short, (0.01, 0.03))
                    ax.text(x + dx, y + dy, result.short, fontsize=5.2, weight="bold", zorder=4)

            if row_idx == 0:
                ax.set_title(class_name, fontweight="bold", pad=12)
                corr_y = 1.015
            else:
                corr_y = 1.015
            ax.text(
                0.5,
                corr_y,
                rf"$r$={r:.2f}, $\rho$={rho:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.3,
                color="#273241",
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([0.3, 0.5, 0.7, 0.9])
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(axis="y", color="#E6E9ED", linewidth=0.55)
            ax.grid(axis="x", color="#F2F4F7", linewidth=0.35)
            ax.tick_params(length=2.0, width=0.65, color="#52606D", pad=1.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.75)
            ax.spines["bottom"].set_linewidth(0.75)
            if col_idx == 0:
                ax.set_ylabel(metric_name)

    fig.text(0.535, 0.045, "CAIL2018 charge Macro-F1", ha="center", va="center", fontsize=8.4)

    if show_legend:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["color"],
                markeredgecolor="#FFFFFF",
                markeredgewidth=0.55,
                markersize=4.8,
                label=style["legend"],
            )
            for row in rows
            for style in (MODEL_STYLE[row.model],)
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=7,
            frameon=False,
            columnspacing=0.72,
            handletextpad=0.25,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    top = 0.885 if show_legend else 0.92
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.11, top=top, hspace=0.30, wspace=0.13)
    fig.savefig(output, bbox_inches="tight")
    if write_pdf:
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    if write_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3x4 scatter plots for LJP-to-PDP-L2 class-wise metric correlations."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "ljp_pdp_l2_metric_correlation.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--subset",
        choices=("all", "closed", "exclude-gpt-oss"),
        default="all",
        help="Model subset to visualize.",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Do not also write a PDF copy.")
    parser.add_argument("--no-svg", action="store_true", help="Do not also write an SVG copy.")
    parser.add_argument("--no-legend", action="store_true", help="Hide the per-model color legend.")
    parser.add_argument("--labels", action="store_true", help="Show short model labels beside points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = filter_results(RESULTS, args.subset)
    plot(
        rows,
        args.output,
        write_pdf=not args.no_pdf,
        write_svg=not args.no_svg,
        show_labels=args.labels,
        show_legend=not args.no_legend,
    )

    print(f"Wrote {args.output}")
    if not args.no_pdf:
        print(f"Wrote {args.output.with_suffix('.pdf')}")
    if not args.no_svg:
        print(f"Wrote {args.output.with_suffix('.svg')}")

    xs = [row.cail_macro_f1 for row in rows]
    for metric_name, class_getters in METRIC_ROWS:
        for class_name, getter in class_getters:
            ys = [getter(row) for row in rows]
            print(
                f"{metric_name} {class_name}: "
                f"Pearson r={pearson(xs, ys):.2f}, Spearman rho={spearman(xs, ys):.2f}"
            )


if __name__ == "__main__":
    main()
