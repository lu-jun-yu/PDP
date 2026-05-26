from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "RQ3"
BASELINE_DIR = ROOT / "results" / "RQ2_2" / "Qwen3-8B_original_test_r8_20260512_121605"
OUT_DIR = ROOT / "paper" / "figures"

OUTPUT_STEM = "rq3_class_prior_intervention"
CLASSES = ["IENP", "SNP", "DNP", "P"]
TARGETS = ["IENP", "SNP", "DNP"]
BASELINE_X = 10
BASELINE_SPLIT_X = 17.5
RATIOS = [BASELINE_X, 25, 40, 55]
RATIO_LABELS = ["no train", "25%", "40%", "55%"]
METRICS = ["F1", "Precision", "Recall"]

COLORS = {
    "IENP": "#174EA6",
    "SNP": "#F26A21",
    "DNP": "#6F4E9A",
    "P": "#D7655F",
}
METRIC_COLORS = {
    "F1": "#2B303A",
    "Precision": "#336FAE",
    "Recall": "#E0902F",
}
METRIC_MARKERS = {
    "F1": "o",
    "Precision": "s",
    "Recall": "^",
}

mpl.rcParams.update(
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
        "axes.labelsize": 8.8,
        "axes.titlesize": 9.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
    }
)


@dataclass(frozen=True)
class RunMetrics:
    group: str
    target: str | None
    ratio: int
    metrics: dict[str, dict[str, float]]


def parse_group_name(name: str) -> tuple[str, str | None, int]:
    if "_balanced_" in name:
        return "Balanced", None, 25

    match = re.search(r"_((?:IENP)|(?:SNP)|(?:DNP))_(40|55)_", name)
    if not match:
        raise ValueError(f"Cannot infer RQ3 intervention group from directory name: {name}")

    target = match.group(1)
    ratio = int(match.group(2))
    return f"{target}-{ratio}", target, ratio


def parse_level2_metrics(metrics_path: Path) -> dict[str, dict[str, float]]:
    lines = metrics_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("### Level2"):
            start = idx
            break
    if start is None:
        raise ValueError(f"Cannot find Level2 table in {metrics_path}")

    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        if "---" in stripped or "Precision" in stripped or "Count" in stripped:
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) >= 4:
            rows.append(cells)
        if len(rows) == 4:
            break

    if len(rows) != 4:
        raise ValueError(f"Expected four Level2 class rows in {metrics_path}, got {len(rows)}")

    # The metrics table is emitted in the fixed PDP class order:
    # IENP, SNP, DNP, P. This avoids depending on possibly mojibaked Chinese labels.
    return {
        label: {
            "Precision": _parse_metric_cell(row[1]),
            "Recall": _parse_metric_cell(row[2]),
            "F1": _parse_metric_cell(row[3]),
        }
        for label, row in zip(CLASSES, rows)
    }


def _parse_metric_cell(cell: str) -> float:
    # Some metrics.md files emit "0.3440 ± 0.0126"; we only need the point estimate.
    return float(cell.split("±")[0].strip())


def load_baseline_metrics() -> dict[str, dict[str, float]]:
    metrics_path = BASELINE_DIR / "metrics.md"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {metrics_path}")
    return parse_level2_metrics(metrics_path)


def load_metrics() -> tuple[list[RunMetrics], dict[str, dict[str, float]]]:
    runs: list[RunMetrics] = []
    for run_dir in sorted(path for path in RESULT_DIR.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.md"
        if not metrics_path.exists():
            continue
        group, target, ratio = parse_group_name(run_dir.name)
        runs.append(RunMetrics(group=group, target=target, ratio=ratio, metrics=parse_level2_metrics(metrics_path)))

    expected = {"Balanced"} | {f"{target}-{ratio}" for target in TARGETS for ratio in (40, 55)}
    observed = {run.group for run in runs}
    missing = sorted(expected - observed)
    if missing:
        raise FileNotFoundError(f"Missing RQ3 result groups: {', '.join(missing)}")
    return runs, load_baseline_metrics()


def build_panel_data(
    runs: list[RunMetrics],
    baseline_metrics: dict[str, dict[str, float]],
    target: str,
    metric: str,
) -> dict[str, list[float]]:
    balanced = next(run for run in runs if run.group == "Balanced")
    by_ratio = {run.ratio: run for run in runs if run.target == target}
    return {
        cls: [
            baseline_metrics[cls][metric],
            balanced.metrics[cls][metric],
            by_ratio[40].metrics[cls][metric],
            by_ratio[55].metrics[cls][metric],
        ]
        for cls in CLASSES
    }


def build_target_metric_series(
    runs: list[RunMetrics],
    baseline_metrics: dict[str, dict[str, float]],
    target: str,
) -> dict[str, list[float]]:
    return {metric: build_panel_data(runs, baseline_metrics, target, metric)[target] for metric in METRICS}


def plot(runs: list[RunMetrics], baseline_metrics: dict[str, dict[str, float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.05), sharex=True, sharey=True)

    for ax, target in zip(axes, TARGETS):
        panel_data = build_target_metric_series(runs, baseline_metrics, target)
        for metric in METRICS:
            ax.plot(
                RATIOS,
                panel_data[metric],
                color=METRIC_COLORS[metric],
                linestyle="-",
                linewidth=1.45,
                marker=METRIC_MARKERS[metric],
                markersize=3.3,
                markerfacecolor="white",
                markeredgewidth=0.9,
                alpha=0.95,
                label=metric,
                zorder=3,
            )

        ax.axvline(
            BASELINE_SPLIT_X,
            color="#C5CBD3",
            linewidth=0.7,
            linestyle=(0, (2, 2)),
            zorder=1,
        )

        ax.set_title(target, fontweight="bold", pad=3)
        ax.set_xticks(RATIOS)
        ax.set_xticklabels(RATIO_LABELS)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#E6E9ED", linewidth=0.55)
        ax.grid(axis="x", color="#F2F4F7", linewidth=0.35)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.tick_params(length=2.0, width=0.65, color="#52606D", pad=1.8)

    axes[0].set_ylabel("Target-class score")
    axes[1].set_xlabel("Target-class training ratio", labelpad=3)
    axes[0].set_xlim(7, 58)
    axes[0].set_ylim(0.0, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.24, top=0.76, wspace=0.13)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUTPUT_STEM}.{ext}", bbox_inches="tight", pad_inches=0.01, dpi=600)
    plt.close(fig)


def main() -> None:
    runs, baseline_metrics = load_metrics()
    plot(runs, baseline_metrics)


if __name__ == "__main__":
    main()
