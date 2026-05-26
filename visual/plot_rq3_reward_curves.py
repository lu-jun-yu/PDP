from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "RQ3" / "wandb_export_2026-05-21T17_46_06.825+08_00.csv"
OUT_DIR = ROOT / "paper" / "figures"
OUTPUT_STEM = "rq3_reward_curves"

TARGETS = ["IENP", "SNP", "DNP"]
SERIES = ["Balanced", "40%", "55%"]

RUN_COLUMNS = {
    "Balanced": "RQ3_DAPO_Qwen3-8B_balanced_seed42_original - train/reward",
    "IENP-40": "RQ3_DAPO_Qwen3-8B_IENP_40_seed42_original - train/reward",
    "IENP-55": "RQ3_DAPO_Qwen3-8B_IENP_55_seed42_original - train/reward",
    "SNP-40": "RQ3_DAPO_Qwen3-8B_SNP_40_seed42_original - train/reward",
    "SNP-55": "RQ3_DAPO_Qwen3-8B_SNP_55_seed42_original - train/reward",
    "DNP-40": "RQ3_DAPO_Qwen3-8B_DNP_40_seed42_original - train/reward",
    "DNP-55": "RQ3_DAPO_Qwen3-8B_DNP_55_seed42_original - train/reward",
}

COLORS = {
    "Balanced": "#5b6472",
    "40%": "#2F6FB0",
    "55%": "#D7655F",
}
LINESTYLES = {
    "Balanced": "--",
    "40%": "-",
    "55%": "-",
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


def read_wandb_export() -> tuple[list[int], dict[str, list[float]]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = [column for column in ["train/global_step", *RUN_COLUMNS.values()] if column not in reader.fieldnames]
        if missing:
            raise KeyError(f"Missing expected columns in {CSV_PATH}: {missing}")

        steps: list[int] = []
        values = {key: [] for key in RUN_COLUMNS}
        for row in reader:
            if not row.get("train/global_step"):
                continue
            steps.append(int(float(row["train/global_step"])))
            for key, column in RUN_COLUMNS.items():
                raw = row.get(column, "")
                values[key].append(float(raw) if raw else float("nan"))

    return steps, values


def series_for_target(values: dict[str, list[float]], target: str) -> dict[str, list[float]]:
    return {
        "Balanced": values["Balanced"],
        "40%": values[f"{target}-40"],
        "55%": values[f"{target}-55"],
    }


def plot() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps, values = read_wandb_export()

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.95), sharex=True, sharey=True)
    for ax, target in zip(axes, TARGETS):
        panel = series_for_target(values, target)
        for label in SERIES:
            ax.plot(
                steps,
                panel[label],
                color=COLORS[label],
                linestyle=LINESTYLES[label],
                linewidth=1.35,
                alpha=0.82 if label == "Balanced" else 0.92,
                label=f"{label} (25%)" if label == "Balanced" else label,
            )

        ax.set_title(target, fontweight="bold", pad=3)
        ax.grid(axis="y", color="#E6E9ED", linewidth=0.55)
        ax.grid(axis="x", color="#F2F4F7", linewidth=0.35)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.tick_params(length=2.0, width=0.65, color="#52606D", pad=1.8)

    axes[0].set_ylabel("Training reward")
    axes[1].set_xlabel("Global step", labelpad=3)
    ymin = min(min(series) for series in values.values())
    ymax = max(max(series) for series in values.values())
    pad = max((ymax - ymin) * 0.10, 0.02)
    axes[0].set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.80, wspace=0.13)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUTPUT_STEM}.{ext}", bbox_inches="tight", pad_inches=0.01, dpi=600)
    plt.close(fig)


def main() -> None:
    plot()


if __name__ == "__main__":
    main()
