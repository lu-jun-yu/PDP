from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
)
FONT_SIZE = 12

LABELS = [
    "Non-Prosecution for Insufficient Evidence",
    "Statutory Non-Prosecution",
    "Discretionary Non-Prosecution",
    "Prosecution",
]
COUNTS = [101, 41, 480, 4008]
# Colorblind-friendly, print-safe palette with enough contrast for small slices.
COLORS = ["#2F6FB0", "#E6A23A", "#8E72BD", "#D7655F"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = sum(COUNTS)
    fig = plt.figure(figsize=(5.3, 1.92), facecolor="white")

    ax = fig.add_axes([0.00, 0.01, 0.38, 0.98])
    ax.axis("equal")
    ax.pie(
        COUNTS,
        startangle=90,
        counterclock=False,
        colors=COLORS,
        radius=1.08,
        wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 0.75},
    )
    ax.text(0, 0.080, f"{total:,}", ha="center", va="center", fontsize=FONT_SIZE, color="#4b5563")
    ax.text(0, -0.105, "samples", ha="center", va="center", fontsize=FONT_SIZE, color="#4b5563")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_ax = fig.add_axes([0.405, 0.025, 0.585, 0.95])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_y = [0.86, 0.61, 0.36, 0.11]
    for color, label, count, y in zip(COLORS, LABELS, COUNTS, legend_y):
        legend_ax.add_patch(plt.Rectangle((0.00, y - 0.011), 0.024, 0.022, facecolor=color, edgecolor="none"))
        legend_ax.text(0.040, y + 0.045, label, ha="left", va="center", fontsize=FONT_SIZE, color="#273241")
        legend_ax.text(0.040, y - 0.052, f"{count:,} ({count / total * 100:.2f}%)", ha="left", va="center", fontsize=FONT_SIZE, color="#4b5563")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"pdp_bench_overview.{ext}", bbox_inches="tight", pad_inches=0.0, dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
