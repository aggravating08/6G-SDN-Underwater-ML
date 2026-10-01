#!/usr/bin/env python3
"""Plot the fixed C20-SVM, depth-3-DTC, fractional-stump-RF comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "underwater_partner_style_equivalent_2000" / "PARTNER_COMPATIBLE_FINAL_C20_DTC3_RF_FRACTIONAL_STUMP"
FIGURES = RESULTS / "figures"
ORDER = ["SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline worst-OC"]
DISPLAY = {
    "SVM-selected OC": "SVM",
    "DTC-selected OC": "DTC",
    "RF-selected OC": "RF",
    "Baseline worst-OC": "Worst-OC baseline",
}
STYLE = {
    "SVM-selected OC": ("#e41a1c", "s"),
    "DTC-selected OC": ("#ff7f00", "o"),
    "RF-selected OC": ("#377eb8", "o"),
    "Baseline worst-OC": ("#c51b7d", "D"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_line_plot(rows: list[dict[str, str]], column: str, ylabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=170)
    nodes = sorted({int(row["Nodes"]) for row in rows})
    for model in ORDER:
        series = {int(row["Nodes"]): float(row[column]) for row in rows if row["Model"] == model}
        if not series:
            continue
        color, marker = STYLE[model]
        values = [series[node] for node in nodes]
        ax.plot(nodes, values, color=color, marker=marker, linewidth=2.0,
                markersize=6, label=DISPLAY[model])
    ax.set_title(ylabel + " vs Number of Nodes", fontweight="bold")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_xticks(nodes)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_accuracy_plot(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=170)
    names = [row["Model"] for row in rows]
    row_accuracy = [100.0 * float(row["Row-level binary accuracy"]) for row in rows]
    x = np.arange(len(names))
    colors = ["#e41a1c", "#ff7f00", "#377eb8"]
    bars = ax.bar(x, row_accuracy, color=colors, width=0.62)
    for bar, value in zip(bars, row_accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Partner-style Row-Level OC Classification Accuracy", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_row_level_accuracy.png", bbox_inches="tight")
    fig.savefig(FIGURES / "ml_row_level_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(RESULTS / "PARTNER_COMPATIBLE_final_network_metrics_by_node.csv")
    accuracy = read_csv(RESULTS / "PARTNER_COMPATIBLE_final_ml_accuracy_comparison.csv")
    write_line_plot(metrics, "Mean PDR (%)", "Packet delivery ratio (%)", "pdr_vs_nodes")
    write_line_plot(metrics, "Mean Delay ms", "Completion-aware delay (ms)", "delay_vs_nodes")
    write_line_plot(metrics, "Mean ROR", "Routing overhead ratio", "ror_vs_nodes")
    write_accuracy_plot(accuracy)
    print(FIGURES)


if __name__ == "__main__":
    main()
