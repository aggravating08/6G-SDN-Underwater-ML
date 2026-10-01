#!/usr/bin/env python3
"""Create raw-metric figures for the latest aggregated-control sanity run."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "underwater_partner_style_equivalent_2000"
RESULTS = BASE / "AGGREGATED_ROR_SANITY_CAPACITY_40_SVM_C20_G005_DTC3_L30_S60_WORST_BASELINE"
ML_RESULTS = BASE / "PARTNER_COMPATIBLE_C20_G005_DTC3_L30_S60_RF_FRACTIONAL_STUMP"
FIGURES = RESULTS / "figures"
ORDER = ("SVM", "DTC", "RF", "Baseline worst-OC")
LABELS = {
    "SVM": "SVM",
    "DTC": "DTC",
    "RF": "RF",
    "Baseline worst-OC": "Worst-OC baseline",
}
STYLE = {
    "SVM": ("#d62728", "s"),
    "DTC": ("#ff7f0e", "o"),
    "RF": ("#1f77b4", "o"),
    "Baseline worst-OC": ("#b52b65", "D"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_metric(ax: plt.Axes, rows: list[dict[str, str]], column: str, ylabel: str) -> None:
    nodes = sorted({int(row["Nodes"]) for row in rows})
    for model in ORDER:
        series = {int(row["Nodes"]): float(row[column]) for row in rows if row["Model"] == model}
        color, marker = STYLE[model]
        ax.plot(nodes, [series[node] for node in nodes], color=color, marker=marker,
                linewidth=2.1, markersize=6.5, label=LABELS[model])
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_xticks(nodes)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.28)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    network = read_rows(RESULTS / "model_metrics_by_node.csv")
    accuracy = read_rows(ML_RESULTS / "PARTNER_COMPATIBLE_final_ml_accuracy_comparison.csv")

    figures = (("PDR vs Number of Nodes", "PDR (%)", "Packet delivery ratio (%)", "pdr_vs_nodes"),
               ("Loss-aware delay vs Number of Nodes", "Loss-aware delay (ms)",
                "Completion delay with 10 s packet-loss timeout (ms)", "loss_aware_delay_vs_nodes"),
               ("ROR vs Number of Nodes", "ROR", "Routing overhead ratio", "ror_vs_nodes"))
    for title, column, ylabel, name in figures:
        fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
        plot_metric(ax, network, column, ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
        fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    names = [row["Model"] for row in accuracy]
    values = [100.0 * float(row["Row-level binary accuracy"]) for row in accuracy]
    bars = ax.bar(np.arange(len(names)), values, color=["#d62728", "#ff7f0e", "#1f77b4"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center")
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Binary classification accuracy (%)")
    ax.set_title("Held-out ML Accuracy", fontweight="bold")
    ax.grid(axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_binary_accuracy.png", bbox_inches="tight")
    fig.savefig(FIGURES / "ml_binary_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), dpi=180)
    for ax, (title, column, ylabel, _) in zip(axes.flat[:3], figures):
        plot_metric(ax, network, column, ylabel)
        ax.set_title(title, fontweight="bold")
    ax = axes.flat[3]
    bars = ax.bar(np.arange(len(names)), values, color=["#d62728", "#ff7f0e", "#1f77b4"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("Binary accuracy (%)")
    ax.set_title("Held-out ML Accuracy", fontweight="bold")
    ax.grid(axis="y", alpha=0.28)
    axes.flat[0].legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "combined_model_comparison.png", bbox_inches="tight")
    fig.savefig(FIGURES / "combined_model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(FIGURES)


if __name__ == "__main__":
    main()
