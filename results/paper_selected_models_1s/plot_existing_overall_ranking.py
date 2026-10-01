#!/usr/bin/env python3
"""Plot overall model ranking from existing evaluation CSV files only."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent
NETWORK_CSV = OUT / "real_baseline_network_results_with_95ci.csv"
CLASSIFICATION_CSV = OUT / "real_baseline_classification_results.csv"
METHODS = ["SVM", "DTC", "RF", "Baseline"]
COLORS = ["#E52521", "#FF7F0E", "#2878B5", "#B52B65"]


def network_averages():
    columns = {
        "PDR": "PDR (%)_estimate",
        "Delay": "Loss-aware delay (ms)_estimate",
        "ROR": "Routing overhead ratio_estimate",
    }
    collected = defaultdict(lambda: defaultdict(list))
    with NETWORK_CSV.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for metric, column in columns.items():
                collected[row["Method"]][metric].append(float(row[column]))
    return {
        method: {
            metric: sum(values) / len(values)
            for metric, values in collected[method].items()
        }
        for method in METHODS
    }


def top1_accuracy():
    values = {}
    with CLASSIFICATION_CSV.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values[row["method"]] = float(row["top1_accuracy_percent"])
    values["Baseline"] = 25.0
    return values


def bar_panel(ax, values, ylabel, title, decimals, ylim):
    bars = ax.bar(
        METHODS, values, color=COLORS, edgecolor="black",
        linewidth=0.6, width=0.64,
    )
    offset = (ylim[1] - ylim[0]) * 0.022
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.{decimals}f}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
    ax.set_axisbelow(True)


def save(fig):
    stem = "overall_model_ranking_existing_data"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    network = network_averages()
    accuracy = top1_accuracy()

    pdr = [network[method]["PDR"] for method in METHODS]
    delay = [network[method]["Delay"] for method in METHODS]
    ror = [network[method]["ROR"] for method in METHODS]
    top1 = [accuracy[method] for method in METHODS]

    with plt.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9,
        "axes.titlesize": 10.5, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
        bar_panel(
            axes[0, 0], pdr, "Mean PDR (%)",
            "(a) Packet Delivery Ratio (higher is better)", 2, (0, 82),
        )
        bar_panel(
            axes[0, 1], delay, "Mean loss-aware delay (ms)",
            "(b) Loss-Aware Delay (lower is better)", 2, (0, 570),
        )
        bar_panel(
            axes[1, 0], ror, "Mean routing overhead ratio",
            "(c) Routing Overhead (lower is better)", 4, (0, 0.31),
        )
        bar_panel(
            axes[1, 1], top1, "Correct OC selection accuracy (%)",
            "(d) Top-1 OC Selection (higher is better)", 2, (0, 100),
        )
        fig.suptitle(
            "Overall Results Averaged Across 25, 50, 75, and 100 Nodes",
            fontsize=12,
        )
        save(fig)


if __name__ == "__main__":
    main()
