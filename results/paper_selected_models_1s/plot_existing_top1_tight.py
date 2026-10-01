#!/usr/bin/env python3
"""Replot saved network results with top-1 OC-selection accuracy.

Only existing CSV summaries are read. No model is fitted and no simulation,
bootstrap, or evaluation data is regenerated.
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(os.environ.get(
    "TOP1_RESULTS_DIRECTORY", str(Path(__file__).resolve().parent)
))
NETWORK_CSV = OUT / "real_baseline_network_results_with_95ci.csv"
CLASSIFICATION_CSV = OUT / "real_baseline_classification_results.csv"
NODES = [25, 50, 75, 100]
METHODS = ["SVM", "DTC", "RF", "Baseline"]
COLORS = {"SVM": "#E52521", "DTC": "#FF7F0E", "RF": "#2878B5", "Baseline": "#B52B65"}
MARKERS = {"SVM": "s", "DTC": "o", "RF": "^", "Baseline": "D"}


def read_network():
    values = {}
    with NETWORK_CSV.open(newline="") as stream:
        for row in csv.DictReader(stream):
            method = row["Method"]
            nodes = int(row["nodes"])
            values[(method, nodes)] = {
                "PDR": float(row["PDR (%)_estimate"]),
                "Delay": float(row["Loss-aware delay (ms)_estimate"]),
                "ROR": float(row["Routing overhead ratio_estimate"]),
            }
    return values


def read_top1():
    values = {}
    with CLASSIFICATION_CSV.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values[row["method"]] = float(row["top1_accuracy_percent"])
    values["Baseline"] = 25.0
    return values


def data_limits(network, metric, padding_fraction=0.035):
    values = [network[(method, nodes)][metric] for method in METHODS for nodes in NODES]
    span = max(values) - min(values)
    padding = span * padding_fraction
    return min(values) - padding, max(values) + padding


def line_panel(ax, network, metric, ylabel, title):
    for method in METHODS:
        y = [network[(method, nodes)][metric] for nodes in NODES]
        ax.plot(
            NODES, y,
            color=COLORS[method], marker=MARKERS[method],
            linewidth=2.0, markersize=6.0,
            markeredgecolor="white", markeredgewidth=0.55,
            label=method,
        )
    ax.set_xticks(NODES)
    ax.set_xlim(20, 105)
    ax.set_ylim(*data_limits(network, metric))
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#B0B0B0", linewidth=0.6, alpha=0.38)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#333333")


def save(fig):
    stem = "real_baseline_combined_top1_tight_axes"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    network = read_network()
    top1 = read_top1()

    with plt.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
        line_panel(
            axes[0, 0], network, "PDR", "Packet delivery ratio (%)",
            "(a) Packet Delivery Ratio",
        )
        line_panel(
            axes[0, 1], network, "Delay", "Loss-aware delay (ms)",
            "(b) Loss-Aware Delay",
        )
        line_panel(
            axes[1, 0], network, "ROR", "Routing overhead ratio",
            "(c) Routing Overhead",
        )

        ax = axes[1, 1]
        bars = ax.bar(
            METHODS, [top1[method] for method in METHODS],
            color=[COLORS[method] for method in METHODS],
            edgecolor="black", linewidth=0.55,
        )
        for bar, method in zip(bars, METHODS):
            value = top1[method]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.2f}%",
                ha="center", fontsize=8,
            )
        ax.set_ylim(0, 100)
        ax.set_ylabel("Correct OC selection accuracy (%)")
        ax.set_title("(d) Top-1 OC Selection Accuracy")
        ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
        ax.set_axisbelow(True)

        save(fig)


if __name__ == "__main__":
    main()
