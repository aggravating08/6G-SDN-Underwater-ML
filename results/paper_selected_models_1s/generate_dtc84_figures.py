#!/usr/bin/env python3
"""Preferred-style figures using the original matched 20-scenario ledger.

SVM, DTC, and RF values are unchanged from the preferred figure. Baseline
always labels OC0 as the selected controller and OC1--OC3 as not selected.
OC0 is the weakest fixed OC overall on this 20-scenario evaluation.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
NODES = np.array([25, 50, 75, 100])

ACCURACY = {
    "SVM": 87.25,
    "DTC": 86.4166666667,
    "RF": 77.9166666667,
    "Baseline": 61.0,
}

NETWORK = {
    "PDR (%)": {
        "SVM": [25.6, 75.0, 90.9, 91.0],
        "DTC": [23.6, 72.0, 86.3, 90.1],
        "RF": [21.6, 62.7, 83.6, 82.6],
        "Baseline": [25.3, 58.1, 88.8, 84.3],
    },
    "Loss-aware delay (ms)": {
        "SVM": [799.067, 480.518, 324.503, 305.974],
        "DTC": [817.453, 511.034, 361.853, 312.535],
        "RF": [828.325, 573.709, 379.270, 373.241],
        "Baseline": [804.867, 604.058, 338.918, 357.170],
    },
    "Routing overhead ratio": {
        "SVM": [0.428767123, 0.143606109, 0.161371429, 0.175217812],
        "DTC": [0.447010870, 0.156326988, 0.182176839, 0.178929357],
        "RF": [0.552667579, 0.205078600, 0.196893670, 0.217676768],
        "Baseline": [0.495828367, 0.239878543, 0.174316180, 0.205934121],
    },
}

COLORS = {
    "SVM": "#E52521",
    "DTC": "#FF7F0E",
    "RF": "#2878B5",
    "Baseline": "#B52B65",
}
MARKERS = {
    "SVM": "s",
    "DTC": "o",
    "RF": "^",
    "Baseline": "D",
}


def save(fig, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axis(ax):
    ax.set_xticks(NODES)
    ax.set_xlim(20, 105)
    ax.grid(True, color="#B0B0B0", linewidth=0.6, alpha=0.38)
    ax.set_axisbelow(True)


def plot_line(ax, metric, ylabel, title, ylim, legend_location):
    for method, values in NETWORK[metric].items():
        ax.plot(
            NODES,
            values,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=method,
        )
    ax.set_ylim(*ylim)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)
    ax.legend(loc=legend_location, frameon=True, fancybox=False, edgecolor="#333333")


def individual_figures():
    specifications = [
        ("PDR (%)", "Packet delivery ratio (%)", "PDR vs Number of Nodes", (12, 98), "upper left", "dtc84_pdr_vs_nodes"),
        ("Loss-aware delay (ms)", "Loss-aware delay (ms)", "Delay vs Number of Nodes", (250, 890), "upper right", "dtc84_delay_vs_nodes"),
        ("Routing overhead ratio", "Routing overhead ratio", "Routing Overhead vs Number of Nodes", (0.12, 0.64), "upper right", "dtc84_ror_vs_nodes"),
    ]
    for metric, ylabel, title, ylim, legend_location, stem in specifications:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        plot_line(ax, metric, ylabel, title, ylim, legend_location)
        save(fig, stem)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    methods = list(ACCURACY)
    values = [ACCURACY[method] for method in methods]
    bars = ax.bar(
        methods,
        values,
        color=[COLORS[method] for method in methods],
        edgecolor="black",
        linewidth=0.6,
        width=0.62,
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Binary accuracy (%)")
    ax.set_title("Held-Out Binary Classification Accuracy")
    ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
    ax.set_axisbelow(True)
    save(fig, "dtc84_binary_accuracy")


def combined_figure():
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    plot_line(axes[0, 0], "PDR (%)", "Packet delivery ratio (%)", "(a) Packet Delivery Ratio", (12, 98), "upper left")
    plot_line(axes[0, 1], "Loss-aware delay (ms)", "Loss-aware delay (ms)", "(b) Loss-Aware Delay", (250, 890), "upper right")
    plot_line(axes[1, 0], "Routing overhead ratio", "Routing overhead ratio", "(c) Routing Overhead", (0.12, 0.64), "upper right")

    ax = axes[1, 1]
    methods = list(ACCURACY)
    values = [ACCURACY[method] for method in methods]
    bars = ax.bar(methods, values, color=[COLORS[method] for method in methods], edgecolor="black", linewidth=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Binary accuracy (%)")
    ax.set_title("(d) Held-Out Binary Accuracy")
    ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
    ax.set_axisbelow(True)
    save(fig, "dtc84_combined_four_metrics")


if __name__ == "__main__":
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    ):
        individual_figures()
        combined_figure()
    print(f"DTC-84 figures written to {OUT}")
