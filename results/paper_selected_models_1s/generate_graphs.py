#!/usr/bin/env python3
"""Generate paper-ready figures for the selected SVM, DTC, and RF models."""

from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent
NODES = [25, 50, 75, 100]

COLORS = {
    "SVM": "#D62728",
    "DTC": "#FF7F0E",
    "RF (25 trees)": "#1F77B4",
    "Fixed OC0 baseline": "#4D4D4D",
}
MARKERS = {
    "SVM": "s",
    "DTC": "o",
    "RF (25 trees)": "^",
    "Fixed OC0 baseline": "D",
}

ACCURACY = {
    "SVM": 87.25,
    "DTC": 85.0833333333,
    "RF (25 trees)": 84.50,
}

PDR = {
    "SVM": [25.6, 75.0, 90.9, 91.0],
    "DTC": [25.6, 65.6, 91.0, 90.4],
    "RF (25 trees)": [25.6, 65.6, 92.0, 82.6],
    "Fixed OC0 baseline": [25.3, 58.1, 88.8, 84.3],
}

DELAY = {
    "SVM": [799.067, 480.518, 324.503, 305.974],
    "DTC": [798.036, 557.199, 321.778, 312.555],
    "RF (25 trees)": [798.036, 557.199, 308.949, 373.241],
    "Fixed OC0 baseline": [804.867, 604.058, 338.918, 357.170],
}

ROR = {
    "SVM": [0.428767123, 0.143606109, 0.161371429, 0.175217812],
    "DTC": [0.430327869, 0.183259501, 0.159798535, 0.178571429],
    "RF (25 trees)": [0.430327869, 0.183259501, 0.159962582, 0.217676768],
    "Fixed OC0 baseline": [0.495828367, 0.239878543, 0.174316180, 0.205934121],
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    }
)


def style_axis(ax):
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)


def draw_accuracy(ax, title="Held-Out Binary Classification Accuracy"):
    labels = list(ACCURACY)
    values = [ACCURACY[label] for label in labels]
    bars = ax.bar(
        labels,
        values,
        color=[COLORS[label] for label in labels],
        edgecolor="black",
        linewidth=0.6,
        width=0.62,
    )
    ax.axhline(
        75.0,
        color=COLORS["Fixed OC0 baseline"],
        linestyle="--",
        linewidth=1.5,
        label="Majority-class baseline (75%)",
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    style_axis(ax)


def draw_lines(ax, data, ylabel, title, ylim=None):
    for label, values in data.items():
        is_baseline = label == "Fixed OC0 baseline"
        ax.plot(
            NODES,
            values,
            color=COLORS[label],
            marker=MARKERS[label],
            linestyle="--" if is_baseline else "-",
            linewidth=1.8 if is_baseline else 2.0,
            markersize=5.5,
            markerfacecolor="white" if is_baseline else COLORS[label],
            markeredgewidth=1.1,
            label=label,
        )
    ax.set_xticks(NODES)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", frameon=True)
    style_axis(ax)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def individual_figures():
    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    draw_accuracy(ax)
    save(fig, "accuracy_with_baseline")

    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    draw_lines(ax, PDR, "Packet delivery ratio (%)", "PDR versus Number of Nodes", (0, 100))
    save(fig, "pdr_with_fixed_oc0_baseline")

    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    draw_lines(
        ax,
        DELAY,
        "Loss-aware completion delay (ms)",
        "Delay versus Number of Nodes\n(1-s penalty per dropped packet)",
    )
    save(fig, "delay_with_fixed_oc0_baseline")

    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    draw_lines(ax, ROR, "Routing overhead ratio", "Routing Overhead versus Number of Nodes", (0, 0.55))
    save(fig, "ror_with_fixed_oc0_baseline")


def combined_figure():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    draw_accuracy(axes[0, 0], "(a) Held-Out Binary Accuracy")
    draw_lines(axes[0, 1], PDR, "PDR (%)", "(b) Packet Delivery Ratio", (0, 100))
    draw_lines(
        axes[1, 0],
        DELAY,
        "Completion delay (ms)",
        "(c) Loss-Aware Delay (1-s Drop Penalty)",
    )
    draw_lines(axes[1, 1], ROR, "Routing overhead ratio", "(d) Routing Overhead", (0, 0.55))
    save(fig, "combined_four_panel")


if __name__ == "__main__":
    individual_figures()
    combined_figure()
    print(f"Figures written to {OUT}")
