#!/usr/bin/env python3
"""Create clearer views without changing any measured result."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from generate_graphs import ACCURACY, COLORS, DELAY, NODES, PDR, ROR


OUT = Path(__file__).resolve().parent
MODELS = ["SVM", "DTC", "RF (25 trees)"]
LINESTYLES = {"SVM": "-", "DTC": "--", "RF (25 trees)": "-."}
MARKERS = {"SVM": "s", "DTC": "o", "RF (25 trees)": "^"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 300,
    }
)


def finish_axis(ax):
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)


def plot_accuracy(ax):
    y = np.arange(len(MODELS))
    for yi, model in zip(y, MODELS):
        value = ACCURACY[model]
        ax.plot(value, yi, marker=MARKERS[model], color=COLORS[model], markersize=8)
        ax.text(value + 0.18, yi, f"{value:.1f}%", va="center", fontsize=8)
    ax.axvline(75, color="#4D4D4D", linestyle=":", linewidth=2)
    ax.text(75.15, 1.72, "75% majority baseline", fontsize=7.5, color="#4D4D4D")
    ax.set_yticks(y, MODELS)
    ax.invert_yaxis()
    ax.set_xlim(73.5, 89.2)
    ax.set_xlabel("Held-out accuracy (%)")
    ax.set_title("(a) Binary Classification Accuracy")
    finish_axis(ax)


def plot_absolute(ax, data, ylabel, title, ylim, overlap_note=None):
    for model in MODELS:
        ax.plot(
            NODES,
            data[model],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            marker=MARKERS[model],
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            linewidth=2.0,
            label=model,
        )
    ax.plot(
        NODES,
        data["Fixed OC0 baseline"],
        color="#4D4D4D",
        linestyle=":",
        marker="D",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.0,
        label="Fixed OC0 baseline",
    )
    ax.set_xticks(NODES)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.98, 0.04, "Zoomed axis", transform=ax.transAxes, ha="right", fontsize=7, color="#555555")
    if overlap_note:
        ax.text(0.03, 0.95, overlap_note, transform=ax.transAxes, va="top", fontsize=7.2)
    finish_axis(ax)


def save(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def clarity_absolute():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9), constrained_layout=True)
    plot_accuracy(axes[0, 0])
    plot_absolute(
        axes[0, 1], PDR, "PDR (%)", "(b) Packet Delivery Ratio", (23, 94),
        "At 25 nodes, all ML results are 25.6%",
    )
    plot_absolute(
        axes[1, 0], DELAY, "Completion delay (ms)",
        "(c) Loss-Aware Delay (1-s Drop Penalty)", (285, 825),
        "DTC and RF overlap at 25 and 50 nodes",
    )
    plot_absolute(
        axes[1, 1], ROR, "Routing overhead ratio", "(d) Routing Overhead", (0.13, 0.51),
        "DTC and RF overlap at 25 and 50 nodes",
    )
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=True)
    save(fig, "combined_four_panel_clarity")


def grouped_improvement(ax, data, ylabel, title, better_when_lower=False):
    x = np.arange(len(NODES))
    width = 0.23
    baseline = np.asarray(data["Fixed OC0 baseline"], dtype=float)
    for index, model in enumerate(MODELS):
        values = np.asarray(data[model], dtype=float)
        improvement = baseline - values if better_when_lower else values - baseline
        ax.bar(
            x + (index - 1) * width,
            improvement,
            width,
            color=COLORS[model],
            edgecolor="black",
            linewidth=0.45,
            label=model,
        )
    ax.axhline(0, color="#4D4D4D", linestyle="--", linewidth=1.5, label="Fixed OC0 baseline")
    ax.set_xticks(x, NODES)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    finish_axis(ax)


def improvement_figure():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), constrained_layout=True)
    grouped_improvement(axes[0], PDR, "PDR gain (percentage points)", "(a) PDR Gain")
    grouped_improvement(axes[1], DELAY, "Delay reduction (ms)", "(b) Delay Reduction", True)
    grouped_improvement(axes[2], ROR, "Overhead-ratio reduction", "(c) Overhead Reduction", True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=True)
    save(fig, "improvement_over_fixed_oc0_baseline")


if __name__ == "__main__":
    clarity_absolute()
    improvement_figure()
    print(f"Clarity figures written to {OUT}")
