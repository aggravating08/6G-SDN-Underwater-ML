#!/usr/bin/env python3
"""Plot baseline-relative improvements from the saved evaluation summaries.

This script does not fit models, rerun simulations, resample scenarios, or alter
measurements.  It only transforms the already-saved point estimates into
differences from the uniform-random Baseline for a clearer companion view.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUT = Path(__file__).resolve().parent
RESULTS = OUT / "real_baseline_network_results_with_95ci.csv"
CLASSIFICATION = OUT / "real_baseline_classification_results.csv"
NODES = [25, 50, 75, 100]
METHODS = ["SVM", "DTC", "RF"]
COLORS = {"SVM": "#E52521", "DTC": "#FF7F0E", "RF": "#2878B5"}
MARKERS = {"SVM": "s", "DTC": "o", "RF": "^"}


def save(fig, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def improvement_table(results):
    baseline = (
        results[results.Method.eq("Baseline")]
        .set_index("nodes")
        .sort_index()
    )
    frames = []
    for method in METHODS:
        current = results[results.Method.eq(method)].set_index("nodes").sort_index()
        frame = pd.DataFrame({
            "nodes": current.index,
            "Method": method,
            "PDR gain (percentage points)": (
                current["PDR (%)_estimate"] - baseline["PDR (%)_estimate"]
            ),
            "Delay reduction (ms)": (
                baseline["Loss-aware delay (ms)_estimate"]
                - current["Loss-aware delay (ms)_estimate"]
            ),
            "Overhead reduction": (
                baseline["Routing overhead ratio_estimate"]
                - current["Routing overhead ratio_estimate"]
            ),
        }).reset_index(drop=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def line_panel(ax, values, metric, ylabel, title, decimals):
    label_offsets = {"SVM": (-7, 9), "DTC": (7, -14), "RF": (0, 8)}
    for method in METHODS:
        subset = values[values.Method.eq(method)].sort_values("nodes")
        ax.plot(
            subset.nodes,
            subset[metric],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.2,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=method,
        )
        for x, y in zip(subset.nodes, subset[metric]):
            ax.annotate(
                f"{y:.{decimals}f}",
                (x, y),
                xytext=label_offsets[method],
                textcoords="offset points",
                ha="center",
                fontsize=6.7,
                color=COLORS[method],
            )
    ax.axhline(0, color="#555555", linewidth=1.0, label="Baseline")
    ax.set_xticks(NODES)
    ax.set_xlim(20, 105)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#B0B0B0", linewidth=0.6, alpha=0.38)
    ax.set_axisbelow(True)
    ax.margins(y=0.18)


def main():
    results = pd.read_csv(RESULTS)
    classification = pd.read_csv(CLASSIFICATION)
    values = improvement_table(results)
    values.to_csv(OUT / "real_baseline_relative_improvements.csv", index=False)

    accuracy = dict(zip(
        classification.method,
        classification.binary_accuracy_percent,
    ))
    accuracy["Baseline"] = 62.5

    with plt.rc_context({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6), constrained_layout=True)
        line_panel(
            axes[0, 0], values,
            "PDR gain (percentage points)",
            "PDR gain (percentage points)",
            "(a) PDR Improvement over Baseline", 1,
        )
        line_panel(
            axes[0, 1], values,
            "Delay reduction (ms)",
            "Delay reduction (ms)",
            "(b) Delay Reduction from Baseline", 0,
        )
        line_panel(
            axes[1, 0], values,
            "Overhead reduction",
            "Routing-overhead reduction",
            "(c) Overhead Reduction from Baseline", 3,
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[0, 0].legend(handles, labels, loc="best", frameon=True, fancybox=False)

        ax = axes[1, 1]
        names = ["SVM", "DTC", "RF", "Baseline"]
        bars = ax.bar(
            names,
            [accuracy[name] for name in names],
            color=[COLORS.get(name, "#B52B65") for name in names],
            edgecolor="black",
            linewidth=0.55,
        )
        for bar, name in zip(bars, names):
            value = accuracy[name]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.2f}%",
                ha="center",
                fontsize=8,
            )
        ax.set_ylim(0, 100)
        ax.set_ylabel("Binary accuracy (%)")
        ax.set_title("(d) Independent Binary Accuracy")
        ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
        ax.set_axisbelow(True)

        save(fig, "real_baseline_relative_improvement_clarity")


if __name__ == "__main__":
    main()
