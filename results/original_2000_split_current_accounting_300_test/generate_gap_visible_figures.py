#!/usr/bin/env python3
"""Show the measured SVM/DTC differences without changing any result values."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
DATA = pd.read_csv(OUT / "network_metrics_by_node.csv")
NODES = [25, 50, 75, 100]
METHODS = ["SVM", "DTC", "RF", "Baseline"]
COLORS = {"SVM": "#E52521", "DTC": "#FF7F0E", "RF": "#2878B5", "Baseline": "#B52B65"}
MARKERS = {"SVM": "s", "DTC": "o", "RF": "^", "Baseline": "D"}


def values(method: str, column: str) -> np.ndarray:
    return (
        DATA.loc[DATA.Method.eq(method)]
        .set_index("Nodes")
        .loc[NODES, column]
        .to_numpy(float)
    )


def make(column: str, ylabel: str, title: str, stem: str, higher_is_better: bool) -> None:
    with plt.rc_context({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }):
        fig, (main, difference) = plt.subplots(
            2,
            1,
            figsize=(7.4, 6.0),
            gridspec_kw={"height_ratios": [4.2, 1.35]},
            constrained_layout=True,
        )
        all_values = []
        for method in METHODS:
            y = values(method, column)
            all_values.extend(y)
            main.plot(
                NODES,
                y,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.2,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=method,
            )
        low, high = min(all_values), max(all_values)
        margin = (high - low) * 0.06
        main.set_xlim(20, 105)
        main.set_ylim(low - margin, high + margin)
        main.set_xticks(NODES)
        main.set_ylabel(ylabel)
        main.set_title(title)
        main.grid(True, alpha=0.32)
        main.legend(loc="best", frameon=True, fancybox=False, edgecolor="#333333")

        svm = values("SVM", column)
        dtc = values("DTC", column)
        advantage = svm - dtc if higher_is_better else dtc - svm
        bar_colors = [COLORS["SVM"] if item >= 0 else COLORS["DTC"] for item in advantage]
        difference.axhline(0, color="#333333", linewidth=1.0)
        bars = difference.bar(NODES, advantage, width=10, color=bar_colors, alpha=0.88)
        span = max(abs(advantage).max(), 1e-9)
        for bar, item in zip(bars, advantage):
            difference.text(
                bar.get_x() + bar.get_width() / 2,
                item + (0.08 * span if item >= 0 else -0.08 * span),
                f"{item:+.3f}",
                ha="center",
                va="bottom" if item >= 0 else "top",
                fontsize=8.5,
            )
        difference.set_xlim(20, 105)
        difference.set_ylim(-1.35 * span, 1.35 * span)
        difference.set_xticks(NODES)
        difference.set_xlabel("Number of nodes")
        difference.set_ylabel("SVM\nadvantage")
        difference.grid(axis="y", alpha=0.25)
        difference.text(
            0.995,
            0.94,
            "Positive = SVM better",
            transform=difference.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
        )

        for extension in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


make("PDR (%)", "Packet delivery ratio (%)", "PDR vs Number of Nodes", "pdr_with_svm_dtc_difference", True)
make(
    "Loss-aware delay (ms)",
    "Loss-aware delay (ms)",
    "Delay vs Number of Nodes",
    "delay_with_svm_dtc_difference",
    False,
)
make(
    "Routing overhead ratio",
    "Routing overhead ratio",
    "ROR vs Number of Nodes",
    "ror_with_svm_dtc_difference",
    False,
)
