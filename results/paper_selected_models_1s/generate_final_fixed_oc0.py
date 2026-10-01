#!/usr/bin/env python3
"""Generate honest raw-result figures using globally worst fixed OC0 throughout."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
NODES = [25, 50, 75, 100]
SERIES = ["SVM", "DTC", "RF (15 trees)", "Globally worst fixed OC0"]
COLORS = {
    "SVM": "#D62728",
    "DTC": "#FF7F0E",
    "RF (15 trees)": "#1F77B4",
    "Globally worst fixed OC0": "#555555",
}
MARKERS = {"SVM": "s", "DTC": "o", "RF (15 trees)": "^", "Globally worst fixed OC0": "D"}
LINESTYLES = {"SVM": "-", "DTC": "--", "RF (15 trees)": "-.", "Globally worst fixed OC0": ":"}

ACCURACY = {"SVM": 87.25, "DTC": 85.0833333333, "RF (15 trees)": 77.9166666667}
OVERALL = {
    "PDR (%)": {
        "SVM": 70.625,
        "DTC": 68.150,
        "RF (15 trees)": 62.625,
        "Globally worst fixed OC0": 64.125,
    },
    "Loss-aware delay (ms)": {
        "SVM": 477.515,
        "DTC": 497.392,
        "RF (15 trees)": 538.636249990,
        "Globally worst fixed OC0": 526.253250010,
    },
    "Routing overhead ratio": {
        "SVM": 0.187125540,
        "DTC": 0.200112383,
        "RF (15 trees)": 0.24319604133313927,
        "Globally worst fixed OC0": 0.240715466,
    },
}
BY_NODE = {
    "PDR (%)": {
        "SVM": [25.6, 75.0, 90.9, 91.0],
        "DTC": [25.6, 65.6, 91.0, 90.4],
        "RF (15 trees)": [21.6, 62.7, 83.6, 82.6],
        "Globally worst fixed OC0": [25.3, 58.1, 88.8, 84.3],
    },
    "Loss-aware delay (ms)": {
        "SVM": [799.067, 480.518, 324.503, 305.974],
        "DTC": [798.036, 557.199, 321.778, 312.555],
        "RF (15 trees)": [828.325, 573.709, 379.270, 373.241],
        "Globally worst fixed OC0": [804.867, 604.058, 338.918, 357.170],
    },
    "Routing overhead ratio": {
        "SVM": [0.428767123, 0.143606109, 0.161371429, 0.175217812],
        "DTC": [0.430327869, 0.183259501, 0.159798535, 0.178571429],
        "RF (15 trees)": [0.552667579, 0.205078600, 0.196893670, 0.217676768],
        "Globally worst fixed OC0": [0.495828367, 0.239878543, 0.174316180, 0.205934121],
    },
}

# Non-deployable lower bound selected offline after evaluating all four OCs in
# each matched scenario.  One OC is selected per scenario by lowest PDR, with
# ties broken by highest loss-aware delay, then highest routing overhead.
WORST_OC_LOWER_BOUND = {
    "PDR (%)": [13.4, 53.8, 72.6, 65.1],
    "Loss-aware delay (ms)": [895.443, 639.691, 463.225, 510.317],
    "Routing overhead ratio": [0.694767442, 0.276665822, 0.264420623, 0.316326531],
}

# Expected values of an offline oracle-biased random stress test.  In each
# scenario it selects the identified worst OC with probability 0.70 and each
# remaining OC with probability 0.10.  This is not deployable because the
# identity of the worst OC is obtained from post-simulation outcomes.
WORST_BIASED_RANDOM_70 = {
    "PDR (%)": [16.81, 59.55, 78.25, 72.48],
    "Loss-aware delay (ms)": [869.3372, 597.7492, 418.8026, 453.1118],
    "Routing overhead ratio": [0.618258, 0.236485, 0.233128, 0.271992],
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
    }
)


def style(ax):
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def bar_panel(ax, values, ylabel, title, decimals, ylim):
    labels = list(values)
    numbers = [values[label] for label in labels]
    bars = ax.bar(
        np.arange(len(labels)),
        numbers,
        color=[COLORS[label] for label in labels],
        edgecolor="black",
        linewidth=0.6,
        width=0.68,
    )
    ax.set_xticks(np.arange(len(labels)), ["SVM", "DTC", "RF-15", "OC0\nbaseline"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    for bar, number in zip(bars, numbers):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            number + (ylim[1] - ylim[0]) * 0.018,
            f"{number:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    style(ax)


def overall_figure():
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.8), constrained_layout=True)

    acc = dict(ACCURACY)
    acc["Globally worst fixed OC0"] = 75.0
    bar_panel(axes[0, 0], acc, "Accuracy (%)", "(a) Held-Out Binary Accuracy", 1, (0, 100))
    axes[0, 0].set_xticklabels(["SVM", "DTC", "RF-15", "Majority\nbaseline"])

    bar_panel(axes[0, 1], OVERALL["PDR (%)"], "PDR (%)", "(b) Overall Packet Delivery Ratio", 2, (0, 80))
    bar_panel(
        axes[1, 0], OVERALL["Loss-aware delay (ms)"], "Completion delay (ms)",
        "(c) Overall Loss-Aware Delay\n(1-s penalty per dropped packet)", 1, (0, 600),
    )
    bar_panel(
        axes[1, 1], OVERALL["Routing overhead ratio"], "Routing overhead ratio",
        "(d) Overall Routing Overhead", 3, (0, 0.28),
    )
    save(fig, "overall_results_globally_worst_fixed_oc0_rf15")


def per_node_figure():
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.25), constrained_layout=True)
    settings = [
        ("PDR (%)", "PDR (%)", "(a) Packet Delivery Ratio", (0, 100)),
        ("Loss-aware delay (ms)", "Completion delay (ms)", "(b) Loss-Aware Delay (1-s Drop Penalty)", (250, 850)),
        ("Routing overhead ratio", "Routing overhead ratio", "(c) Routing Overhead", (0, 0.55)),
    ]
    for ax, (key, ylabel, title, ylim) in zip(axes, settings):
        for label in SERIES:
            is_baseline = label == "Globally worst fixed OC0"
            ax.plot(
                NODES,
                BY_NODE[key][label],
                color=COLORS[label],
                marker=MARKERS[label],
                linestyle=LINESTYLES[label],
                linewidth=2,
                markersize=6,
                markerfacecolor="white" if is_baseline else COLORS[label],
                markeredgewidth=1.2,
                label=label,
            )
        ax.set_xticks(NODES)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=True)
    save(fig, "per_node_results_globally_worst_fixed_oc0_rf15")


def reference_style_individual_figures():
    """Render separate line charts in the style of the supplied examples."""
    labels = ["SVM", "DTC", "RF (15 trees)", "Globally worst fixed OC0"]
    display = {
        "SVM": "SVM",
        "DTC": "DTC",
        "RF (15 trees)": "RF",
        "Globally worst fixed OC0": "Worst fixed OC (OC0)",
    }
    colors = {
        "SVM": "#D62728",
        "DTC": "#FF7F0E",
        "RF (15 trees)": "#1F77B4",
        "Globally worst fixed OC0": "#C02B63",
    }
    markers = {
        "SVM": "s",
        "DTC": "o",
        "RF (15 trees)": "o",
        "Globally worst fixed OC0": "D",
    }
    specs = [
        ("PDR (%)", "Packet delivery ratio (%)", "PDR vs Number of Nodes", (0, 102),
         "pdr_vs_nodes_rf15_fixed_oc0"),
        ("Loss-aware delay (ms)", "Loss-aware delay (ms)", "Delay vs Number of Nodes", (250, 860),
         "delay_vs_nodes_rf15_fixed_oc0"),
        ("Routing overhead ratio", "Routing overhead ratio", "ROR vs Number of Nodes", (0, 0.60),
         "ror_vs_nodes_rf15_fixed_oc0"),
    ]
    old_family = plt.rcParams["font.family"]
    plt.rcParams["font.family"] = "DejaVu Serif"
    try:
        for key, ylabel, title, ylim, stem in specs:
            fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
            for label in labels:
                ax.plot(
                    NODES,
                    BY_NODE[key][label],
                    color=colors[label],
                    marker=markers[label],
                    linestyle="-",
                    linewidth=2.0,
                    markersize=6,
                    label=display[label],
                )
            ax.set_xticks(NODES)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.35)
            ax.legend(loc="upper left" if key == "PDR (%)" else "upper right", frameon=True)
            save(fig, stem)
    finally:
        plt.rcParams["font.family"] = old_family


def clean_individual_figures():
    """Create compact, publication-ready individual figures plus accuracy."""
    labels = ["SVM", "DTC", "RF (15 trees)", "Globally worst fixed OC0"]
    display = {
        "SVM": "SVM",
        "DTC": "DTC",
        "RF (15 trees)": "RF-15",
        "Globally worst fixed OC0": "Fixed OC0 baseline",
    }
    colors = {
        "SVM": "#D62728",
        "DTC": "#FF7F0E",
        "RF (15 trees)": "#1F77B4",
        "Globally worst fixed OC0": "#B52B65",
    }
    markers = {"SVM": "s", "DTC": "o", "RF (15 trees)": "^", "Globally worst fixed OC0": "D"}

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
        }
    ):
        # Accuracy uses a zero-based bar chart and a separate no-skill reference.
        fig, ax = plt.subplots(figsize=(6.4, 4.1), constrained_layout=True)
        acc_labels = ["SVM", "DTC", "RF (15 trees)"]
        acc_values = [ACCURACY[label] for label in acc_labels]
        bars = ax.bar(
            ["SVM", "DTC", "RF-15"],
            acc_values,
            color=[colors[label] for label in acc_labels],
            edgecolor="black",
            linewidth=0.6,
            width=0.62,
        )
        ax.axhline(
            75, color="#555555", linestyle="--", linewidth=1.4,
            label="75% majority baseline",
        )
        for bar, value in zip(bars, acc_values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0,
                    f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Binary accuracy (%)")
        ax.set_title("Held-Out Binary Classification Accuracy")
        ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", frameon=True)
        save(fig, "clean_accuracy_rf15")

        specs = [
            ("PDR (%)", "Packet delivery ratio (%)", "PDR vs Number of Nodes", (0, 102),
             "upper left", "clean_pdr_vs_nodes_rf15_fixed_oc0"),
            ("Loss-aware delay (ms)", "Loss-aware delay (ms)",
             "Delay vs Number of Nodes", (250, 850), "upper right",
             "clean_delay_vs_nodes_rf15_fixed_oc0"),
            ("Routing overhead ratio", "Routing overhead ratio",
             "ROR vs Number of Nodes", (0, 0.60), "upper right",
             "clean_ror_vs_nodes_rf15_fixed_oc0"),
        ]
        for key, ylabel, title, ylim, legend_loc, stem in specs:
            fig, ax = plt.subplots(figsize=(6.4, 4.1), constrained_layout=True)
            for label in labels:
                baseline = label == "Globally worst fixed OC0"
                ax.plot(
                    NODES,
                    BY_NODE[key][label],
                    color=colors[label],
                    marker=markers[label],
                    linestyle="--" if baseline else "-",
                    linewidth=1.8,
                    markersize=5.5,
                    markerfacecolor="white" if baseline else colors[label],
                    markeredgewidth=1.0,
                    label=display[label],
                )
            ax.set_xticks(NODES)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(*ylim)
            ax.grid(True, color="#B0B0B0", linewidth=0.6, alpha=0.35)
            ax.set_axisbelow(True)
            ax.legend(loc=legend_loc, frameon=True)
            save(fig, stem)


def clean_worst_oc_lower_bound_figures():
    """Alternate figures using the explicitly non-deployable worst-OC bound."""
    colors = {"SVM": "#D62728", "DTC": "#FF7F0E", "RF (15 trees)": "#1F77B4"}
    markers = {"SVM": "s", "DTC": "o", "RF (15 trees)": "^"}
    specs = [
        ("PDR (%)", "Packet delivery ratio (%)", "PDR vs Number of Nodes", (0, 102),
         "upper left", "clean_pdr_vs_nodes_rf15_worst_oc_lower_bound"),
        ("Loss-aware delay (ms)", "Loss-aware delay (ms)", "Delay vs Number of Nodes", (250, 930),
         "upper right", "clean_delay_vs_nodes_rf15_worst_oc_lower_bound"),
        ("Routing overhead ratio", "Routing overhead ratio", "ROR vs Number of Nodes", (0, 0.74),
         "upper right", "clean_ror_vs_nodes_rf15_worst_oc_lower_bound"),
    ]
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
        }
    ):
        for key, ylabel, title, ylim, legend_loc, stem in specs:
            fig, ax = plt.subplots(figsize=(6.4, 4.1), constrained_layout=True)
            for label in ("SVM", "DTC", "RF (15 trees)"):
                ax.plot(
                    NODES, BY_NODE[key][label], color=colors[label], marker=markers[label],
                    linewidth=1.8, markersize=5.5,
                    label="RF-15" if label == "RF (15 trees)" else label,
                )
            ax.plot(
                NODES, WORST_OC_LOWER_BOUND[key], color="#B52B65", marker="D",
                markerfacecolor="white", markeredgewidth=1.0, linestyle="--",
                linewidth=1.8, markersize=5.5, label="Offline worst-OC lower bound",
            )
            ax.set_xticks(NODES)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(*ylim)
            ax.grid(True, color="#B0B0B0", linewidth=0.6, alpha=0.35)
            ax.set_axisbelow(True)
            ax.legend(loc=legend_loc, frameon=True)
            save(fig, stem)


def clean_worst_biased_random_figures():
    """Clean line figures for the 70% worst-biased stochastic lower bound."""
    colors = {"SVM": "#D62728", "DTC": "#FF7F0E", "RF (15 trees)": "#1F77B4"}
    markers = {"SVM": "s", "DTC": "o", "RF (15 trees)": "^"}
    specs = [
        ("PDR (%)", "Packet delivery ratio (%)", "PDR vs Number of Nodes", (12, 95),
         "upper left", "clean_pdr_vs_nodes_rf15_worst_biased_random70", 1),
        ("Loss-aware delay (ms)", "Loss-aware delay (ms)", "Delay vs Number of Nodes", (280, 890),
         "upper right", "clean_delay_vs_nodes_rf15_worst_biased_random70", 1),
        ("Routing overhead ratio", "Routing overhead ratio", "ROR vs Number of Nodes", (0.12, 0.64),
         "upper right", "clean_ror_vs_nodes_rf15_worst_biased_random70", 3),
    ]
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 11,
            "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9,
            "ytick.labelsize": 9, "axes.linewidth": 0.9,
        }
    ):
        for key, ylabel, title, ylim, legend_loc, stem, decimals in specs:
            fig, ax = plt.subplots(figsize=(6.4, 4.1), constrained_layout=True)
            for label in ("SVM", "DTC", "RF (15 trees)"):
                ax.plot(
                    NODES, BY_NODE[key][label], color=colors[label], marker=markers[label],
                    linestyle="-", linewidth=2.1, markersize=6.5,
                    markeredgecolor="white", markeredgewidth=0.65,
                    label="RF-15" if label == "RF (15 trees)" else label,
                )
            ax.plot(
                NODES, WORST_BIASED_RANDOM_70[key], color="#B52B65", marker="D",
                markerfacecolor="white", markeredgewidth=1.0, linestyle="-",
                linewidth=1.9, markersize=6.0, label="70% worst-biased random bound",
            )
            ax.set_xticks(NODES)
            ax.set_xlim(20, 105)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(*ylim)
            ax.grid(True, color="#B0B0B0", linewidth=0.6, alpha=0.35)
            ax.set_axisbelow(True)
            ax.legend(loc=legend_loc, frameon=True)
            save(fig, stem)


def combined_four_metric_figure():
    """Compact 2-by-2 figure matching the supplied multi-panel example."""
    methods = ("RF (15 trees)", "DTC", "SVM")
    colors = {"RF (15 trees)": "#4169E1", "DTC": "#FF7F0E", "SVM": "#FF3030"}
    markers = {"RF (15 trees)": "P", "DTC": "o", "SVM": "s"}
    labels = {"RF (15 trees)": "RF-15", "DTC": "DTC", "SVM": "SVM"}
    specs = [
        ("PDR (%)", "Packet delivery ratio (%)", "(a) Packet Delivery Ratio", (12, 95),
         "upper left", (90.3, 91.1), [0.57, 0.10, 0.38, 0.25]),
        ("Loss-aware delay (ms)", "Loss-aware delay (ms)", "(b) Loss-Aware Delay", (280, 890),
         "upper right", (303, 327), [0.08, 0.10, 0.38, 0.25]),
        ("Routing overhead ratio", "Routing overhead ratio", "(c) Routing Overhead", (0.12, 0.64),
         "upper right", (0.158, 0.181), [0.08, 0.10, 0.38, 0.25]),
    ]
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif", "font.size": 8.5,
            "axes.titlesize": 9.5, "axes.labelsize": 9,
            "legend.fontsize": 7.5, "xtick.labelsize": 8,
            "ytick.labelsize": 8, "axes.linewidth": 0.9,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3), constrained_layout=True)
        for ax, (key, ylabel, title, ylim, legend_loc, zoom_ylim, inset_bounds) in zip(
            (axes[0, 0], axes[0, 1], axes[1, 0]), specs
        ):
            for method in methods:
                ax.plot(
                    NODES, BY_NODE[key][method], color=colors[method],
                    marker=markers[method], linestyle="-", linewidth=1.7,
                    markersize=5.2, markerfacecolor="white" if method == "RF (15 trees)" else colors[method],
                    markeredgewidth=0.9, label=labels[method],
                )
            ax.plot(
                NODES, WORST_BIASED_RANDOM_70[key], color="#B52B65",
                marker="D", linestyle="-", linewidth=1.7, markersize=5.0,
                markerfacecolor="#B52B65", label="70% lower bound",
            )
            ax.set_xlim(20, 105)
            ax.set_ylim(*ylim)
            ax.set_xticks(NODES)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, color="#B0B0B0", linewidth=0.55, alpha=0.4)
            ax.set_axisbelow(True)
            ax.legend(loc=legend_loc, frameon=True, fancybox=False, edgecolor="#333333")

            zoom_ax = ax.inset_axes(inset_bounds)
            for method in ("DTC", "SVM"):
                zoom_ax.plot(
                    NODES[-2:], BY_NODE[key][method][-2:], color=colors[method],
                    marker=markers[method], linestyle="-", linewidth=1.25,
                    markersize=3.5,
                )
            zoom_ax.set_xlim(72, 103)
            zoom_ax.set_ylim(*zoom_ylim)
            zoom_ax.set_xticks(NODES[-2:])
            zoom_ax.tick_params(axis="both", labelsize=5.8, pad=1)
            zoom_ax.grid(True, color="#B0B0B0", linewidth=0.4, alpha=0.45)
            zoom_ax.set_title("SVM–DTC zoom", fontsize=6.3, pad=1.5)

        accuracy_ax = axes[1, 1]
        accuracy_methods = ("RF (15 trees)", "DTC", "SVM")
        accuracy_values = [ACCURACY[method] for method in accuracy_methods]
        bars = accuracy_ax.bar(
            [labels[method] for method in accuracy_methods], accuracy_values,
            color=[colors[method] for method in accuracy_methods],
            edgecolor="black", linewidth=0.55, width=0.62,
        )
        accuracy_ax.axhline(
            75, color="#B52B65", linestyle="-", linewidth=1.7,
            label="Majority baseline",
        )
        for bar, value in zip(bars, accuracy_values):
            accuracy_ax.text(
                bar.get_x() + bar.get_width() / 2, value + 1.1,
                f"{value:.1f}%", ha="center", va="bottom", fontsize=7.5,
            )
        accuracy_ax.set_ylim(0, 100)
        accuracy_ax.set_ylabel("Binary accuracy (%)")
        accuracy_ax.set_title("(d) Held-Out Binary Accuracy")
        accuracy_ax.grid(axis="y", color="#B0B0B0", linewidth=0.55, alpha=0.4)
        accuracy_ax.set_axisbelow(True)
        accuracy_ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#333333")
        save(fig, "combined_four_metrics_rf15_worst_biased_random70")


if __name__ == "__main__":
    overall_figure()
    per_node_figure()
    reference_style_individual_figures()
    clean_individual_figures()
    clean_worst_oc_lower_bound_figures()
    clean_worst_biased_random_figures()
    combined_four_metric_figure()
    print(f"Final fixed-OC0 figures written to {OUT}")
