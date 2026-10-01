#!/usr/bin/env python3
"""Generate independent-evaluation figures with a real uniform-random baseline.

Models are fitted on the 1,700 development scenarios and evaluated on 120
independently generated scenarios (30 per node count).  For network results,
the Baseline selects each of the four OC candidates with probability 0.25,
without observing any network outcome.  Confidence intervals bootstrap whole
scenarios and recompute each aggregate metric.
"""

from pathlib import Path
import os
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", message="Unable to import Axes3D")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get(
    "REAL_BASELINE_OUTPUT_DIRECTORY", str(Path(__file__).resolve().parent)
))
OUT.mkdir(parents=True, exist_ok=True)
DEVELOPMENT = ROOT / "results/underwater_partner_style_equivalent_2000/underwater_feature_rule_2000_scenarios_8000_candidates.csv"
FRESH_FEATURES = Path(os.environ.get(
    "REAL_BASELINE_FEATURES_FILE",
    str(ROOT / "results/fresh_locked_evaluation_30_per_node_1s/fresh_features_120_scenarios_480_candidates.csv"),
))
FRESH_LEDGER = Path(os.environ.get(
    "REAL_BASELINE_LEDGER_FILE",
    str(ROOT / "results/fresh_locked_evaluation_30_per_node_1s/fresh_all_four_oc_ledger.csv"),
))
FEATURES = ["x", "y", "local_density", "speed"]
NODES = [25, 50, 75, 100]
LOSS_PENALTY_MS = 1000.0
BOOTSTRAP_REPLICATES = 5000

COLORS = {"SVM": "#E52521", "DTC": "#FF7F0E", "RF": "#2878B5", "Baseline": "#B52B65"}
MARKERS = {"SVM": "s", "DTC": "o", "RF": "^", "Baseline": "D"}


def models():
    return {
        "SVM": Pipeline([("scaler", StandardScaler()), ("classifier", SVC())]),
        "DTC": DecisionTreeClassifier(
            criterion="gini", max_depth=8, min_samples_split=20,
            min_samples_leaf=2, max_features=3, random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=15, criterion="gini", max_depth=2,
            min_samples_split=20, min_samples_leaf=2, max_features=1,
            bootstrap=True, max_samples=0.35, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }


def development_fit_rows(data):
    selected = []
    for nodes in NODES:
        identifiers = sorted(data.loc[data.node_count.eq(nodes), "scenario_id"].unique())
        random.Random(91_000 + nodes).shuffle(identifiers)
        selected.extend(identifiers[:425])
    return data[data.scenario_id.isin(selected)]


def controller_choices(model, rows):
    try:
        scores = model.predict_proba(rows[FEATURES])[:, list(model.classes_).index(1)]
    except AttributeError:
        scores = model.decision_function(rows[FEATURES])
    scored = rows[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    scored["score"] = scores
    return scored.loc[scored.groupby("scenario_id").score.idxmax()].rename(columns={"auv_id": "selected_oc"})


def expected_baseline_rows(ledger):
    """Expected counters when every OC has equal selection probability."""
    work = ledger.copy()
    work["delivered_delay_sum"] = work.E2ED_ms.fillna(0) * work.delivered_packets
    expected = work.groupby(["scenario_id", "node_count"], as_index=False).agg(
        generated_packets=("generated_packets", "mean"),
        delivered_packets=("delivered_packets", "mean"),
        delivered_delay_sum=("delivered_delay_sum", "mean"),
        control_transmissions=("control_transmissions", "mean"),
        data_hops=("data_hops", "mean"),
    )
    expected["Method"] = "Baseline"
    return expected


def selected_method_rows(name, choices, ledger):
    mapped = choices[["scenario_id", "selected_oc"]].merge(
        ledger, on=["scenario_id", "selected_oc"], validate="one_to_one"
    )
    mapped["delivered_delay_sum"] = mapped.E2ED_ms.fillna(0) * mapped.delivered_packets
    mapped["Method"] = name
    return mapped[[
        "scenario_id", "node_count", "generated_packets", "delivered_packets",
        "delivered_delay_sum", "control_transmissions", "data_hops", "Method",
    ]]


def metrics(frame):
    generated = frame.generated_packets.sum()
    delivered = frame.delivered_packets.sum()
    return {
        "PDR (%)": 100.0 * delivered / generated,
        "Loss-aware delay (ms)": (
            frame.delivered_delay_sum.sum() + (generated - delivered) * LOSS_PENALTY_MS
        ) / generated,
        "Routing overhead ratio": frame.control_transmissions.sum() / (
            frame.control_transmissions.sum() + frame.data_hops.sum()
        ),
    }


def bootstrap_summary(frame, seed):
    point = metrics(frame)
    values = frame[[
        "generated_packets", "delivered_packets", "delivered_delay_sum",
        "control_transmissions", "data_hops",
    ]].to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = {key: [] for key in point}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = values[rng.integers(0, len(values), len(values))]
        generated, delivered, delivered_delay, control, data_hops = sampled.sum(axis=0)
        estimates["PDR (%)"].append(100.0 * delivered / generated)
        estimates["Loss-aware delay (ms)"].append(
            (delivered_delay + (generated - delivered) * LOSS_PENALTY_MS) / generated
        )
        estimates["Routing overhead ratio"].append(control / (control + data_hops))
    return {
        key: (point[key], *np.percentile(estimates[key], [2.5, 97.5]))
        for key in point
    }


def save(fig, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def line_panel(ax, result, metric, ylabel, title, ylim, legend=True, show_ci=True):
    for method in ("SVM", "DTC", "RF", "Baseline"):
        subset = result[result.Method.eq(method)].sort_values("nodes")
        y = subset[f"{metric}_estimate"].to_numpy()
        low = subset[f"{metric}_low"].to_numpy()
        high = subset[f"{metric}_high"].to_numpy()
        ax.plot(
            subset.nodes, y, color=COLORS[method], marker=MARKERS[method],
            linewidth=2.0, markersize=6.0, markeredgecolor="white",
            markeredgewidth=0.55, label=method,
        )
        if show_ci:
            ax.fill_between(subset.nodes, low, high, color=COLORS[method], alpha=0.10)
    ax.set_xticks(NODES)
    ax.set_xlim(20, 105)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#B0B0B0", linewidth=0.6, alpha=0.38)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#333333")


def data_driven_limits(result, metric, padding_fraction=0.035):
    """Tight, honest limits based only on the displayed point estimates."""
    values = result[f"{metric}_estimate"].to_numpy(float)
    span = values.max() - values.min()
    padding = span * padding_fraction
    return values.min() - padding, values.max() + padding


def main():
    development = pd.read_csv(DEVELOPMENT)
    fresh = pd.read_csv(FRESH_FEATURES)
    ledger = pd.read_csv(FRESH_LEDGER)
    numeric = [
        "scenario_id", "node_count", "selected_oc", "generated_packets",
        "delivered_packets", "E2ED_ms", "control_transmissions", "data_hops",
    ]
    for column in numeric:
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce")

    fit_rows = development_fit_rows(development)
    network_rows = []
    classification_rows = []
    fitted = models()
    for name, model in fitted.items():
        model.fit(fit_rows[FEATURES], fit_rows.is_oc)
        binary = model.predict(fresh[FEATURES])
        precision, recall, f1, _ = precision_recall_fscore_support(
            fresh.is_oc, binary, average="binary", zero_division=0
        )
        choices = controller_choices(model, fresh)
        truth = fresh.loc[fresh.is_oc.eq(1)].set_index("scenario_id").auv_id
        choices["true_oc"] = truth.loc[choices.scenario_id].to_numpy()
        top1 = choices.selected_oc.eq(choices.true_oc)
        classification_rows.append({
            "method": name,
            "binary_accuracy_percent": 100.0 * accuracy_score(fresh.is_oc, binary),
            "precision": precision,
            "recall": recall,
            "binary_f1": f1,
            "top1_accuracy_percent": 100.0 * top1.mean(),
            "top1_macro_f1": f1_score(
                choices.true_oc, choices.selected_oc, labels=[0, 1, 2, 3],
                average="macro", zero_division=0,
            ),
        })
        network_rows.append(selected_method_rows(name, choices, ledger))
    network_rows.append(expected_baseline_rows(ledger))
    selected = pd.concat(network_rows, ignore_index=True)

    rows = []
    for method_index, method in enumerate(("SVM", "DTC", "RF", "Baseline")):
        for node_index, nodes in enumerate(NODES):
            frame = selected[selected.Method.eq(method) & selected.node_count.eq(nodes)]
            summary = bootstrap_summary(frame, 42_000 + method_index * 100 + node_index)
            row = {"nodes": nodes, "Method": method, "scenarios": len(frame)}
            for metric, (estimate, low, high) in summary.items():
                row[f"{metric}_estimate"] = estimate
                row[f"{metric}_low"] = low
                row[f"{metric}_high"] = high
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "real_baseline_network_results_with_95ci.csv", index=False)
    classification = pd.DataFrame(classification_rows)
    classification.to_csv(OUT / "real_baseline_classification_results.csv", index=False)

    with plt.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }):
        specs = [
            ("PDR (%)", "Packet delivery ratio (%)", "PDR vs Number of Nodes", (20, 100), "real_baseline_pdr"),
            ("Loss-aware delay (ms)", "Loss-aware delay (ms)", "Delay vs Number of Nodes", (250, 800), "real_baseline_delay"),
            ("Routing overhead ratio", "Routing overhead ratio", "Routing Overhead vs Number of Nodes", (0.12, 0.52), "real_baseline_ror"),
        ]
        for metric, ylabel, title, ylim, stem in specs:
            fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
            line_panel(ax, result, metric, ylabel, title, ylim)
            save(fig, stem)

        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        accuracy = dict(zip(classification.method, classification.binary_accuracy_percent))
        # Uniformly choosing one of four candidates predicts one positive and
        # three negatives. With one true positive per scenario, its expected
        # candidate-level binary accuracy is 62.5%.
        accuracy["Baseline"] = 62.5
        bars = ax.bar(
            list(accuracy), list(accuracy.values()),
            color=[COLORS[name] for name in accuracy], edgecolor="black",
            linewidth=0.6, width=0.62,
        )
        for bar, value in zip(bars, accuracy.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Binary accuracy (%)")
        ax.set_title("Independent Binary Classification Accuracy")
        ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
        ax.set_axisbelow(True)
        save(fig, "real_baseline_accuracy")

        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
        line_panel(axes[0, 0], result, "PDR (%)", "Packet delivery ratio (%)", "(a) Packet Delivery Ratio", (20, 100))
        line_panel(axes[0, 1], result, "Loss-aware delay (ms)", "Loss-aware delay (ms)", "(b) Loss-Aware Delay", (250, 800))
        line_panel(axes[1, 0], result, "Routing overhead ratio", "Routing overhead ratio", "(c) Routing Overhead", (0.12, 0.52))
        ax = axes[1, 1]
        bars = ax.bar(
            list(accuracy), list(accuracy.values()),
            color=[COLORS[name] for name in accuracy], edgecolor="black", linewidth=0.55,
        )
        for bar, value in zip(bars, accuracy.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Binary accuracy (%)")
        ax.set_title("(d) Independent Binary Accuracy")
        ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
        ax.set_axisbelow(True)
        save(fig, "real_baseline_combined_four_metrics")

        # A clean companion version preserves the same independent values but
        # omits the confidence shading so the small model differences remain
        # visually readable. The CSV retains all confidence intervals.
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
        line_panel(axes[0, 0], result, "PDR (%)", "Packet delivery ratio (%)", "(a) Packet Delivery Ratio", (20, 100), show_ci=False)
        line_panel(axes[0, 1], result, "Loss-aware delay (ms)", "Loss-aware delay (ms)", "(b) Loss-Aware Delay", (250, 800), show_ci=False)
        line_panel(axes[1, 0], result, "Routing overhead ratio", "Routing overhead ratio", "(c) Routing Overhead", (0.12, 0.52), show_ci=False)
        ax = axes[1, 1]
        bars = ax.bar(
            list(accuracy), list(accuracy.values()),
            color=[COLORS[name] for name in accuracy], edgecolor="black", linewidth=0.55,
        )
        for bar, value in zip(bars, accuracy.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Binary accuracy (%)")
        ax.set_title("(d) Independent Binary Accuracy")
        ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
        ax.set_axisbelow(True)
        save(fig, "real_baseline_combined_clean")

        # Tight-axis presentation analogous to the supplied reference.  The
        # network axes are derived from the displayed evaluation data; the
        # accuracy bars retain a zero origin to avoid exaggerating bar height.
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
        line_panel(
            axes[0, 0], result, "PDR (%)", "Packet delivery ratio (%)",
            "(a) Packet Delivery Ratio", data_driven_limits(result, "PDR (%)"),
            show_ci=False,
        )
        line_panel(
            axes[0, 1], result, "Loss-aware delay (ms)", "Loss-aware delay (ms)",
            "(b) Loss-Aware Delay", data_driven_limits(result, "Loss-aware delay (ms)"),
            show_ci=False,
        )
        line_panel(
            axes[1, 0], result, "Routing overhead ratio", "Routing overhead ratio",
            "(c) Routing Overhead", data_driven_limits(result, "Routing overhead ratio"),
            show_ci=False,
        )
        ax = axes[1, 1]
        bars = ax.bar(
            list(accuracy), list(accuracy.values()),
            color=[COLORS[name] for name in accuracy], edgecolor="black", linewidth=0.55,
        )
        for bar, value in zip(bars, accuracy.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Binary accuracy (%)")
        ax.set_title("(d) Independent Binary Accuracy")
        ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
        ax.set_axisbelow(True)
        save(fig, "real_baseline_combined_tight_axes")

    print(classification.to_string(index=False))
    print(result[["nodes", "Method", "PDR (%)_estimate", "Loss-aware delay (ms)_estimate", "Routing overhead ratio_estimate"]].to_string(index=False))


if __name__ == "__main__":
    main()
