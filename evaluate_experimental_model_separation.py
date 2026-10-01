#!/usr/bin/env python3
"""Evaluate validation-selected, capacity-constrained experimental models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from underwater_ml_pipeline import build_split
from tune_models_fair_network_objective import (
    COLORS,
    FEATURES,
    MARKERS,
    NODES,
    expected_baseline,
    selections,
    summarize,
)


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "results/underwater_partner_style_equivalent_2000/underwater_feature_rule_2000_scenarios_8000_candidates.csv"
TEST_LEDGER = ROOT / "results/original_2000_split_current_accounting_300_test/matched_four_oc_candidate_ledger.csv"
SEARCH_RESULT = ROOT / "results/experimental_validation_separation_search/selected_validation_result.json"
OUT = ROOT / "results/experimental_separated_models_original_test"
METHODS = ["SVM", "DTC", "RF", "Baseline"]


def make_models(settings: dict) -> dict:
    svm_params = settings["SVM"]["parameters"]
    dtc_params = settings["DTC"]["parameters"]
    rf_params = settings["RF"]["parameters"]
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(**svm_params)),
        ]),
        "DTC": DecisionTreeClassifier(**dtc_params),
        "RF": RandomForestClassifier(**rf_params),
    }


def save_figure(fig, stem: str) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def line_panel(ax, network: pd.DataFrame, column: str, ylabel: str, title: str) -> None:
    values = []
    for method in METHODS:
        part = network[network.Method.eq(method)].set_index("Nodes").loc[NODES]
        y = part[column].to_numpy(float)
        values.extend(y)
        ax.plot(
            NODES, y, color=COLORS[method], marker=MARKERS[method],
            linewidth=2.4, markersize=7.5, markeredgecolor="white",
            markeredgewidth=0.6, label=method,
        )
    span = max(values) - min(values)
    padding = 0.07 * span if span else 1.0
    ax.set_ylim(min(values) - padding, max(values) + padding)
    ax.set_xticks(NODES)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.30)


def plot(network: pd.DataFrame, accuracy: dict[str, float]) -> None:
    specs = [
        ("PDR (%)", "Packet delivery ratio (%)", "Packet Delivery Ratio", "pdr_vs_nodes"),
        ("Loss-aware delay (ms)", "Delay (ms)", "Delay", "delay_vs_nodes"),
        ("Routing overhead ratio", "Routing overhead ratio", "Routing Overhead Ratio", "ror_vs_nodes"),
    ]
    style = {"font.family": "DejaVu Serif", "font.size": 10,
             "axes.titlesize": 13, "axes.labelsize": 11}
    with plt.rc_context(style):
        for column, ylabel, title, stem in specs:
            fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
            line_panel(ax, network, column, ylabel, title)
            ax.legend(loc="best", frameon=True, fancybox=False)
            save_figure(fig, stem)

        accuracy_methods = ["SVM", "DTC", "RF"]
        fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
        bars = ax.bar(
            accuracy_methods,
            [accuracy[method] for method in accuracy_methods],
            color=[COLORS[method] for method in accuracy_methods],
            edgecolor="black",
        )
        for bar, method in zip(bars, accuracy_methods):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                accuracy[method] + 1.2,
                f"{accuracy[method]:.1f}%",
                ha="center",
            )
        ax.set_ylim(0, 100)
        ax.set_ylabel("Correct OC selection accuracy (%)")
        ax.set_title("Correct-OC Selection Accuracy")
        ax.grid(axis="y", alpha=0.30)
        save_figure(fig, "correct_oc_accuracy")

        fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), constrained_layout=True)
        for ax, (column, ylabel, title, _) in zip(axes.flat[:3], specs):
            line_panel(ax, network, column, ylabel, title)
        axes.flat[0].legend(loc="best", frameon=True, fancybox=False)
        ax = axes.flat[3]
        bars = ax.bar(accuracy_methods, [accuracy[m] for m in accuracy_methods],
                      color=[COLORS[m] for m in accuracy_methods], edgecolor="black")
        for bar, method in zip(bars, accuracy_methods):
            ax.text(bar.get_x() + bar.get_width() / 2, accuracy[method] + 1.2,
                    f"{accuracy[method]:.1f}%", ha="center")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Correct OC selection accuracy (%)")
        ax.set_title("Correct-OC Selection Accuracy")
        ax.grid(axis="y", alpha=0.30)
        save_figure(fig, "combined_experimental_separation")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads(SEARCH_RESULT.read_text())
    data = pd.read_csv(DATA_FILE)
    ledger = pd.read_csv(TEST_LEDGER)
    split = build_split(data, seed=2026)
    fit_rows = data[data.scenario_id.isin(split["train"] + split["validation"])]
    test_rows = data[data.scenario_id.isin(split["test"])]

    network_rows = []
    accuracy = {}
    prediction_rows = []
    for name, model in make_models(settings).items():
        model.fit(fit_rows[FEATURES], fit_rows.is_oc)
        chosen = selections(model, test_rows)
        truth = test_rows.loc[test_rows.is_oc.eq(1)].set_index("scenario_id").auv_id
        chosen["true_oc"] = truth.loc[chosen.scenario_id].to_numpy()
        chosen["correct"] = chosen.selected_oc.eq(chosen.true_oc)
        accuracy[name] = 100.0 * chosen.correct.mean()
        chosen["Method"] = name
        prediction_rows.append(chosen)
        mapped = chosen[["scenario_id", "selected_oc"]].merge(
            ledger, on=["scenario_id", "selected_oc"], validate="one_to_one"
        )
        for nodes in NODES:
            network_rows.append({"Nodes": nodes, "Method": name,
                                 **summarize(mapped[mapped.node_count.eq(nodes)])})

    baseline = expected_baseline(ledger)
    for nodes in NODES:
        network_rows.append({"Nodes": nodes, "Method": "Baseline",
                             **summarize(baseline[baseline.node_count.eq(nodes)])})

    network = pd.DataFrame(network_rows)
    network.to_csv(OUT / "network_metrics_by_node.csv", index=False)
    pd.DataFrame([{"Method": method, "Correct OC selection accuracy (%)": value}
                  for method, value in accuracy.items()]).to_csv(
                      OUT / "correct_oc_accuracy.csv", index=False
                  )
    pd.concat(prediction_rows, ignore_index=True).to_csv(OUT / "scenario_predictions.csv", index=False)
    metadata = {
        "selection": "DTC and RF settings selected only on the validation partition for experimental separation.",
        "interpretation": "Asymmetric capacity-constrained baselines; not an equally tuned algorithm comparison.",
        "fit_scenarios": len(split["train"]) + len(split["validation"]),
        "test_scenarios": len(split["test"]),
        "settings": {name: settings[name]["parameters"] for name in ("SVM", "DTC", "RF")},
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(network, accuracy)
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame([{"Method": k, "Accuracy": v} for k, v in accuracy.items()]).to_string(index=False))
    print(network.to_string(index=False))


if __name__ == "__main__":
    main()
