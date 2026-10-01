#!/usr/bin/env python3
"""Plot the requested exploratory SVM/DTC/RF comparison on 500 scenarios.

This is deliberately separate from confirmatory paper outputs because the
RF configuration was evaluated after the 500-scenario outcomes were seen.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from underwater_ml_pipeline import build_split  # noqa: E402


OUT = Path(os.environ.get("EXPLORATORY_OUTPUT_DIRECTORY", Path(__file__).resolve().parent))
SOURCE_MODELS = (
    ROOT
    / "results/underwater_partner_style_equivalent_2000"
    / "PARTNER_COMPATIBLE_C20_G005_DTC3_L30_S60_RF_FRACTIONAL_STUMP"
)
DEVELOPMENT = (
    ROOT
    / "results/underwater_partner_style_equivalent_2000"
    / "underwater_feature_rule_2000_scenarios_8000_candidates.csv"
)
FRESH = Path(os.environ.get(
    "EXPLORATORY_FRESH_FEATURES_FILE",
    ROOT / "results/fresh_locked_evaluation_125_per_node_1s/fresh_features_500_scenarios_2000_candidates.csv",
))
LEDGER = Path(os.environ.get(
    "EXPLORATORY_FRESH_LEDGER_FILE",
    ROOT / "results/fresh_locked_evaluation_125_per_node_1s/fresh_all_four_oc_ledger.csv",
))
FEATURES = ["x", "y", "local_density", "speed"]
NODES = [25, 50, 75, 100]
METHODS = ["SVM", "DTC", "RF", "Baseline"]
COLORS = {"SVM": "#E52521", "DTC": "#FF7F0E", "RF": "#2878B5", "Baseline": "#B52B65"}
MARKERS = {"SVM": "s", "DTC": "o", "RF": "^", "Baseline": "D"}
LOSS_PENALTY_MS = 1000.0
RF_VARIANT = os.environ.get("EXPLORATORY_RF_VARIANT", "rf10")
DTC_VARIANT = os.environ.get("EXPLORATORY_DTC_VARIANT", "old")
SVM_VARIANT = os.environ.get("EXPLORATORY_SVM_VARIANT", "old")
EVALUATION_SPLIT = os.environ.get("EXPLORATORY_EVALUATION_SPLIT")
FIT_TRAIN_VALIDATION = os.environ.get("EXPLORATORY_FIT_TRAIN_VALIDATION", "false").lower() == "true"


def positive_scores(model, rows: pd.DataFrame):
    try:
        classes = list(model.classes_)
        return model.predict_proba(rows[FEATURES])[:, classes.index(1)]
    except AttributeError:
        return model.decision_function(rows[FEATURES])


def select_oc(model, rows: pd.DataFrame) -> pd.DataFrame:
    scored = rows[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    scored["score"] = positive_scores(model, rows)
    return scored.loc[scored.groupby("scenario_id").score.idxmax()].rename(
        columns={"auv_id": "selected_oc"}
    )


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    generated = rows.generated_packets.sum()
    delivered = rows.delivered_packets.sum()
    delivered_delay = (rows.E2ED_ms.fillna(0) * rows.delivered_packets).sum()
    control = rows.control_transmissions.sum()
    data_hops = rows.data_hops.sum()
    return {
        "PDR (%)": 100.0 * delivered / generated,
        "Loss-aware delay (ms)": (
            delivered_delay + (generated - delivered) * LOSS_PENALTY_MS
        ) / generated,
        "Routing overhead ratio": control / (control + data_hops),
    }


def padded_limits(values, fraction: float = 0.06):
    low, high = min(values), max(values)
    margin = (high - low) * fraction
    return low - margin, high + margin


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(DEVELOPMENT)
    fresh = pd.read_csv(FRESH)
    ledger = pd.read_csv(LEDGER)

    split = build_split(development, seed=2026)
    fit_ids = split["train"] + (split["validation"] if FIT_TRAIN_VALIDATION else [])
    training = development[development.scenario_id.isin(fit_ids)]
    if EVALUATION_SPLIT:
        if EVALUATION_SPLIT not in split:
            raise ValueError(f"Unknown EXPLORATORY_EVALUATION_SPLIT: {EVALUATION_SPLIT}")
        evaluation_ids = set(split[EVALUATION_SPLIT])
        fresh = fresh[fresh.scenario_id.isin(evaluation_ids)].copy()
        ledger = ledger[ledger.scenario_id.isin(evaluation_ids)].copy()

    if RF_VARIANT == "rf15":
        rf = RandomForestClassifier(
            n_estimators=15,
            criterion="gini",
            max_depth=2,
            min_samples_split=20,
            min_samples_leaf=2,
            max_features=1,
            bootstrap=True,
            max_samples=0.35,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif RF_VARIANT == "rf10":
        rf = RandomForestClassifier(
            n_estimators=10,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            max_samples=None,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown EXPLORATORY_RF_VARIANT: {RF_VARIANT}")

    if DTC_VARIANT == "strong":
        dtc = DecisionTreeClassifier(
            criterion="gini",
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=2,
            max_features=3,
            random_state=42,
        )
    elif DTC_VARIANT == "old":
        dtc = joblib.load(SOURCE_MODELS / "partner_compatible_dtc_pipeline.joblib")
    else:
        raise ValueError(f"Unknown EXPLORATORY_DTC_VARIANT: {DTC_VARIANT}")

    if SVM_VARIANT == "c5_g002":
        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=5.0, gamma=0.02)),
        ])
    elif SVM_VARIANT == "c100_g002_balanced":
        svm = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(kernel="rbf", C=100.0, gamma=0.02, class_weight="balanced"),
            ),
        ])
    elif SVM_VARIANT == "old":
        svm = joblib.load(SOURCE_MODELS / "partner_compatible_svm_pipeline.joblib")
    else:
        raise ValueError(f"Unknown EXPLORATORY_SVM_VARIANT: {SVM_VARIANT}")

    models = {
        "SVM": svm,
        "DTC": dtc,
        "RF": rf,
    }
    if SVM_VARIANT in {"c5_g002", "c100_g002_balanced"}:
        models["SVM"].fit(training[FEATURES], training.is_oc)
    if DTC_VARIANT == "strong":
        models["DTC"].fit(training[FEATURES], training.is_oc)
    models["RF"].fit(training[FEATURES], training.is_oc)

    truth = fresh.loc[fresh.is_oc.eq(1)].set_index("scenario_id").auv_id
    choices = {name: select_oc(model, fresh) for name, model in models.items()}
    accuracy = {"Baseline": 25.0}
    macro_f1 = {}
    binary_metrics = {}
    for name, selected in choices.items():
        binary_predictions = models[name].predict(fresh[FEATURES])
        precision, recall, binary_f1, _ = precision_recall_fscore_support(
            fresh.is_oc,
            binary_predictions,
            average="binary",
            zero_division=0,
        )
        binary_metrics[name] = {
            "Binary accuracy (%)": 100.0 * accuracy_score(fresh.is_oc, binary_predictions),
            "Binary precision": precision,
            "Binary recall": recall,
            "Binary F1": binary_f1,
        }
        expected = truth.loc[selected.scenario_id].to_numpy()
        accuracy[name] = 100.0 * selected.selected_oc.eq(expected).mean()
        macro_f1[name] = f1_score(
            expected,
            selected.selected_oc,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )

    network_rows = []
    for nodes in NODES:
        for name, selected in choices.items():
            mapped = selected.loc[
                selected.node_count.eq(nodes), ["scenario_id", "selected_oc"]
            ].merge(ledger, on=["scenario_id", "selected_oc"], validate="one_to_one")
            network_rows.append({"Nodes": nodes, "Method": name, **summarize(mapped)})

        # All candidates have equal weight: exact expected uniform-random result.
        baseline = ledger.loc[ledger.node_count.eq(nodes)]
        network_rows.append({"Nodes": nodes, "Method": "Baseline", **summarize(baseline)})

    network = pd.DataFrame(network_rows)
    classification = pd.DataFrame(
        [
            {
                "Method": method,
                **binary_metrics.get(method, {}),
                "Correct OC selection accuracy (%)": accuracy[method],
                "Top-1 macro F1": macro_f1.get(method),
            }
            for method in METHODS
        ]
    )
    network.to_csv(OUT / "network_metrics_by_node.csv", index=False)
    classification.to_csv(OUT / "correct_oc_accuracy.csv", index=False)

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.3), constrained_layout=True)
        panels = [
            ("PDR (%)", "Packet delivery ratio (%)", "(a) Packet Delivery Ratio"),
            ("Loss-aware delay (ms)", "Loss-aware delay (ms)", "(b) Loss-Aware Delay"),
            ("Routing overhead ratio", "Routing overhead ratio", "(c) Routing Overhead"),
        ]
        for ax, (column, ylabel, title) in zip(axes.flat[:3], panels):
            all_values = []
            for method in METHODS:
                part = network.loc[network.Method.eq(method)].sort_values("Nodes")
                values = part[column].tolist()
                all_values.extend(values)
                ax.plot(
                    part.Nodes,
                    values,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linewidth=2.2,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=method,
                )
            ax.set_xlim(20, 105)
            ax.set_ylim(*padded_limits(all_values))
            ax.set_xticks(NODES)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.32)
            ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#333333")

        ax = axes[1, 1]
        bars = ax.bar(
            METHODS,
            [accuracy[m] for m in METHODS],
            color=[COLORS[m] for m in METHODS],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, method in zip(bars, METHODS):
            value = accuracy[method]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylim(0, 100)
        ax.set_ylabel("Correct OC selection accuracy (%)")
        ax.set_title("(d) Correct-OC Selection Accuracy")
        ax.grid(axis="y", alpha=0.32)

        for extension in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"combined_correct_oc_metrics.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

        individual_specs = [
            ("PDR (%)", "Packet delivery ratio (%)", "PDR vs Number of Nodes", "pdr_vs_nodes"),
            ("Loss-aware delay (ms)", "Loss-aware delay (ms)", "Delay vs Number of Nodes", "delay_vs_nodes"),
            ("Routing overhead ratio", "Routing overhead ratio", "ROR vs Number of Nodes", "ror_vs_nodes"),
        ]
        for column, ylabel, title, stem in individual_specs:
            line_fig, line_ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
            all_values = []
            for method in METHODS:
                part = network.loc[network.Method.eq(method)].sort_values("Nodes")
                values = part[column].tolist()
                all_values.extend(values)
                line_ax.plot(
                    part.Nodes,
                    values,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linewidth=2.2,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=method,
                )
            line_ax.set_xlim(20, 105)
            line_ax.set_ylim(*padded_limits(all_values))
            line_ax.set_xticks(NODES)
            line_ax.set_xlabel("Number of nodes")
            line_ax.set_ylabel(ylabel)
            line_ax.set_title(title)
            line_ax.grid(True, alpha=0.32)
            line_ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#333333")
            for extension in ("png", "pdf", "svg"):
                line_fig.savefig(OUT / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
            plt.close(line_fig)

        accuracy_fig, accuracy_ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
        accuracy_bars = accuracy_ax.bar(
            METHODS,
            [accuracy[method] for method in METHODS],
            color=[COLORS[method] for method in METHODS],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, method in zip(accuracy_bars, METHODS):
            value = accuracy[method]
            accuracy_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )
        accuracy_ax.set_ylim(0, 100)
        accuracy_ax.set_ylabel("Correct OC selection accuracy (%)")
        accuracy_ax.set_title("Held-Out Correct-OC Accuracy")
        accuracy_ax.grid(axis="y", alpha=0.32)
        for extension in ("png", "pdf", "svg"):
            accuracy_fig.savefig(
                OUT / f"held_out_correct_oc_accuracy.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(accuracy_fig)

    metadata = {
        "status": os.environ.get("EXPERIMENT_STATUS", "exploratory_not_confirmatory"),
        "reason": os.environ.get(
            "EXPERIMENT_REASON",
            f"{RF_VARIANT.upper()} was selected after inspecting this 500-scenario evaluation set.",
        ),
        "evaluation_split": EVALUATION_SPLIT,
        "training_scenarios": int(training.scenario_id.nunique()),
        "evaluation_scenarios": int(fresh.scenario_id.nunique()),
        "scenarios_per_node": {
            str(int(nodes)): int(count)
            for nodes, count in fresh.groupby("node_count").scenario_id.nunique().items()
        },
        "loss_penalty_ms": LOSS_PENALTY_MS,
        "baseline": "exact expected uniform-random selection among four candidates",
        "svm": models["SVM"].get_params(),
        "dtc": models["DTC"].get_params(),
        "rf": models["RF"].get_params(),
    }
    (OUT / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )

    print(classification.to_string(index=False))
    print(network.to_string(index=False))


if __name__ == "__main__":
    main()
