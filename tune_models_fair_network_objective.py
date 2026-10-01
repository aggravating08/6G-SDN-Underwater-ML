#!/usr/bin/env python3
"""Fair validation-only tuning for SVM, DTC, and RF network performance."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
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

from underwater_ml_pipeline import build_split


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/underwater_partner_style_equivalent_2000"
VALIDATION_LEDGER = ROOT / "results/original_2000_split_current_accounting_300_validation/matched_four_oc_candidate_ledger.csv"
TEST_LEDGER = ROOT / "results/original_2000_split_current_accounting_300_test/matched_four_oc_candidate_ledger.csv"
OUT = ROOT / "results/fair_validation_tuned_original_split_current_accounting"
FEATURES = ["x", "y", "local_density", "speed"]
NODES = [25, 50, 75, 100]
METHODS = ["SVM", "DTC", "RF", "Baseline"]
COLORS = {"SVM": "#E52521", "DTC": "#FF7F0E", "RF": "#2878B5", "Baseline": "#B52B65"}
MARKERS = {"SVM": "s", "DTC": "o", "RF": "^", "Baseline": "D"}
LOSS_PENALTY_MS = 1000.0
SEARCH_BUDGET = 64


def model_configs() -> dict[str, list[dict]]:
    svm = [
        {"C": c, "gamma": gamma, "class_weight": weight}
        for c, gamma, weight in itertools.product(
            [0.5, 1, 2, 5, 10, 20, 50, 100],
            [0.005, 0.02, 0.05, 0.1],
            [None, "balanced"],
        )
    ]
    dtc_all = [
        {
            "max_depth": depth,
            "min_samples_split": split,
            "min_samples_leaf": leaf,
            "max_features": max_features,
            "class_weight": weight,
        }
        for depth, split, leaf, max_features, weight in itertools.product(
            [2, 3, 4, 5, 6, 8, None],
            [2, 10, 20, 40],
            [1, 2, 5, 10, 20],
            [2, 3, 4, None],
            [None, "balanced"],
        )
        if split > leaf
    ]
    rf_all = [
        {
            "n_estimators": trees,
            "max_depth": depth,
            "min_samples_split": split,
            "min_samples_leaf": leaf,
            "max_features": max_features,
            "max_samples": samples,
            "class_weight": weight,
        }
        for trees, depth, split, leaf, max_features, samples, weight in itertools.product(
            [15, 25, 50, 100],
            [2, 4, 8, None],
            [2, 10, 20],
            [1, 2, 5],
            [1, 2, 3, "sqrt"],
            [0.35, 0.60, None],
            [None, "balanced"],
        )
        if split > leaf
    ]
    rng = np.random.default_rng(20260915)
    dtc = [dtc_all[index] for index in rng.choice(len(dtc_all), SEARCH_BUDGET - 1, replace=False)]
    rf = [rf_all[index] for index in rng.choice(len(rf_all), SEARCH_BUDGET - 1, replace=False)]
    dtc.append({"max_depth": 8, "min_samples_split": 20, "min_samples_leaf": 2, "max_features": 3, "class_weight": None})
    rf.append({"n_estimators": 15, "max_depth": 2, "min_samples_split": 20, "min_samples_leaf": 2, "max_features": 1, "max_samples": 0.35, "class_weight": "balanced"})
    return {"SVM": svm, "DTC": dtc, "RF": rf}


def make_model(name: str, params: dict):
    if name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=False, random_state=42, **params)),
        ])
    if name == "DTC":
        return DecisionTreeClassifier(criterion="gini", random_state=42, **params)
    return RandomForestClassifier(
        criterion="gini", bootstrap=True, random_state=42, n_jobs=-1, **params
    )


def positive_scores(model, rows: pd.DataFrame) -> np.ndarray:
    try:
        return model.predict_proba(rows[FEATURES])[:, list(model.classes_).index(1)]
    except AttributeError:
        return model.decision_function(rows[FEATURES])


def selections(model, rows: pd.DataFrame) -> pd.DataFrame:
    scored = rows[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    scored["score"] = positive_scores(model, rows)
    return scored.loc[scored.groupby("scenario_id").score.idxmax()].rename(
        columns={"auv_id": "selected_oc"}
    )


def add_candidate_utility(ledger: pd.DataFrame) -> pd.DataFrame:
    work = ledger.copy()
    work["pdr_fraction"] = work.delivered_packets / work.generated_packets
    work["loss_aware_delay"] = (
        work.E2ED_ms.fillna(0) * work.delivered_packets
        + (work.generated_packets - work.delivered_packets) * LOSS_PENALTY_MS
    ) / work.generated_packets
    work["routing_ratio"] = work.control_transmissions / (
        work.control_transmissions + work.data_hops
    )
    scores = []
    for _, group in work.groupby("scenario_id", sort=False):
        parts = []
        for column, higher in (("pdr_fraction", True), ("loss_aware_delay", False), ("routing_ratio", False)):
            values = group[column].to_numpy(float)
            span = values.max() - values.min()
            normalized = np.ones(len(values)) if span == 0 else (values - values.min()) / span
            parts.append(normalized if higher else 1.0 - normalized)
        scores.extend(np.mean(parts, axis=0))
    work["network_utility"] = scores
    return work


def validation_result(model, rows: pd.DataFrame, ledger: pd.DataFrame) -> tuple[float, float, float]:
    chosen = selections(model, rows)
    mapped = chosen[["scenario_id", "selected_oc"]].merge(
        ledger[["scenario_id", "selected_oc", "node_count", "network_utility"]],
        on=["scenario_id", "selected_oc"],
        validate="one_to_one",
    )
    by_node = mapped.groupby("node_count").network_utility.mean()
    truth = rows.loc[rows.is_oc.eq(1)].set_index("scenario_id").auv_id
    top1 = chosen.selected_oc.eq(truth.loc[chosen.scenario_id].to_numpy()).mean()
    return float(mapped.network_utility.mean()), float(by_node.min()), float(top1)


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    generated = rows.generated_packets.sum()
    delivered = rows.delivered_packets.sum()
    return {
        "PDR (%)": 100.0 * delivered / generated,
        "Loss-aware delay (ms)": (
            (rows.E2ED_ms.fillna(0) * rows.delivered_packets).sum()
            + (generated - delivered) * LOSS_PENALTY_MS
        ) / generated,
        "Routing overhead ratio": rows.control_transmissions.sum()
        / (rows.control_transmissions.sum() + rows.data_hops.sum()),
    }


def expected_baseline(ledger: pd.DataFrame) -> pd.DataFrame:
    work = ledger.copy()
    work["delay_total"] = work.E2ED_ms.fillna(0) * work.delivered_packets
    result = work.groupby(["scenario_id", "node_count"], as_index=False).agg(
        generated_packets=("generated_packets", "mean"),
        delivered_packets=("delivered_packets", "mean"),
        delay_total=("delay_total", "mean"),
        control_transmissions=("control_transmissions", "mean"),
        data_hops=("data_hops", "mean"),
    )
    result["E2ED_ms"] = result.delay_total / result.delivered_packets
    return result


def plot_results(network: pd.DataFrame, accuracy: dict[str, float]) -> None:
    specs = [
        ("PDR (%)", "Packet delivery ratio (%)", "PDR vs Number of Nodes", "pdr"),
        ("Loss-aware delay (ms)", "Loss-aware delay (ms)", "Delay vs Number of Nodes", "delay"),
        ("Routing overhead ratio", "Routing overhead ratio", "ROR vs Number of Nodes", "ror"),
    ]
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11}):
        for column, ylabel, title, stem in specs:
            fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
            all_values = []
            for method in METHODS:
                part = network[network.Method.eq(method)].set_index("Nodes").loc[NODES]
                y = part[column].to_numpy(float)
                all_values.extend(y)
                ax.plot(NODES, y, color=COLORS[method], marker=MARKERS[method], linewidth=2.2,
                        markersize=7, markeredgecolor="white", markeredgewidth=0.6, label=method)
            span = max(all_values) - min(all_values)
            ax.set_ylim(min(all_values) - 0.06 * span, max(all_values) + 0.06 * span)
            ax.set_xticks(NODES); ax.set_xlabel("Number of nodes"); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.grid(True, alpha=0.32); ax.legend(loc="best", frameon=True, fancybox=False)
            for extension in ("png", "pdf", "svg"):
                fig.savefig(OUT / f"{stem}_vs_nodes.{extension}", dpi=300, bbox_inches="tight")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
        bars = ax.bar(METHODS, [accuracy[m] for m in METHODS], color=[COLORS[m] for m in METHODS], edgecolor="black")
        for bar, method in zip(bars, METHODS):
            ax.text(bar.get_x() + bar.get_width()/2, accuracy[method] + 1.2, f"{accuracy[method]:.1f}%", ha="center")
        ax.set_ylim(0, 100); ax.set_ylabel("Correct OC selection accuracy (%)"); ax.set_title("Held-Out Correct-OC Accuracy")
        ax.grid(axis="y", alpha=0.32)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"correct_oc_accuracy.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE / "underwater_feature_rule_2000_scenarios_8000_candidates.csv")
    validation_ledger = add_candidate_utility(pd.read_csv(VALIDATION_LEDGER))
    test_ledger = pd.read_csv(TEST_LEDGER)
    split = build_split(data, seed=2026)
    training = data[data.scenario_id.isin(split["train"])]
    validation = data[data.scenario_id.isin(split["validation"])]
    testing = data[data.scenario_id.isin(split["test"])]
    searches = []
    selected_params = {}
    for name, configs in model_configs().items():
        family = []
        for index, params in enumerate(configs):
            model = make_model(name, params)
            model.fit(training[FEATURES], training.is_oc)
            mean_utility, worst_node_utility, top1 = validation_result(model, validation, validation_ledger)
            row = {"model": name, "candidate": index, "mean_network_utility": mean_utility,
                   "worst_node_network_utility": worst_node_utility, "top1_accuracy": top1,
                   "parameters": json.dumps(params, sort_keys=True)}
            searches.append(row); family.append(row)
        best = sorted(
            family,
            key=lambda row: (row["mean_network_utility"], row["worst_node_network_utility"], row["top1_accuracy"]),
            reverse=True,
        )[0]
        selected_params[name] = json.loads(best["parameters"])
    pd.DataFrame(searches).to_csv(OUT / "validation_search.csv", index=False)
    (OUT / "selected_settings.json").write_text(json.dumps(selected_params, indent=2) + "\n")

    fit_rows = data[data.scenario_id.isin(split["train"] + split["validation"])]
    network_rows = []
    classification_rows = []
    accuracy = {"Baseline": 25.0}
    for name in ("SVM", "DTC", "RF"):
        model = make_model(name, selected_params[name])
        model.fit(fit_rows[FEATURES], fit_rows.is_oc)
        joblib.dump(model, OUT / f"{name.lower()}_model.joblib")
        binary = model.predict(testing[FEATURES])
        precision, recall, binary_f1, _ = precision_recall_fscore_support(
            testing.is_oc, binary, average="binary", zero_division=0
        )
        chosen = selections(model, testing)
        truth = testing.loc[testing.is_oc.eq(1)].set_index("scenario_id").auv_id
        actual = truth.loc[chosen.scenario_id].to_numpy()
        accuracy[name] = 100.0 * chosen.selected_oc.eq(actual).mean()
        classification_rows.append({"Method": name, "Binary accuracy (%)": 100.0 * accuracy_score(testing.is_oc, binary),
                                    "Precision": precision, "Recall": recall, "Binary F1": binary_f1,
                                    "Correct OC selection accuracy (%)": accuracy[name],
                                    "Top-1 macro F1": f1_score(actual, chosen.selected_oc, labels=[0,1,2,3], average="macro", zero_division=0)})
        mapped = chosen[["scenario_id", "selected_oc"]].merge(
            test_ledger, on=["scenario_id", "selected_oc"], validate="one_to_one"
        )
        for nodes in NODES:
            network_rows.append({"Nodes": nodes, "Method": name, **summarize(mapped[mapped.node_count.eq(nodes)])})
    baseline = expected_baseline(test_ledger)
    for nodes in NODES:
        network_rows.append({"Nodes": nodes, "Method": "Baseline", **summarize(baseline[baseline.node_count.eq(nodes)])})
    network = pd.DataFrame(network_rows)
    classification = pd.DataFrame(classification_rows)
    network.to_csv(OUT / "network_metrics_by_node.csv", index=False)
    classification.to_csv(OUT / "classification_metrics.csv", index=False)
    plot_results(network, accuracy)
    print(json.dumps(selected_params, indent=2))
    print(classification.to_string(index=False))
    print(network.to_string(index=False))


if __name__ == "__main__":
    main()
