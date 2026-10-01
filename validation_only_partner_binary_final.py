#!/usr/bin/env python3
"""Validation-only final partner-style binary OC-selection experiment.

No ns-3 execution occurs here.  Models see only pre-routing observables; the
matched candidate ledger is used for the binary utility label, validation
network-tradeoff selection, and final selected-OC metric lookup.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "results" / "underwater_rebuild" / "current"
PARENT = CURRENT / "partner_style_binary_is_oc_experiment"
OUT = PARENT / "validation_only_final_svm_first_results"
OUT.mkdir(parents=True, exist_ok=True)
DATASET = PARENT / "binary_utility_is_oc_dataset.csv"
LEDGER = CURRENT / "final_paper_style_results" / "utility_label_experiment" / "utility_label_candidate_diagnostics.csv"
MANIFEST = CURRENT / "non_cognitive_ml_models" / "split_manifest.json"

# Exact user-specified input contract: all pre-routing, no IDs/outcomes.
FEATURES = [
    "x", "y", "local_density", "speed",
    "source_to_oc_distance", "destination_to_oc_distance",
    "sensors_in_oc_range", "gateways_in_oc_range",
    "source_covered_by_oc", "destination_covered_by_oc",
    "estimated_local_path_exists", "estimated_hop_count",
    "mean_endpoint_distance", "endpoint_distance_balance",
    "local_view_sensor_fraction",
]
ORDER = ["SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0"]


def raw_scores(model, x: pd.DataFrame, kind: str) -> np.ndarray:
    return model.decision_function(x) if kind == "SVM" else model.predict_proba(x)[:, 1]


def rank_oc(data: pd.DataFrame, scores: np.ndarray, model: str) -> pd.DataFrame:
    ranked = data[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    ranked["raw_score"] = scores
    chosen = (ranked.sort_values(["scenario_id", "raw_score", "auv_id"], ascending=[True, False, True])
              .groupby("scenario_id", as_index=False).head(1).copy())
    truth = data[data.is_oc.eq(1)].set_index("scenario_id").auv_id
    chosen["true_oc"] = truth.loc[chosen.scenario_id].to_numpy()
    chosen["correct"] = (chosen.auv_id == chosen.true_oc).astype(int)
    chosen["Model"] = model
    return chosen.rename(columns={"auv_id": "predicted_oc"})


def threshold_candidates(kind: str, scores: np.ndarray) -> np.ndarray:
    if kind == "SVM":
        # Decision scores have no fixed probabilistic threshold.
        return np.unique(np.r_[np.linspace(-2.0, 2.0, 161), np.quantile(scores, np.linspace(0.01, 0.99, 99))])
    return np.unique(np.r_[np.linspace(0.01, 0.99, 99), np.quantile(scores, np.linspace(0.01, 0.99, 99))])


def choose_threshold(y: pd.Series, scores: np.ndarray, kind: str) -> tuple[float, float, float]:
    """Choose on validation F1, with candidate binary accuracy as tie-break."""
    best = (-1.0, -1.0)
    best_threshold = 0.0
    for threshold in threshold_candidates(kind, scores):
        predicted = (scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
        acc = accuracy_score(y, predicted)
        if (f1, acc) > best:
            best, best_threshold = (f1, acc), float(threshold)
    return best_threshold, best[0], best[1]


def attach_network(selected: pd.DataFrame, ledger: pd.DataFrame, method: str) -> pd.DataFrame:
    m = selected.merge(ledger[["scenario_id", "auv_id", "PDR", "E2ED_ms", "ROR_total", "ROR_reactive"]],
                       left_on=["scenario_id", "predicted_oc"], right_on=["scenario_id", "auv_id"],
                       how="left", validate="one_to_one")
    if m.PDR.isna().any():
        raise ValueError("Selected OC lacks a matched candidate metric row.")
    m["Method"] = method
    return m


def validation_utility(selected: pd.DataFrame, ledger: pd.DataFrame, baseline: pd.DataFrame,
                       f1: float) -> tuple[float, float, float, float]:
    rows = attach_network(selected, ledger, "temporary")
    pdr_gain = (rows.PDR.mean() - baseline.PDR.mean()) / 100.0
    ror_reduction = baseline.ROR_total.mean() - rows.ROR_total.mean()
    top1 = selected.correct.mean()
    # All terms are benefits, expressed on compatible roughly [0,1] scales.
    # The hyphens in the request are interpreted as list bullets, not as
    # penalties for higher F1/PDR gain/ROR reduction.
    score = 0.40 * top1 + 0.25 * f1 + 0.20 * pdr_gain + 0.15 * ror_reduction
    return float(score), float(top1), float(pdr_gain), float(ror_reduction)


def make_model(kind: str, params: dict[str, object]):
    if kind == "SVM":
        return Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf", probability=False, random_state=7, **params))])
    if kind == "DTC":
        return DecisionTreeClassifier(random_state=7, **params)
    return RandomForestClassifier(random_state=7, n_jobs=-1, **params)


def parameter_grid(kind: str):
    if kind == "SVM":
        for c, gamma in itertools.product([20, 50, 100, 200], [0.002, 0.005, 0.01, 0.02]):
            yield {"C": c, "gamma": gamma, "class_weight": None}
    elif kind == "DTC":
        for depth, split, leaf in itertools.product([4, 5, 6], [10, 20], [10, 15, 20]):
            yield {"criterion": "gini", "max_depth": depth, "min_samples_split": split,
                   "min_samples_leaf": leaf, "class_weight": None}
    elif kind == "RF":
        for depth, split, leaf in itertools.product([4, 5, 6], [20, 30], [30, 50]):
            yield {"n_estimators": 100, "criterion": "gini", "max_depth": depth,
                   "min_samples_split": split, "min_samples_leaf": leaf,
                   "max_features": "sqrt", "class_weight": None}
    else:
        raise ValueError(kind)


def tune(kind: str, train: pd.DataFrame, validation: pd.DataFrame, ledger: pd.DataFrame,
         baseline_validation: pd.DataFrame):
    rows = []
    best_key = (-np.inf, -np.inf, -np.inf)
    best = None
    for params in parameter_grid(kind):
        model = make_model(kind, params)
        model.fit(train[FEATURES], train.is_oc)
        scores = raw_scores(model, validation[FEATURES], kind)
        threshold, f1, acc = choose_threshold(validation.is_oc, scores, kind)
        chosen = rank_oc(validation, scores, kind)
        utility, top1, pdr_gain, ror_reduction = validation_utility(chosen, ledger, baseline_validation, f1)
        record = {**params, "threshold": threshold, "validation_score": utility,
                  "validation_top1": top1, "validation_binary_f1": f1,
                  "validation_binary_accuracy": acc, "validation_pdr_gain": pdr_gain,
                  "validation_ror_reduction": ror_reduction}
        rows.append(record)
        # Utility is primary.  Top-1 and F1 determine exact ties.
        if (utility, top1, f1) > best_key:
            best_key = (utility, top1, f1)
            best = record
    pd.DataFrame(rows).sort_values(["validation_score", "validation_top1", "validation_binary_f1"],
                                   ascending=False).to_csv(OUT / f"{kind.lower()}_validation_search.csv", index=False)
    return best


def test_report(kind: str, model, threshold: float, test: pd.DataFrame, ledger: pd.DataFrame):
    scores = raw_scores(model, test[FEATURES], kind)
    binary_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test.is_oc, binary_pred, average="binary", zero_division=0
    )
    selected = rank_oc(test, scores, kind)
    mp, mr, macro_f1, _ = precision_recall_fscore_support(
        selected.true_oc, selected.predicted_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )
    binary = {"Model": kind, "Binary accuracy": accuracy_score(test.is_oc, binary_pred),
              "Binary precision": precision, "Binary recall": recall, "Binary F1": f1,
              "Top-1 OC accuracy": selected.correct.mean(), "Top-1 macro F1": macro_f1,
              "Top-1 macro precision": mp, "Top-1 macro recall": mr}
    network = attach_network(selected, ledger, f"{kind}-selected OC")
    selected.to_csv(OUT / f"{kind.lower()}_test_selected_oc_predictions.csv", index=False)
    return binary, network


def main() -> None:
    data = pd.read_csv(DATASET)
    ledger = pd.read_csv(LEDGER)
    required = set(FEATURES) | {"scenario_id", "node_count", "is_oc"}
    if not required.issubset(data.columns):
        raise ValueError(f"Dataset lacks requested feature(s): {sorted(required - set(data.columns))}")
    with MANIFEST.open() as f:
        manifest = json.load(f)
    train = data[data.scenario_id.isin(manifest["train"])].copy()
    validation = data[data.scenario_id.isin(manifest["validation"])].copy()
    test = data[data.scenario_id.isin(manifest["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("Expected frozen grouped 1400/300/300 split.")
    base_val = ledger[ledger.scenario_id.isin(manifest["validation"]) & ledger.auv_id.eq(0)].copy()

    best_rows, metric_rows, network_rows = [], [], []
    for kind in ("SVM", "DTC", "RF"):
        print(f"Validation-tuning {kind}...", flush=True)
        best = tune(kind, train, validation, ledger, base_val)
        best_rows.append({"Model": kind, "Best parameters": json.dumps(
            {k: v for k, v in best.items() if k not in {"validation_score", "validation_top1", "validation_binary_f1", "validation_binary_accuracy", "validation_pdr_gain", "validation_ror_reduction"}},
            default=str), "Validation score": best["validation_score"]})
        fit = pd.concat([train, validation], ignore_index=True)
        model = make_model(kind, {k: best[k] for k in best if k in {
            "C", "gamma", "class_weight", "criterion", "max_depth", "min_samples_split", "min_samples_leaf", "n_estimators", "max_features"}})
        model.fit(fit[FEATURES], fit.is_oc)
        dump(model, OUT / f"{kind.lower()}_pipeline.joblib")
        metrics, network = test_report(kind, model, best["threshold"], test, ledger)
        metric_rows.append(metrics)
        network_rows.append(network)

    baseline = ledger[ledger.scenario_id.isin(manifest["test"]) & ledger.auv_id.eq(0)].copy()
    baseline["Method"] = "Baseline OC0"
    all_network = pd.concat(network_rows + [baseline], ignore_index=True, sort=False)
    overall = pd.DataFrame([{
        "Method": method,
        "Mean PDR %": all_network[all_network.Method.eq(method)].PDR.mean(),
        "Mean E2ED ms": all_network[all_network.Method.eq(method)].E2ED_ms.mean(skipna=True),
        "Mean ROR total": all_network[all_network.Method.eq(method)].ROR_total.mean(),
        "Mean ROR reactive": all_network[all_network.Method.eq(method)].ROR_reactive.mean(),
    } for method in ORDER])
    per_node = (all_network.groupby(["node_count", "Method"], as_index=False)
                .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                       "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    per_node["_order"] = per_node.Method.map({m: i for i, m in enumerate(ORDER)})
    per_node = per_node.sort_values(["node_count", "_order"]).drop(columns="_order")
    best_table = pd.DataFrame(best_rows)
    metrics_table = pd.DataFrame(metric_rows)
    best_table.to_csv(OUT / "best_hyperparameters.csv", index=False)
    metrics_table.to_csv(OUT / "test_ml_metrics.csv", index=False)
    overall.to_csv(OUT / "test_network_metrics.csv", index=False)
    per_node.to_csv(OUT / "per_node_test_network_metrics.csv", index=False)
    all_network.to_csv(OUT / "selected_oc_test_rows_with_network_metrics.csv", index=False)
    interpretation = [
        "Validation-only grouped binary is_oc experiment.",
        "All model settings and binary thresholds were selected on validation scenarios only.",
        "Scenario-level Top-1 selection ranks the four raw candidate scores; its result never uses the binary threshold.",
        "SVM is the primary proposed model only if it has the highest held-out Top-1 accuracy and the best or near-best network tradeoff.",
        "No oracle, train metric, simulation rerun, or outcome-derived ML input is included.",
    ]
    (OUT / "final_interpretation.txt").write_text("\n".join(interpretation) + "\n")
    print("\nBest validation settings\n", best_table.to_string(index=False))
    print("\nHeld-out test ML metrics\n", metrics_table.to_string(index=False))
    print("\nHeld-out test network metrics\n", overall.to_string(index=False))
    print("\nHeld-out per-node network metrics\n", per_node.to_string(index=False))


if __name__ == "__main__":
    main()
