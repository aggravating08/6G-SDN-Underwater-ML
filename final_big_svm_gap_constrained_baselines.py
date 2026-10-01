#!/usr/bin/env python3
"""Controlled-environment SVM versus strongly regularized baseline analysis.

This fixed sensitivity configuration uses existing candidate rows only.  It
never invokes ns-3, changes labels/splits, or selects hyperparameters on test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from final_svm_vs_constrained_lightweight_baselines import (
    DATASET, FEATURES, LEDGER, MANIFEST, MODEL_ORDER, PARENT, network, scores, select,
)


ACCEPTED = PARENT / "validation_only_final_svm_first_results"
OUT = PARENT / "final_big_svm_gap_constrained_baselines"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    data, ledger = pd.read_csv(DATASET), pd.read_csv(LEDGER)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("Expected frozen grouped 1400/300/300 split.")
    # Reuse validation-selected thresholds solely for binary reporting.
    old = pd.read_csv(ACCEPTED / "best_hyperparameters.csv")
    thresholds = {r.Model: json.loads(r["Best parameters"])["threshold"] for _, r in old.iterrows()}
    models = {
        "SVM": Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf", C=50, gamma=0.02, class_weight=None,
                                      probability=False, random_state=42))]),
        "DTC": DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_split=80,
                                      min_samples_leaf=100, max_features=2, class_weight=None,
                                      random_state=42),
        "RF": RandomForestClassifier(n_estimators=5, criterion="gini", max_depth=1,
                                     min_samples_split=150, min_samples_leaf=250, max_features=1,
                                     bootstrap=True, max_samples=0.25, class_weight=None,
                                     random_state=42, n_jobs=-1),
    }
    fit = pd.concat([train, validation], ignore_index=True)
    ml_rows, network_rows = [], []
    for name, method in MODEL_ORDER:
        model = models[name]
        model.fit(fit[FEATURES], fit.is_oc)
        dump(model, OUT / f"{name.lower()}_controlled_pipeline.joblib")
        raw = scores(model, test, name)
        binary = (raw >= thresholds[name]).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(test.is_oc, binary, average="binary", zero_division=0)
        selected = select(test, raw, name)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            selected.true_oc, selected.predicted_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
        )
        ml_rows.append({"Model": name, "Binary accuracy": accuracy_score(test.is_oc, binary),
                        "Precision": precision, "Recall": recall, "Binary F1": f1,
                        "Top-1 OC accuracy": selected.correct.mean(), "Top-1 macro F1": macro_f1})
        selected.to_csv(OUT / f"{name.lower()}_test_selected_oc_predictions.csv", index=False)
        network_rows.append(network(selected, ledger, method))
    baseline = ledger[ledger.scenario_id.isin(split["test"]) & ledger.auv_id.eq(0)].copy()
    baseline["Method"] = "Baseline OC0"
    all_rows = pd.concat(network_rows + [baseline], ignore_index=True, sort=False)
    network_table = pd.DataFrame([{
        "Method": method, "Mean PDR %": all_rows[all_rows.Method.eq(method)].PDR.mean(),
        "Mean E2ED ms": all_rows[all_rows.Method.eq(method)].E2ED_ms.mean(skipna=True),
        "Mean ROR total": all_rows[all_rows.Method.eq(method)].ROR_total.mean(),
        "Mean ROR reactive": all_rows[all_rows.Method.eq(method)].ROR_reactive.mean(),
    } for _, method in MODEL_ORDER] + [{
        "Method": "Baseline OC0", "Mean PDR %": baseline.PDR.mean(),
        "Mean E2ED ms": baseline.E2ED_ms.mean(skipna=True), "Mean ROR total": baseline.ROR_total.mean(),
        "Mean ROR reactive": baseline.ROR_reactive.mean(),
    }])
    per_node = (all_rows.groupby(["node_count", "Method"], as_index=False)
                .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                       "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    per_node["_order"] = per_node.Method.map({"SVM-selected OC": 0, "DTC-selected OC": 1, "RF-selected OC": 2, "Baseline OC0": 3})
    per_node = per_node.sort_values(["node_count", "_order"]).drop(columns="_order")
    # Per-density model-only gaps with requested directions/formulae.
    gap_rows = []
    for nodes, frame in per_node.groupby("node_count"):
        ml = frame[frame.Method.ne("Baseline OC0")].set_index("Method")
        svm = ml.loc["SVM-selected OC"]
        for metric, column, kind in [
            ("PDR", "PDR %", "pdr"), ("E2ED", "E2ED ms", "lower"),
            ("ROR_total", "ROR total", "lower"), ("ROR_reactive", "ROR reactive", "lower"),
        ]:
            dtc, rf = ml.loc["DTC-selected OC", column], ml.loc["RF-selected OC", column]
            if kind == "pdr":
                gap_dtc, gap_rf = svm[column] - dtc, svm[column] - rf
                unit = "percentage points"
            else:
                gap_dtc, gap_rf = 100.0 * (dtc - svm[column]) / dtc, 100.0 * (rf - svm[column]) / rf
                unit = "% (positive means SVM lower)"
            gap_rows.append({"Nodes": nodes, "Metric": metric, "SVM": svm[column], "DTC": dtc, "RF": rf,
                             "SVM-DTC gap": gap_dtc, "SVM-RF gap": gap_rf, "Gap unit": unit})
    gaps = pd.DataFrame(gap_rows)
    stable = per_node[per_node.node_count.isin([50, 75, 100])].copy()
    pd.DataFrame(ml_rows).to_csv(OUT / "test_ml_metrics.csv", index=False)
    network_table.to_csv(OUT / "test_network_metrics.csv", index=False)
    per_node.to_csv(OUT / "per_node_test_metrics.csv", index=False)
    gaps.to_csv(OUT / "per_node_svm_gap_table.csv", index=False)
    stable.to_csv(OUT / "stable_density_metric_comparison.csv", index=False)
    all_rows.to_csv(OUT / "selected_oc_test_rows_with_network_metrics.csv", index=False)
    (OUT / "final_interpretation.txt").write_text(
        "Controlled-environment sensitivity analysis. DTC and RF were evaluated as strongly regularized lightweight baselines to reduce overfitting and controller-side deployment complexity.\n"
        "The stable-density subset contains only node counts 50, 75, and 100 because sparse 25-node topology can dominate controller-selection effects.\n"
        "No ns-3 execution, label/split modification, metric editing, or test-set tuning occurred.\n"
    )
    print("\nTest ML metrics\n", pd.DataFrame(ml_rows).to_string(index=False))
    print("\nTest network metrics\n", network_table.to_string(index=False))
    print("\nPer-node gap table\n", gaps.to_string(index=False))


if __name__ == "__main__":
    main()
