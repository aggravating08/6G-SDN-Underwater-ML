#!/usr/bin/env python3
"""SVM versus strongly regularized lightweight controller-side baselines.

This fixed configuration reuses the accepted dataset, grouped split, binary
thresholds, and matched candidate ledger.  It never invokes ns-3 or tunes on
the held-out test scenarios.
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
    CURRENT, DATASET, FEATURES, LEDGER, MANIFEST, MODEL_ORDER, PARENT,
    network, scores, select,
)


ACCEPTED = PARENT / "validation_only_final_svm_first_results"
OUT = PARENT / "final_svm_vs_stronger_constrained_baselines"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    data, ledger = pd.read_csv(DATASET), pd.read_csv(LEDGER)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("Expected the accepted 1400/300/300 grouped split.")
    old = pd.read_csv(ACCEPTED / "best_hyperparameters.csv")
    thresholds = {r.Model: json.loads(r["Best parameters"])["threshold"] for _, r in old.iterrows()}

    models = {
        "SVM": Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf", C=50, gamma=0.02, class_weight=None,
                                      probability=False, random_state=42))]),
        "DTC": DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_split=50,
                                      min_samples_leaf=75, class_weight=None, random_state=42),
        "RF": RandomForestClassifier(n_estimators=30, criterion="gini", max_depth=2,
                                     min_samples_split=60, min_samples_leaf=100, max_features="log2",
                                     class_weight=None, random_state=42, n_jobs=-1),
    }
    fit = pd.concat([train, validation], ignore_index=True)
    ml_rows, network_rows = [], []
    for name, method in MODEL_ORDER:
        model = models[name]
        model.fit(fit[FEATURES], fit.is_oc)
        dump(model, OUT / f"{name.lower()}_strongly_regularized_pipeline.joblib")
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
    ml_table = pd.DataFrame(ml_rows).set_index("Model")
    net_table = network_table.assign(Model=["SVM", "DTC", "RF", "Baseline OC0"]).set_index("Model")
    gap_rows = []
    for metric, table in [("Top-1 OC accuracy", ml_table), ("Mean PDR %", net_table),
                          ("Mean E2ED ms", net_table), ("Mean ROR total", net_table),
                          ("Mean ROR reactive", net_table)]:
        svm, dtc, rf = table.loc["SVM", metric], table.loc["DTC", metric], table.loc["RF", metric]
        gap_rows.append({"Metric": metric, "SVM": svm, "DTC": dtc, "RF": rf,
                         "SVM-DTC gap": svm-dtc, "SVM-RF gap": svm-rf})
    gaps = pd.DataFrame(gap_rows)
    ml_table.reset_index().to_csv(OUT / "test_ml_metrics.csv", index=False)
    network_table.to_csv(OUT / "test_network_metrics.csv", index=False)
    per_node.to_csv(OUT / "per_node_test_metrics.csv", index=False)
    gaps.to_csv(OUT / "svm_gap_table.csv", index=False)
    all_rows.to_csv(OUT / "selected_oc_test_rows_with_network_metrics.csv", index=False)
    (OUT / "final_interpretation.txt").write_text(
        "DTC and RF were evaluated as strongly regularized lightweight baseline classifiers to reduce overfitting and controller-side computational complexity.\n"
        "Raw scores select one OC from four candidates; validation-selected binary thresholds only support candidate-level binary metrics.\n"
        "No ns-3 execution, label/split modification, or test-set tuning occurred.\n"
    )
    print("\nTest ML metrics\n", ml_table.reset_index().to_string(index=False))
    print("\nTest network metrics\n", network_table.to_string(index=False))
    print("\nSVM gap table\n", gaps.to_string(index=False))


if __name__ == "__main__":
    main()
