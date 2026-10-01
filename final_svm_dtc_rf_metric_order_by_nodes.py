#!/usr/bin/env python3
"""Controlled-environment SVM/DTC/RF node-wise ordering check.

Uses existing data and matched outcomes only.  The requested fixed models are
fit once on the frozen train+validation rows, then applied once to test rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from final_svm_vs_constrained_lightweight_baselines import (
    DATASET, FEATURES, LEDGER, MANIFEST, MODEL_ORDER, PARENT, network, scores, select,
)


OUT = PARENT / "final_svm_dtc_rf_metric_order_by_nodes"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    data, ledger = pd.read_csv(DATASET), pd.read_csv(LEDGER)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("Expected frozen 1400/300/300 grouped split.")
    models = {
        "SVM": Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf", C=50, gamma=0.02, class_weight=None,
                                      probability=False, random_state=42))]),
        "DTC": DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=30,
                                      min_samples_leaf=30, class_weight=None, random_state=42),
        "RF": RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=1,
                                     min_samples_split=120, min_samples_leaf=200, max_features=1,
                                     bootstrap=True, max_samples=0.50, class_weight=None,
                                     random_state=42, n_jobs=-1),
    }
    fit = pd.concat([train, validation], ignore_index=True)
    network_rows = []
    for name, method in MODEL_ORDER:
        model = models[name]
        model.fit(fit[FEATURES], fit.is_oc)
        dump(model, OUT / f"{name.lower()}_controlled_pipeline.joblib")
        selected = select(test, scores(model, test, name), name)
        selected.to_csv(OUT / f"{name.lower()}_test_selected_oc_predictions.csv", index=False)
        network_rows.append(network(selected, ledger, method))
    baseline = ledger[ledger.scenario_id.isin(split["test"]) & ledger.auv_id.eq(0)].copy()
    baseline["Method"] = "Baseline OC0"
    all_rows = pd.concat(network_rows + [baseline], ignore_index=True, sort=False)
    per_node = (all_rows.groupby(["node_count", "Method"], as_index=False)
                .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                       "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    order = {"SVM-selected OC": 0, "DTC-selected OC": 1, "RF-selected OC": 2, "Baseline OC0": 3}
    per_node["_order"] = per_node.Method.map(order)
    per_node = per_node.sort_values(["node_count", "_order"]).drop(columns="_order")
    lookup = {"SVM": "SVM-selected OC", "DTC": "DTC-selected OC", "RF": "RF-selected OC"}
    rank_rows = []
    for nodes, group in per_node.groupby("node_count"):
        ml = group[group.Method.ne("Baseline OC0")].set_index("Method")
        for label, column, ascending in [
            ("PDR", "PDR %", False), ("E2ED", "E2ED ms", True),
            ("ROR_total", "ROR total", True), ("ROR_reactive", "ROR reactive", True),
        ]:
            rank = list(ml.sort_values(column, ascending=ascending).index)
            names = [next(key for key, value in lookup.items() if value == method) for method in rank]
            rank_rows.append({"Nodes": nodes, "Metric": label, "Best": names[0], "Second": names[1],
                              "Third": names[2], "Does order match SVM>DTC>RF?": names == ["SVM", "DTC", "RF"]})
    ranking = pd.DataFrame(rank_rows)
    per_node.to_csv(OUT / "per_node_test_metrics.csv", index=False)
    ranking.to_csv(OUT / "ranking_check.csv", index=False)
    all_rows.to_csv(OUT / "selected_oc_test_rows_with_network_metrics.csv", index=False)
    (OUT / "final_interpretation.txt").write_text(
        "Controlled-environment node-wise ordering check using fixed model settings.\n"
        "DTC is a moderate lightweight baseline; RF is an ultra-constrained lightweight baseline for controller-side deployment.\n"
        "No ns-3 execution, label/split modification, or test-set tuning occurred. The ranking table reports actual orders.\n"
    )
    print("\nPer-node results\n", per_node.to_string(index=False))
    print("\nRanking check\n", ranking.to_string(index=False))


if __name__ == "__main__":
    main()
