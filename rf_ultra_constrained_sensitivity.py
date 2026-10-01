#!/usr/bin/env python3
"""Fixed ultra-lightweight RF controller-selection sensitivity run.

No ns-3 process is started; outcomes are looked up from the matched ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from final_svm_vs_constrained_lightweight_baselines import DATASET, FEATURES, LEDGER, MANIFEST, PARENT, network, select


OUT = PARENT / "rf_ultra_constrained_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
ACCEPTED = PARENT / "validation_only_final_svm_first_results"


def main() -> None:
    data, ledger = pd.read_csv(DATASET), pd.read_csv(LEDGER)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("Expected frozen grouped 1400/300/300 split.")
    threshold_table = pd.read_csv(ACCEPTED / "best_hyperparameters.csv")
    threshold = json.loads(threshold_table.loc[threshold_table.Model.eq("RF"), "Best parameters"].iat[0])["threshold"]
    model = RandomForestClassifier(
        n_estimators=3, max_depth=1, min_samples_leaf=1000, min_samples_split=2,
        max_samples=0.20, max_features=1, bootstrap=True, class_weight=None,
        random_state=42, n_jobs=-1,
    )
    fit = pd.concat([train, validation], ignore_index=True)
    model.fit(fit[FEATURES], fit.is_oc)
    dump(model, OUT / "rf_ultra_constrained_pipeline.joblib")
    scores = model.predict_proba(test[FEATURES])[:, 1]
    binary = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(test.is_oc, binary, average="binary", zero_division=0)
    selected = select(test, scores, "RF")
    _, _, macro_f1, _ = precision_recall_fscore_support(
        selected.true_oc, selected.predicted_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )
    metrics = pd.DataFrame([{
        "Model": "RF ultra-constrained", "Binary accuracy": accuracy_score(test.is_oc, binary),
        "Precision": precision, "Recall": recall, "Binary F1": f1,
        "Top-1 OC accuracy": selected.correct.mean(), "Top-1 macro F1": macro_f1,
    }])
    selected.to_csv(OUT / "rf_test_selected_oc_predictions.csv", index=False)
    results = network(selected, ledger, "RF-selected OC")
    summary = pd.DataFrame([{
        "Method": "RF-selected OC", "Mean PDR %": results.PDR.mean(),
        "Mean E2ED ms": results.E2ED_ms.mean(skipna=True), "Mean ROR total": results.ROR_total.mean(),
        "Mean ROR reactive": results.ROR_reactive.mean(),
    }])
    per_node = (results.groupby("node_count", as_index=False)
                .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                       "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    metrics.to_csv(OUT / "rf_test_ml_metrics.csv", index=False)
    summary.to_csv(OUT / "rf_test_network_metrics.csv", index=False)
    per_node.to_csv(OUT / "rf_per_node_test_metrics.csv", index=False)
    results.to_csv(OUT / "rf_selected_oc_test_rows_with_network_metrics.csv", index=False)
    (OUT / "interpretation.txt").write_text(
        "RF ultra-constrained controlled sensitivity: 3 depth-1 trees with large leaf-size regularization.\n"
        "This result is a controller-side complexity stress baseline, not a general RF benchmark.\n"
        "No ns-3 execution, label/split change, or test-set tuning occurred.\n"
    )
    print(metrics.to_string(index=False))
    print(summary.to_string(index=False))
    print(per_node.to_string(index=False))


if __name__ == "__main__":
    main()
