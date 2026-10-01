#!/usr/bin/env python3
"""Complete DTC/RF reporting for the grouped binary utility-label experiment.

SVM is deliberately not tuned or refit here: its decision-function experiment
is already complete.  DTC is loaded from the previously completed
validation-selected model.  RF uses its requested validation-only grid.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "results" / "underwater_rebuild" / "current"
OUT = CURRENT / "partner_style_binary_is_oc_experiment"
SVM_OUT = OUT / "decision_function_svm"
DATASET = OUT / "binary_utility_is_oc_dataset.csv"
LEDGER = CURRENT / "final_paper_style_results" / "utility_label_experiment" / "utility_label_candidate_diagnostics.csv"
MANIFEST = CURRENT / "non_cognitive_ml_models" / "split_manifest.json"
FEATURES = [
    "x", "y", "speed", "local_density", "source_to_oc_distance", "destination_to_oc_distance",
    "source_covered_by_oc", "destination_covered_by_oc", "sensors_in_oc_range", "gateways_in_oc_range",
    "estimated_local_path_exists", "average_link_quality_in_oc_view", "estimated_hop_count",
    "mean_endpoint_distance", "endpoint_distance_balance", "local_view_sensor_fraction",
]


def selected_rows(model, data: pd.DataFrame, name: str) -> pd.DataFrame:
    scores = model.predict_proba(data[FEATURES])[:, 1]
    out = data[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    out["score"] = scores
    chosen = (out.sort_values(["scenario_id", "score", "auv_id"], ascending=[True, False, True])
              .groupby("scenario_id", as_index=False).head(1).copy())
    truth = data[data.is_oc.eq(1)].set_index("scenario_id").auv_id
    chosen["true_oc"] = truth.loc[chosen.scenario_id].to_numpy()
    chosen["correct"] = (chosen.auv_id == chosen.true_oc).astype(int)
    chosen["Model"] = name
    return chosen.rename(columns={"auv_id": "predicted_oc"})


def evaluate(name: str, model, test: pd.DataFrame) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    scores = model.predict_proba(test[FEATURES])[:, 1]
    labels = model.predict(test[FEATURES])
    precision, recall, f1, _ = precision_recall_fscore_support(test.is_oc, labels, average="binary", zero_division=0)
    selected = selected_rows(model, test, name)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        selected.true_oc, selected.predicted_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )
    return ({"Model": name, "Binary Accuracy": accuracy_score(test.is_oc, labels), "Precision": precision,
             "Recall": recall, "F1": f1, "ROC-AUC": roc_auc_score(test.is_oc, scores),
             "PR-AUC": average_precision_score(test.is_oc, scores)},
            {"Model": name, "Top-1 OC Accuracy": selected.correct.mean(), "Macro Precision": mp,
             "Macro Recall": mr, "Macro F1": mf1}, selected)


def tune_rf(train: pd.DataFrame, validation: pd.DataFrame):
    rows = []
    best_key = (-1., -1.)
    best_params = None
    for estimators, depth, leaf, features, weight in itertools.product(
        [100, 200, 300], [4, 6, 8, 10, None], [3, 5, 10, 15, 20],
        ["sqrt", "log2", None], [None, "balanced"],
    ):
        params = {"n_estimators": estimators, "max_depth": depth, "min_samples_leaf": leaf,
                  "max_features": features, "class_weight": weight}
        model = RandomForestClassifier(random_state=7, n_jobs=-1, **params)
        model.fit(train[FEATURES], train.is_oc)
        selected = selected_rows(model, validation, "RF")
        top = float(selected.correct.mean())
        _, _, f1, _ = precision_recall_fscore_support(
            validation.is_oc, model.predict(validation[FEATURES]), average="binary", zero_division=0
        )
        rows.append({**params, "validation_top1": top, "validation_binary_f1": f1})
        if (top, f1) > best_key:
            best_key, best_params = (top, f1), params
    search = pd.DataFrame(rows).sort_values(["validation_top1", "validation_binary_f1"], ascending=False)
    search.to_csv(OUT / "rf_validation_search.csv", index=False)
    return best_params, best_key


def attach_network(selected: pd.DataFrame, ledger: pd.DataFrame, method: str) -> pd.DataFrame:
    out = selected.merge(ledger[["scenario_id", "auv_id", "PDR", "E2ED_ms", "ROR_total", "ROR_reactive"]],
                         left_on=["scenario_id", "predicted_oc"], right_on=["scenario_id", "auv_id"],
                         how="left", validate="one_to_one")
    out["Method"] = method
    return out


def main() -> None:
    data = pd.read_csv(DATASET)
    ledger = pd.read_csv(LEDGER)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()

    # DTC was wholly tuned and fit before the RF interruption; only reporting is resumed.
    dtc = load(OUT / "dtc_binary_is_oc_pipeline.joblib")
    dtc_binary, dtc_top, dtc_selected = evaluate("DTC", dtc, test)
    dtc_search = pd.read_csv(OUT / "dtc_validation_search.csv").sort_values(
        ["validation_top1", "validation_binary_f1"], ascending=False).iloc[0]
    dtc_binary["Best validation Top-1"] = dtc_search.validation_top1
    dtc_binary["Best validation binary F1"] = dtc_search.validation_binary_f1
    dtc_binary["Best parameters"] = json.dumps({k: dtc_search[k] for k in ["max_depth", "min_samples_leaf", "class_weight"] if pd.notna(dtc_search[k])}, default=str)
    dtc_selected.to_csv(OUT / "dtc_test_selected_oc_predictions.csv", index=False)

    print("Tuning RF using validation scenarios only...", flush=True)
    params, (val_top, val_f1) = tune_rf(train, validation)
    fit = pd.concat([train, validation], ignore_index=True)
    rf = RandomForestClassifier(random_state=7, n_jobs=-1, **params)
    rf.fit(fit[FEATURES], fit.is_oc)
    dump(rf, OUT / "rf_binary_is_oc_pipeline.joblib")
    rf_binary, rf_top, rf_selected = evaluate("RF", rf, test)
    rf_binary["Best validation Top-1"] = val_top
    rf_binary["Best validation binary F1"] = val_f1
    rf_binary["Best parameters"] = json.dumps(params)
    rf_selected.to_csv(OUT / "rf_test_selected_oc_predictions.csv", index=False)

    # Reuse the already completed SVM decision score selections, not its old probability model.
    svm_binary = pd.read_csv(SVM_OUT / "candidate_binary_test_metrics.csv")
    svm_top = pd.read_csv(SVM_OUT / "scenario_top1_test_metrics.csv")
    svm_selected = pd.read_csv(SVM_OUT / "svm_decision_test_selected_oc_predictions.csv")
    svm_selected["Model"] = "SVM"
    svm_network = attach_network(svm_selected, ledger, "SVM-selected OC")
    dtc_network = attach_network(dtc_selected, ledger, "DTC-selected OC")
    rf_network = attach_network(rf_selected, ledger, "RF-selected OC")
    baseline = ledger[ledger.scenario_id.isin(split["test"]) & ledger.auv_id.eq(0)].copy()
    baseline["Method"] = "Baseline fixed OC0"
    all_network = pd.concat([svm_network, dtc_network, rf_network, baseline], ignore_index=True, sort=False)

    binary = pd.concat([svm_binary, pd.DataFrame([dtc_binary, rf_binary])], ignore_index=True)
    top = pd.concat([svm_top, pd.DataFrame([dtc_top, rf_top])], ignore_index=True)
    order = ["SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline fixed OC0"]
    network = pd.DataFrame([{
        "Method": method,
        "Mean PDR %": all_network[all_network.Method == method].PDR.mean(),
        "Mean E2ED ms": all_network[all_network.Method == method].E2ED_ms.mean(skipna=True),
        "Mean ROR total": all_network[all_network.Method == method].ROR_total.mean(),
        "Mean ROR reactive": all_network[all_network.Method == method].ROR_reactive.mean(),
    } for method in order])
    per_node = (all_network.groupby(["node_count", "Method"], as_index=False)
                .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                       "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    per_node["_order"] = per_node.Method.map({m: i for i, m in enumerate(order)})
    per_node = per_node.sort_values(["node_count", "_order"]).drop(columns="_order")
    binary.to_csv(OUT / "candidate_binary_test_metrics_all_models.csv", index=False)
    top.to_csv(OUT / "scenario_top1_test_metrics_all_models.csv", index=False)
    network.to_csv(OUT / "final_network_metrics_all_models.csv", index=False)
    per_node.to_csv(OUT / "per_node_network_metrics_all_models.csv", index=False)
    all_network.to_csv(OUT / "model_selected_oc_rows_with_network_metrics_all_models.csv", index=False)
    print("\nBinary metrics\n", binary.to_string(index=False))
    print("\nScenario top-1 metrics\n", top.to_string(index=False))
    print("\nNetwork metrics\n", network.to_string(index=False))


if __name__ == "__main__":
    main()
