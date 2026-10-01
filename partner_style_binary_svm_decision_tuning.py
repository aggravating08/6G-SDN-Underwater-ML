#!/usr/bin/env python3
"""Fast validation-only SVM selection using decision-function ranking.

This is intentionally separate from the interrupted probability-enabled grid.
It does not simulate, change labels, or overwrite the prior experiment.  An
RBF SVC decision score is sufficient for choosing the highest-ranked OC among
the four candidates and for ROC/PR AUC.  Probability calibration is therefore
not performed during the large hyperparameter sweep.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "results" / "underwater_rebuild" / "current"
PARENT = CURRENT / "partner_style_binary_is_oc_experiment"
OUT = PARENT / "decision_function_svm"
OUT.mkdir(parents=True, exist_ok=True)
DATASET = PARENT / "binary_utility_is_oc_dataset.csv"
LEDGER = CURRENT / "final_paper_style_results" / "utility_label_experiment" / "utility_label_candidate_diagnostics.csv"
MANIFEST = CURRENT / "non_cognitive_ml_models" / "split_manifest.json"

FEATURES = [
    "x", "y", "speed", "local_density", "source_to_oc_distance",
    "destination_to_oc_distance", "source_covered_by_oc", "destination_covered_by_oc",
    "sensors_in_oc_range", "gateways_in_oc_range", "estimated_local_path_exists",
    "average_link_quality_in_oc_view", "estimated_hop_count", "mean_endpoint_distance",
    "endpoint_distance_balance", "local_view_sensor_fraction",
]


def selection_from_scores(data: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = data[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    out["decision_score"] = scores
    chosen = (out.sort_values(["scenario_id", "decision_score", "auv_id"], ascending=[True, False, True])
              .groupby("scenario_id", as_index=False).head(1).copy())
    truth = data[data.is_oc.eq(1)].set_index("scenario_id").auv_id
    chosen["true_oc"] = truth.loc[chosen.scenario_id].to_numpy()
    chosen["correct"] = (chosen.auv_id == chosen.true_oc).astype(int)
    return chosen.rename(columns={"auv_id": "predicted_oc"})


def top1(model, data: pd.DataFrame) -> float:
    return float(selection_from_scores(model, data))


def main() -> None:
    data = pd.read_csv(DATASET)
    ledger = pd.read_csv(LEDGER)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("Expected frozen 1400/300/300 grouped split.")

    best: tuple[float, float] = (-1., -1.)
    best_params: dict[str, object] | None = None
    rows = []
    for c, gamma, weight in itertools.product(
        [1, 2, 5, 10, 20, 50, 100, 200, 500],
        [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        [None, "balanced"],
    ):
        params = {"C": c, "gamma": gamma, "class_weight": weight}
        model = Pipeline([("scale", StandardScaler()),
                          ("svc", SVC(kernel="rbf", probability=False, random_state=7, **params))])
        model.fit(train[FEATURES], train.is_oc)
        selected = selection_from_scores(validation, model.decision_function(validation[FEATURES]))
        top = float(selected.correct.mean())
        _, _, f1, _ = precision_recall_fscore_support(
            validation.is_oc, model.predict(validation[FEATURES]), average="binary", zero_division=0
        )
        rows.append({**params, "validation_top1": top, "validation_binary_f1": f1})
        if (top, f1) > best:
            best, best_params = (top, f1), params
    search = pd.DataFrame(rows).sort_values(["validation_top1", "validation_binary_f1"], ascending=False)
    search.to_csv(OUT / "svm_decision_validation_search.csv", index=False)

    # One final train+validation fit; the held-out test is evaluated below once.
    fit = pd.concat([train, validation], ignore_index=True)
    final = Pipeline([("scale", StandardScaler()),
                      ("svc", SVC(kernel="rbf", probability=False, random_state=7, **best_params))])
    final.fit(fit[FEATURES], fit.is_oc)
    dump(final, OUT / "svm_decision_function_pipeline.joblib")

    test_scores = final.decision_function(test[FEATURES])
    test_labels = final.predict(test[FEATURES])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test.is_oc, test_labels, average="binary", zero_division=0
    )
    selected = selection_from_scores(test, test_scores)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        selected.true_oc, selected.predicted_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )
    binary = pd.DataFrame([{
        "Model": "SVM decision-function", "Binary Accuracy": accuracy_score(test.is_oc, test_labels),
        "Precision": precision, "Recall": recall, "F1": f1,
        "ROC-AUC": roc_auc_score(test.is_oc, test_scores),
        "PR-AUC": average_precision_score(test.is_oc, test_scores),
        "Best validation Top-1": best[0], "Best validation binary F1": best[1],
        "Best parameters": json.dumps(best_params),
    }])
    scenario = pd.DataFrame([{
        "Model": "SVM decision-function", "Top-1 OC Accuracy": selected.correct.mean(),
        "Macro Precision": mp, "Macro Recall": mr, "Macro F1": mf1,
    }])
    selected.insert(0, "Model", "SVM-selected OC")
    selected.to_csv(OUT / "svm_decision_test_selected_oc_predictions.csv", index=False)
    network = selected.merge(
        ledger[["scenario_id", "auv_id", "PDR", "E2ED_ms", "ROR_total", "ROR_reactive"]],
        left_on=["scenario_id", "predicted_oc"], right_on=["scenario_id", "auv_id"],
        how="left", validate="one_to_one",
    )
    network["Method"] = "SVM-selected OC"
    baseline = ledger[ledger.scenario_id.isin(split["test"]) & ledger.auv_id.eq(0)].copy()
    baseline["Method"] = "Baseline fixed OC0"
    all_network = pd.concat([network, baseline], ignore_index=True, sort=False)
    overall = (all_network.groupby("Method", as_index=False)
               .agg(**{"Mean PDR %": ("PDR", "mean"), "Mean E2ED ms": ("E2ED_ms", "mean"),
                      "Mean ROR total": ("ROR_total", "mean"), "Mean ROR reactive": ("ROR_reactive", "mean")}))
    per_node = (all_network.groupby(["node_count", "Method"], as_index=False)
                .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                       "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    binary.to_csv(OUT / "candidate_binary_test_metrics.csv", index=False)
    scenario.to_csv(OUT / "scenario_top1_test_metrics.csv", index=False)
    overall.to_csv(OUT / "network_metrics.csv", index=False)
    per_node.to_csv(OUT / "per_node_network_metrics.csv", index=False)
    network.to_csv(OUT / "model_selected_oc_rows_with_network_metrics.csv", index=False)
    print("Candidate binary metrics\n", binary.to_string(index=False))
    print("\nScenario top-1 metrics\n", scenario.to_string(index=False))
    print("\nNetwork metrics\n", overall.to_string(index=False))


if __name__ == "__main__":
    main()
