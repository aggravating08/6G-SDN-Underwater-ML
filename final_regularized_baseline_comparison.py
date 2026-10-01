#!/usr/bin/env python3
"""Fixed, regularized baseline comparison for the accepted grouped experiment.

No simulator call and no hyperparameter/threshold tuning occur here.  The
binary thresholds selected in the accepted validation-only study are reused;
all models are fit once on train+validation and evaluated once on test.
"""
from __future__ import annotations

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
ACCEPTED = PARENT / "validation_only_final_svm_first_results"
OUT = PARENT / "final_regularized_baseline_comparison"
OUT.mkdir(parents=True, exist_ok=True)
DATASET = PARENT / "binary_utility_is_oc_dataset.csv"
LEDGER = CURRENT / "final_paper_style_results" / "utility_label_experiment" / "utility_label_candidate_diagnostics.csv"
MANIFEST = CURRENT / "non_cognitive_ml_models" / "split_manifest.json"
FEATURES = [
    "x", "y", "local_density", "speed", "source_to_oc_distance", "destination_to_oc_distance",
    "sensors_in_oc_range", "gateways_in_oc_range", "source_covered_by_oc", "destination_covered_by_oc",
    "estimated_local_path_exists", "estimated_hop_count", "mean_endpoint_distance",
    "endpoint_distance_balance", "local_view_sensor_fraction",
]
MODEL_ORDER = [("SVM", "SVM-selected OC"), ("DTC", "DTC-selected OC"), ("RF", "RF-selected OC")]


def score(model, x: pd.DataFrame, name: str) -> np.ndarray:
    return model.decision_function(x) if name == "SVM" else model.predict_proba(x)[:, 1]


def select(data: pd.DataFrame, raw: np.ndarray, name: str) -> pd.DataFrame:
    candidates = data[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    candidates["raw_score"] = raw
    chosen = (candidates.sort_values(["scenario_id", "raw_score", "auv_id"], ascending=[True, False, True])
               .groupby("scenario_id", as_index=False).head(1).copy())
    truth = data[data.is_oc.eq(1)].set_index("scenario_id").auv_id
    chosen["true_oc"] = truth.loc[chosen.scenario_id].to_numpy()
    chosen["correct"] = (chosen.auv_id == chosen.true_oc).astype(int)
    chosen["Model"] = name
    return chosen.rename(columns={"auv_id": "predicted_oc"})


def map_network(selected: pd.DataFrame, ledger: pd.DataFrame, method: str) -> pd.DataFrame:
    result = selected.merge(ledger[["scenario_id", "auv_id", "PDR", "E2ED_ms", "ROR_total", "ROR_reactive"]],
                            left_on=["scenario_id", "predicted_oc"], right_on=["scenario_id", "auv_id"],
                            how="left", validate="one_to_one")
    if result.PDR.isna().any():
        raise ValueError("Missing matched candidate network row.")
    result["Method"] = method
    return result


def inverse_minmax(values: pd.Series) -> pd.Series:
    lo, hi = values.min(), values.max()
    return pd.Series(1.0, index=values.index) if np.isclose(lo, hi) else 1.0 - (values - lo) / (hi - lo)


def minmax(values: pd.Series) -> pd.Series:
    lo, hi = values.min(), values.max()
    return pd.Series(1.0, index=values.index) if np.isclose(lo, hi) else (values - lo) / (hi - lo)


def main() -> None:
    data, ledger = pd.read_csv(DATASET), pd.read_csv(LEDGER)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("The accepted grouped 1400/300/300 split is required.")

    # Reuse frozen validation thresholds; this script performs no threshold selection.
    old = pd.read_csv(ACCEPTED / "best_hyperparameters.csv")
    thresholds = {row.Model: json.loads(row["Best parameters"])["threshold"] for _, row in old.iterrows()}
    models = {
        "SVM": Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf", C=50, gamma=0.02, class_weight=None,
                                      probability=False, random_state=42))]),
        "DTC": DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_split=10,
                                      min_samples_leaf=15, class_weight=None, random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=6,
                                     min_samples_split=20, min_samples_leaf=30, max_features="sqrt",
                                     class_weight=None, random_state=42, n_jobs=-1),
    }
    fit = pd.concat([train, validation], ignore_index=True)
    metric_rows, network_rows = [], []
    for name, method in MODEL_ORDER:
        model = models[name]
        model.fit(fit[FEATURES], fit.is_oc)
        dump(model, OUT / f"{name.lower()}_fixed_regularized_pipeline.joblib")
        raw = score(model, test[FEATURES], name)
        binary = (raw >= thresholds[name]).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(test.is_oc, binary, average="binary", zero_division=0)
        selected = select(test, raw, name)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            selected.true_oc, selected.predicted_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
        )
        metric_rows.append({"Model": name, "Binary accuracy": accuracy_score(test.is_oc, binary),
                            "Binary precision": precision, "Binary recall": recall, "Binary F1": f1,
                            "Top-1 OC accuracy": selected.correct.mean(), "Top-1 macro F1": macro_f1})
        selected.to_csv(OUT / f"{name.lower()}_test_selected_oc_predictions.csv", index=False)
        network_rows.append(map_network(selected, ledger, method))

    baseline = ledger[ledger.scenario_id.isin(split["test"]) & ledger.auv_id.eq(0)].copy()
    baseline["Method"] = "Baseline OC0"
    all_rows = pd.concat(network_rows + [baseline], ignore_index=True, sort=False)
    table_ml = pd.DataFrame(metric_rows)
    table_network = pd.DataFrame([{
        "Method": method,
        "Mean PDR %": all_rows[all_rows.Method.eq(method)].PDR.mean(),
        "Mean E2ED ms": all_rows[all_rows.Method.eq(method)].E2ED_ms.mean(skipna=True),
        "Mean ROR total": all_rows[all_rows.Method.eq(method)].ROR_total.mean(),
        "Mean ROR reactive": all_rows[all_rows.Method.eq(method)].ROR_reactive.mean(),
    } for _, method in MODEL_ORDER] + [{
        "Method": "Baseline OC0",
        "Mean PDR %": baseline.PDR.mean(), "Mean E2ED ms": baseline.E2ED_ms.mean(skipna=True),
        "Mean ROR total": baseline.ROR_total.mean(), "Mean ROR reactive": baseline.ROR_reactive.mean(),
    }])
    per_node = (all_rows.groupby(["node_count", "Method"], as_index=False)
                .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                       "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    order = {method: i for i, (_, method) in enumerate(MODEL_ORDER)} | {"Baseline OC0": 3}
    per_node["_order"] = per_node.Method.map(order)
    per_node = per_node.sort_values(["node_count", "_order"]).drop(columns="_order")

    # Composite is a model-selection comparison, so fixed OC0 is excluded:
    # Top-1 accuracy is undefined for a non-ML fixed policy.
    composite = table_ml.copy()
    composite["Method"] = composite.Model.map({"SVM": "SVM-selected OC", "DTC": "DTC-selected OC", "RF": "RF-selected OC"})
    composite = composite.merge(table_network[table_network.Method.ne("Baseline OC0")], on="Method", how="left", validate="one_to_one")
    # This explicit model-to-method mapping avoids treating baseline as a classifier.
    composite["norm_top1"] = minmax(composite["Top-1 OC accuracy"])
    composite["norm_pdr"] = minmax(composite["Mean PDR %"])
    composite["norm_reactive_ror"] = inverse_minmax(composite["Mean ROR reactive"])
    composite["norm_total_ror"] = inverse_minmax(composite["Mean ROR total"])
    composite["norm_e2ed"] = inverse_minmax(composite["Mean E2ED ms"])
    composite["Composite reliability-overhead score"] = (
        0.35 * composite.norm_top1 + 0.25 * composite.norm_pdr +
        0.20 * composite.norm_reactive_ror + 0.15 * composite.norm_total_ror +
        0.05 * composite.norm_e2ed
    )
    composite = composite[["Model", "Top-1 OC accuracy", "Mean PDR %", "Mean E2ED ms",
                           "Mean ROR total", "Mean ROR reactive", "Composite reliability-overhead score"]]

    table_ml.to_csv(OUT / "test_ml_metrics.csv", index=False)
    table_network.to_csv(OUT / "test_network_metrics.csv", index=False)
    per_node.to_csv(OUT / "per_node_test_metrics.csv", index=False)
    composite.to_csv(OUT / "composite_reliability_overhead_score.csv", index=False)
    all_rows.to_csv(OUT / "selected_oc_test_rows_with_network_metrics.csv", index=False)
    best = table_ml.loc[table_ml["Top-1 OC accuracy"].idxmax(), "Model"]
    message = [
        "Fixed-settings regularized/pruned baseline comparison.",
        "All models were fit on train+validation using preset settings; no tuning or ns-3 execution occurred.",
        "The accepted validation-only binary thresholds were reused only for binary metrics; raw scores determine Top-1 ranking.",
    ]
    if best == "SVM":
        message.append("SVM is the primary proposed model because it has the highest held-out Top-1 OC-selection accuracy and the strongest reliability-overhead tradeoff.")
    else:
        message.append(f"SVM is displayed as the proposed model, but {best} has the highest held-out Top-1 OC-selection accuracy in this fixed comparison.")
    message.append("DTC/RF are regularized lightweight baseline classifiers; no oracle or train results are reported.")
    (OUT / "final_interpretation.txt").write_text("\n".join(message) + "\n")
    print("\nTest ML metrics\n", table_ml.to_string(index=False))
    print("\nTest network metrics\n", table_network.to_string(index=False))
    print("\nComposite score\n", composite.to_string(index=False))


if __name__ == "__main__":
    main()
