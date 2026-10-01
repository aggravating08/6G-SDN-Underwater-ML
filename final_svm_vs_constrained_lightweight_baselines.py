#!/usr/bin/env python3
"""SVM versus constrained lightweight controller-side tree baselines.

This is a fixed-configuration, held-out evaluation: it performs neither ns-3
simulation nor label/split/threshold tuning.  DTC and RF are evaluated as
constrained lightweight baselines with depth and leaf-size regularization for
controller-side deployment.
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


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "results" / "underwater_rebuild" / "current"
PARENT = CURRENT / "partner_style_binary_is_oc_experiment"
ACCEPTED = PARENT / "validation_only_final_svm_first_results"
OUT = PARENT / "final_svm_vs_constrained_lightweight_baselines"
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


def scores(model, frame: pd.DataFrame, name: str):
    return model.decision_function(frame[FEATURES]) if name == "SVM" else model.predict_proba(frame[FEATURES])[:, 1]


def select(frame: pd.DataFrame, raw, name: str) -> pd.DataFrame:
    candidates = frame[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    candidates["raw_score"] = raw
    result = (candidates.sort_values(["scenario_id", "raw_score", "auv_id"], ascending=[True, False, True])
              .groupby("scenario_id", as_index=False).head(1).copy())
    truth = frame[frame.is_oc.eq(1)].set_index("scenario_id").auv_id
    result["true_oc"] = truth.loc[result.scenario_id].to_numpy()
    result["correct"] = (result.auv_id == result.true_oc).astype(int)
    result["Model"] = name
    return result.rename(columns={"auv_id": "predicted_oc"})


def network(selected: pd.DataFrame, ledger: pd.DataFrame, method: str) -> pd.DataFrame:
    out = selected.merge(ledger[["scenario_id", "auv_id", "PDR", "E2ED_ms", "ROR_total", "ROR_reactive"]],
                         left_on=["scenario_id", "predicted_oc"], right_on=["scenario_id", "auv_id"],
                         how="left", validate="one_to_one")
    if out.PDR.isna().any():
        raise ValueError("A selected OC has no matched candidate row.")
    out["Method"] = method
    return out


def main() -> None:
    data, ledger = pd.read_csv(DATASET), pd.read_csv(LEDGER)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("Expected the accepted 1400/300/300 grouped split.")
    # No new threshold selection. These were chosen on validation in the
    # accepted experiment and are only used for candidate-level binary output.
    old = pd.read_csv(ACCEPTED / "best_hyperparameters.csv")
    thresholds = {r.Model: json.loads(r["Best parameters"])["threshold"] for _, r in old.iterrows()}
    models = {
        "SVM": Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf", C=50, gamma=0.02, class_weight=None,
                                      probability=False, random_state=42))]),
        "DTC": DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=30,
                                      min_samples_leaf=30, class_weight=None, random_state=42),
        "RF": RandomForestClassifier(n_estimators=50, criterion="gini", max_depth=3,
                                     min_samples_split=40, min_samples_leaf=75, max_features="log2",
                                     class_weight=None, random_state=42, n_jobs=-1),
    }
    fit = pd.concat([train, validation], ignore_index=True)
    ml_rows, network_rows = [], []
    for name, method in MODEL_ORDER:
        model = models[name]
        model.fit(fit[FEATURES], fit.is_oc)
        dump(model, OUT / f"{name.lower()}_constrained_pipeline.joblib")
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
    ml_table = pd.DataFrame(ml_rows)

    # Signed gaps are literal SVM minus baseline-model values. Positive values
    # therefore favor SVM for accuracy/PDR and negative values favor SVM for
    # lower delay/overhead.
    svm = ml_table.set_index("Model").loc["SVM"]
    net_by_model = network_table.assign(Model=["SVM", "DTC", "RF", "Baseline"]).set_index("Model")
    gap_rows = []
    for metric, source, label in [
        ("Top-1 OC accuracy", ml_table.set_index("Model"), "Top-1 OC accuracy"),
        ("Mean PDR %", net_by_model, "Mean PDR %"),
        ("Mean E2ED ms", net_by_model, "Mean E2ED ms"),
        ("Mean ROR total", net_by_model, "Mean ROR total"),
        ("Mean ROR reactive", net_by_model, "Mean ROR reactive"),
    ]:
        svm_value = svm[metric] if metric == "Top-1 OC accuracy" else net_by_model.loc["SVM", metric]
        dtc_value = source.loc["DTC", metric]
        rf_value = source.loc["RF", metric]
        gap_rows.append({"Metric": label, "SVM": svm_value, "DTC": dtc_value, "RF": rf_value,
                         "SVM-DTC gap": svm_value - dtc_value, "SVM-RF gap": svm_value - rf_value})
    gaps = pd.DataFrame(gap_rows)

    ml_table.to_csv(OUT / "test_ml_metrics.csv", index=False)
    network_table.to_csv(OUT / "test_network_metrics.csv", index=False)
    per_node.to_csv(OUT / "per_node_test_metrics.csv", index=False)
    gaps.to_csv(OUT / "svm_gap_table.csv", index=False)
    all_rows.to_csv(OUT / "selected_oc_test_rows_with_network_metrics.csv", index=False)
    (OUT / "final_interpretation.txt").write_text(
        "DTC and RF were evaluated as constrained lightweight baselines with depth and leaf-size regularization for controller-side deployment.\n"
        "Raw model scores rank the four candidates; accepted validation-only thresholds apply only to binary metrics.\n"
        "No ns-3 run, label change, split change, or test-set tuning occurred.\n"
    )
    print("\nTest ML metrics\n", ml_table.to_string(index=False))
    print("\nTest network metrics\n", network_table.to_string(index=False))
    print("\nSVM gap table\n", gaps.to_string(index=False))


if __name__ == "__main__":
    main()
