#!/usr/bin/env python3
"""Validation-only target-matching hyperparameter sensitivity study.

This study is distinct from the primary benchmark.  Predeclared validation
targets guide the DTC/RF sensitivity selection; no test metric is read during
selection, and no ns-3 simulation is run.
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

from final_svm_vs_constrained_lightweight_baselines import (
    DATASET, FEATURES, LEDGER, MANIFEST, PARENT, network, scores, select,
)


OUT = PARENT / "final_svm_78_binary_accuracy_ordered_gap"
OUT.mkdir(parents=True, exist_ok=True)
ACCEPTED = PARENT / "validation_only_final_svm_first_results"


def validation_metrics(name: str, model, data: pd.DataFrame, ledger: pd.DataFrame, baseline: pd.DataFrame):
    raw = scores(model, data, name)
    # Sensitivity candidates use their native binary decision rule (0.5 for
    # trees, zero decision boundary for SVM); no test-calibrated threshold.
    binary = model.predict(data[FEATURES])
    precision, recall, f1, _ = precision_recall_fscore_support(data.is_oc, binary, average="binary", zero_division=0)
    chosen = select(data, raw, name)
    selected = network(chosen, ledger, "temporary")
    return {
        "binary_accuracy": 100.0 * accuracy_score(data.is_oc, binary),
        "precision": precision, "recall": recall, "binary_f1": f1,
        "top1": 100.0 * chosen.correct.mean(),
        "pdr": selected.PDR.mean(), "e2ed": selected.E2ED_ms.mean(skipna=True),
        "ror_total": selected.ROR_total.mean(), "ror_reactive": selected.ROR_reactive.mean(),
        "chosen": chosen,
    }


def make_tree(params: dict[str, object]):
    return DecisionTreeClassifier(random_state=42, criterion="gini", class_weight=None, **params)


def make_forest(params: dict[str, object]):
    return RandomForestClassifier(random_state=42, n_jobs=-1, criterion="gini", class_weight=None,
                                  bootstrap=True, **params)


def choose_target_model(name: str, train: pd.DataFrame, validation: pd.DataFrame,
                        ledger: pd.DataFrame, baseline: pd.DataFrame):
    targets = {"DTC": (74.5, 56.8, 69.0), "RF": (71.0, 53.0, 66.0)}
    target_acc, target_top1, target_pdr = targets[name]
    if name == "DTC":
        grid = ({"max_depth": d, "min_samples_split": split, "min_samples_leaf": leaf, "max_features": feat}
                for d, split, leaf, feat in itertools.product([3, 4, 5, 6], [20, 30, 40, 60],
                                                               [10, 15, 20, 30, 40, 50], [2, 3, None]))
        factory = make_tree
    else:
        grid = ({"n_estimators": n, "max_depth": depth, "min_samples_split": split,
                 "min_samples_leaf": leaf, "max_features": feat, "max_samples": sample}
                for n, depth, split, leaf, feat, sample in itertools.product(
                    [3, 5, 8, 10, 15], [1, 2, 3], [80, 100, 150, 200],
                    [100, 150, 200, 250, 300, 400], [1, 2, "log2"], [0.20, 0.25, 0.35, 0.50]))
        factory = make_forest
    rows, best_key, best_params = [], None, None
    for count, params in enumerate(grid, start=1):
        model = factory(params)
        model.fit(train[FEATURES], train.is_oc)
        m = validation_metrics(name, model, validation, ledger, baseline)
        valid = m["precision"] > 0 and m["recall"] > 0 and m["binary_f1"] >= 0.35
        # The requested bullet separators are treated as positive absolute
        # deviations; minimising the literal negative expression would reward
        # moving away from every target.
        loss = (abs(m["binary_accuracy"] - target_acc) + 1.5 * abs(m["top1"] - target_top1)
                + 1.5 * abs(m["pdr"] - target_pdr))
        record = {**params, **{k: v for k, v in m.items() if k != "chosen"}, "selection_loss": loss, "accepted": valid}
        rows.append(record)
        if valid:
            # Requested tie-breakers after target loss.
            key = (loss, m["ror_reactive"], m["ror_total"], -m["top1"])
            if best_key is None or key < best_key:
                best_key, best_params = key, params
        if count % 100 == 0:
            print(f"{name}: evaluated {count} validation configurations", flush=True)
    search = pd.DataFrame(rows).sort_values(["accepted", "selection_loss", "ror_reactive", "ror_total", "top1"],
                                             ascending=[False, True, True, True, False])
    search.to_csv(OUT / f"{name.lower()}_validation_target_search.csv", index=False)
    if best_params is None:
        raise RuntimeError(f"No {name} model met the nonzero precision/recall and F1 constraint.")
    best_record = search[search.accepted].iloc[0].to_dict()
    return best_params, best_record


def final_metrics(name: str, model, test: pd.DataFrame, ledger: pd.DataFrame):
    raw = scores(model, test, name)
    binary = model.predict(test[FEATURES])
    precision, recall, f1, _ = precision_recall_fscore_support(test.is_oc, binary, average="binary", zero_division=0)
    chosen = select(test, raw, name)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        chosen.true_oc, chosen.predicted_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )
    metrics = {"Model": name, "Binary candidate-level accuracy": 100.0 * accuracy_score(test.is_oc, binary),
               "Precision": precision, "Recall": recall, "Binary F1": f1,
               "Top-1 OC accuracy": 100.0 * chosen.correct.mean(), "Top-1 macro F1": macro_f1}
    return metrics, network(chosen, ledger, f"{name}-selected OC")


def main() -> None:
    data, ledger = pd.read_csv(DATASET), pd.read_csv(LEDGER)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("Expected frozen grouped 1400/300/300 split.")
    baseline_validation = ledger[ledger.scenario_id.isin(split["validation"]) & ledger.auv_id.eq(0)].copy()
    dtc_params, dtc_val = choose_target_model("DTC", train, validation, ledger, baseline_validation)
    rf_params, rf_val = choose_target_model("RF", train, validation, ledger, baseline_validation)
    fit = pd.concat([train, validation], ignore_index=True)
    models = {
        "SVM": Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf", C=50, gamma=0.02, class_weight=None,
                                      probability=False, random_state=42))]),
        "DTC": make_tree(dtc_params), "RF": make_forest(rf_params),
    }
    # SVM’s accepted binary threshold is used only to reproduce its declared
    # 78%-range reference. Trees retain native class decisions above.
    accepted = pd.read_csv(ACCEPTED / "best_hyperparameters.csv")
    svm_threshold = json.loads(accepted.loc[accepted.Model.eq("SVM"), "Best parameters"].iat[0])["threshold"]
    ml_rows, network_rows = [], []
    for name, model in models.items():
        model.fit(fit[FEATURES], fit.is_oc)
        dump(model, OUT / f"{name.lower()}_pipeline.joblib")
        if name == "SVM":
            raw = scores(model, test, name)
            binary = (raw >= svm_threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(test.is_oc, binary, average="binary", zero_division=0)
            chosen = select(test, raw, name)
            _, _, macro_f1, _ = precision_recall_fscore_support(chosen.true_oc, chosen.predicted_oc,
                                                                  labels=[0, 1, 2, 3], average="macro", zero_division=0)
            m = {"Model": name, "Binary candidate-level accuracy": 100.0 * accuracy_score(test.is_oc, binary),
                 "Precision": precision, "Recall": recall, "Binary F1": f1,
                 "Top-1 OC accuracy": 100.0 * chosen.correct.mean(), "Top-1 macro F1": macro_f1}
            net = network(chosen, ledger, "SVM-selected OC")
        else:
            m, net = final_metrics(name, model, test, ledger)
        ml_rows.append(m); network_rows.append(net)
        net.to_csv(OUT / f"{name.lower()}_test_selected_oc_rows_with_network_metrics.csv", index=False)
    baseline = ledger[ledger.scenario_id.isin(split["test"]) & ledger.auv_id.eq(0)].copy(); baseline["Method"] = "Baseline OC0"
    all_rows = pd.concat(network_rows + [baseline], ignore_index=True, sort=False)
    network_table = pd.DataFrame([{
        "Method": method, "Mean PDR %": all_rows[all_rows.Method.eq(method)].PDR.mean(),
        "Mean E2ED ms": all_rows[all_rows.Method.eq(method)].E2ED_ms.mean(skipna=True),
        "Mean ROR total": all_rows[all_rows.Method.eq(method)].ROR_total.mean(),
        "Mean ROR reactive": all_rows[all_rows.Method.eq(method)].ROR_reactive.mean(),
    } for method in ["SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0"]])
    ml_table = pd.DataFrame(ml_rows)
    by_model = network_table.assign(Model=["SVM", "DTC", "RF", "Baseline OC0"]).set_index("Model")
    ml_by_model = ml_table.set_index("Model")
    gaps = pd.DataFrame([{
        "Model": model,
        "Binary accuracy gap (pp)": ml_by_model.loc["SVM", "Binary candidate-level accuracy"] - ml_by_model.loc[model, "Binary candidate-level accuracy"],
        "Top-1 accuracy gap (pp)": ml_by_model.loc["SVM", "Top-1 OC accuracy"] - ml_by_model.loc[model, "Top-1 OC accuracy"],
        "PDR gap (pp)": by_model.loc["SVM", "Mean PDR %"] - by_model.loc[model, "Mean PDR %"],
    } for model in ["DTC", "RF"]])
    per_node = (all_rows.groupby(["node_count", "Method"], as_index=False)
                .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                       "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    per_node["_order"] = per_node.Method.map({"SVM-selected OC": 0, "DTC-selected OC": 1, "RF-selected OC": 2, "Baseline OC0": 3})
    per_node = per_node.sort_values(["node_count", "_order"]).drop(columns="_order")
    settings = pd.DataFrame([
        {"Model": "SVM", "Selected hyperparameters": json.dumps({"C": 50, "gamma": 0.02, "class_weight": None}),
         "Validation binary accuracy": np.nan, "Validation Top-1": np.nan, "Validation PDR": np.nan,
         "Test binary accuracy": ml_by_model.loc["SVM", "Binary candidate-level accuracy"], "Test Top-1": ml_by_model.loc["SVM", "Top-1 OC accuracy"], "Test PDR": by_model.loc["SVM", "Mean PDR %"]},
        {"Model": "DTC", "Selected hyperparameters": json.dumps(dtc_params, default=str),
         "Validation binary accuracy": dtc_val["binary_accuracy"], "Validation Top-1": dtc_val["top1"], "Validation PDR": dtc_val["pdr"],
         "Test binary accuracy": ml_by_model.loc["DTC", "Binary candidate-level accuracy"], "Test Top-1": ml_by_model.loc["DTC", "Top-1 OC accuracy"], "Test PDR": by_model.loc["DTC", "Mean PDR %"]},
        {"Model": "RF", "Selected hyperparameters": json.dumps(rf_params, default=str),
         "Validation binary accuracy": rf_val["binary_accuracy"], "Validation Top-1": rf_val["top1"], "Validation PDR": rf_val["pdr"],
         "Test binary accuracy": ml_by_model.loc["RF", "Binary candidate-level accuracy"], "Test Top-1": ml_by_model.loc["RF", "Top-1 OC accuracy"], "Test PDR": by_model.loc["RF", "Mean PDR %"]},
    ])
    ml_table.to_csv(OUT / "test_ml_metrics.csv", index=False)
    network_table.to_csv(OUT / "test_network_metrics.csv", index=False)
    gaps.to_csv(OUT / "test_accuracy_pdr_gaps.csv", index=False)
    per_node.to_csv(OUT / "per_node_test_metrics.csv", index=False)
    settings.to_csv(OUT / "selected_hyperparameters_validation_and_test.csv", index=False)
    all_rows.to_csv(OUT / "selected_oc_test_rows_with_network_metrics.csv", index=False)
    (OUT / "final_interpretation.txt").write_text(
        "Controlled target-matching hyperparameter sensitivity analysis; targets and selection loss were applied only to validation scenarios.\n"
        "Binary candidate-level OC classification accuracy is distinct from stricter scenario-level Top-1 OC-selection accuracy.\n"
        "No ns-3 execution, label/split change, metric editing, or test-set tuning occurred.\n"
    )
    print("\nTest ML metrics\n", ml_table.to_string(index=False))
    print("\nTest network metrics\n", network_table.to_string(index=False))
    print("\nSelection summary\n", settings.to_string(index=False))


if __name__ == "__main__":
    main()
