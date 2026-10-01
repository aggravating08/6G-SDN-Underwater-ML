#!/usr/bin/env python3
"""Grouped, partner-style binary ``is_oc`` experiment.

Candidate outcomes are used *only* to create the utility-optimal binary
target.  Every classifier input is available before route installation and
before any packet is generated.  Rows from one scenario always stay together
in the frozen 1400/300/300 scenario split.

This program deliberately reuses the completed matched candidate ledger; it
does not invoke ns-3 and does not overwrite the earlier utility-label study.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "results" / "underwater_rebuild" / "current"
UTILITY = CURRENT / "final_paper_style_results" / "utility_label_experiment"
OUT = CURRENT / "partner_style_binary_is_oc_experiment"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = UTILITY / "utility_label_candidate_diagnostics.csv"
PREROUTING = UTILITY / "prerouting_candidate_features.csv"
MANIFEST = CURRENT / "non_cognitive_ml_models" / "split_manifest.json"

# These fields are all available before controller selection.  The final three
# are deterministic transforms of existing pre-routing fields, not outcomes.
FEATURES = [
    "x", "y", "speed", "local_density",
    "source_to_oc_distance", "destination_to_oc_distance",
    "source_covered_by_oc", "destination_covered_by_oc",
    "sensors_in_oc_range", "gateways_in_oc_range",
    "estimated_local_path_exists", "average_link_quality_in_oc_view",
    "estimated_hop_count",
    "mean_endpoint_distance", "endpoint_distance_balance",
    "local_view_sensor_fraction",
]


def finite_number(value: object) -> object:
    """Stable CSV representation for non-finite metric values."""
    return value if np.isfinite(value) else np.nan


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    if not LEDGER.exists() or not PREROUTING.exists():
        raise FileNotFoundError("The completed utility candidate ledger/features are required.")
    ledger = pd.read_csv(LEDGER)
    pre = pd.read_csv(PREROUTING)
    if ledger.duplicated(["scenario_id", "auv_id"]).any():
        raise ValueError("Candidate ledger has duplicate scenario/AUV rows.")
    if not ledger.groupby("scenario_id").size().eq(4).all():
        raise ValueError("Every scenario must have OC0–OC3 candidate outcomes.")
    if not ledger.groupby("scenario_id").is_oc.sum().eq(1).all():
        raise ValueError("Utility target must select exactly one OC per scenario.")

    keep = ["scenario_id", "scenario_seed", "node_count", "auv_id", "is_oc"]
    data = pre.merge(ledger[keep], on=["scenario_id", "scenario_seed", "node_count", "auv_id"],
                     how="inner", validate="one_to_one")
    if len(data) != len(ledger):
        raise ValueError("Missing pre-routing candidate feature rows.")
    data["mean_endpoint_distance"] = (
        data["source_to_oc_distance"] + data["destination_to_oc_distance"]
    ) / 2.0
    data["endpoint_distance_balance"] = np.abs(
        data["source_to_oc_distance"] - data["destination_to_oc_distance"]
    )
    data["local_view_sensor_fraction"] = data["sensors_in_oc_range"] / data["node_count"]
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite pre-routing feature value encountered.")

    with MANIFEST.open() as f:
        manifest = json.load(f)
    splits = {name: [int(x) for x in manifest[name]] for name in ("train", "validation", "test")}
    all_ids = set(data.scenario_id)
    for name, ids in splits.items():
        if not set(ids).issubset(all_ids):
            raise ValueError(f"{name} split has IDs absent from candidate data.")
    if set(splits["train"]) & set(splits["validation"]) or set(splits["train"]) & set(splits["test"]) or set(splits["validation"]) & set(splits["test"]):
        raise ValueError("Split manifest overlaps scenarios.")
    return data, ledger, splits


def split_rows(data: pd.DataFrame, scenario_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = data[data.scenario_id.isin(scenario_ids)].copy()
    return rows[FEATURES], rows["is_oc"]


def top1_rows(model, data: pd.DataFrame) -> pd.DataFrame:
    scored = data[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    scored["probability_is_oc"] = model.predict_proba(data[FEATURES])[:, 1]
    # Stable lower-AUV-ID tie-break only for mathematically identical scores.
    selected = (scored.sort_values(["scenario_id", "probability_is_oc", "auv_id"],
                                   ascending=[True, False, True])
                     .groupby("scenario_id", as_index=False).head(1).copy())
    selected["true_oc"] = (data.loc[data.is_oc.eq(1), ["scenario_id", "auv_id"]]
                             .set_index("scenario_id").loc[selected.scenario_id, "auv_id"].to_numpy())
    selected["correct"] = (selected.auv_id == selected.true_oc).astype(int)
    return selected.rename(columns={"auv_id": "predicted_oc"})


def validation_score(model, validation: pd.DataFrame) -> tuple[float, float]:
    """Primary selection criterion is grouped scenario top-1; F1 breaks ties."""
    selected = top1_rows(model, validation)
    _, _, f1, _ = precision_recall_fscore_support(
        validation.is_oc, model.predict(validation[FEATURES]), average="binary", zero_division=0
    )
    return float(selected.correct.mean()), float(f1)


def choose_best(train: pd.DataFrame, validation: pd.DataFrame, name: str):
    x_train, y_train = split_rows(train, train.scenario_id.unique().tolist())
    search_rows: list[dict[str, object]] = []
    best_key: tuple[float, float] = (-1.0, -1.0)
    best_params: dict[str, object] | None = None

    if name == "SVM":
        grid = itertools.product([1, 2, 5, 10, 20, 50, 100, 200, 500],
                                 [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
                                 [None, "balanced"])
        for c, gamma, weight in grid:
            params = {"C": c, "gamma": gamma, "class_weight": weight}
            model = Pipeline([("scale", StandardScaler()),
                              ("svc", SVC(kernel="rbf", probability=True, random_state=7,
                                          **params))])
            model.fit(x_train, y_train)
            top1, f1 = validation_score(model, validation)
            search_rows.append({"C": c, "gamma": gamma, "class_weight": weight,
                                "validation_top1": top1, "validation_binary_f1": f1})
            if (top1, f1) > best_key:
                best_key, best_params = (top1, f1), params
        final = Pipeline([("scale", StandardScaler()),
                          ("svc", SVC(kernel="rbf", probability=True, random_state=7,
                                      **best_params))])
    elif name == "DTC":
        grid = itertools.product([3, 4, 5, 6, 8, 10, None], [1, 3, 5, 10, 20], [None, "balanced"])
        for depth, leaf, weight in grid:
            params = {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight}
            model = DecisionTreeClassifier(random_state=7, **params)
            model.fit(x_train, y_train)
            top1, f1 = validation_score(model, validation)
            search_rows.append({**params, "validation_top1": top1, "validation_binary_f1": f1})
            if (top1, f1) > best_key:
                best_key, best_params = (top1, f1), params
        final = DecisionTreeClassifier(random_state=7, **best_params)
    elif name == "RF":
        grid = itertools.product([100, 200, 300], [4, 6, 8, 10, None], [3, 5, 10, 15, 20],
                                 ["sqrt", "log2", None], [None, "balanced"])
        for estimators, depth, leaf, max_features, weight in grid:
            params = {"n_estimators": estimators, "max_depth": depth,
                      "min_samples_leaf": leaf, "max_features": max_features,
                      "class_weight": weight}
            model = RandomForestClassifier(random_state=7, n_jobs=-1, **params)
            model.fit(x_train, y_train)
            top1, f1 = validation_score(model, validation)
            search_rows.append({**params, "validation_top1": top1, "validation_binary_f1": f1})
            if (top1, f1) > best_key:
                best_key, best_params = (top1, f1), params
        final = RandomForestClassifier(random_state=7, n_jobs=-1, **best_params)
    else:
        raise ValueError(name)
    pd.DataFrame(search_rows).sort_values(["validation_top1", "validation_binary_f1"], ascending=False).to_csv(
        OUT / f"{name.lower()}_validation_search.csv", index=False)
    return final, best_params, best_key


def candidate_metrics(name: str, model, test: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    labels = model.predict(test[FEATURES])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test.is_oc, labels, average="binary", zero_division=0
    )
    binary = {
        "Model": name,
        "Binary Accuracy": accuracy_score(test.is_oc, labels),
        "Precision": precision, "Recall": recall, "F1": f1,
        "ROC-AUC": roc_auc_score(test.is_oc, probabilities),
        "PR-AUC": average_precision_score(test.is_oc, probabilities),
    }
    selected = top1_rows(model, test)
    true = selected.true_oc
    predicted = selected.predicted_oc
    p, r, macro_f1, _ = precision_recall_fscore_support(
        true, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )
    top = {"Model": name, "Top-1 OC Accuracy": selected.correct.mean(),
           "Macro Precision": p, "Macro Recall": r, "Macro F1": macro_f1}
    selected.insert(0, "Model", name)
    selected.to_csv(OUT / f"{name.lower()}_test_selected_oc_predictions.csv", index=False)
    return {**binary, **{f"Top1 {k}": v for k, v in top.items() if k != "Model"}}, selected


def selected_network(name: str, selected: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    network_cols = ["scenario_id", "auv_id", "PDR", "E2ED_ms", "ROR_total", "ROR_reactive"]
    out = selected.merge(ledger[network_cols], left_on=["scenario_id", "predicted_oc"],
                         right_on=["scenario_id", "auv_id"], how="left", validate="one_to_one")
    if out.PDR.isna().any():
        raise ValueError(f"Network lookup failed for {name}.")
    out["Method"] = f"{name}-selected OC"
    return out


def summarize_network(rows: pd.DataFrame) -> dict[str, object]:
    return {"Method": rows.Method.iat[0], "Mean PDR %": rows.PDR.mean(),
            "Mean E2ED ms": rows.E2ED_ms.mean(skipna=True),
            "Mean ROR total": rows.ROR_total.mean(),
            "Mean ROR reactive": rows.ROR_reactive.mean()}


def main() -> None:
    data, ledger, splits = load_dataset()
    # Persist both the binary data and the exact input column contract.
    data.to_csv(OUT / "binary_utility_is_oc_dataset.csv", index=False)
    (OUT / "feature_contract.json").write_text(json.dumps({
        "features": FEATURES,
        "unavailable_requested_features": [
            "oc_to_source_dest_midpoint_distance", "oc_to_network_centroid_distance",
            "source_destination_distance", "local_view_edge_count", "local_view_density",
            "estimated_min_path_cost", "source_component_reachable",
            "destination_component_reachable", "local_connected_component_size",
            "min_link_quality_on_estimated_path", "common_idle_channel_opportunities",
        ],
        "reason": "These values were not exported in the completed ledger; ns-3 was not rerun."
    }, indent=2))

    train = data[data.scenario_id.isin(splits["train"])].copy()
    validation = data[data.scenario_id.isin(splits["validation"])].copy()
    test = data[data.scenario_id.isin(splits["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise ValueError("Frozen 1400/300/300 grouped split did not yield four rows per scenario.")

    trained: dict[str, object] = {}
    selected_rows: list[pd.DataFrame] = []
    binary_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for name in ("SVM", "DTC", "RF"):
        model_path = OUT / f"{name.lower()}_binary_is_oc_pipeline.joblib"
        search_path = OUT / f"{name.lower()}_validation_search.csv"
        if model_path.exists() and search_path.exists():
            # Resume safely after an interrupted reporting step: the existing
            # model was selected using validation data only and was not tuned
            # against test data.
            print(f"Reusing completed validation-selected {name} model...", flush=True)
            model = load(model_path)
            search = pd.read_csv(search_path).sort_values(
                ["validation_top1", "validation_binary_f1"], ascending=False
            ).iloc[0]
            params = {k: search[k] for k in search.index
                      if k not in {"validation_top1", "validation_binary_f1"} and pd.notna(search[k])}
            val_top1, val_f1 = float(search.validation_top1), float(search.validation_binary_f1)
        else:
            print(f"Tuning {name} using validation scenarios only...", flush=True)
            model, params, (val_top1, val_f1) = choose_best(train, validation, name)
            # The one final fit includes train+validation, then test is evaluated once below.
            fit = pd.concat([train, validation], ignore_index=True)
            model.fit(fit[FEATURES], fit.is_oc)
            dump(model, model_path)
        trained[name] = model
        binary, selected = candidate_metrics(name, model, test)
        binary["Best validation Top-1"] = val_top1
        binary["Best validation binary F1"] = val_f1
        binary["Best parameters"] = json.dumps(params, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        binary_rows.append(binary)
        p, r, f1, _ = precision_recall_fscore_support(
            selected.true_oc, selected.predicted_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
        )
        top_rows.append({"Model": name, "Top-1 OC Accuracy": selected.correct.mean(),
                         "Macro Precision": p, "Macro Recall": r, "Macro F1": f1})
        selected_rows.append(selected_network(name, selected, ledger))

    selected_all = pd.concat(selected_rows, ignore_index=True)
    baseline = ledger[ledger.scenario_id.isin(splits["test"]) & ledger.auv_id.eq(0)].copy()
    baseline["Method"] = "Baseline fixed OC0"
    baseline = baseline.rename(columns={"auv_id": "predicted_oc"})
    network_all = pd.concat([selected_all, baseline], ignore_index=True, sort=False)

    binary_table = pd.DataFrame(binary_rows)
    top_table = pd.DataFrame(top_rows)
    overall = pd.DataFrame([summarize_network(network_all[network_all.Method == method])
                            for method in ["SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline fixed OC0"]])
    by_node = (network_all.groupby(["node_count", "Method"], as_index=False)
               .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                      "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    order = {"SVM-selected OC": 0, "DTC-selected OC": 1, "RF-selected OC": 2, "Baseline fixed OC0": 3}
    by_node["_order"] = by_node.Method.map(order)
    by_node = by_node.sort_values(["node_count", "_order"]).drop(columns="_order")

    binary_table.to_csv(OUT / "candidate_binary_test_metrics.csv", index=False)
    top_table.to_csv(OUT / "scenario_top1_test_metrics.csv", index=False)
    overall.to_csv(OUT / "final_network_metrics.csv", index=False)
    by_node.to_csv(OUT / "per_node_network_metrics.csv", index=False)
    network_all.to_csv(OUT / "model_selected_oc_rows_with_network_metrics.csv", index=False)

    print("\nCandidate-level binary metrics")
    print(binary_table.to_string(index=False))
    print("\nScenario-level top-1 metrics")
    print(top_table.to_string(index=False))
    print("\nNetwork metrics: test scenarios only")
    print(overall.to_string(index=False))
    print("\nPer-node network metrics: test scenarios only")
    print(by_node.to_string(index=False))
    print(f"\nSaved separate experiment to {OUT}")


if __name__ == "__main__":
    main()
