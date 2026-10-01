#!/usr/bin/env python3
"""Aligned ML and network evaluation using one balanced-distance scenario ledger.

This program never invokes ns-3: the completed v8 ledger already has 2,000
scenarios, exactly four matched OC candidates per scenario, and their actual
network outcomes.  It creates a fresh, self-contained reporting directory
whose labels, feature rows, grouped split, model predictions, and selected OC
network metrics all join on the same scenario ID, seed, node count, and OC.
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


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "results" / "underwater_rebuild" / "current"
SOURCE = CURRENT / "v8_paper_trend_balanced_pdr"
DATASET = SOURCE / "balanced_utility_labeled_dataset.csv"
LEDGER = SOURCE / "balanced_matched_four_oc_outcomes.csv"
SOURCE_SPLIT = SOURCE / "split_manifest.json"
OUT = CURRENT / "final_aligned_paper_faithful_real_metrics"

# All values are available before the selected OC is installed.  In
# particular, no PDR/E2ED/ROR or outcome/counter is an input feature.
FEATURES = [
    "x", "y", "local_density", "speed",
    "source_to_oc_distance", "destination_to_oc_distance",
    "sensors_in_oc_range", "gateways_in_oc_range",
    "source_covered_by_oc", "destination_covered_by_oc",
    "estimated_local_path_exists", "estimated_hop_count",
    "mean_endpoint_distance", "endpoint_distance_balance",
    "local_view_sensor_fraction",
]


def raw_score(model, rows: pd.DataFrame, name: str) -> np.ndarray:
    if name == "SVM":
        return np.asarray(model.decision_function(rows[FEATURES]), dtype=float)
    return np.asarray(model.predict_proba(rows[FEATURES])[:, 1], dtype=float)


def threshold_from_validation(rows: pd.DataFrame, scores: np.ndarray) -> float:
    """Pick candidate-level threshold by validation binary F1 only.

    The threshold is solely for binary precision/recall/F1 reporting.  Top-1
    OC selection always ranks unthresholded scores inside each scenario.
    """
    unique = np.unique(scores)
    if unique.size == 1:
        return float(unique[0])
    candidates = np.r_[unique[0] - 1.0, (unique[:-1] + unique[1:]) / 2.0, unique[-1] + 1.0]

    def key(t: float) -> tuple[float, float, float]:
        predicted = (scores >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(rows.is_oc, predicted, average="binary", zero_division=0)
        return float(f1), float(accuracy_score(rows.is_oc, predicted)), -float(t)

    return float(max(candidates, key=key))


def binary_stats(rows: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(rows.is_oc, predicted, average="binary", zero_division=0)
    return {"Binary accuracy": 100.0 * float(accuracy_score(rows.is_oc, predicted)),
            "Precision": float(precision), "Recall": float(recall), "Binary F1": float(f1)}


def select_oc(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    selected = rows[["scenario_id", "scenario_seed", "node_count", "auv_id", "is_oc"]].copy()
    selected["raw_score"] = scores
    # The lower candidate ID resolves an exact numerical tie deterministically.
    selected = (selected.sort_values(["scenario_id", "raw_score", "auv_id"], ascending=[True, False, True])
                        .groupby("scenario_id", as_index=False).head(1).copy())
    truth = rows[rows.is_oc.eq(1)].set_index("scenario_id").auv_id
    selected["true_best_oc"] = truth.loc[selected.scenario_id].to_numpy()
    selected["correct_top1"] = (selected.auv_id == selected.true_best_oc).astype(int)
    return selected.rename(columns={"auv_id": "selected_oc"})


def top1_stats(selected: pd.DataFrame) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        selected.true_best_oc, selected.selected_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )
    return {"Top-1 OC accuracy": 100.0 * float(selected.correct_top1.mean()),
            "Top-1 macro F1": float(f1),
            "_macro_precision": float(precision), "_macro_recall": float(recall)}


def make_model(name: str, params: dict):
    if name == "SVM":
        return Pipeline([("scaler", StandardScaler()),
                         ("svc", SVC(kernel="rbf", probability=False, random_state=42, **params))])
    if name == "DTC":
        return DecisionTreeClassifier(criterion="gini", random_state=42, class_weight=None, **params)
    return RandomForestClassifier(criterion="gini", random_state=42, class_weight=None,
                                  n_jobs=-1, bootstrap=True, **params)


def parameter_grid(name: str):
    if name == "SVM":
        for c, gamma in itertools.product([1, 5, 10, 20, 50, 100], [.001, .005, .01, .02, .05]):
            yield {"C": c, "gamma": gamma, "class_weight": None}
    elif name == "DTC":
        for depth, split, leaf, max_features in itertools.product(
            [3, 4, 5, 6], [10, 20, 30, 40], [5, 10, 15, 20, 30], [2, 3, None]
        ):
            yield {"max_depth": depth, "min_samples_split": split,
                   "min_samples_leaf": leaf, "max_features": max_features}
    elif name == "RF":
        # Deliberately finite/pruned ranges: controller-side baseline models
        # remain reasonable, lightweight alternatives without using test data.
        for trees, depth, split, leaf, max_features, sample in itertools.product(
            [50, 100], [3, 4, 5, 6], [10, 20, 30], [5, 10, 15, 20], ["sqrt", 2], [.5, .75]
        ):
            yield {"n_estimators": trees, "max_depth": depth, "min_samples_split": split,
                   "min_samples_leaf": leaf, "max_features": max_features, "max_samples": sample}
    else:
        raise ValueError(name)


def select_on_validation(name: str, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, float, pd.DataFrame]:
    records, best = [], None
    for params in parameter_grid(name):
        model = make_model(name, params)
        model.fit(train[FEATURES], train.is_oc)
        scores = raw_score(model, validation, name)
        threshold = threshold_from_validation(validation, scores)
        binary = binary_stats(validation, scores, threshold)
        top = top1_stats(select_oc(validation, scores))
        # Primary criterion is the actual four-way task: scenario Top-1.  F1
        # and candidate accuracy make genuine validation ties deterministic.
        key = (top["Top-1 OC accuracy"], binary["Binary F1"], binary["Binary accuracy"])
        record = {**params, "Binary threshold": threshold, **binary,
                  "Top-1 OC accuracy": top["Top-1 OC accuracy"], "Top-1 macro F1": top["Top-1 macro F1"]}
        records.append(record)
        if best is None or key > best[0]:
            best = (key, params, threshold)
    search = pd.DataFrame(records).sort_values(
        ["Top-1 OC accuracy", "Binary F1", "Binary accuracy"], ascending=False
    )
    if best is None:
        raise RuntimeError(f"No validation candidate for {name}")
    return best[1], best[2], search


def map_actual_metrics(selected: pd.DataFrame, ledger: pd.DataFrame, model_label: str) -> pd.DataFrame:
    selected_keys = ["scenario_id", "scenario_seed", "node_count", "selected_oc"]
    metrics = ["PDR", "E2ED_ms", "ROR_total", "generated_packets", "delivered_packets",
               "control_transmissions", "data_hops", "OC_data_hops", "architecture_violations",
               "mean_source_destination_distance_m", "mean_delivered_hop_count", "mean_delivered_path_length_m"]
    mapped = selected.merge(ledger[selected_keys + metrics], on=selected_keys, how="left", validate="one_to_one")
    if mapped[["PDR", "generated_packets", "control_transmissions", "data_hops"]].isna().any().any():
        raise RuntimeError(f"Missing candidate metric mapping for {model_label}")
    mapped["Model"] = model_label
    return mapped


def metric_summary(rows: pd.DataFrame) -> dict[str, float]:
    rx = float(rows.delivered_packets.sum())
    tx = float(rows.generated_packets.sum())
    control = float(rows.control_transmissions.sum())
    data_tx = float(rows.data_hops.sum())
    # PDR, E2ED and ROR are pooled over the selected test candidates.  This
    # directly implements their packet-level definitions rather than averaging
    # ratios with different denominators.
    e2ed = float((rows.E2ED_ms.fillna(0.0) * rows.delivered_packets).sum() / rx) if rx else np.nan
    ror = control / (control + data_tx) if control + data_tx else np.nan
    return {"Mean PDR (%)": 100.0 * rx / tx if tx else np.nan,
            "Mean E2ED (ms)": e2ed, "Mean ROR": ror}


def audit_summary(rows: pd.DataFrame, nodes: int, model: str) -> dict[str, float]:
    rx = float(rows.delivered_packets.sum())
    delivered = rows.delivered_packets.gt(0)
    # Weight hop/path values by actual delivered packet count.  A failed flow
    # contributes no delivered path to either diagnostic.
    weighted_hops = float((rows.mean_delivered_hop_count.fillna(0.0) * rows.delivered_packets).sum() / rx) if rx else np.nan
    weighted_path = float((rows.mean_delivered_path_length_m.fillna(0.0) * rows.delivered_packets).sum() / rx) if rx else np.nan
    return {"Nodes": nodes, "Model": model,
            "TX data packets": int(rows.generated_packets.sum()), "RX data packets": int(rx),
            "Control packets": float(rows.control_transmissions.sum()),
            "Total packets": float(rows.control_transmissions.sum() + rows.data_hops.sum()),
            "Delivered scenarios": int(delivered.sum()), "Failed scenarios": int((~delivered).sum()),
            "Mean selected OC": float(rows.selected_oc.mean()),
            "Mean S-D distance": float(rows.mean_source_destination_distance_m.mean()),
            "Mean delivered hops": weighted_hops, "Mean path length": weighted_path}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATASET)
    ledger = pd.read_csv(LEDGER)
    with SOURCE_SPLIT.open() as f:
        split = json.load(f)
    (OUT / "split_manifest.json").write_text(json.dumps(split, indent=2))

    if len(data) != 8000 or len(ledger) != 8000:
        raise RuntimeError("Expected 2,000 scenarios and 8,000 candidate rows in both aligned inputs.")
    if not data.groupby("scenario_id").size().eq(4).all() or not data.groupby("scenario_id").is_oc.sum().eq(1).all():
        raise RuntimeError("Dataset must have exactly four rows and one label per scenario.")
    keys = ["scenario_id", "scenario_seed", "node_count", "auv_id"]
    if data.duplicated(keys).any() or ledger.duplicated(["scenario_id", "scenario_seed", "node_count", "selected_oc"]).any():
        raise RuntimeError("Duplicate aligned candidate keys.")
    aligned_hashes = data.merge(
        ledger.rename(columns={"selected_oc": "auv_id"})[keys + ["topology_hash"]],
        on=keys, suffixes=("", "_ledger"), validate="one_to_one"
    )
    if not aligned_hashes.topology_hash.eq(aligned_hashes.topology_hash_ledger).all():
        raise RuntimeError("Feature and candidate ledgers have mismatched topology hashes.")
    if not ledger.generated_packets.eq(200).all() or not ledger.OC_data_hops.eq(0).all() or not ledger.architecture_violations.eq(0).all():
        raise RuntimeError("Candidate ledger violates packet-count or architecture invariants.")
    if not ledger.mean_source_destination_distance_m.between(300.0 - 1e-6, 450.0 + 1e-6).all():
        raise RuntimeError("The 300--450 m source-destination rule failed in the candidate ledger.")

    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise RuntimeError("Grouped split must be 1400/300/300 scenarios.")
    test_counts = test.drop_duplicates("scenario_id").node_count.value_counts().to_dict()
    if test_counts != {25: 75, 50: 75, 75: 75, 100: 75}:
        raise RuntimeError(f"Expected 75 test scenarios per node count, got {test_counts}")

    # Hyperparameters and binary thresholds are selected without ever reading
    # test outputs.  After this loop, settings are locked and test is read once.
    chosen, validation_rows = {}, []
    for name in ("SVM", "DTC", "RF"):
        params, threshold, search = select_on_validation(name, train, validation)
        search.to_csv(OUT / f"{name.lower()}_validation_search.csv", index=False)
        top = search.iloc[0]
        chosen[name] = (params, threshold)
        validation_rows.append({"Model": name, "Selected hyperparameters": json.dumps(params),
                                "Binary threshold": threshold, "Validation Binary accuracy": top["Binary accuracy"],
                                "Validation Top-1 OC accuracy": top["Top-1 OC accuracy"],
                                "Validation Binary F1": top["Binary F1"]})

    fitted = pd.concat([train, validation], ignore_index=True)
    ml_rows, selected_outputs, mapped_outputs, audit_rows = [], [], [], []
    for name in ("SVM", "DTC", "RF"):
        params, threshold = chosen[name]
        model = make_model(name, params)
        model.fit(fitted[FEATURES], fitted.is_oc)
        dump(model, OUT / f"{name.lower()}_pipeline.joblib")
        scores = raw_score(model, test, name)
        binary = binary_stats(test, scores, threshold)
        selected = select_oc(test, scores)
        top = top1_stats(selected)
        ml_rows.append({"Model": name, **binary, "Top-1 OC accuracy": top["Top-1 OC accuracy"],
                        "Top-1 macro F1": top["Top-1 macro F1"]})
        mapped = map_actual_metrics(selected, ledger, f"{name}-selected OC")
        mapped_outputs.append(mapped)
        selected_outputs.append(mapped)

    # Fixed OC0 has the exact same test scenarios and candidate ledger.
    baseline = test[test.auv_id.eq(0)][["scenario_id", "scenario_seed", "node_count", "auv_id", "is_oc"]].copy()
    baseline = baseline.rename(columns={"auv_id": "selected_oc", "is_oc": "_label"})
    baseline["true_best_oc"] = test[test.is_oc.eq(1)].set_index("scenario_id").auv_id.loc[baseline.scenario_id].to_numpy()
    baseline["correct_top1"] = (baseline.selected_oc == baseline.true_best_oc).astype(int)
    baseline["raw_score"] = np.nan
    baseline = map_actual_metrics(baseline, ledger, "Baseline OC0")
    selected_outputs.append(baseline)

    ml_table = pd.DataFrame(ml_rows)
    summary_rows, by_node_rows = [], []
    order = ["SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0"]
    combined = pd.concat(selected_outputs, ignore_index=True)
    for method in order:
        rows = combined[combined.Model.eq(method)].copy()
        summary_rows.append({"Model": method, **metric_summary(rows)})
        for nodes in (25, 50, 75, 100):
            part = rows[rows.node_count.eq(nodes)]
            by_node_rows.append({"Nodes": nodes, "Model": method, **metric_summary(part)})
            audit_rows.append(audit_summary(part, nodes, method))

    final_selected = combined.rename(columns={"node_count": "nodes", "E2ED_ms": "E2ED", "ROR_total": "ROR",
                                               "mean_source_destination_distance_m": "source_destination_distance"})
    final_selected = final_selected[["scenario_id", "nodes", "Model", "selected_oc", "true_best_oc", "correct_top1",
                                     "PDR", "E2ED", "ROR", "source_destination_distance"]]
    final_selected.to_csv(OUT / "selected_oc_per_test_scenario.csv", index=False)
    ml_table.to_csv(OUT / "final_ml_accuracy_comparison.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "final_network_metrics_overall.csv", index=False)
    pd.DataFrame(by_node_rows).to_csv(OUT / "final_network_metrics_by_node.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "final_network_audit_table.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_selected_hyperparameters.csv", index=False)

    summary = [
        "FINAL ALIGNED PAPER-FAITHFUL EVALUATION",
        "",
        "All ML features, utility-best labels, candidate outcomes, and selected-OC PDR/E2ED/ROR metrics come from the same 2,000-scenario, 8,000-row balanced-distance ledger.",
        "The same saved grouped split is used for ML and network evaluation: 1,400 train, 300 validation, and 300 test scenarios (75 test scenarios per node count).",
        "Every candidate run generated 200 packets, had zero OC payload hops and zero architecture violations. Every recorded source-destination distance is between 300 and 450 m.",
        "Hyperparameters and binary thresholds were selected only with training/validation data. The held-out test split was not used for selection.",
        "PDR is delivered data packets divided by generated data packets. E2ED is the delivered-packet weighted source-generation-to-destination-arrival delay. ROR is pooled control packet transmissions divided by control plus realized data-packet transmissions.",
        "The SVM, DTC, and RF each rank the four candidates per held-out scenario, and each selected row is joined to the exact matching candidate network outcome. Baseline OC0 selects candidate 0 for those same scenarios.",
    ]
    (OUT / "final_results_summary.txt").write_text("\n".join(summary) + "\n")
    (OUT / "experiment_configuration.json").write_text(json.dumps({
        "source_dataset": str(DATASET), "source_candidate_ledger": str(LEDGER),
        "source_split": str(SOURCE_SPLIT), "traffic_rule": "source-destination distance 300--450 m",
        "scenario_count": 2000, "candidate_rows": 8000, "packets_per_candidate_run": 200,
        "test_scenarios": 300, "test_scenarios_per_node_count": 75,
        "ror_definition": "control packet transmissions/(control packet transmissions + realized data-packet transmissions)",
        "features": FEATURES, "test_used_for_hyperparameter_selection": False,
    }, indent=2))

    print("FINAL ML ACCURACY")
    print(ml_table.to_string(index=False))
    print("\nFINAL NETWORK OVERALL")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nFINAL NETWORK BY NODE")
    print(pd.DataFrame(by_node_rows).to_string(index=False))


if __name__ == "__main__":
    main()
