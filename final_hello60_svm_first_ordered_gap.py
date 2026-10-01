#!/usr/bin/env python3
"""Controlled validation-target sensitivity under the HELLO=60 s protocol.

This is intentionally a separate, outcome-targeted *controlled sensitivity*
experiment.  The DTC/RF configurations are selected only on validation rows;
the held-out test split is read only after selection.  SVM, labels, and the
grouped 1400/300/300 scenario split remain fixed.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
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
CURRENT = ROOT / "results/underwater_rebuild/current"
PARENT = CURRENT / "partner_style_binary_is_oc_experiment"
DATASET = PARENT / "binary_utility_is_oc_dataset.csv"
MANIFEST = CURRENT / "non_cognitive_ml_models/split_manifest.json"
PREVIOUS = CURRENT / "hello60_reactive_suppression_ror_below_04/raw_unique_candidate_runs.csv"
OUT = CURRENT / "final_hello60_svm_first_ordered_gap_v2"
RUNS = OUT / "candidate_runs"
PREEXISTING_RUNS = CURRENT / "final_hello60_svm_first_ordered_gap/candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
FEATURES = [
    "x", "y", "local_density", "speed", "source_to_oc_distance", "destination_to_oc_distance",
    "sensors_in_oc_range", "gateways_in_oc_range", "source_covered_by_oc", "destination_covered_by_oc",
    "estimated_local_path_exists", "estimated_hop_count", "mean_endpoint_distance",
    "endpoint_distance_balance", "local_view_sensor_fraction",
]
METHODS = [("SVM", "SVM-selected OC"), ("DTC", "DTC-selected OC"), ("RF", "RF-selected OC")]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def candidate_key(row: dict | pd.Series) -> tuple[int, int, int, int]:
    return (int(row["node_count"]), int(row["scenario_id"]), int(row["scenario_seed"]), int(row["auv_id"]))


def execute(key: tuple[int, int, int, int]) -> dict[str, str]:
    n, sid, seed, oc = key
    output = RUNS / f"n{n}_sid{sid}_seed{seed}_oc{oc}.csv"
    if not output.exists():
        subprocess.run([str(EXE), "--mode=run", f"--nodeCount={n}", f"--scenarioId={sid}",
                        f"--scenarioSeed={seed}", f"--selectedOc={oc}", f"--output={output}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    rows = csv_rows(output)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {output}")
    return rows[0]


def fill_missing(existing: dict, needed: set[tuple[int, int, int, int]]) -> dict:
    missing = sorted(needed - set(existing))
    print(f"Candidate reuse={len(needed)-len(missing)}, missing simulations={len(missing)}", flush=True)
    if not missing:
        return existing
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(execute, key): key for key in missing}
        for i, job in enumerate(as_completed(jobs), start=1):
            row = job.result()
            if int(row["generated_packets"]) != 200 or int(row["OC_data_hops"]) != 0 or int(row["architecture_violations"]) != 0:
                raise RuntimeError(f"Architecture/counter invariant failed for {jobs[job]}")
            existing[candidate_key({**row, "auv_id": row["selected_oc"]})] = row
            if i % 100 == 0 or i == len(jobs):
                print(f"Completed {i}/{len(jobs)} missing candidate trials", flush=True)
    return existing


def score(model, data: pd.DataFrame, name: str) -> np.ndarray:
    return model.decision_function(data[FEATURES]) if name == "SVM" else model.predict_proba(data[FEATURES])[:, 1]


def select(data: pd.DataFrame, raw: np.ndarray, name: str) -> pd.DataFrame:
    out = data[["scenario_id", "scenario_seed", "node_count", "auv_id", "is_oc"]].copy()
    out["raw_score"] = raw
    out = (out.sort_values(["scenario_id", "raw_score", "auv_id"], ascending=[True, False, True])
              .groupby("scenario_id", as_index=False).head(1).copy())
    truth = data[data.is_oc.eq(1)].set_index("scenario_id").auv_id
    out["true_oc"] = truth.loc[out.scenario_id].to_numpy()
    out["correct"] = (out.auv_id == out.true_oc).astype(int)
    out["Model"] = name
    return out.rename(columns={"auv_id": "predicted_oc"})


def map_network(selected: pd.DataFrame, ledger: pd.DataFrame, method: str) -> pd.DataFrame:
    cols = ["scenario_id", "scenario_seed", "node_count", "selected_oc", "PDR", "E2ED_ms", "ROR_total",
            "ROR_reactive", "hello_tx", "route_request_tx", "route_reply_tx", "mc_control_tx",
            "data_hops", "control_transmissions"]
    have = [c for c in cols if c in ledger.columns]
    ledger = ledger.copy()
    for column in ("scenario_id", "scenario_seed", "node_count", "selected_oc"):
        ledger[column] = pd.to_numeric(ledger[column], errors="raise").astype(int)
    for column in ("PDR", "E2ED_ms", "ROR_total", "ROR_reactive", "hello_tx", "route_request_tx",
                   "route_reply_tx", "mc_control_tx", "data_hops", "control_transmissions"):
        if column in ledger:
            ledger[column] = pd.to_numeric(ledger[column], errors="coerce")
    result = selected.merge(ledger[have], left_on=["scenario_id", "scenario_seed", "node_count", "predicted_oc"],
                            right_on=["scenario_id", "scenario_seed", "node_count", "selected_oc"],
                            how="left", validate="one_to_one")
    if result.PDR.isna().any():
        raise RuntimeError("A selected candidate has no HELLO=60 matched ledger metric")
    result["Method"] = method
    return result


def binary_metrics(data: pd.DataFrame, raw: np.ndarray, threshold: float) -> dict:
    pred = (raw >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(data.is_oc, pred, average="binary", zero_division=0)
    return {"Binary accuracy": 100. * accuracy_score(data.is_oc, pred), "Precision": precision,
            "Recall": recall, "Binary F1": f1, "Binary threshold": threshold}


def calibrated_threshold(data: pd.DataFrame, raw: np.ndarray) -> float:
    """Select only a validation binary threshold; ranking always uses raw scores."""
    candidates = np.arange(.20, .61, .01)
    # F1 is the primary criterion; accuracy makes exact ties deterministic.
    return max(candidates, key=lambda t: (binary_metrics(data, raw, float(t))["Binary F1"],
                                           binary_metrics(data, raw, float(t))["Binary accuracy"], -float(t)))


def network_means(frame: pd.DataFrame) -> dict:
    return {"PDR": frame.PDR.mean(), "E2ED_ms": frame.E2ED_ms.mean(skipna=True),
            "ROR_total": frame.ROR_total.mean(), "ROR_reactive": frame.ROR_reactive.mean()}


def evaluate_validation(name: str, model, train: pd.DataFrame, validation: pd.DataFrame,
                        validation_ledger: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    model.fit(train[FEATURES], train.is_oc)
    raw = score(model, validation, name)
    threshold = 0.0 if name == "SVM" else calibrated_threshold(validation, raw)
    binary = binary_metrics(validation, raw, threshold)
    chosen = select(validation, raw, name)
    network = map_network(chosen, validation_ledger, "temporary")
    m = {**binary, "Top-1": 100. * chosen.correct.mean(), **network_means(network)}
    return m, chosen


def select_calibrated_model(name: str, train: pd.DataFrame, validation: pd.DataFrame,
                            validation_ledger: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    if name == "DTC":
        target = (74.5, 56.5, 68.8, .375)
        grid = ({"max_depth": d, "min_samples_split": split, "min_samples_leaf": leaf,
                 "max_features": feat, "class_weight": weight}
                for d, split, leaf, feat, weight in itertools.product(
                    [3, 4, 5, 6], [10, 20, 30, 40, 60], [5, 10, 15, 20, 30, 40, 50],
                    [2, 3, None], [None, "balanced"]))
        def make(p): return DecisionTreeClassifier(criterion="gini", random_state=42, **p)
    else:
        target = (71.0, 53.0, 66.5, .385)
        grid = ({"n_estimators": n, "max_depth": depth, "min_samples_split": split,
                 "min_samples_leaf": leaf, "max_samples": sample}
                for n, depth, split, leaf, sample in itertools.product(
                    [3, 5, 8, 10], [1, 2], [100, 150, 200, 300], [150, 200, 250, 300, 400, 500], [.20, .25, .35]))
        def make(p): return RandomForestClassifier(criterion="gini", class_weight=None, random_state=42,
                                                    n_jobs=-1, max_features=1, bootstrap=True, **p)
    rows, best = [], None
    for params in grid:
        metrics, _ = evaluate_validation(name, make(params), train, validation, validation_ledger)
        # This is a controlled target-deviation loss.  Absolute deviations are
        # additive; the dash bullets in the request are not arithmetic minus.
        loss = (abs(metrics["Binary accuracy"]-target[0]) + 1.5*abs(metrics["Top-1"]-target[1]) +
                2.*abs(metrics["PDR"]-target[2]) + 100.*abs(metrics["ROR_total"]-target[3]))
        top1_gate = 45. if name == "DTC" else 35.
        valid = (metrics["Precision"] > 0 and metrics["Recall"] > 0 and metrics["Binary F1"] >= .35
                 and metrics["Top-1"] >= top1_gate)
        record = {**params, **metrics, "selection_loss": loss, "accepted": valid}
        rows.append(record)
        key = (loss, metrics["ROR_reactive"], metrics["ROR_total"], -metrics["Top-1"], -metrics["Binary F1"])
        if valid and (best is None or key < best[0]):
            best = (key, params, metrics)
    search = pd.DataFrame(rows).sort_values(["accepted", "selection_loss", "ROR_reactive", "Top-1"],
                                             ascending=[False, True, True, False])
    if best is None:
        raise RuntimeError(f"No valid {name} configuration satisfied the F1 constraint")
    return best[1], best[2], search


def load_saved_validation_selection(name: str) -> tuple[dict, dict, pd.DataFrame] | None:
    """Reuse an already completed validation-only search after a later I/O failure."""
    path = OUT / f"{name.lower()}_validation_controlled_search.csv"
    if not path.exists():
        return None
    search = pd.read_csv(path)
    accepted = search[search.accepted.astype(str).str.lower().eq("true")]
    if accepted.empty:
        return None
    row = accepted.iloc[0]
    if name == "DTC":
        params = {"max_depth": int(row.max_depth), "min_samples_split": int(row.min_samples_split),
                  "min_samples_leaf": int(row.min_samples_leaf),
                  "max_features": int(row.max_features),
                  "class_weight": None if pd.isna(row.class_weight) else row.class_weight}
    else:
        params = {"n_estimators": int(row.n_estimators), "max_depth": int(row.max_depth),
                  "min_samples_split": int(row.min_samples_split), "min_samples_leaf": int(row.min_samples_leaf),
                  "max_samples": float(row.max_samples)}
    metrics = {key: float(row[key]) for key in ["Binary accuracy", "Precision", "Recall", "Binary F1",
                                                 "Binary threshold", "Top-1", "PDR", "E2ED_ms",
                                                 "ROR_total", "ROR_reactive"]}
    print(f"Reusing completed validation-only {name} search from {path.name}", flush=True)
    return params, metrics, search


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    data = pd.read_csv(DATASET)
    with MANIFEST.open() as f:
        split = json.load(f)
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise RuntimeError("Expected frozen grouped 1400/300/300 split")

    # Existing 60-s test runs are reusable.  Every validation scenario needs
    # all four candidates because hyperparameter choices can select any OC.
    existing_rows = csv_rows(PREVIOUS)
    existing = {(int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"]), int(r["selected_oc"])): r
                for r in existing_rows}
    # A previous interrupted run may already have produced individual new
    # candidate files; load them before deciding whether a simulation is
    # actually missing.
    for path in list(PREEXISTING_RUNS.glob("*.csv")) + list(RUNS.glob("*.csv")):
        rows = csv_rows(path)
        if len(rows) == 1:
            row = rows[0]
            existing[(int(row["node_count"]), int(row["scenario_id"]), int(row["scenario_seed"]),
                      int(row["selected_oc"]))] = row
    validation_needed = {candidate_key(row) for _, row in validation.iterrows()}
    existing = fill_missing(existing, validation_needed)
    validation_ledger = pd.DataFrame([existing[k] for k in validation_needed])

    svm = Pipeline([("scale", StandardScaler()),
                    ("svc", SVC(kernel="rbf", C=50, gamma=.02, class_weight=None,
                                 probability=False, random_state=42))])
    svm_validation, _ = evaluate_validation("SVM", svm, train, validation, validation_ledger)
    dtc_saved = load_saved_validation_selection("DTC")
    rf_saved = load_saved_validation_selection("RF")
    dtc_params, dtc_validation, dtc_search = dtc_saved or select_calibrated_model("DTC", train, validation, validation_ledger)
    rf_params, rf_validation, rf_search = rf_saved or select_calibrated_model("RF", train, validation, validation_ledger)
    write_csv(OUT / "dtc_validation_controlled_search.csv", dtc_search)
    write_csv(OUT / "rf_validation_controlled_search.csv", rf_search)

    models = {
        "SVM": svm,
        "DTC": DecisionTreeClassifier(criterion="gini", random_state=42, **dtc_params),
        "RF": RandomForestClassifier(criterion="gini", class_weight=None, random_state=42, n_jobs=-1,
                                       max_features=1, bootstrap=True, **rf_params),
    }
    fit = pd.concat([train, validation], ignore_index=True)
    # Make candidate rows available for the selected model OCs, running only
    # test candidates absent from the existing HELLO=60 ledger.
    test_predictions: dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        model.fit(fit[FEATURES], fit.is_oc)
        dump(model, OUT / f"{name.lower()}_pipeline.joblib")
        test_predictions[name] = select(test, score(model, test, name), name)
    selected_test_needed = {(int(r.node_count), int(r.scenario_id), int(r.scenario_seed), int(r.predicted_oc))
                            for frame in test_predictions.values() for _, r in frame.iterrows()}
    baseline_needed = {(int(r.node_count), int(r.scenario_id), int(r.scenario_seed), 0)
                       for _, r in test.drop_duplicates("scenario_id").iterrows()}
    existing = fill_missing(existing, selected_test_needed | baseline_needed)
    all_test_needed = selected_test_needed | baseline_needed
    test_ledger = pd.DataFrame([existing[k] for k in all_test_needed])
    for column in ("scenario_id", "scenario_seed", "node_count", "selected_oc", "PDR", "E2ED_ms",
                   "ROR_total", "ROR_reactive", "hello_tx", "route_request_tx", "route_reply_tx",
                   "mc_control_tx", "data_hops", "control_transmissions"):
        if column in test_ledger:
            test_ledger[column] = pd.to_numeric(test_ledger[column], errors="coerce")

    validation_records = {"SVM": svm_validation, "DTC": dtc_validation, "RF": rf_validation}
    thresholds = {"SVM": svm_validation["Binary threshold"], "DTC": dtc_validation["Binary threshold"],
                  "RF": rf_validation["Binary threshold"]}
    ml_rows, network_rows, selected_rows = [], [], []
    for name, method in METHODS:
        model = models[name]
        raw = score(model, test, name)
        # Ranking always uses raw scores; thresholds were fixed using only
        # validation rows and affect candidate-level binary reporting only.
        binary = binary_metrics(test, raw, thresholds[name])
        chosen = test_predictions[name]
        _, _, macro_f1, _ = precision_recall_fscore_support(chosen.true_oc, chosen.predicted_oc,
                                                              labels=[0, 1, 2, 3], average="macro", zero_division=0)
        mapped = map_network(chosen, test_ledger, method)
        selected_rows.append(mapped)
        ml_rows.append({"Model": name, **binary, "Top-1 OC accuracy": 100.*chosen.correct.mean(),
                        "Top-1 macro F1": macro_f1})
        network_rows.append(mapped)
    baseline = test_ledger[test_ledger.selected_oc.astype(int).eq(0)].copy()
    baseline["Method"] = "Baseline OC0"
    all_selected = pd.concat(network_rows + [baseline], ignore_index=True, sort=False)
    ml_table = pd.DataFrame(ml_rows)
    network_table = pd.DataFrame([{"Method": m, "Mean PDR %": g.PDR.mean(),
        "Mean E2ED ms": g.E2ED_ms.mean(skipna=True), "Mean ROR total": g.ROR_total.mean(),
        "Mean ROR reactive": g.ROR_reactive.mean()} for m, g in all_selected.groupby("Method")])
    order = {method: i for i, (_, method) in enumerate(METHODS)} | {"Baseline OC0": 3}
    network_table["_order"] = network_table.Method.map(order)
    network_table = network_table.sort_values("_order").drop(columns="_order")
    per_node = (all_selected.groupby(["node_count", "Method"], as_index=False)
                .agg(**{"PDR %": ("PDR", "mean"), "E2ED ms": ("E2ED_ms", "mean"),
                        "ROR total": ("ROR_total", "mean"), "ROR reactive": ("ROR_reactive", "mean")}))
    per_node["_order"] = per_node.Method.map(order)
    per_node = per_node.sort_values(["node_count", "_order"]).drop(columns="_order")
    chain = []
    prior = None
    for method in [m for _, m in METHODS]:
        row = network_table[network_table.Method.eq(method)].iloc[0]
        chain.append({"Method": method, "Compared with previous model": prior["Method"] if prior is not None else "N/A",
                      "PDR gap vs previous model": math.nan if prior is None else row["Mean PDR %"]-prior["Mean PDR %"],
                      "ROR_total gap vs previous model": math.nan if prior is None else row["Mean ROR total"]-prior["Mean ROR total"],
                      "ROR_reactive gap vs previous model": math.nan if prior is None else row["Mean ROR reactive"]-prior["Mean ROR reactive"]})
        prior = row
    setting_rows = []
    for name, params, valid in [("SVM", {"C":50, "gamma":.02, "class_weight":None}, svm_validation),
                                ("DTC", dtc_params, dtc_validation), ("RF", rf_params, rf_validation)]:
        test_ml = ml_table[ml_table.Model.eq(name)].iloc[0]
        test_net = network_table[network_table.Method.eq(f"{name}-selected OC")].iloc[0]
        setting_rows.append({"Model": name, "Selected hyperparameters": json.dumps(params),
            "Selected binary threshold": valid["Binary threshold"],
            "Validation binary accuracy": valid["Binary accuracy"], "Validation F1": valid["Binary F1"], "Validation Top-1": valid["Top-1"],
            "Validation PDR": valid["PDR"], "Validation ROR_total": valid["ROR_total"],
            "Test binary accuracy": test_ml["Binary accuracy"], "Test Top-1": test_ml["Top-1 OC accuracy"],
            "Test PDR": test_net["Mean PDR %"], "Test ROR_total": test_net["Mean ROR total"]})
    write_csv(OUT / "test_ml_metrics.csv", ml_table)
    write_csv(OUT / "test_network_metrics.csv", network_table)
    write_csv(OUT / "pdr_ror_gaps_by_display_order.csv", pd.DataFrame(chain))
    write_csv(OUT / "per_node_test_metrics.csv", per_node)
    write_csv(OUT / "selected_hyperparameters_validation_and_test.csv", pd.DataFrame(setting_rows))
    write_csv(OUT / "model_selected_oc_rows_with_network_metrics.csv", pd.concat(selected_rows, ignore_index=True))
    write_csv(OUT / "candidate_ledger_validation_and_required_test.csv", pd.DataFrame([existing[k] for k in validation_needed | all_test_needed]))
    (OUT / "interpretation.txt").write_text(
        "Controlled target-deviation sensitivity experiment. DTC/RF were selected using validation-only target deviations; the held-out test split was not used during selection.\n"
        "SVM is the proposed OC-selection classifier. DTC and RF are validation-calibrated lightweight baseline classifiers evaluated under the HELLO=60s reactive-suppression protocol.\n"
        "This output is not an unbiased primary model-ranking benchmark because validation targets deliberately guided baseline configuration selection.\n")
    print("\nTEST ML\n", ml_table.to_string(index=False))
    print("\nTEST NETWORK\n", network_table.to_string(index=False))
    print("\nBY NODE\n", per_node.to_string(index=False))
    print("\nSETTINGS\n", pd.DataFrame(setting_rows).to_string(index=False))


if __name__ == "__main__":
    main()
