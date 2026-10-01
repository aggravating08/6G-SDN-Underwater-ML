#!/usr/bin/env python3
"""Full validation-only ML tuning and aligned network evaluation for v8.

Inputs are exclusively the v8 300--450 m balanced-distance feature, label,
candidate-outcome and grouped-split files.  PDR/E2ED/ROR stay out of every ML
input and out of validation hyperparameter selection.  They are looked up only
after models have selected a candidate OC on held-out scenarios.
"""
from __future__ import annotations

import itertools
import json
from concurrent.futures import ThreadPoolExecutor
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
OUT = CURRENT / "final_aligned_paper_faithful_real_metrics" / "v8_validation_only_full_grid"
DATA = SOURCE / "balanced_utility_labeled_dataset.csv"
LEDGER = SOURCE / "balanced_matched_four_oc_outcomes.csv"
SPLIT = SOURCE / "split_manifest.json"
FEATURES = [
    "x", "y", "local_density", "speed", "source_to_oc_distance", "destination_to_oc_distance",
    "sensors_in_oc_range", "gateways_in_oc_range", "source_covered_by_oc", "destination_covered_by_oc",
    "estimated_local_path_exists", "estimated_hop_count", "mean_endpoint_distance",
    "endpoint_distance_balance", "local_view_sensor_fraction",
]


def score(model, rows: pd.DataFrame, name: str) -> np.ndarray:
    return (np.asarray(model.decision_function(rows[FEATURES]), dtype=float)
            if name == "SVM" else np.asarray(model.predict_proba(rows[FEATURES])[:, 1], dtype=float))


def threshold_f1(rows: pd.DataFrame, scores: np.ndarray) -> float:
    """Validation-only binary threshold. Raw scores always drive Top-1."""
    # Sort once, then evaluate the F1/accuracy of every distinct score cut in
    # O(n log n).  This is algebraically identical to thresholding each score
    # and calling a metric routine repeatedly, but avoids a large CPU cost in
    # the requested RF validation grid.
    y = rows.is_oc.to_numpy(dtype=int)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    predicted_positive = ends + 1
    true_positive = np.cumsum(y)[ends]
    total_positive = int(y.sum())
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive
    denom = 2.0 * true_positive + false_positive + false_negative
    f1 = np.divide(2.0 * true_positive, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
    true_negative = len(y) - predicted_positive - false_negative
    accuracy = (true_positive + true_negative) / float(len(y))
    # The final criterion reproduces the earlier lower-threshold tie-break.
    best = max(range(len(ends)), key=lambda i: (float(f1[i]), float(accuracy[i]), -float(s[ends[i]])))
    end = int(ends[best])
    if end == len(s) - 1:
        # Same all-positive candidate used by the original implementation.
        return float(s[end] - 1.0)
    # Any value between adjacent distinct scores gives the same validation
    # predictions; the midpoint matches the original exhaustive search.
    return float((s[end] + s[end + 1]) / 2.0)


def binary(rows: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(rows.is_oc, pred, average="binary", zero_division=0)
    return {"Binary accuracy": 100. * float(accuracy_score(rows.is_oc, pred)),
            "Precision": float(precision), "Recall": float(recall), "Binary F1": float(f1)}


def selected(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = rows[["scenario_id", "scenario_seed", "node_count", "auv_id", "is_oc"]].copy()
    out["raw_score"] = scores
    out = (out.sort_values(["scenario_id", "raw_score", "auv_id"], ascending=[True, False, True])
              .groupby("scenario_id", as_index=False).head(1).copy())
    truth = rows[rows.is_oc.eq(1)].set_index("scenario_id").auv_id
    out["true_best_oc"] = truth.loc[out.scenario_id].to_numpy()
    out["correct_top1"] = (out.auv_id == out.true_best_oc).astype(int)
    return out.rename(columns={"auv_id": "selected_oc"})


def top1(rows: pd.DataFrame, scores: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    selected_rows = selected(rows, scores)
    _, _, f1, _ = precision_recall_fscore_support(
        selected_rows.true_best_oc, selected_rows.selected_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )
    return {"Top-1 OC accuracy": 100. * float(selected_rows.correct_top1.mean()),
            "Top-1 macro F1": float(f1)}, selected_rows


def build(name: str, p: dict):
    if name == "SVM":
        return Pipeline([("scaler", StandardScaler()),
                         ("svc", SVC(kernel="rbf", probability=False, random_state=42, **p))])
    if name == "DTC":
        return DecisionTreeClassifier(random_state=42, **p)
    # Cached RF family fits are parallelized at the configuration level below,
    # so each individual forest stays single-threaded and avoids nested pools.
    return RandomForestClassifier(random_state=42, n_jobs=1, bootstrap=True, **p)


def grid(name: str):
    if name == "SVM":
        for c, gamma, weight in itertools.product(
            [1, 2, 5, 10, 20, 50, 100, 200], [.001, .002, .005, .01, .02, .03, .05, .08, .1], [None, "balanced"]
        ):
            yield {"C": c, "gamma": gamma, "class_weight": weight}
    elif name == "DTC":
        for criterion, depth, split, leaf, feat in itertools.product(
            ["gini", "entropy"], [3, 4, 5, 6, 8, 10, None], [2, 5, 10, 20, 40], [1, 5, 10, 20], [None, 2, 3, 4]
        ):
            yield {"criterion": criterion, "max_depth": depth, "min_samples_split": split,
                   "min_samples_leaf": leaf, "max_features": feat, "class_weight": None}
    elif name == "RF":
        for trees, depth, split, leaf, feat in itertools.product(
            [20, 50, 100, 200], [2, 3, 4, 5, 6, 8, None], [2, 5, 10, 20, 40], [1, 5, 10, 20],
            ["sqrt", "log2", 2, 3, 4, None]
        ):
            yield {"n_estimators": trees, "max_depth": depth, "min_samples_split": split,
                   "min_samples_leaf": leaf, "max_features": feat, "class_weight": None}
    else:
        raise ValueError(name)


def parameter_key(p: dict) -> str:
    return json.dumps(p, sort_keys=True, separators=(",", ":"))


def validation_search(name: str, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, float, pd.DataFrame]:
    """Evaluate every requested grid point; save progress so interruption is safe."""
    OUT.mkdir(parents=True, exist_ok=True)
    progress = OUT / f"validation_search_{name.lower()}_progress.csv"
    existing = pd.read_csv(progress) if progress.exists() else pd.DataFrame()
    known = set(existing.parameter_key) if not existing.empty else set()
    records = existing.to_dict("records") if not existing.empty else []
    params = list(grid(name))
    print(f"{name}: {len(known)}/{len(params)} validation configurations already available", flush=True)
    pending = [p for p in params if parameter_key(p) not in known]

    def evaluate_one(p: dict) -> dict:
        key = parameter_key(p)
        model = build(name, p)
        model.fit(train[FEATURES], train.is_oc)
        values = score(model, validation, name)
        threshold = threshold_f1(validation, values)
        b = binary(validation, values, threshold)
        t, _ = top1(validation, values)
        return {"parameter_key": key, "parameters": json.dumps(p, sort_keys=True),
                "Binary threshold": threshold, **b, **t}

    # Individual forests are explicitly single-threaded.  Two independent
    # validation fits therefore use the available CPU without nested worker
    # pools or altering any estimator's random state/results.
    workers = 2 if name == "RF" else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(evaluate_one, pending):
            records.append(record)
            known.add(record["parameter_key"])
            if len(records) % 20 == 0 or len(records) == len(params):
                pd.DataFrame(records).to_csv(progress, index=False)
                print(f"{name}: saved {len(records)}/{len(params)} validation configurations", flush=True)
    search = pd.DataFrame(records)
    # Requested lexicographic validation objective: Top-1, Top-1 macro F1,
    # binary F1, then binary accuracy.
    search = search.sort_values(["Top-1 OC accuracy", "Top-1 macro F1", "Binary F1", "Binary accuracy"],
                                ascending=False).reset_index(drop=True)
    search.to_csv(OUT / f"validation_search_{name.lower()}.csv", index=False)
    best = search.iloc[0]
    return json.loads(best.parameters), float(best["Binary threshold"]), search


def rf_prefix_probability(forest: RandomForestClassifier, x: pd.DataFrame, count: int) -> np.ndarray:
    """Exact RF probability using the first ``count`` deterministic trees.

    With a fixed ``random_state``, sklearn generates the same first N trees
    whether ``n_estimators`` is N or a larger value.  This lets the requested
    20/50/100/200 estimator grid reuse one 200-tree fit per otherwise identical
    parameter group.  It is computation caching, not a model approximation.
    """
    classes = forest.classes_
    # Tree.predict_proba(check_input=False) requires the validated ndarray
    # representation used internally by RandomForestClassifier.
    x_values = x.to_numpy(dtype=np.float32, copy=False)
    accumulator = np.zeros((len(x_values), len(classes)), dtype=float)
    for estimator in forest.estimators_[:count]:
        probabilities = estimator.predict_proba(x_values, check_input=False)
        class_indices = np.searchsorted(classes, estimator.classes_)
        accumulator[:, class_indices] += probabilities
    positive = int(np.flatnonzero(classes == 1)[0])
    return accumulator[:, positive] / float(count)


def validation_search_rf_cached(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, float, pd.DataFrame]:
    """Evaluate all 3,360 requested RF grid points with exact tree-prefix reuse."""
    progress = OUT / "validation_search_rf_cached_progress.csv"
    existing = pd.read_csv(progress) if progress.exists() else pd.DataFrame()
    known = set(existing.parameter_key) if not existing.empty else set()
    records = existing.to_dict("records") if not existing.empty else []
    all_params = list(grid("RF"))
    print(f"RF cached grid: {len(known)}/{len(all_params)} configurations already available", flush=True)
    families = []
    for depth, split, leaf, feat in itertools.product(
        [2, 3, 4, 5, 6, 8, None], [2, 5, 10, 20, 40], [1, 5, 10, 20], ["sqrt", "log2", 2, 3, 4, None]
    ):
        family = [{"n_estimators": n, "max_depth": depth, "min_samples_split": split,
                   "min_samples_leaf": leaf, "max_features": feat, "class_weight": None}
                  for n in [20, 50, 100, 200]]
        if any(parameter_key(p) not in known for p in family):
            families.append(family)

    def evaluate_family(family: list[dict]) -> list[dict]:
        needed = [p for p in family if parameter_key(p) not in known]
        full = build("RF", family[-1])
        full.fit(train[FEATURES], train.is_oc)
        output = []
        for p in needed:
            values = rf_prefix_probability(full, validation[FEATURES], int(p["n_estimators"]))
            threshold = threshold_f1(validation, values)
            b = binary(validation, values, threshold)
            t, _ = top1(validation, values)
            output.append({"parameter_key": parameter_key(p), "parameters": json.dumps(p, sort_keys=True),
                           "Binary threshold": threshold, **b, **t})
        return output

    # Four independent, single-threaded families use the available cores.  The
    # parameter grid, deterministic random state, and validation scores are
    # unchanged; only independent computation is concurrent.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for family_output in pool.map(evaluate_family, families):
            for record in family_output:
                records.append(record)
                known.add(record["parameter_key"])
            if len(records) % 20 == 0 or len(records) == len(all_params):
                pd.DataFrame(records).to_csv(progress, index=False)
                print(f"RF cached grid: saved {len(records)}/{len(all_params)} validation configurations", flush=True)
    search = pd.DataFrame(records).sort_values(
        ["Top-1 OC accuracy", "Top-1 macro F1", "Binary F1", "Binary accuracy"], ascending=False
    ).reset_index(drop=True)
    search.to_csv(OUT / "validation_search_rf.csv", index=False)
    best = search.iloc[0]
    return json.loads(best.parameters), float(best["Binary threshold"]), search


def map_metrics(chosen: pd.DataFrame, ledger: pd.DataFrame, model: str) -> pd.DataFrame:
    keys = ["scenario_id", "scenario_seed", "node_count", "selected_oc"]
    needed = ["PDR", "E2ED_ms", "ROR_total", "generated_packets", "delivered_packets", "control_transmissions", "data_hops",
              "mean_source_destination_distance_m", "OC_data_hops", "architecture_violations"]
    result = chosen.merge(ledger[keys + needed], on=keys, how="left", validate="one_to_one")
    # E2ED is intentionally N/A when a candidate delivers no packets.  Its
    # absence is not a failed key join; all identifiers and packet counters
    # must nevertheless be present.
    required_mapping = [c for c in needed if c != "E2ED_ms"]
    if result[required_mapping].isna().any().any():
        raise RuntimeError(f"Missing same-ledger metric for {model}")
    result["Model"] = model
    return result


def network_metrics(rows: pd.DataFrame) -> dict[str, float]:
    delivered = rows.delivered_packets.sum()
    generated = rows.generated_packets.sum()
    control = rows.control_transmissions.sum()
    data_hops = rows.data_hops.sum()
    return {"Mean PDR": 100. * delivered / generated,
            "Mean E2ED": float((rows.E2ED_ms * rows.delivered_packets).sum() / delivered) if delivered else np.nan,
            "Mean ROR": float(control / (control + data_hops)) if control + data_hops else np.nan}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, ledger = pd.read_csv(DATA), pd.read_csv(LEDGER)
    split = json.load(SPLIT.open())
    (OUT / "split_manifest.json").write_text(json.dumps(split, indent=2))
    if len(data) != 8000 or len(ledger) != 8000 or not data.groupby("scenario_id").size().eq(4).all():
        raise RuntimeError("The v8 aligned 8,000-row dataset/ledger is incomplete")
    if not data.groupby("scenario_id").is_oc.sum().eq(1).all():
        raise RuntimeError("Every aligned scenario must have exactly one utility label")
    if not ledger.generated_packets.eq(200).all() or not ledger.OC_data_hops.eq(0).all() or not ledger.architecture_violations.eq(0).all():
        raise RuntimeError("v8 ledger invariant failed")
    if not ledger.mean_source_destination_distance_m.between(300., 450.).all():
        raise RuntimeError("v8 300--450 m traffic rule failed")
    train = data[data.scenario_id.isin(split["train"])].copy()
    validation = data[data.scenario_id.isin(split["validation"])].copy()
    test = data[data.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise RuntimeError("Expected 1,400/300/300 grouped scenario split")
    if test.drop_duplicates("scenario_id").node_count.value_counts().to_dict() != {25:75,50:75,75:75,100:75}:
        raise RuntimeError("Expected 75 held-out scenarios per density")

    choices = {}
    for name in ("SVM", "DTC"):
        choices[name] = validation_search(name, train, validation)[:2]
    choices["RF"] = validation_search_rf_cached(train, validation)[:2]
    # All settings are frozen here.  The following is the one held-out test
    # evaluation, and no test metric feeds back into parameter selection.
    fit = pd.concat([train, validation], ignore_index=True)
    ml_rows, selection_rows = [], []
    for name in ("SVM", "DTC", "RF"):
        params, threshold = choices[name]
        model = build(name, params)
        model.fit(fit[FEATURES], fit.is_oc)
        dump(model, OUT / f"{name.lower()}_pipeline.joblib")
        values = score(model, test, name)
        b = binary(test, values, threshold)
        t, chosen = top1(test, values)
        mapped = map_metrics(chosen, ledger, name)
        mapped.to_csv(OUT / f"{name.lower()}_selected_oc_test_rows.csv", index=False)
        ml_rows.append({"Model": name, **b, **t, **network_metrics(mapped)})
        selection_rows.append(mapped)
    all_selected = pd.concat(selection_rows, ignore_index=True)
    ml = pd.DataFrame(ml_rows)
    ml.to_csv(OUT / "final_aligned_ml_accuracy_comparison.csv", index=False)
    network = ml[["Model", "Mean PDR", "Mean E2ED", "Mean ROR"]]
    network.to_csv(OUT / "final_aligned_network_metrics_by_model.csv", index=False)
    node_rows = []
    for name in ("SVM", "DTC", "RF"):
        group = all_selected[all_selected.Model.eq(name)]
        for n in (25, 50, 75, 100):
            node_rows.append({"Nodes":n,"Model":name,**network_metrics(group[group.node_count.eq(n)])})
    pd.DataFrame(node_rows).to_csv(OUT / "final_aligned_network_metrics_by_node.csv", index=False)
    all_selected.rename(columns={"node_count":"nodes", "E2ED_ms":"E2ED", "ROR_total":"ROR"}).to_csv(
        OUT / "selected_oc_per_test_scenario.csv", index=False)
    (OUT / "experiment_protocol.json").write_text(json.dumps({
        "dataset": str(DATA), "candidate_ledger": str(LEDGER), "split": str(SPLIT),
        "scenarios": 2000, "candidate_rows":8000, "held_out_test_scenarios":300,
        "traffic_rule":"source-destination distance 300--450 m", "test_used_for_tuning":False,
        "features":FEATURES,
        "ror":"control packet transmissions/(control packet transmissions + realized data packet transmissions)"
    },indent=2))
    print("FINAL ALIGNED V8 RESULTS")
    print(ml.to_string(index=False))


if __name__ == "__main__":
    main()
