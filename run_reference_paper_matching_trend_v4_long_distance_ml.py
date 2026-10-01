#!/usr/bin/env python3
"""Long-distance, reference-style OC-selection experiment (v4).

This is deliberately independent of the prior v3 network-only trend run.  It
creates a new long-distance traffic population, simulates all four OCs per
scenario, labels the utility-best candidate from those matched outcomes, and
trains fresh grouped-split ML models.  Network outcomes are used *only* to
make the target label and to evaluate an OC after it has been selected; they
are never model inputs.
"""
from __future__ import annotations

import csv
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from joblib import dump


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_v4_long_distance_ml"
RUNS = OUT / "candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
NODES = (25, 50, 75, 100)
# A balanced, deterministic 2,000-scenario population: 500 scenarios/density.
SPECS = {25: (0, 29), 50: (500, 1_000_029), 75: (1000, 2_000_029), 100: (1500, 3_000_029)}
METHODS = ("SVM", "DTC", "RF", "Baseline OC0")
LONG_DISTANCE_M = 424.3

# Strictly pre-routing observables.  Scenario-relative values are legitimate:
# all four candidate positions/speeds and the scheduled endpoints are known at
# the controller-selection instant, before a route is computed or packets run.
FEATURES = [
    "x", "y", "local_density", "speed", "source_to_oc_distance",
    "destination_to_oc_distance", "sensors_in_oc_range", "gateways_in_oc_range",
    "source_covered_by_oc", "destination_covered_by_oc", "estimated_local_path_exists",
    "estimated_hop_count", "mean_endpoint_distance", "endpoint_distance_balance",
    "local_view_sensor_fraction",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NA", "NaN", ""])


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run(args: list[str], output: Path) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(EXE), *args, f"--output={output}"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def protocol_args(nodes: int, scenario_id: int, seed: int, selected_oc: int) -> list[str]:
    return ["--mode=run", f"--nodeCount={nodes}", f"--scenarioId={scenario_id}",
            f"--scenarioSeed={seed}", f"--selectedOc={selected_oc}", "--runs=500",
            "--fixedControlPolicy=true", "--helloSeconds=30", "--routeTtlSeconds=60",
            "--negativeRouteTtlSeconds=120", "--referenceUpdateAccounting=true",
            "--bufferRouteDiscovery=true", "--routeDiscoveryRetryLimit=3",
            "--packetBufferTimeoutSeconds=60", "--longDistanceFlows=true"]


def generate_ledgers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate independently reproducible pre-routing and four-OC ledgers."""
    feature_files: list[Path] = []
    candidate_files: list[Path] = []
    for nodes, (sid, seed) in SPECS.items():
        feature_file = OUT / "topology_features" / f"n{nodes}_features.csv"
        if not feature_file.exists():
            run(["--mode=topology-features", f"--nodeCount={nodes}", f"--scenarioId={sid}",
                 f"--baseSeed={seed}", "--runs=500", "--longDistanceFlows=true"], feature_file)
        feature_files.append(feature_file)
        for oc in range(4):
            candidate_file = RUNS / f"n{nodes}_oc{oc}.csv"
            run(protocol_args(nodes, sid, seed, oc), candidate_file)
            candidate_files.append(candidate_file)
    features = pd.concat([read_csv(path) for path in feature_files], ignore_index=True)
    candidates = pd.concat([read_csv(path) for path in candidate_files], ignore_index=True)
    write_csv(OUT / "long_distance_candidate_features.csv", features)
    write_csv(OUT / "long_distance_matched_four_oc_outcomes.csv", candidates)
    return features, candidates


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def minmax_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    # Undefined delivered-packet E2ED is the least favourable value whenever a
    # scenario has at least one defined E2ED candidate.  If none are defined,
    # delay cannot discriminate and all receive the same score.
    finite = x.dropna()
    if finite.empty or math.isclose(float(finite.min()), float(finite.max())):
        return pd.Series(1.0, index=x.index)
    fill = float(finite.min()) if higher_is_better else float(finite.max())
    x = x.fillna(fill)
    z = (x - x.min()) / (x.max() - x.min())
    return z if higher_is_better else 1.0 - z


def label_outcomes(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"scenario_id", "selected_oc", "PDR", "E2ED_ms", "ROR_generated",
                "ROR_reactive", "mc_fallbacks", "final_no_path_drops", "generated_packets"}
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"Candidate ledger lacks label fields: {sorted(missing)}")
    labeled: list[pd.DataFrame] = []
    for sid, group in candidates.groupby("scenario_id", sort=False):
        if len(group) != 4 or set(group.selected_oc) != {0, 1, 2, 3}:
            raise RuntimeError(f"Scenario {sid} does not have exactly OC0..OC3")
        g = group.copy()
        g["mc_fallback_rate"] = g.mc_fallbacks / g.generated_packets
        g["route_failure_rate"] = g.final_no_path_drops / g.generated_packets
        # Predeclared v4 utility.  Lower overhead, fallback, delay and route
        # failures are inverted per matched scenario; PDR remains primary.
        g["utility_score"] = (
            1.0 * minmax_score(g.PDR, True)
            + 0.7 * minmax_score(g.ROR_generated, False)
            + 0.5 * minmax_score(g.ROR_reactive, False)
            + 0.3 * minmax_score(g.mc_fallback_rate, False)
            + 0.2 * minmax_score(g.E2ED_ms, False)
            + 0.5 * minmax_score(g.route_failure_rate, False)
        )
        # Deterministic utility ties resolve to the lower AUV identifier.
        best = g.sort_values(["utility_score", "selected_oc"], ascending=[False, True]).iloc[0].selected_oc
        g["is_oc"] = (g.selected_oc == best).astype(int)
        labeled.append(g)
    return pd.concat(labeled, ignore_index=True)


def validate_matched(features: pd.DataFrame, candidates: pd.DataFrame) -> None:
    if len(features) != 8000 or len(candidates) != 8000:
        raise RuntimeError(f"Expected 8,000 feature and candidate rows, got {len(features)} and {len(candidates)}")
    for sid, group in candidates.groupby("scenario_id"):
        if group.topology_hash.nunique() != 1 or group.acoustic_state_hash.nunique() != 1:
            raise RuntimeError(f"Unmatched topology/acoustic state across OCs in scenario {sid}")
        if (not (group.generated_packets == 200).all() or
                not (group.OC_data_hops == 0).all() or
                not (group.architecture_violations == 0).all()):
            raise RuntimeError(f"Payload/invariant violation in scenario {sid}")
        if group.mean_source_destination_distance_m.nunique(dropna=False) != 1:
            raise RuntimeError(f"Flow-distance mismatch across OCs in scenario {sid}")
    if (features.mean_source_destination_distance_m < LONG_DISTANCE_M - 1e-6).any():
        raise RuntimeError("A generated scenario violates the >=424.3 m mean long-distance rule")


def make_dataset(features: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    out = features.merge(
        labeled[["scenario_id", "selected_oc", "is_oc", "utility_score"]],
        left_on=["scenario_id", "auv_id"], right_on=["scenario_id", "selected_oc"], how="inner",
    ).drop(columns="selected_oc")
    if len(out) != 8000 or out.groupby("scenario_id").is_oc.sum().ne(1).any():
        raise RuntimeError("Label merge did not produce exactly one OC label per scenario")
    # Pre-routing, endpoint-relative derived features.
    out["mean_endpoint_distance"] = (out.source_to_oc_distance + out.destination_to_oc_distance) / 2.0
    out["endpoint_distance_balance"] = np.abs(out.source_to_oc_distance - out.destination_to_oc_distance)
    out["local_view_sensor_fraction"] = out.sensors_in_oc_range / out.node_count
    return out


def split_scenarios(dataset: pd.DataFrame) -> dict[str, list[int]]:
    """Fresh deterministic grouped 1400/300/300 split, balanced by density."""
    manifest: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for nodes in NODES:
        ids = sorted(dataset.loc[dataset.node_count.eq(nodes), "scenario_id"].unique().tolist())
        if len(ids) != 500:
            raise RuntimeError(f"Expected 500 scenarios at {nodes}, got {len(ids)}")
        rng = random.Random(20_260 + nodes)
        rng.shuffle(ids)
        manifest["train"].extend(ids[:350])
        manifest["validation"].extend(ids[350:425])
        manifest["test"].extend(ids[425:])
    return manifest


def score(model, frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "SVM":
        return model.predict_proba(frame[FEATURES])[:, 1]
    return model.predict_proba(frame[FEATURES])[:, 1]


def scenario_predictions(model, frame: pd.DataFrame, name: str) -> pd.DataFrame:
    ranked = frame[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    ranked["score"] = score(model, frame, name)
    chosen = ranked.loc[ranked.groupby("scenario_id").score.idxmax()].copy()
    truth = frame.loc[frame.is_oc.eq(1), ["scenario_id", "auv_id"]].rename(columns={"auv_id": "true_oc"})
    chosen = chosen.rename(columns={"auv_id": "predicted_oc"}).merge(truth, on="scenario_id", validate="one_to_one")
    chosen["correct"] = (chosen.predicted_oc == chosen.true_oc).astype(int)
    return chosen


def top1_macro_f1(predictions: pd.DataFrame) -> float:
    return float(f1_score(predictions.true_oc, predictions.predicted_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0))


def validate_model(model, train: pd.DataFrame, validation: pd.DataFrame, name: str, params: dict) -> dict:
    model.fit(train[FEATURES], train.is_oc)
    pred = model.predict(validation[FEATURES])
    scen = scenario_predictions(model, validation, name)
    return {
        "params": params,
        "binary_f1": float(f1_score(validation.is_oc, pred, zero_division=0)),
        "top1": float(scen.correct.mean()),
        "top1_macro_f1": top1_macro_f1(scen),
    }


def choose_models(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    candidates: dict[str, list[tuple[object, dict]]] = {"SVM": [], "DTC": [], "RF": []}
    for c in (1, 5, 20, 50):
        for gamma in (0.001, 0.005, 0.02, 0.05):
            for class_weight in (None, "balanced"):
                candidates["SVM"].append((Pipeline([("scale", StandardScaler()),
                    ("svc", SVC(kernel="rbf", C=c, gamma=gamma, class_weight=class_weight,
                                 probability=True, random_state=42))]),
                    {"C": c, "gamma": gamma, "class_weight": class_weight}))
    for depth in (3, 5, 8, None):
        for leaf in (3, 10, 20):
            for class_weight in (None, "balanced"):
                candidates["DTC"].append((DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                    class_weight=class_weight, random_state=42),
                    {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": class_weight}))
    for depth in (4, 8, None):
        for leaf in (3, 10, 20):
            for features in ("sqrt", "log2"):
                candidates["RF"].append((RandomForestClassifier(n_estimators=100, max_depth=depth,
                    min_samples_leaf=leaf, max_features=features, class_weight="balanced",
                    n_jobs=-1, random_state=42),
                    {"n_estimators": 100, "max_depth": depth, "min_samples_leaf": leaf,
                     "max_features": features, "class_weight": "balanced"}))
    selected: dict[str, tuple[object, dict]] = {}
    audit: list[dict] = []
    for name, choices in candidates.items():
        best: tuple[float, object, dict, dict] | None = None
        for model, params in choices:
            result = validate_model(model, train, validation, name, params)
            # Validation-only selection: top-1 is primary; candidate binary F1 breaks ties.
            objective = result["top1"] + 1e-3 * result["binary_f1"]
            audit.append({"Model": name, "objective": objective, **result})
            if best is None or objective > best[0]:
                best = (objective, model, params, result)
        assert best is not None
        selected[name] = (best[1], best[2])
    return selected, pd.DataFrame(audit)


def ci95(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 2:
        return math.nan
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def aggregate(mapped: pd.DataFrame, method: str, nodes: int | None = None) -> dict:
    use = mapped.loc[mapped.Method.eq(method)]
    if nodes is not None:
        use = use.loc[use.node_count.eq(nodes)]
    return {
        "Method": method, "Nodes": "All" if nodes is None else nodes, "Scenarios": len(use),
        "PDR %": finite_mean(use.PDR), "PDR 95% CI": ci95(use.PDR),
        "E2ED ms": finite_mean(use.E2ED_ms), "E2ED 95% CI": ci95(use.E2ED_ms),
        "ROR offered": finite_mean(use.ROR_generated), "ROR offered 95% CI": ci95(use.ROR_generated),
        "ROR transmission": finite_mean(use.ROR_total), "ROR transmission 95% CI": ci95(use.ROR_total),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    features, candidates = generate_ledgers()
    validate_matched(features, candidates)
    labeled = label_outcomes(candidates)
    dataset = make_dataset(features, labeled)
    write_csv(OUT / "long_distance_utility_labeled_dataset.csv", dataset)
    write_csv(OUT / "long_distance_candidate_outcomes_with_labels.csv", labeled)

    manifest = split_scenarios(dataset)
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    train = dataset[dataset.scenario_id.isin(manifest["train"])].copy()
    validation = dataset[dataset.scenario_id.isin(manifest["validation"])].copy()
    test = dataset[dataset.scenario_id.isin(manifest["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise RuntimeError("Grouped split size is not 1400/300/300 scenarios")

    models, validation_audit = choose_models(train, validation)
    write_csv(OUT / "validation_model_search.csv", validation_audit.sort_values(["Model", "objective"], ascending=[True, False]))
    trainval = pd.concat([train, validation], ignore_index=True)
    ml_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    test_candidates = labeled.rename(columns={"selected_oc": "auv_id"}).copy()
    # Candidate outcomes include all metric columns needed for model-to-network mapping.
    for name, (model, params) in models.items():
        model.fit(trainval[FEATURES], trainval.is_oc)
        dump(model, OUT / f"{name.lower()}_pipeline.joblib")
        binary = model.predict(test[FEATURES])
        precision, recall, _, _ = precision_recall_fscore_support(test.is_oc, binary, average="binary", zero_division=0)
        scen = scenario_predictions(model, test, name)
        scen["Model"] = name
        prediction_rows.append(scen)
        ml_rows.append({"Model": name, "Binary accuracy": accuracy_score(test.is_oc, binary),
                        "Precision": precision, "Recall": recall,
                        "Binary F1": f1_score(test.is_oc, binary, zero_division=0),
                        "Top-1 OC accuracy": scen.correct.mean(),
                        "Top-1 macro F1": top1_macro_f1(scen),
                        "Selected hyperparameters": json.dumps(params, sort_keys=True)})
    predictions = pd.concat(prediction_rows, ignore_index=True)
    write_csv(OUT / "test_model_oc_predictions.csv", predictions)

    mapped_parts: list[pd.DataFrame] = []
    for name in ("SVM", "DTC", "RF"):
        pred = predictions[predictions.Model.eq(name)].copy()
        candidate_map = test_candidates.drop(columns=["node_count", "is_oc"])
        mapped = pred.merge(candidate_map, left_on=["scenario_id", "predicted_oc"],
                            right_on=["scenario_id", "auv_id"], how="left", validate="one_to_one")
        if mapped.PDR.isna().any():
            raise RuntimeError(f"Unmapped candidate outcome for {name}")
        mapped["Method"] = name
        mapped_parts.append(mapped)
    baseline = labeled[labeled.selected_oc.eq(0) & labeled.scenario_id.isin(manifest["test"])].copy()
    baseline["Method"] = "Baseline OC0"
    mapped = pd.concat([*mapped_parts, baseline], ignore_index=True, sort=False)
    write_csv(OUT / "model_selected_oc_rows_with_network_metrics.csv", mapped)

    ml_table = pd.DataFrame(ml_rows)
    network_table = pd.DataFrame([aggregate(mapped, m) for m in METHODS])
    per_node = pd.DataFrame([aggregate(mapped, m, n) for n in NODES for m in METHODS])
    write_csv(OUT / "final_ml_metrics.csv", ml_table)
    write_csv(OUT / "final_network_metrics.csv", network_table)
    write_csv(OUT / "final_network_metrics_by_node.csv", per_node)

    svm_by_node = per_node[per_node.Method.eq("SVM")].copy()
    # Add raw traffic/path diagnostics from exactly the model-selected test rows.
    extras = []
    for n in NODES:
        use = mapped[(mapped.Method == "SVM") & (mapped.node_count == n)]
        extras.append({"Nodes": n,
                       "mean source-destination distance m": finite_mean(use.mean_source_destination_distance_m),
                       "mean delivered hop count": finite_mean(use.mean_delivered_hop_count),
                       "mean delivered path length m": finite_mean(use.mean_delivered_path_length_m),
                       "delivered-E2ED scenarios": int(use.E2ED_ms.notna().sum())})
    svm_density = svm_by_node.merge(pd.DataFrame(extras), on="Nodes", validate="one_to_one")
    write_csv(OUT / "svm_density_trend.csv", svm_density)
    seq = svm_density.sort_values("Nodes")
    pdr = seq["PDR %"].tolist(); e2e = seq["E2ED ms"].tolist(); offered = seq["ROR offered"].tolist()
    source_distance = seq["mean source-destination distance m"].tolist()
    checks = pd.DataFrame([
        {"Metric": "PDR %", **dict(zip(map(str, NODES), pdr)),
         "Expected trend": "increases then saturates (0.5 pp tolerance)",
         "Pass/Fail": "PASS" if all(b >= a - .5 for a, b in zip(pdr, pdr[1:])) else "FAIL"},
        {"Metric": "E2ED ms", **dict(zip(map(str, NODES), e2e)), "Expected trend": "generally decreases",
         "Pass/Fail": "PASS" if all(b <= a for a, b in zip(e2e, e2e[1:])) else "FAIL"},
        {"Metric": "ROR offered", **dict(zip(map(str, NODES), offered)), "Expected trend": "increases",
         "Pass/Fail": "PASS" if all(b >= a for a, b in zip(offered, offered[1:])) else "FAIL"},
        {"Metric": "source-destination distance m", **dict(zip(map(str, NODES), source_distance)),
         "Expected trend": "all >= 424.3 m and max/min <= 1.05", "Pass/Fail": "PASS" if min(source_distance) >= LONG_DISTANCE_M and max(source_distance) / min(source_distance) <= 1.05 else "FAIL"},
    ])
    write_csv(OUT / "trend_gate.csv", checks)
    (OUT / "configuration.json").write_text(json.dumps({
        "experiment": "reference_paper_matching_trend_v4_long_distance_ml",
        "area_m": "500 x 500", "minimum_source_destination_distance_m": LONG_DISTANCE_M,
        "traffic": "10 matched flows x 20 packets; every selected pair is >= 424.3 m",
        "fixed_protocol": {"hello_seconds": 30, "route_ttl_seconds": 60,
                            "negative_cache_seconds": 120, "route_discovery_retries": 3,
                            "route_discovery_buffer_timeout_seconds": 60,
                            "reference_topology_update_accounting": True},
        "labels": "utility-best OC from matched four-OC outcomes; PDR primary, then inverse offered/reactive overhead, MC fallback rate, delivered E2ED, and route-failure rate",
        "model_inputs": FEATURES,
        "outcome_leakage": "PDR, E2ED, ROR and control counters are label/evaluation-only and are not model inputs",
        "split": "fresh deterministic grouped split, 350/75/75 scenarios per density = 1400/300/300",
        "ror_offered": "control packets / (control packets + 200 generated source packets)",
        "ror_transmission": "control packets / (control packets + realized data-hop transmissions)",
        "e2ed": "destination arrival time minus source generation time for delivered packets only; successful controller/retry waiting is included",
    }, indent=2) + "\n")

    print("\nML TABLE")
    print(ml_table.to_string(index=False))
    print("\nNETWORK TABLE")
    print(network_table.to_string(index=False))
    print("\nSVM DENSITY TREND")
    print(svm_density.to_string(index=False))
    print("\nTREND GATE")
    print(checks.to_string(index=False))
    print(f"\nSaved v4 experiment to {OUT}")


if __name__ == "__main__":
    main()
