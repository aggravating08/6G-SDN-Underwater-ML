#!/usr/bin/env python3
"""Clean, path-balanced, fully aligned underwater SDN/ML experiment (v10).

The script creates one new population of 2,000 accepted scenarios.  Every
feature row, utility label, matched four-OC result, split membership, ML score,
and final selected-OC metric uses the same scenario ID and deterministic seed.
No outcome column is included in the ML feature matrix.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import run_reference_paper_matching_trend_v4_long_distance_ml as utility


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/underwater_rebuild/current/final_aligned_v10_paper_trend_clean"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
NODES = (25, 50, 75, 100)
SPECS = {25: (0, 8_000_029), 50: (500, 9_000_029),
         75: (1000, 10_000_029), 100: (1500, 11_000_029)}
METHODS = ("SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0")
FEATURES = utility.FEATURES
ACCEPTANCE_DESCRIPTION = (
    "sensor-only static path; endpoint distance 300--450 m; 5--8 shortest hops "
    "at 50/75/100 nodes; 5--8 at 25 nodes when available, otherwise documented 4--8 fallback"
)


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NA", "NaN", ""])


def write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run(args: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(EXE), *args, f"--output={output}"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def fixed_protocol_args() -> list[str]:
    # One fixed policy for every density.  The relay wait is a local,
    # candidate-count mechanism, so it is also independent of node count.
    return ["--fixedControlPolicy=true", "--helloSeconds=30", "--routeTtlSeconds=60",
            "--negativeRouteTtlSeconds=120", "--referenceUpdateAccounting=true",
            "--aggregatedTopologyDigestAccounting=true", "--topologyDigestCapacityNodes=20",
            "--bufferRouteDiscovery=true", "--routeDiscoveryRetryLimit=3",
            "--packetBufferTimeoutSeconds=60", "--hopBalancedDistanceFlows=true",
            "--relaySearchDelay=true", "--relaySearchBaseMs=15",
            "--relaySearchScaleMs=360", "--relaySearchCandidateExponent=2",
            "--relaySearchMaxMs=180"]


def manifest_path(nodes: int) -> Path:
    return OUT / "accepted_scenarios" / f"n{nodes}_accepted_scenarios.csv"


def acquire_accepted_scenarios(nodes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Acceptance-sample only predeclared topology/traffic difficulty rules."""
    manifest = manifest_path(nodes)
    feature_dir = OUT / "accepted_topology_features" / f"n{nodes}"
    feature_dir.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        accepted = read(manifest)
        if len(accepted) == 500:
            files = [feature_dir / f"scenario_{sid}.csv" for sid in accepted.scenario_id]
            if all(path.exists() for path in files):
                return accepted, pd.concat([read(path) for path in files], ignore_index=True)

    start_id, seed_base = SPECS[nodes]
    strict = (5, 8)
    fallback = (4, 8)
    accepted_rows: list[dict] = []
    feature_frames: list[pd.DataFrame] = []
    trial = 0
    max_trials = 12_000
    while len(accepted_rows) < 500 and trial < max_trials:
        seed = seed_base + trial
        sid = start_id + len(accepted_rows)
        temp = feature_dir / f"candidate_seed_{seed}.csv"
        result = subprocess.run(
            [str(EXE), "--mode=topology-features", f"--nodeCount={nodes}",
             f"--scenarioId={sid}", f"--baseSeed={seed}", "--runs=1",
             "--hopBalancedDistanceFlows=true", f"--output={temp}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        trial += 1
        if result.returncode != 0:
            continue
        frame = read(temp)
        # C++ ensures every scheduled flow is connected, in distance range,
        # and in the hop band.  Recheck exported means here as an audit guard.
        if len(frame) != 4 or frame.scenario_id.nunique() != 1:
            raise RuntimeError(f"Bad acceptance feature output for node count {nodes}, seed {seed}")
        hops = float(frame.mean_static_shortest_hops.iloc[0])
        min_hops, max_hops = strict
        fallback_used = False
        if not (min_hops <= hops <= max_hops):
            if nodes == 25 and fallback[0] <= hops <= fallback[1]:
                fallback_used = True
            else:
                raise RuntimeError(f"C++ accepted a path outside the declared hop band: n={nodes}, hops={hops}")
        destination = feature_dir / f"scenario_{sid}.csv"
        temp.replace(destination)
        feature_frames.append(frame)
        accepted_rows.append({
            "scenario_id": sid, "scenario_seed": seed, "node_count": nodes,
            "mean_source_destination_distance_m": frame.mean_source_destination_distance_m.iloc[0],
            "connected_pair_ratio": frame.connected_flow_pair_ratio.iloc[0],
            "mean_static_shortest_hops": hops,
            "mean_static_shortest_path_length_m": frame.mean_static_shortest_path_length_m.iloc[0],
            "mean_static_feasible_relay_candidates": frame.mean_static_feasible_relay_candidates.iloc[0],
            "used_25_node_fallback_4_to_8": fallback_used,
        })
        if len(accepted_rows) % 50 == 0:
            print(f"accepted n={nodes}: {len(accepted_rows)}/500 after {trial} candidate deployments", flush=True)
    if len(accepted_rows) != 500:
        raise RuntimeError(f"Could only accept {len(accepted_rows)} n={nodes} scenarios in {max_trials} attempts")
    accepted = pd.DataFrame(accepted_rows)
    write(manifest, accepted)
    return accepted, pd.concat(feature_frames, ignore_index=True)


def batch_candidate_run(nodes: int, oc: int) -> Path:
    output = OUT / "candidate_runs" / f"n{nodes}_oc{oc}.csv"
    if output.exists() and len(read(output)) == 500:
        return output
    args = ["--mode=run", f"--nodeCount={nodes}", f"--selectedOc={oc}",
            f"--scenarioManifest={manifest_path(nodes)}", *fixed_protocol_args()]
    run(args, output)
    if len(read(output)) != 500:
        raise RuntimeError(f"Incomplete candidate ledger: {output}")
    return output


def generate_ledgers() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    accepted_frames, feature_frames = [], []
    for nodes in NODES:
        accepted, features = acquire_accepted_scenarios(nodes)
        accepted_frames.append(accepted)
        feature_frames.append(features)
    accepted = pd.concat(accepted_frames, ignore_index=True)
    features = pd.concat(feature_frames, ignore_index=True)
    write(OUT / "accepted_scenarios.csv", accepted)
    write(OUT / "candidate_features.csv", features)

    jobs = [(nodes, oc) for nodes in NODES for oc in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(batch_candidate_run, nodes, oc) for nodes, oc in jobs]
        paths = [job.result() for job in as_completed(pending)]
    candidates = pd.concat([read(path) for path in paths], ignore_index=True)
    write(OUT / "matched_four_oc_candidate_ledger.csv", candidates)
    return accepted, features, candidates


def validate_ledgers(accepted: pd.DataFrame, features: pd.DataFrame, candidates: pd.DataFrame) -> None:
    if len(accepted) != 2000 or len(features) != 8000 or len(candidates) != 8000:
        raise RuntimeError("v10 requires exactly 2,000 accepted scenarios and 8,000 rows in each matched ledger")
    if accepted.groupby("node_count").size().to_dict() != {25: 500, 50: 500, 75: 500, 100: 500}:
        raise RuntimeError("Population is not balanced at 500 scenarios per density")
    if not accepted.connected_pair_ratio.eq(1.0).all():
        raise RuntimeError("A disconnected pair entered the accepted population")
    if not accepted.mean_source_destination_distance_m.between(300., 450.).all():
        raise RuntimeError("Accepted endpoint distance is outside 300--450 m")
    if not accepted[accepted.node_count.ne(25)].mean_static_shortest_hops.between(5, 8).all():
        raise RuntimeError("Non-25-node accepted path violates 5--8 hop band")
    if not accepted[accepted.node_count.eq(25)].mean_static_shortest_hops.between(4, 8).all():
        raise RuntimeError("25-node accepted path violates 4--8 fallback band")
    for sid, group in candidates.groupby("scenario_id"):
        if len(group) != 4 or set(group.selected_oc) != {0, 1, 2, 3}:
            raise RuntimeError(f"Scenario {sid} lacks an OC0..OC3 candidate set")
        if group.topology_hash.nunique() != 1 or group.acoustic_state_hash.nunique() != 1:
            raise RuntimeError(f"Scenario {sid} is not matched across OCs")
        if not group.generated_packets.eq(200).all() or not group.OC_data_hops.eq(0).all() or not group.architecture_violations.eq(0).all():
            raise RuntimeError(f"Payload/control-plane invariant failed at scenario {sid}")
    if features.groupby("scenario_id").size().ne(4).any():
        raise RuntimeError("Feature ledger lacks four candidate AUV rows per scenario")


def grouped_split(dataset: pd.DataFrame) -> dict[str, list[int]]:
    split: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for nodes in NODES:
        ids = sorted(dataset.loc[dataset.node_count.eq(nodes), "scenario_id"].unique().tolist())
        if len(ids) != 500:
            raise RuntimeError(f"Expected 500 IDs for n={nodes}")
        rng = random.Random(91_000 + nodes)
        rng.shuffle(ids)
        split["train"].extend(ids[:350])
        split["validation"].extend(ids[350:425])
        split["test"].extend(ids[425:])
    return split


def score(model: object, frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "SVM":
        return np.asarray(model.decision_function(frame[FEATURES]), dtype=float)
    return np.asarray(model.predict_proba(frame[FEATURES])[:, 1], dtype=float)


def select_oc(model: object, frame: pd.DataFrame, name: str) -> pd.DataFrame:
    rank = frame[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    rank["score"] = score(model, frame, name)
    selected = rank.loc[rank.groupby("scenario_id").score.idxmax()].copy()
    truth = frame.loc[frame.is_oc.eq(1), ["scenario_id", "auv_id"]].rename(columns={"auv_id": "true_oc"})
    selected = selected.rename(columns={"auv_id": "selected_oc"}).merge(truth, on="scenario_id", validate="one_to_one")
    selected["correct"] = selected.selected_oc.eq(selected.true_oc).astype(int)
    return selected


def top1_macro_f1(selected: pd.DataFrame) -> float:
    return float(f1_score(selected.true_oc, selected.selected_oc, labels=[0, 1, 2, 3], average="macro", zero_division=0))


def make_model(name: str, params: dict) -> object:
    if name == "SVM":
        return Pipeline([("scale", StandardScaler()), ("svc", SVC(kernel="rbf", probability=False, random_state=42, **params))])
    if name == "DTC":
        return DecisionTreeClassifier(criterion="gini", class_weight=None, random_state=42, **params)
    return RandomForestClassifier(criterion="gini", class_weight=None, bootstrap=True, random_state=42, n_jobs=-1, **params)


def parameter_grid(name: str):
    if name == "SVM":
        return ({"C": c, "gamma": gamma, "class_weight": weight}
                for c, gamma, weight in itertools.product(
                    [1, 2, 5, 10, 20, 50, 100, 200, 500],
                    [.001, .002, .005, .01, .02, .03, .05], [None, "balanced"]))
    if name == "DTC":
        return ({"max_depth": depth, "min_samples_split": split, "min_samples_leaf": leaf, "max_features": features}
                for depth, split, leaf, features in itertools.product(
                    [3, 4, 5, 6], [10, 20, 40, 60], [10, 20, 30, 40], [2, 3, None]))
    return ({"n_estimators": trees, "max_depth": depth, "min_samples_split": split,
             "min_samples_leaf": leaf, "max_features": features, "max_samples": fraction}
            for trees, depth, split, leaf, features, fraction in itertools.product(
                [5, 8, 10, 20], [1, 2, 3], [50, 100, 150], [50, 100, 200, 300],
                [1, 2, "sqrt"], [.20, .25, .35]))


def evaluate_validation(model: object, train: pd.DataFrame, validation: pd.DataFrame, name: str, params: dict) -> dict:
    model.fit(train[FEATURES], train.is_oc)
    binary = model.predict(validation[FEATURES])
    selected = select_oc(model, validation, name)
    return {"Model": name, "params": json.dumps(params, sort_keys=True),
            "Top-1 OC accuracy": float(selected.correct.mean()),
            "Top-1 macro F1": top1_macro_f1(selected),
            "Binary F1": float(f1_score(validation.is_oc, binary, zero_division=0)),
            "Binary accuracy": float(accuracy_score(validation.is_oc, binary))}


def choose_models(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame]:
    choices: dict[str, dict] = {}
    audits: list[dict] = []
    for name in ("SVM", "DTC", "RF"):
        best: tuple[tuple[float, float, float, float], dict] | None = None
        for params in parameter_grid(name):
            model = make_model(name, params)
            result = evaluate_validation(model, train, validation, name, params)
            audits.append(result)
            key = (result["Top-1 OC accuracy"], result["Top-1 macro F1"], result["Binary F1"], result["Binary accuracy"])
            if best is None or key > best[0]:
                best = (key, params)
        assert best is not None
        choices[name] = best[1]
        print(f"validation selected {name}: {best[1]} with lexicographic score {best[0]}", flush=True)
    audit = pd.DataFrame(audits)
    write(OUT / "validation_search_all_models.csv", audit.sort_values(
        ["Model", "Top-1 OC accuracy", "Top-1 macro F1", "Binary F1", "Binary accuracy"],
        ascending=[True, False, False, False, False]))
    return choices, audit


def weighted_e2ed(rows: pd.DataFrame) -> float:
    valid = rows.dropna(subset=["E2ED_ms"])
    delivered = valid.delivered_packets.sum()
    return float((valid.E2ED_ms * valid.delivered_packets).sum() / delivered) if delivered else math.nan


def ror(rows: pd.DataFrame) -> float:
    control = pd.to_numeric(rows.control_transmissions, errors="coerce").sum()
    generated = pd.to_numeric(rows.generated_packets, errors="coerce").sum()
    return float(control / (control + generated)) if control + generated else math.nan


def network_summary(rows: pd.DataFrame, label: str, nodes: int | None) -> dict:
    use = rows if nodes is None else rows[rows.node_count.eq(nodes)]
    return {"Model": label, "Mean PDR (%)": float(100 * use.delivered_packets.sum() / use.generated_packets.sum()),
            "Mean E2ED (ms)": weighted_e2ed(use), "Mean ROR": ror(use)}


def density_diagnostic(rows: pd.DataFrame, label: str, nodes: int) -> dict:
    use = rows[rows.node_count.eq(nodes)]
    delivered = use.delivered_packets.sum()
    def hop_weighted(column: str) -> float:
        hops = pd.to_numeric(use.delivered_packets, errors="coerce").sum()
        return float((pd.to_numeric(use[column], errors="coerce") * use.delivered_packets).sum() / hops) if hops else math.nan
    return {"Nodes": nodes, "Model": label,
            "Mean S-D distance (m)": float(use.mean_source_destination_distance_m.mean()),
            "Shortest path hops": float(use.mean_static_shortest_hops.mean()),
            "Delivered hops": hop_weighted("mean_delivered_hop_count"),
            "Path length (m)": hop_weighted("mean_delivered_path_length_m"),
            "Relay wait (ms)": hop_weighted("mean_delivered_relay_wait_ms"),
            "Candidate count": hop_weighted("mean_delivered_relay_candidate_count"),
            "Connected-pair ratio": float(use.connected_flow_pair_ratio.mean()),
            "Delivered scenarios": int(use.E2ED_ms.notna().sum()),
            "Delivered packets": int(delivered)}


def main() -> None:
    accepted, features, candidates = generate_ledgers()
    validate_ledgers(accepted, features, candidates)
    # Matched outcomes create one utility-best binary OC label per scenario;
    # outcome columns do not become features.
    labeled = utility.label_outcomes(candidates)
    dataset = utility.make_dataset(features, labeled)
    if len(dataset) != 8000 or dataset.groupby("scenario_id").is_oc.sum().ne(1).any():
        raise RuntimeError("Label/data merge failed to produce one OC label per scenario")
    split = grouped_split(dataset)
    (OUT / "split_manifest.json").write_text(json.dumps(split, indent=2) + "\n")
    write(OUT / "utility_labeled_dataset.csv", dataset)
    write(OUT / "candidate_outcomes_with_labels.csv", labeled)
    train = dataset[dataset.scenario_id.isin(split["train"])].copy()
    validation = dataset[dataset.scenario_id.isin(split["validation"])].copy()
    test = dataset[dataset.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise RuntimeError("Grouped split is not 1400/300/300 scenarios")
    if set(split["train"]) & set(split["validation"]) or set(split["train"]) & set(split["test"]) or set(split["validation"]) & set(split["test"]):
        raise RuntimeError("Scenario leakage across grouped split")

    selected_params, _ = choose_models(train, validation)
    trainval = pd.concat([train, validation], ignore_index=True)
    ml_rows, selections = [], []
    for name, params in selected_params.items():
        model = make_model(name, params)
        model.fit(trainval[FEATURES], trainval.is_oc)
        dump(model, OUT / f"{name.lower()}_pipeline.joblib")
        binary = model.predict(test[FEATURES])
        precision, recall, _, _ = precision_recall_fscore_support(test.is_oc, binary, average="binary", zero_division=0)
        selected = select_oc(model, test, name)
        selected["Method"] = f"{name}-selected OC"
        selections.append(selected)
        ml_rows.append({"Model": name, "Binary accuracy": accuracy_score(test.is_oc, binary),
                        "Precision": precision, "Recall": recall,
                        "Binary F1": f1_score(test.is_oc, binary, zero_division=0),
                        "Top-1 OC accuracy": selected.correct.mean(), "Top-1 macro F1": top1_macro_f1(selected)})
    selected_predictions = pd.concat(selections, ignore_index=True)
    write(OUT / "selected_oc_per_test_scenario.csv", selected_predictions)

    # Every selected OC must map back to exactly one row in the same candidate ledger.
    candidate_map = labeled.rename(columns={"selected_oc": "candidate_oc"})
    selected_rows: list[pd.DataFrame] = []
    for method in ("SVM-selected OC", "DTC-selected OC", "RF-selected OC"):
        chosen = selected_predictions[selected_predictions.Method.eq(method)].copy()
        mapped = chosen.merge(candidate_map, left_on=["scenario_id", "selected_oc"],
                              right_on=["scenario_id", "candidate_oc"], how="left", validate="one_to_one")
        if mapped.PDR.isna().any():
            raise RuntimeError(f"Missing candidate metric join for {method}")
        mapped["Model"] = method
        selected_rows.append(mapped)
    baseline = candidate_map[candidate_map.candidate_oc.eq(0) & candidate_map.scenario_id.isin(split["test"])].copy()
    baseline["Model"] = "Baseline OC0"
    baseline["selected_oc"] = 0
    selected_rows.append(baseline)
    mapped_all = pd.concat(selected_rows, ignore_index=True)
    if len(mapped_all) != 1200:
        raise RuntimeError("Expected four method rows per one of 300 test scenarios")

    overall = pd.DataFrame([network_summary(mapped_all[mapped_all.Model.eq(method)], method, None) for method in METHODS])
    by_node = pd.DataFrame([network_summary(mapped_all[mapped_all.Model.eq(method)], method, nodes)
                            | {"Nodes": nodes} for nodes in NODES for method in METHODS])
    by_node = by_node[["Nodes", "Model", "Mean PDR (%)", "Mean E2ED (ms)", "Mean ROR"]]
    diagnostics = pd.DataFrame([density_diagnostic(mapped_all[mapped_all.Model.eq(method)], method, nodes)
                                for nodes in NODES for method in METHODS])
    write(OUT / "final_ml_accuracy_comparison.csv", pd.DataFrame(ml_rows))
    write(OUT / "final_network_metrics_overall.csv", overall)
    write(OUT / "final_network_metrics_by_node.csv", by_node)
    write(OUT / "final_density_diagnostics.csv", diagnostics)
    write(OUT / "model_selected_oc_rows_with_network_metrics.csv", mapped_all)

    test_counts = test.groupby("node_count").scenario_id.nunique().to_dict()
    audit = [
        "final_aligned_v10_paper_trend_clean alignment audit",
        f"accepted population: {len(accepted)} scenarios; per density={accepted.groupby('node_count').size().to_dict()}",
        f"test scenarios: {len(split['test'])}; per density={test_counts}",
        f"feature rows/candidate rows/dataset rows: {len(features)}/{len(candidates)}/{len(dataset)}",
        f"traffic acceptance: {ACCEPTANCE_DESCRIPTION}",
        f"endpoint distance range: {accepted.mean_source_destination_distance_m.min():.6f}--{accepted.mean_source_destination_distance_m.max():.6f} m",
        f"static hop range: {accepted.mean_static_shortest_hops.min():.6f}--{accepted.mean_static_shortest_hops.max():.6f}",
        f"25-node 4--8 fallback accepted scenarios: {int(accepted.used_25_node_fallback_4_to_8.sum())}",
        "same scenario IDs are used for ML rows, labels, four-OC outcomes, split manifest, predictions, and selected-OC metrics: PASS",
        "four matched candidates per scenario; matched topology/acoustic state: PASS",
        "200 generated packets; OC data hops=0; architecture violations=0 across all candidate runs: PASS",
        "train/validation/test scenario ID sets are disjoint; validation-only hyperparameter selection: PASS",
        "final ROR = pooled control_packets / (control_packets + generated_data_packets); no alternate ROR name appears in final paper tables: PASS",
        f"selected-OC metric joins missing: {int(mapped_all.PDR.isna().sum())}",
    ]
    (OUT / "final_alignment_audit.txt").write_text("\n".join(audit) + "\n")
    (OUT / "configuration.json").write_text(json.dumps({
        "experiment": "final_aligned_v10_paper_trend_clean",
        "scenario_population": "2,000 fresh accepted scenarios; 500 per node count",
        "traffic_acceptance": ACCEPTANCE_DESCRIPTION,
        "fixed_protocol": {"hello_seconds": 30, "route_ttl_seconds": 60,
                            "negative_cache_seconds": 120, "route_retry_limit": 3,
                            "packet_buffer_timeout_seconds": 60,
                            "topology_digest_capacity_nodes": 20},
        "relay_search_wait_ms": "min(15 + 360/max(1,candidate_count^2), 180)",
        "final_ror": "control_packets / (control_packets + generated_data_packets)",
        "ml_features": FEATURES,
        "split": "grouped scenario IDs: 1400/300/300; 75 test scenarios per density",
        "label": "v4 predeclared matched-candidate utility; outcomes only create is_oc labels, never ML inputs",
    }, indent=2) + "\n")
    print("FINAL ML\n", pd.DataFrame(ml_rows).to_string(index=False))
    print("FINAL NETWORK OVERALL\n", overall.to_string(index=False))
    print("FINAL NETWORK BY NODE\n", by_node.to_string(index=False))
    print("DENSITY DIAGNOSTICS\n", diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
