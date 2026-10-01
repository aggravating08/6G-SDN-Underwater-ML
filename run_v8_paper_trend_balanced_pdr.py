#!/usr/bin/env python3
"""v8 balanced moderate-long-distance, fresh ML and network experiment."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import run_reference_paper_matching_trend_v4_long_distance_ml as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/underwater_rebuild/current/v8_paper_trend_balanced_pdr"
RUNS = OUT / "candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
NODES = (25, 50, 75, 100)
# Fresh topology/traffic population, distinct from v4/v5/v6/v7.
SPECS = {25: (0, 4_000_029), 50: (500, 5_000_029), 75: (1000, 6_000_029), 100: (1500, 7_000_029)}
METHODS = ("SVM", "DTC", "RF")
FEATURES = base.FEATURES


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NA", "NaN", ""])


def write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def invoke(args: list[str], output: Path) -> None:
    if output.exists():
        # Batch output must be complete: header plus 500 scenario rows.
        if sum(1 for _ in output.open()) == 501:
            return
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(EXE), *args, f"--output={output}"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def fixed_args(nodes: int, sid: int, seed: int, oc: int) -> list[str]:
    return ["--mode=run", f"--nodeCount={nodes}", f"--scenarioId={sid}",
            f"--scenarioSeed={seed}", f"--selectedOc={oc}", "--runs=500",
            "--fixedControlPolicy=true", "--helloSeconds=30", "--routeTtlSeconds=60",
            "--negativeRouteTtlSeconds=120", "--referenceUpdateAccounting=true",
            "--aggregatedTopologyDigestAccounting=true", "--topologyDigestCapacityNodes=25",
            "--bufferRouteDiscovery=true", "--routeDiscoveryRetryLimit=3",
            "--packetBufferTimeoutSeconds=60", "--balancedDistanceFlows=true",
            "--relaySearchDelay=true", "--relaySearchBaseMs=5",
            "--relaySearchScaleMs=180", "--relaySearchMaxMs=80"]


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_files, candidate_files = [], []
    for nodes, (sid, seed) in SPECS.items():
        feature = OUT / "topology_features" / f"n{nodes}_features.csv"
        invoke(["--mode=topology-features", f"--nodeCount={nodes}", f"--scenarioId={sid}",
                f"--baseSeed={seed}", "--runs=500", "--balancedDistanceFlows=true"], feature)
        feature_files.append(feature)
        for oc in range(4):
            candidate = RUNS / f"n{nodes}_oc{oc}.csv"
            invoke(fixed_args(nodes, sid, seed, oc), candidate)
            candidate_files.append(candidate)
    features = pd.concat([read(p) for p in feature_files], ignore_index=True)
    candidates = pd.concat([read(p) for p in candidate_files], ignore_index=True)
    write(OUT / "balanced_candidate_features.csv", features)
    write(OUT / "balanced_matched_four_oc_outcomes.csv", candidates)
    return features, candidates


def validate(features: pd.DataFrame, candidates: pd.DataFrame) -> None:
    if len(features) != 8000 or len(candidates) != 8000:
        raise RuntimeError("v8 requires 8,000 feature and candidate rows")
    for sid, group in candidates.groupby("scenario_id"):
        if len(group) != 4 or set(group.selected_oc) != {0, 1, 2, 3}:
            raise RuntimeError(f"Missing matched OC candidate at scenario {sid}")
        if group.topology_hash.nunique() != 1 or group.acoustic_state_hash.nunique() != 1:
            raise RuntimeError(f"Unmatched topology/acoustic state at scenario {sid}")
        if not (group.generated_packets == 200).all() or not (group.OC_data_hops == 0).all() or not (group.architecture_violations == 0).all():
            raise RuntimeError(f"Architecture/counter invariant failure at scenario {sid}")
        if group.mean_source_destination_distance_m.nunique(dropna=False) != 1:
            raise RuntimeError(f"Flow-distance mismatch at scenario {sid}")
    d = candidates.mean_source_destination_distance_m
    if (d < 300 - 1e-6).any() or (d > 450 + 1e-6).any():
        raise RuntimeError("v8 traffic violates the 300--450 m source/destination rule")


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def weighted_e2ed(rows: pd.DataFrame) -> float:
    valid = rows.dropna(subset=["E2ED_ms"])
    delivered = valid.delivered_packets.sum()
    return float((valid.E2ED_ms * valid.delivered_packets).sum() / delivered) if delivered else math.nan


def network_summary(rows: pd.DataFrame, method: str, nodes: int | str) -> dict:
    use = rows[rows.Method.eq(method)]
    if nodes != "All": use = use[use.node_count.eq(nodes)]
    return {"Nodes": nodes, "Model": method, "Scenarios": len(use),
            "PDR %": float(100 * use.delivered_packets.sum() / use.generated_packets.sum()),
            "E2ED ms": weighted_e2ed(use), "ROR offered": finite_mean(use.ROR_generated),
            "ROR transmission": finite_mean(use.ROR_total)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); RUNS.mkdir(exist_ok=True)
    features, candidates = generate()
    validate(features, candidates)
    labeled = base.label_outcomes(candidates)
    dataset = base.make_dataset(features, labeled)
    write(OUT / "balanced_utility_labeled_dataset.csv", dataset)
    write(OUT / "balanced_candidate_outcomes_with_labels.csv", labeled)
    manifest = base.split_scenarios(dataset)
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    train = dataset[dataset.scenario_id.isin(manifest["train"])].copy()
    validation = dataset[dataset.scenario_id.isin(manifest["validation"])].copy()
    test = dataset[dataset.scenario_id.isin(manifest["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise RuntimeError("Expected 1400/300/300 grouped-scenario split")
    models, audit = base.choose_models(train, validation)
    write(OUT / "validation_model_search.csv", audit.sort_values(["Model", "objective"], ascending=[True, False]))
    trainval = pd.concat([train, validation], ignore_index=True)
    ml_rows, predictions = [], []
    for name, (model, params) in models.items():
        model.fit(trainval[FEATURES], trainval.is_oc)
        dump(model, OUT / f"{name.lower()}_pipeline.joblib")
        binary = model.predict(test[FEATURES])
        precision, recall, _, _ = precision_recall_fscore_support(test.is_oc, binary, average="binary", zero_division=0)
        chosen = base.scenario_predictions(model, test, name)
        chosen["Model"] = name; predictions.append(chosen)
        ml_rows.append({"Model": name, "Binary accuracy": accuracy_score(test.is_oc,binary),
                        "Precision": precision, "Recall": recall, "Binary F1": f1_score(test.is_oc,binary,zero_division=0),
                        "Top-1 OC accuracy": chosen.correct.mean(), "Top-1 macro F1": base.top1_macro_f1(chosen),
                        "Selected hyperparameters": json.dumps(params,sort_keys=True)})
    predicted = pd.concat(predictions, ignore_index=True)
    write(OUT / "test_model_oc_predictions.csv", predicted)
    # Join each frozen test prediction to the matched candidate selected by it.
    candidate_map = labeled.rename(columns={"selected_oc":"auv_id"}).drop(columns=["node_count","is_oc"])
    mapped_parts=[]
    for name in METHODS:
        picks=predicted[predicted.Model.eq(name)]
        mapped=picks.merge(candidate_map,left_on=["scenario_id","predicted_oc"],right_on=["scenario_id","auv_id"],how="left",validate="one_to_one")
        if mapped.PDR.isna().any(): raise RuntimeError(f"Missing candidate mapping for {name}")
        mapped["Method"]=name; mapped_parts.append(mapped)
    mapped=pd.concat(mapped_parts,ignore_index=True)
    write(OUT / "model_selected_oc_rows_with_network_metrics.csv",mapped)
    ml=pd.DataFrame(ml_rows)
    bynode=pd.DataFrame([network_summary(mapped,m,n) for n in NODES for m in METHODS])
    overall=pd.DataFrame([network_summary(mapped,m,"All") for m in METHODS])
    write(OUT / "final_ml_metrics.csv",ml); write(OUT / "final_network_metrics_by_node.csv",bynode); write(OUT / "final_network_metrics.csv",overall)
    svm=mapped[mapped.Method.eq("SVM")]
    trend=[]
    for n in NODES:
        use=svm[svm.node_count.eq(n)]
        base_row=network_summary(mapped,"SVM",n)
        base_row.update({"Mean S-D distance m":finite_mean(use.mean_source_destination_distance_m),
                         "Connected-pair ratio":finite_mean(use.connected_flow_pair_ratio),
                         "Delivered hop count":finite_mean(use.mean_delivered_hop_count),
                         "Delivered path length m":finite_mean(use.mean_delivered_path_length_m),
                         "Relay wait ms":finite_mean(use.mean_delivered_relay_wait_ms),
                         "Mean candidates":finite_mean(use.mean_delivered_relay_candidate_count),
                         "Delivered scenario count":int(use.E2ED_ms.notna().sum())})
        trend.append(base_row)
    trend_frame=pd.DataFrame(trend); write(OUT / "svm_density_trend.csv",trend_frame)
    pdr=trend_frame["PDR %"].tolist(); e2e=trend_frame["E2ED ms"].tolist(); ror=trend_frame["ROR offered"].tolist()
    gates=pd.DataFrame([
        {"Metric":"PDR %",**dict(zip(map(str,NODES),pdr)),"Expected trend":"increases then saturates (0.5 pp tolerance)","Pass/Fail":"PASS" if all(b>=a-.5 for a,b in zip(pdr,pdr[1:])) else "FAIL"},
        {"Metric":"E2ED ms",**dict(zip(map(str,NODES),e2e)),"Expected trend":"generally decreases","Pass/Fail":"PASS" if all(b<=a for a,b in zip(e2e,e2e[1:])) else "FAIL"},
        {"Metric":"ROR offered",**dict(zip(map(str,NODES),ror)),"Expected trend":"increases","Pass/Fail":"PASS" if all(b>=a for a,b in zip(ror,ror[1:])) else "FAIL"},
        {"Metric":"25-node PDR","25":pdr[0],"50":"","75":"","100":"","Expected trend":">=20%","Pass/Fail":"PASS" if pdr[0]>=20 else "FAIL"},
    ]); write(OUT / "trend_gate.csv",gates)
    (OUT / "configuration.json").write_text(json.dumps({
        "experiment":"v8_paper_trend_balanced_pdr", "population":"fresh 2,000 scenarios; 500/density",
        "traffic":"10 flows x 20 packets; each pair 300--450 m; static connected pairs preferred without forcing connectivity",
        "protocol":{"hello_update_interval_seconds":30,"route_ttl_seconds":60,"negative_cache_seconds":120,"route_retry_limit":3,"buffer_timeout_seconds":60,"topology_digest_capacity_nodes":25},
        "relay_search_delay":"min(5 + 180/max(1, usable positive-progress neighbours), 80) ms",
        "ror_offered":"control/(control+200 generated source packets)","ror_transmission":"control/(control+realized data hops)",
        "models":"fresh validation-tuned SVM/DTC/RF; grouped 1400/300/300 scenario split", "features":FEATURES,
    },indent=2)+"\n")
    print("V8 ML TABLE\n",ml.to_string(index=False)); print("V8 NETWORK BY NODE\n",bynode.to_string(index=False)); print("V8 SVM TREND\n",trend_frame.to_string(index=False)); print("V8 GATES\n",gates.to_string(index=False))


if __name__ == "__main__": main()
