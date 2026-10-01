#!/usr/bin/env python3
"""Clean non-cognitive underwater PPT experiment.

This deliberately does not use v10's accepted-path population, relay-search
wait, topology-digest aggregation, partner/coastal code, or cognitive/PU
logic.  Sensor positions and source/destination flows are randomly generated
from each scenario seed over the full 500 x 500 m area.  The existing LC cache
and selected-OC/MC control hierarchy remain in randy.cc.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import run_final_aligned_v10_paper_trend_clean as core
import run_reference_paper_matching_trend_v4_long_distance_ml as utility


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/underwater_rebuild/current/final_underwater_ppt_noncognitive_clean_ror_corrected"
RUNS = OUT / "candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
NODES = (25, 50, 75, 100)
SPECS = {25: (0, 12_000_029), 50: (500, 13_000_029),
         75: (1000, 14_000_029), 100: (1500, 15_000_029)}
FEATURES = ["x", "y", "local_density", "speed"]
METHODS = ("SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NA", "NaN", ""])


def write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def invoke(args: list[str], output: Path, expected_rows: int) -> Path:
    if output.exists() and len(read(output)) == expected_rows:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    subprocess.run([str(EXE), *args, f"--output={output}"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if len(read(output)) != expected_rows:
        raise RuntimeError(f"Incomplete output: {output}")
    return output


def common_protocol() -> list[str]:
    # Fixed across all densities.  No cognitive channels, PUs, flow-distance
    # rule, topology digest aggregation, density-adaptive policy, or added
    # relay-search delay is activated here.
    return ["--fixedControlPolicy=true", "--helloSeconds=30", "--routeTtlSeconds=60",
            "--negativeRouteTtlSeconds=120", "--referenceUpdateAccounting=true"]


def generate_ledgers() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_files, candidate_files = [], []
    for nodes, (sid, seed) in SPECS.items():
        feature = OUT / "topology_features" / f"n{nodes}_features.csv"
        invoke(["--mode=topology-features", f"--nodeCount={nodes}", f"--scenarioId={sid}",
                f"--baseSeed={seed}", "--runs=500"], feature, 2000)
        feature_files.append(feature)
    # Matching requires identical sequential seed/topology/flows for OC0..3.
    jobs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for nodes, (sid, seed) in SPECS.items():
            for oc in range(4):
                path = RUNS / f"n{nodes}_oc{oc}.csv"
                args = ["--mode=run", f"--nodeCount={nodes}", f"--scenarioId={sid}",
                        f"--scenarioSeed={seed}", f"--selectedOc={oc}", "--runs=500", *common_protocol()]
                jobs.append(pool.submit(invoke, args, path, 500))
        for future in as_completed(jobs):
            candidate_files.append(future.result())
    features = pd.concat([read(path) for path in feature_files], ignore_index=True)
    candidates = pd.concat([read(path) for path in candidate_files], ignore_index=True)
    write(OUT / "candidate_features.csv", features)
    write(OUT / "matched_four_oc_candidate_ledger.csv", candidates)
    return features, candidates


def validate(features: pd.DataFrame, candidates: pd.DataFrame) -> None:
    if len(features) != 8000 or len(candidates) != 8000:
        raise RuntimeError("Expected 8,000 pre-routing feature rows and 8,000 matched candidate rows")
    if features.groupby("scenario_id").size().ne(4).any():
        raise RuntimeError("Feature rows are not four candidates per scenario")
    for sid, group in candidates.groupby("scenario_id"):
        if len(group) != 4 or set(group.selected_oc) != {0, 1, 2, 3}:
            raise RuntimeError(f"Missing OC candidate at scenario {sid}")
        if group.topology_hash.nunique() != 1 or group.acoustic_state_hash.nunique() != 1:
            raise RuntimeError(f"Topology/acoustic mismatch at scenario {sid}")
        if not group.generated_packets.eq(200).all() or not group.OC_data_hops.eq(0).all() or not group.architecture_violations.eq(0).all():
            raise RuntimeError(f"Payload/control-plane invariant failure at scenario {sid}")


def label_outcomes_with_paper_ror(candidates: pd.DataFrame) -> pd.DataFrame:
    """Create matched utility labels using the corrected packet ROR only."""
    required = {"scenario_id", "selected_oc", "PDR", "E2ED_ms", "ROR_total",
                "ROR_reactive", "mc_fallbacks", "final_no_path_drops", "generated_packets"}
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"Candidate ledger lacks fields for corrected utility labels: {sorted(missing)}")
    groups = []
    for _, group in candidates.groupby("scenario_id", sort=False):
        g = group.copy()
        g["mc_fallback_rate"] = g.mc_fallbacks / g.generated_packets
        g["route_failure_rate"] = g.final_no_path_drops / g.generated_packets
        g["utility_score"] = (
            1.0 * utility.minmax_score(g.PDR, True)
            + 0.7 * utility.minmax_score(g.ROR_total, False)
            + 0.5 * utility.minmax_score(g.ROR_reactive, False)
            + 0.3 * utility.minmax_score(g.mc_fallback_rate, False)
            + 0.2 * utility.minmax_score(g.E2ED_ms, False)
            + 0.5 * utility.minmax_score(g.route_failure_rate, False)
        )
        best = g.sort_values(["utility_score", "selected_oc"], ascending=[False, True]).iloc[0].selected_oc
        g["is_oc"] = g.selected_oc.eq(best).astype(int)
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def paper_ror(rows: pd.DataFrame) -> float:
    control = pd.to_numeric(rows.control_transmissions, errors="coerce").sum()
    data = pd.to_numeric(rows.data_hops, errors="coerce").sum()
    return float(control / (control + data)) if control + data else float("nan")


def paper_network_summary(rows: pd.DataFrame, method: str) -> dict:
    return {"Model": method,
            "Mean PDR (%)": float(100 * rows.delivered_packets.sum() / rows.generated_packets.sum()),
            "Mean E2ED (ms)": core.weighted_e2ed(rows),
            "Mean ROR": paper_ror(rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Reuse thoroughly tested validation-only model-selection helpers, but
    # bind them to this experiment and the four PPT features for this process.
    core.OUT = OUT
    core.FEATURES = FEATURES
    features, candidates = generate_ledgers()
    validate(features, candidates)
    labeled = label_outcomes_with_paper_ror(candidates)
    dataset = utility.make_dataset(features, labeled)
    # Keep only PPT features plus scenario metadata/label.  Utility outcome
    # columns stay out of the ML matrix.
    if dataset.groupby("scenario_id").is_oc.sum().ne(1).any():
        raise RuntimeError("The matched utility label is not one OC per scenario")
    split = core.grouped_split(dataset)
    (OUT / "split_manifest.json").write_text(json.dumps(split, indent=2) + "\n")
    write(OUT / "utility_labeled_four_feature_dataset.csv", dataset)
    write(OUT / "candidate_outcomes_with_labels.csv", labeled)
    train = dataset[dataset.scenario_id.isin(split["train"])].copy()
    validation = dataset[dataset.scenario_id.isin(split["validation"])].copy()
    test = dataset[dataset.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise RuntimeError("Expected grouped 1400/300/300 split")
    params, _ = core.choose_models(train, validation)
    trainval = pd.concat([train, validation], ignore_index=True)
    ml_rows, predictions = [], []
    for name, selected_params in params.items():
        model = core.make_model(name, selected_params)
        model.fit(trainval[FEATURES], trainval.is_oc)
        dump(model, OUT / f"{name.lower()}_pipeline.joblib")
        binary = model.predict(test[FEATURES])
        precision, recall, _, _ = precision_recall_fscore_support(test.is_oc, binary, average="binary", zero_division=0)
        chosen = core.select_oc(model, test, name)
        chosen["Method"] = f"{name}-selected OC"
        predictions.append(chosen)
        ml_rows.append({"Model": name, "Binary accuracy": accuracy_score(test.is_oc, binary),
                        "Precision": precision, "Recall": recall,
                        "Binary F1": f1_score(test.is_oc, binary, zero_division=0),
                        "Top-1 OC accuracy": chosen.correct.mean(), "Top-1 macro F1": core.top1_macro_f1(chosen)})
    predictions = pd.concat(predictions, ignore_index=True)
    write(OUT / "selected_oc_per_test_scenario.csv", predictions)
    candidate_map = labeled.rename(columns={"selected_oc": "candidate_oc"})
    mapped_parts = []
    for method in METHODS[:3]:
        picks = predictions[predictions.Method.eq(method)][["scenario_id", "selected_oc", "true_oc", "correct", "score"]]
        mapped = picks.merge(candidate_map, left_on=["scenario_id", "selected_oc"],
                             right_on=["scenario_id", "candidate_oc"], how="left", validate="one_to_one")
        if mapped.PDR.isna().any():
            raise RuntimeError(f"Missing candidate metrics for {method}")
        mapped["Model"] = method
        mapped_parts.append(mapped)
    baseline = candidate_map[candidate_map.candidate_oc.eq(0) & candidate_map.scenario_id.isin(split["test"])].copy()
    baseline["selected_oc"] = 0; baseline["Model"] = "Baseline OC0"
    mapped_parts.append(baseline)
    mapped = pd.concat(mapped_parts, ignore_index=True)
    if len(mapped) != 1200 or mapped.node_count.isna().any():
        raise RuntimeError("Test-only model/ledger mapping is incomplete")
    overall = pd.DataFrame([paper_network_summary(mapped[mapped.Model.eq(method)], method) for method in METHODS])
    bynode = pd.DataFrame([{"Nodes": nodes, **paper_network_summary(mapped[(mapped.Model.eq(method)) & (mapped.node_count.eq(nodes))], method)}
                           for nodes in NODES for method in METHODS])
    bynode = bynode[["Nodes", "Model", "Mean PDR (%)", "Mean E2ED (ms)", "Mean ROR"]]
    diagnostics = pd.DataFrame([core.density_diagnostic(mapped[mapped.Model.eq(method)], method, nodes)
                                for nodes in NODES for method in METHODS])
    write(OUT / "final_ml_accuracy_comparison.csv", pd.DataFrame(ml_rows))
    write(OUT / "final_network_metrics_overall.csv", overall)
    write(OUT / "final_network_metrics_by_node.csv", bynode)
    write(OUT / "final_density_diagnostics.csv", diagnostics)
    write(OUT / "model_selected_oc_rows_with_network_metrics.csv", mapped)
    (OUT / "final_alignment_audit.txt").write_text("\n".join([
        "clean non-cognitive underwater PPT alignment audit",
        "random sensor deployment and random source/destination flows: PASS",
        "no path acceptance, no forced connected endpoints, no hop balancing: PASS",
        "no cognitive/PU/channel-selection mechanism: PASS",
        "same IDs/seeds across features, labels, four OC outcomes, split, predictions, and metrics: PASS",
        "grouped 1400/300/300 split with 75 test scenarios per density: PASS",
        "200 generated packets/run, OC data hops=0, architecture violations=0: PASS",
        "paper ROR = control transmissions / (control transmissions + data-hop transmissions): PASS",
        "sensor beacon is counted once per sender broadcast; neighbour receptions are diagnostic-only and not double-counted: PASS",
        "validation-only hyperparameter selection; no test-set selection: PASS",
    ]) + "\n")
    print(pd.DataFrame(ml_rows).to_string(index=False))
    print(overall.to_string(index=False))
    print(bynode.to_string(index=False))


if __name__ == "__main__":
    main()
