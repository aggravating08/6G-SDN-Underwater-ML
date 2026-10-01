#!/usr/bin/env python3
"""Held-out evaluation of density-adaptive HELLO and no-path suppression.

The selected OC identities are read from the completed v2 ML output.  This
script does not load, fit, tune, or otherwise change SVM/DTC/RF.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/underwater_rebuild/current/final_hello60_svm_first_ordered_gap_v2/model_selected_oc_rows_with_network_metrics.csv"
OUT = ROOT / "results/underwater_rebuild/current/final_density_adaptive_ror_monotonic"
RUNS = OUT / "candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
MODELS = ("SVM", "DTC", "RF")
METHODS = ("SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]) if rows else [], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def n(row: dict, column: str) -> float:
    value = row.get(column, "")
    return math.nan if value in ("", "NA", "NaN") else float(value)


def mean(rows: list[dict], column: str) -> float:
    values = [n(row, column) for row in rows]
    values = [x for x in values if math.isfinite(x)]
    return sum(values) / len(values) if values else math.nan


def run(key: tuple[int, int, int, int]) -> dict[str, str]:
    nodes, scenario_id, seed, oc = key
    output = RUNS / f"n{nodes}_sid{scenario_id}_seed{seed}_oc{oc}.csv"
    if not output.exists():
        subprocess.run([str(EXE), "--mode=run", f"--nodeCount={nodes}", f"--scenarioId={scenario_id}",
                        f"--scenarioSeed={seed}", f"--selectedOc={oc}", f"--output={output}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    records = read(output)
    if len(records) != 1:
        raise RuntimeError(f"Expected exactly one candidate result: {output}")
    return records[0]


def assert_invariants(rows: list[dict]) -> None:
    invalid = [r for r in rows if int(r["generated_packets"]) != 200
               or int(r["OC_data_hops"]) != 0 or int(r["architecture_violations"]) != 0]
    if invalid:
        raise RuntimeError(f"Counter/architecture invariant failed in {len(invalid)} rows")
    seen: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for r in rows:
        seen[(r["node_count"], r["scenario_id"], r["scenario_seed"])].add(
            (r["topology_hash"], r["acoustic_state_hash"]))
    if any(len(hashes) != 1 for hashes in seen.values()):
        raise RuntimeError("Matched topology/acoustic hash invariant failed")


def summary(rows: list[dict], method: str, nodes: int | None) -> dict:
    use = [r for r in rows if r["Method"] == method and (nodes is None or int(r["node_count"]) == nodes)]
    return {"Nodes": "All" if nodes is None else nodes, "Method": method,
            "PDR %": mean(use, "PDR"), "E2ED ms": mean(use, "E2ED_ms"),
            "ROR total": mean(use, "ROR_total"), "ROR reactive": mean(use, "ROR_reactive"),
            "HELLO packets": mean(use, "hello_tx"), "MC fallback": mean(use, "mc_fallbacks"),
            "Failed attempts": mean(use, "failed_route_attempts"),
            "Negative hits": mean(use, "negative_cache_hits"), "Scenarios": len(use)}


def main() -> None:
    if not SOURCE.exists() or not EXE.exists():
        raise RuntimeError("Frozen v2 selected-OC output or compiled simulator is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    selected_source = read(SOURCE)
    if len(selected_source) != 900:
        raise RuntimeError("Expected 900 frozen v2 ML selections")
    selected: dict[tuple[str, int], dict[str, str]] = {}
    need: set[tuple[int, int, int, int]] = set()
    for row in selected_source:
        model = row["Model"]
        if model not in MODELS:
            raise RuntimeError(f"Unexpected model {model}")
        identity = (model, int(row["scenario_id"]))
        if identity in selected:
            raise RuntimeError(f"Duplicate selected-OC row {identity}")
        selected[identity] = row
        base = (int(row["node_count"]), int(row["scenario_id"]), int(row["scenario_seed"]))
        need.add((*base, int(row["predicted_oc"])))
        need.add((*base, 0))
    print(f"Running/reusing {len(need)} unique frozen-selection candidate trials", flush=True)
    candidate_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(run, key): key for key in sorted(need)}
        for done, job in enumerate(as_completed(jobs), start=1):
            candidate_rows.append(job.result())
            if done % 50 == 0 or done == len(jobs):
                print(f"Completed {done}/{len(jobs)}", flush=True)
    candidate_rows.sort(key=lambda r: (int(r["node_count"]), int(r["scenario_id"]), int(r["selected_oc"])))
    assert_invariants(candidate_rows)
    write(OUT / "raw_unique_candidate_runs.csv", candidate_rows)
    index = {(int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"]), int(r["selected_oc"])): r
             for r in candidate_rows}
    mapped: list[dict] = []
    for (model, sid), source in selected.items():
        key = (int(source["node_count"]), sid, int(source["scenario_seed"]), int(source["predicted_oc"]))
        r = dict(index[key])
        r.update({"Method": f"{model}-selected OC", "model": model, "predicted_oc": source["predicted_oc"],
                  "true_oc": source["true_oc"]})
        mapped.append(r)
    scenario_ids = {(int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"])) for r in mapped}
    for nodes, sid, seed in scenario_ids:
        r = dict(index[(nodes, sid, seed, 0)])
        r.update({"Method": "Baseline OC0", "model": "Baseline", "predicted_oc": 0, "true_oc": ""})
        mapped.append(r)
    mapped.sort(key=lambda r: (int(r["node_count"]), int(r["scenario_id"]), METHODS.index(r["Method"])))
    if len(mapped) != 1200:
        raise RuntimeError(f"Expected 1200 model/baseline rows, got {len(mapped)}")
    write(OUT / "model_selected_oc_rows_with_network_metrics.csv", mapped)
    by_node = [summary(mapped, method, nodes) for nodes in (25, 50, 75, 100) for method in METHODS]
    overall = [summary(mapped, method, None) for method in METHODS]
    svm = [r for r in by_node if r["Method"] == "SVM-selected OC"]
    previous = None
    monotonic = []
    for row in svm:
        monotonic.append({"Nodes": row["Nodes"], "SVM ROR total": row["ROR total"],
                          "Change from previous node count": math.nan if previous is None else row["ROR total"]-previous})
        previous = row["ROR total"]
    write(OUT / "final_metrics_by_node.csv", by_node)
    write(OUT / "final_metrics_overall.csv", overall)
    write(OUT / "svm_monotonic_ror_check.csv", monotonic)
    (OUT / "configuration.json").write_text(json.dumps({
        "base_hello_seconds": 60, "route_ttl_seconds": 120,
        "hello_seconds_by_node_count": {"25":60, "50":75, "75":90, "100":120},
        "negative_cache_ttl_seconds_by_node_count": {"25":60, "50":45, "75":30, "100":30},
        "dense_delta_update_mode": "75/100: initial sensor topology discovery then suppress unchanged duplicate sensor updates",
        "model_retraining": False, "unique_candidate_trials": len(candidate_rows),
        "invariants": "generated_packets=200; OC_data_hops=0; architecture_violations=0; matched topology/acoustic hashes",
        "frozen_selection_source": str(SOURCE.relative_to(ROOT)),
    }, indent=2) + "\n")
    print("\nBY NODE")
    for row in by_node: print(row)
    print("\nOVERALL")
    for row in overall: print(row)
    print("\nSVM ROR CHECK")
    for row in monotonic: print(row)


if __name__ == "__main__":
    main()
