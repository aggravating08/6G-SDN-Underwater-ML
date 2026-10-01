#!/usr/bin/env python3
"""Held-out fixed-control, reference-style trend evaluation.

This is deliberately separate from the density-adaptive policy experiment.
It reuses frozen model-selected OC identities, but every candidate trial is
executed with the same 30-second HELLO and 120-second no-path cache policy at
all four densities.  No model is fitted, tuned, or relabelled here.
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
OUT = ROOT / "results/underwater_rebuild/current/final_reference_style_trend_fixed_control"
RUNS = OUT / "candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
MODELS = ("SVM", "DTC", "RF")
METHODS = ("SVM", "DTC", "RF", "Baseline OC0")
NODES = (25, 50, 75, 100)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], column: str) -> float:
    value = row.get(column, "")
    return math.nan if value in ("", "NA", "NaN") else float(value)


def mean_ci(rows: list[dict[str, str]], column: str) -> tuple[float, float]:
    values = [number(row, column) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan
    average = sum(values) / len(values)
    if len(values) < 2:
        return average, math.nan
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return average, 1.96 * math.sqrt(variance / len(values))


def run_candidate(key: tuple[int, int, int, int]) -> dict[str, str]:
    nodes, scenario_id, seed, oc = key
    output = RUNS / f"n{nodes}_sid{scenario_id}_seed{seed}_oc{oc}.csv"
    if not output.exists():
        subprocess.run(
            [str(EXE), "--mode=run", f"--nodeCount={nodes}", f"--scenarioId={scenario_id}",
             f"--scenarioSeed={seed}", f"--selectedOc={oc}",
             "--fixedControlPolicy=true", "--helloSeconds=30",
             "--negativeRouteTtlSeconds=120", f"--output={output}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
    rows = read(output)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one candidate run in {output}")
    return rows[0]


def assert_invariants(rows: list[dict[str, str]]) -> None:
    invalid = [row for row in rows if int(row["generated_packets"]) != 200
               or int(row["OC_data_hops"]) != 0
               or int(row["architecture_violations"]) != 0]
    if invalid:
        raise RuntimeError(f"Packet/control-plane invariant failed in {len(invalid)} runs")
    hashes: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        hashes[(row["node_count"], row["scenario_id"], row["scenario_seed"])].add(
            (row["topology_hash"], row["acoustic_state_hash"]))
    if any(len(values) != 1 for values in hashes.values()):
        raise RuntimeError("Matched topology/acoustic-state hash check failed")


def summary(mapped: list[dict[str, str]], method: str, nodes: int | None) -> dict:
    rows = [row for row in mapped if row["Method"] == method
            and (nodes is None or int(row["node_count"]) == nodes)]
    pdr, pdr_ci = mean_ci(rows, "PDR")
    delay, delay_ci = mean_ci(rows, "E2ED_ms")
    ror, ror_ci = mean_ci(rows, "ROR_total")
    reactive, reactive_ci = mean_ci(rows, "ROR_reactive")
    def simple(column: str) -> float:
        values = [number(row, column) for row in rows]
        return sum(values) / len(values) if values else math.nan
    return {
        "Nodes": "All" if nodes is None else nodes, "Method": method,
        "PDR %": pdr, "PDR 95% CI": pdr_ci,
        "E2ED ms": delay, "E2ED 95% CI": delay_ci,
        "ROR total": ror, "ROR total 95% CI": ror_ci,
        "ROR reactive": reactive, "ROR reactive 95% CI": reactive_ci,
        "HELLO packets": simple("hello_tx"),
        "Route requests": simple("route_request_tx"),
        "MC fallbacks": simple("mc_fallbacks"),
        "Failed attempts": simple("failed_route_attempts"),
        "Scenarios": len(rows),
    }


def trend_check(by_node: list[dict]) -> list[dict]:
    svm = {int(row["Nodes"]): row for row in by_node if row["Method"] == "SVM"}
    metrics = (("PDR %", "increasing"), ("E2ED ms", "decreasing"), ("ROR total", "increasing"))
    output = []
    for metric, expected in metrics:
        values = [svm[n][metric] for n in NODES]
        if expected == "increasing":
            passed = all(right >= left for left, right in zip(values, values[1:]))
        else:
            passed = all(right <= left for left, right in zip(values, values[1:]))
        output.append({"Metric": metric, **{str(n): value for n, value in zip(NODES, values)},
                       "Expected direction": expected, "Pass/Fail": "PASS" if passed else "FAIL"})
    return output


def main() -> None:
    if not SOURCE.exists() or not EXE.exists():
        raise RuntimeError("Frozen selected-OC file or compiled randy executable is missing")
    selections = read(SOURCE)
    if len(selections) != 900:
        raise RuntimeError("Expected 900 frozen held-out model selections")
    selected: dict[tuple[str, int], dict[str, str]] = {}
    required: set[tuple[int, int, int, int]] = set()
    for row in selections:
        if row["Model"] not in MODELS:
            raise RuntimeError(f"Unexpected model: {row['Model']}")
        identity = (row["Model"], int(row["scenario_id"]))
        if identity in selected:
            raise RuntimeError(f"Duplicate selected OC: {identity}")
        selected[identity] = row
        base = (int(row["node_count"]), int(row["scenario_id"]), int(row["scenario_seed"]))
        required.add((*base, int(row["predicted_oc"])))
        required.add((*base, 0))

    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    print(f"Running/reusing {len(required)} fixed-control candidate trials", flush=True)
    candidates: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {executor.submit(run_candidate, key): key for key in sorted(required)}
        for completed, future in enumerate(as_completed(pending), start=1):
            candidates.append(future.result())
            if completed % 50 == 0 or completed == len(pending):
                print(f"Completed {completed}/{len(pending)}", flush=True)
    candidates.sort(key=lambda row: (int(row["node_count"]), int(row["scenario_id"]), int(row["selected_oc"])))
    assert_invariants(candidates)
    write(OUT / "raw_unique_candidate_runs.csv", candidates)
    index = {(int(row["node_count"]), int(row["scenario_id"]), int(row["scenario_seed"]), int(row["selected_oc"])): row
             for row in candidates}

    mapped: list[dict[str, str]] = []
    for (model, scenario_id), source in selected.items():
        key = (int(source["node_count"]), scenario_id, int(source["scenario_seed"]), int(source["predicted_oc"]))
        row = dict(index[key])
        row.update({"Method": model, "model": model, "predicted_oc": source["predicted_oc"],
                    "true_oc": source.get("true_oc", "")})
        mapped.append(row)
    scenarios = {(int(row["node_count"]), int(row["scenario_id"]), int(row["scenario_seed"])) for row in mapped}
    for nodes, scenario_id, seed in scenarios:
        row = dict(index[(nodes, scenario_id, seed, 0)])
        row.update({"Method": "Baseline OC0", "model": "Baseline", "predicted_oc": "0", "true_oc": ""})
        mapped.append(row)
    if len(mapped) != 1200:
        raise RuntimeError(f"Expected 1200 model/baseline outcome rows, found {len(mapped)}")
    mapped.sort(key=lambda row: (int(row["node_count"]), int(row["scenario_id"]), METHODS.index(row["Method"])))
    write(OUT / "model_selected_oc_rows_with_network_metrics.csv", mapped)
    by_node = [summary(mapped, method, nodes) for nodes in NODES for method in METHODS]
    overall = [summary(mapped, method, None) for method in METHODS]
    check = trend_check(by_node)
    write(OUT / "final_metrics_by_node.csv", by_node)
    write(OUT / "final_metrics_overall.csv", overall)
    write(OUT / "svm_trend_check.csv", check)
    (OUT / "configuration.json").write_text(json.dumps({
        "hello_interval_seconds": 30,
        "route_ttl_seconds": 120,
        "negative_cache_ttl_seconds": 120,
        "dense_summary_delta_suppression": False,
        "fixed_control_policy": True,
        "route_objective": "paper-style minimum path duration: serialization + propagation + ENC penalty",
        "ror_definition": "control transmissions / (control transmissions + realized data-hop transmissions)",
        "model_retraining": False,
        "selection_source": str(SOURCE.relative_to(ROOT)),
        "invariants": "generated_packets=200; OC_data_hops=0; architecture_violations=0; matched topology/acoustic hashes",
        "unique_candidate_trials": len(candidates),
    }, indent=2) + "\n")
    print("\nBY NODE")
    for row in by_node:
        print(row)
    print("\nSVM TREND CHECK")
    for row in check:
        print(row)


if __name__ == "__main__":
    main()
