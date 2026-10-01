#!/usr/bin/env python3
"""Re-evaluate frozen held-out OC selections after the HELLO-period change.

The script reads the previously saved 10-s model selections rather than loading,
fitting, or tuning any ML model.  It runs each distinct (scenario, OC) pair once,
then maps SVM/DTC/RF and OC0 baseline rows back to those exact candidate results.
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
OLD = ROOT / "results/underwater_rebuild/current/ttl64_hello10_paper_ror_test_evaluation"
OLD_MODEL_ROWS = OLD / "model_selected_oc_rows_with_fresh_network_metrics.csv"
OLD_RAW = OLD / "raw_matched_four_oc_test_diagnostics.csv"
OUT = ROOT / "results/underwater_rebuild/current/hello_interval_30s_experiment"
RUNS = OUT / "candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
MODEL_ORDER = ("SVM", "DTC", "RF")
METHOD_ORDER = ("SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict, key: str) -> float:
    value = row.get(key, "")
    return math.nan if value in ("", "NA", "NaN") else float(value)


def mean(rows: list[dict], key: str) -> float:
    values = [fnum(r, key) for r in rows]
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def run_candidate(key: tuple[int, int, int, int]) -> dict[str, str]:
    node_count, scenario_id, scenario_seed, oc = key
    target = RUNS / f"n{node_count}_sid{scenario_id}_seed{scenario_seed}_oc{oc}.csv"
    if not target.exists():
        cmd = [str(EXE), "--mode=run", f"--nodeCount={node_count}",
               f"--scenarioId={scenario_id}", f"--scenarioSeed={scenario_seed}",
               f"--selectedOc={oc}", f"--output={target}"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    rows = read_csv(target)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one output row in {target}, got {len(rows)}")
    return rows[0]


def assert_invariants(rows: list[dict]) -> None:
    bad = [r for r in rows if int(r["generated_packets"]) != 200
           or int(r["OC_data_hops"]) != 0 or int(r["architecture_violations"]) != 0]
    if bad:
        raise RuntimeError(f"Architecture/counter invariant failed in {len(bad)} candidate rows")
    by_scenario: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for r in rows:
        by_scenario[(r["node_count"], r["scenario_id"], r["scenario_seed"])].add(
            (r["topology_hash"], r["acoustic_state_hash"]))
    inconsistent = [key for key, hashes in by_scenario.items() if len(hashes) != 1]
    if inconsistent:
        raise RuntimeError(f"Matched topology/acoustic hash mismatch in {len(inconsistent)} scenarios")


def summary(rows: list[dict], method: str, node_count: int | None) -> dict:
    selected = [r for r in rows if r["Method"] == method and (node_count is None or int(r["node_count"]) == node_count)]
    return {
        "Nodes": "All" if node_count is None else node_count,
        "Method": method,
        "PDR %": mean(selected, "PDR"),
        "E2ED ms": mean(selected, "E2ED_ms"),
        "ROR total": mean(selected, "ROR_total"),
        "ROR reactive": mean(selected, "ROR_reactive"),
        "HELLO packets": mean(selected, "hello_tx"),
        "Scenarios": len(selected),
        "Undefined E2ED": sum(not math.isfinite(fnum(r, "E2ED_ms")) for r in selected),
    }


def previous_summary(old_model: list[dict], old_raw: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for r in old_model:
        r = dict(r)
        r["Method"] = f"{r['model']}-selected OC"
        rows.append(r)
    for r in old_raw:
        if int(r["auv_id"]) == 0:
            r = dict(r)
            r["Method"] = "Baseline OC0"
            rows.append(r)
    return rows


def main() -> None:
    if not EXE.exists():
        raise RuntimeError(f"Missing compiled simulator: {EXE}")
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    old_model, old_raw = read_csv(OLD_MODEL_ROWS), read_csv(OLD_RAW)
    if len(old_model) != 900 or len(old_raw) != 1200:
        raise RuntimeError("The required frozen 10-s held-out source ledgers are incomplete.")

    selections: dict[tuple[str, int], dict[str, str]] = {}
    required: set[tuple[int, int, int, int]] = set()
    for r in old_model:
        model = r["model"]
        if model not in MODEL_ORDER:
            raise RuntimeError(f"Unexpected model row: {model}")
        key = (model, int(r["scenario_id"]))
        if key in selections:
            raise RuntimeError(f"Duplicate frozen selection: {key}")
        selections[key] = r
        required.add((int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"]), int(r["predicted_oc"])))
        required.add((int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"]), 0))
    if len(selections) != 900:
        raise RuntimeError("Expected 300 frozen held-out selections for each ML model.")

    print(f"Running/reusing {len(required)} unique candidate trials.", flush=True)
    candidate_rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_candidate, key): key for key in sorted(required)}
        for done, future in enumerate(as_completed(futures), start=1):
            candidate_rows.append(future.result())
            if done % 50 == 0 or done == len(futures):
                print(f"Completed {done}/{len(futures)} candidate trials", flush=True)
    candidate_rows.sort(key=lambda r: (int(r["node_count"]), int(r["scenario_id"]), int(r["selected_oc"])))
    assert_invariants(candidate_rows)
    write_csv(OUT / "raw_unique_candidate_runs_hello30.csv", candidate_rows)

    index = {(int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"]), int(r["selected_oc"])): r
             for r in candidate_rows}
    selected_rows: list[dict] = []
    for (model, scenario_id), selection in sorted(selections.items(), key=lambda v: (v[1]["node_count"], v[0][1], v[0][0])):
        key = (int(selection["node_count"]), scenario_id, int(selection["scenario_seed"]), int(selection["predicted_oc"]))
        result = dict(index[key])
        result.update({"model": model, "Method": f"{model}-selected OC", "predicted_oc": selection["predicted_oc"],
                       "true_oc": selection["true_oc"]})
        selected_rows.append(result)
    seen_scenarios = {(int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"])) for r in selected_rows}
    for n, sid, seed in sorted(seen_scenarios):
        result = dict(index[(n, sid, seed, 0)])
        result.update({"model": "Baseline", "Method": "Baseline OC0", "predicted_oc": 0, "true_oc": ""})
        selected_rows.append(result)
    selected_rows.sort(key=lambda r: (int(r["node_count"]), int(r["scenario_id"]), METHOD_ORDER.index(r["Method"])))
    if len(selected_rows) != 1200:
        raise RuntimeError(f"Expected 1200 model/baseline rows, got {len(selected_rows)}")
    write_csv(OUT / "model_selected_oc_rows_with_hello30_network_metrics.csv", selected_rows)

    by_node = [summary(selected_rows, method, n) for n in (25, 50, 75, 100) for method in METHOD_ORDER]
    overall = [summary(selected_rows, method, None) for method in METHOD_ORDER]
    write_csv(OUT / "final_metrics_by_node.csv", by_node)
    write_csv(OUT / "final_metrics_overall.csv", overall)

    old_rows = previous_summary(old_model, old_raw)
    comparison: list[dict] = []
    for method in METHOD_ORDER:
        for interval, data in ((10, old_rows), (30, selected_rows)):
            s = summary(data, method, None)
            comparison.append({"Method": method, "HELLO interval seconds": interval,
                               "PDR %": s["PDR %"], "E2ED ms": s["E2ED ms"],
                               "ROR total": s["ROR total"], "ROR reactive": s["ROR reactive"],
                               "HELLO packets": s["HELLO packets"], "Scenarios": s["Scenarios"]})
    write_csv(OUT / "hello10_vs_hello30_comparison.csv", comparison)
    config = {"hello_period_seconds": 30, "route_ttl_seconds": 90,
              "source_hello10_selection_ledger": str(OLD_MODEL_ROWS.relative_to(ROOT)),
              "model_retraining": False, "candidate_trials": len(candidate_rows),
              "held_out_scenarios": len(seen_scenarios),
              "invariants": "generated_packets=200, OC_data_hops=0, architecture_violations=0; matched topology/acoustic hashes"}
    (OUT / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    print("\nBY NODE")
    for r in by_node:
        print(r)
    print("\nOVERALL")
    for r in overall:
        print(r)
    print("\nHELLO COMPARISON")
    for r in comparison:
        print(r)


if __name__ == "__main__":
    main()
