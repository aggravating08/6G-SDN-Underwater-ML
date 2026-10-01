#!/usr/bin/env python3
"""Held-out evaluation of the 60-s HELLO/reactive-suppression protocol.

This runner does not load, train, tune, or modify any ML model.  It reuses the
saved SVM/DTC/RF OC choices from the 10-s held-out ledger, runs each distinct
candidate once under the new protocol, and maps the raw candidate result back
to each selected method.
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
SOURCE = ROOT / "results/underwater_rebuild/current/ttl64_hello10_paper_ror_test_evaluation"
SELECTIONS = SOURCE / "model_selected_oc_rows_with_fresh_network_metrics.csv"
OUT = ROOT / "results/underwater_rebuild/current/hello60_reactive_suppression_ror_below_04"
RUN_DIR = OUT / "candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
MODELS = ("SVM", "DTC", "RF")
METHODS = ("SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, keys, extrasaction="ignore")
        out.writeheader()
        out.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    return math.nan if raw in ("", "NA", "NaN") else float(raw)


def avg(rows: list[dict], key: str) -> float:
    xs = [number(r, key) for r in rows]
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else math.nan


def run(key: tuple[int, int, int, int]) -> dict[str, str]:
    n, scenario_id, seed, oc = key
    output = RUN_DIR / f"n{n}_sid{scenario_id}_seed{seed}_oc{oc}.csv"
    if not output.exists():
        subprocess.run([str(EXE), "--mode=run", f"--nodeCount={n}",
                        f"--scenarioId={scenario_id}", f"--scenarioSeed={seed}",
                        f"--selectedOc={oc}", f"--output={output}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    rows = read(output)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one candidate row in {output}")
    return rows[0]


def check(rows: list[dict[str, str]]) -> None:
    invalid = [r for r in rows if int(r["generated_packets"]) != 200
               or int(r["OC_data_hops"]) != 0 or int(r["architecture_violations"]) != 0]
    if invalid:
        raise RuntimeError(f"Packet or architecture invariant failed in {len(invalid)} candidate runs")
    hashes: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for r in rows:
        hashes[(r["node_count"], r["scenario_id"], r["scenario_seed"])].add(
            (r["topology_hash"], r["acoustic_state_hash"]))
    if any(len(v) != 1 for v in hashes.values()):
        raise RuntimeError("Matched topology/acoustic hash invariant failed")


def aggregate(rows: list[dict], method: str, nodes: int | None) -> dict:
    use = [r for r in rows if r["Method"] == method and (nodes is None or int(r["node_count"]) == nodes)]
    return {
        "Nodes": "All" if nodes is None else nodes,
        "Method": method,
        "PDR %": avg(use, "PDR"), "E2ED ms": avg(use, "E2ED_ms"),
        "ROR total": avg(use, "ROR_total"), "ROR reactive": avg(use, "ROR_reactive"),
        "HELLO packets": avg(use, "hello_tx"),
        "Route request packets": avg(use, "route_request_tx"),
        "Route reply packets": avg(use, "route_reply_tx"),
        "MC fallbacks": avg(use, "mc_fallbacks"),
        "Failed route attempts": avg(use, "failed_route_attempts"),
        "Negative-cache hits": avg(use, "negative_cache_hits"),
        "Data transmissions": avg(use, "data_hops"),
        "Control transmissions": avg(use, "control_transmissions"),
        "Scenarios": len(use),
        "Undefined E2ED": sum(not math.isfinite(number(r, "E2ED_ms")) for r in use),
    }


def main() -> None:
    if not EXE.exists() or not SELECTIONS.exists():
        raise RuntimeError("Compiled simulator or frozen selection ledger is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(exist_ok=True)
    selection_rows = read(SELECTIONS)
    if len(selection_rows) != 900:
        raise RuntimeError("Expected 900 frozen ML selections (3 models × 300 test scenarios)")
    frozen: dict[tuple[str, int], dict[str, str]] = {}
    needed: set[tuple[int, int, int, int]] = set()
    for r in selection_rows:
        if r["model"] not in MODELS:
            raise RuntimeError(f"Unexpected model: {r['model']}")
        label = (r["model"], int(r["scenario_id"]))
        if label in frozen:
            raise RuntimeError(f"Duplicate frozen selection: {label}")
        frozen[label] = r
        base = (int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"]))
        needed.add((*base, int(r["predicted_oc"])))
        needed.add((*base, 0))
    if len(frozen) != 900:
        raise RuntimeError("Frozen selection ledger is incomplete")

    print(f"Running/reusing {len(needed)} unique candidate trials", flush=True)
    candidates: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(run, key): key for key in sorted(needed)}
        for completed, job in enumerate(as_completed(jobs), start=1):
            candidates.append(job.result())
            if completed % 50 == 0 or completed == len(jobs):
                print(f"Completed {completed}/{len(jobs)}", flush=True)
    candidates.sort(key=lambda r: (int(r["node_count"]), int(r["scenario_id"]), int(r["selected_oc"])))
    check(candidates)
    write(OUT / "raw_unique_candidate_runs.csv", candidates)

    index = {(int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"]), int(r["selected_oc"])): r
             for r in candidates}
    selected: list[dict] = []
    for (model, sid), s in frozen.items():
        key = (int(s["node_count"]), sid, int(s["scenario_seed"]), int(s["predicted_oc"]))
        r = dict(index[key])
        r.update({"Method": f"{model}-selected OC", "model": model, "predicted_oc": s["predicted_oc"],
                  "true_oc": s["true_oc"]})
        selected.append(r)
    all_scenarios = {(int(r["node_count"]), int(r["scenario_id"]), int(r["scenario_seed"])) for r in selected}
    for n, sid, seed in all_scenarios:
        r = dict(index[(n, sid, seed, 0)])
        r.update({"Method": "Baseline OC0", "model": "Baseline", "predicted_oc": 0, "true_oc": ""})
        selected.append(r)
    selected.sort(key=lambda r: (int(r["node_count"]), int(r["scenario_id"]), METHODS.index(r["Method"])))
    if len(selected) != 1200:
        raise RuntimeError(f"Expected 1200 model/baseline rows, got {len(selected)}")
    write(OUT / "model_selected_oc_rows_with_network_metrics.csv", selected)

    by_node = [aggregate(selected, method, n) for n in (25, 50, 75, 100) for method in METHODS]
    overall = [aggregate(selected, method, None) for method in METHODS]
    stable = [aggregate([r for r in selected if int(r["node_count"]) >= 50], method, None) for method in METHODS]
    for r in stable:
        r["Nodes"] = "50/75/100"
    write(OUT / "final_metrics_by_node.csv", by_node)
    write(OUT / "final_metrics_overall.csv", overall)
    write(OUT / "stable_density_50_75_100_metrics.csv", stable)
    (OUT / "configuration.json").write_text(json.dumps({
        "hello_period_seconds": 60, "route_ttl_seconds": 120,
        "negative_cache_ttl_seconds": 30, "max_mc_fallbacks_per_packet": 1,
        "model_retraining": False, "held_out_scenarios": len(all_scenarios),
        "unique_candidate_trials": len(candidates),
        "invariants": "generated_packets=200; OC_data_hops=0; architecture_violations=0; matched topology/acoustic hashes",
        "frozen_selection_source": str(SELECTIONS.relative_to(ROOT)),
    }, indent=2) + "\n")
    print("\nBY NODE")
    for r in by_node:
        print(r)
    print("\nOVERALL")
    for r in overall:
        print(r)
    print("\nSTABLE DENSITIES (50/75/100)")
    for r in stable:
        print(r)


if __name__ == "__main__":
    main()
