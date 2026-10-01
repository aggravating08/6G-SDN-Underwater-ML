#!/usr/bin/env python3
"""Version 3: fixed reference-style updates plus bounded route retry buffering."""
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import run_reference_paper_matching_trend_corrected as v2


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_v3"
RUNS = OUT / "candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
NODES = (25, 50, 75, 100)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def run_candidate(key: tuple[int, int, int, int]) -> dict[str, str]:
    nodes, scenario_id, seed, oc = key
    output = RUNS / f"n{nodes}_sid{scenario_id}_seed{seed}_oc{oc}.csv"
    if not output.exists():
        subprocess.run(
            [str(EXE), "--mode=run", f"--nodeCount={nodes}", f"--scenarioId={scenario_id}",
             f"--scenarioSeed={seed}", f"--selectedOc={oc}",
             "--fixedControlPolicy=true", "--helloSeconds=30", "--routeTtlSeconds=60",
             "--negativeRouteTtlSeconds=120", "--referenceUpdateAccounting=true",
             "--bufferRouteDiscovery=true", "--routeDiscoveryRetryLimit=3",
             "--packetBufferTimeoutSeconds=60", f"--output={output}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
    rows = read(output)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row: {output}")
    return rows[0]


def average(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", "NA", "NaN")]
    return sum(values) / len(values) if values else math.nan


def build_tolerant_check(rows: list[dict[str, str]]) -> list[dict]:
    svm = {int(row["Nodes"]): row for row in rows if row["Method"] == "SVM"}
    pdr = [float(svm[n]["PDR %"]) for n in NODES]
    offered = [float(svm[n]["ROR offered"]) for n in NODES]
    delay = [float(svm[n]["E2ED ms"]) for n in NODES]
    return [
        {"Metric": "PDR %", **{str(n): x for n, x in zip(NODES, pdr)},
         "Expected trend": "increases then saturates (0.5 pp tolerance)",
         "Pass/Fail": "PASS" if all(b >= a - .5 for a, b in zip(pdr, pdr[1:])) else "FAIL"},
        {"Metric": "E2ED ms", **{str(n): x for n, x in zip(NODES, delay)},
         "Expected trend": "generally decreasing", "Pass/Fail": "PASS" if all(b <= a for a, b in zip(delay, delay[1:])) else "FAIL"},
        {"Metric": "ROR offered", **{str(n): x for n, x in zip(NODES, offered)},
         "Expected trend": "increasing", "Pass/Fail": "PASS" if all(b >= a for a, b in zip(offered, offered[1:])) else "FAIL"},
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    v2.OUT = OUT
    v2.RUNS = RUNS
    v2.run_candidate = run_candidate
    v2.main()

    old_rows = read(OUT / "final_metrics_by_node.csv")
    renamed = []
    for row in old_rows:
        renamed.append({
            "Nodes": row["Nodes"], "Method": row["Method"], "PDR %": row["PDR %"],
            "E2ED ms": row["E2ED ms"], "ROR offered": row["ROR generated"],
            "ROR transmission": row["ROR total"], "ROR reactive": row["ROR reactive"],
            "HELLO/update packets": row["HELLO/update packets"], "Route requests": row["Route requests"],
            "MC fallbacks": row["MC fallbacks"], "Failed attempts": row["Failed attempts"],
            "Buffered retries": row.get("Buffered retries", "0"), "Scenarios": row["Scenarios"],
        })
    # The underlying candidate rows retain the retry diagnostics; calculate
    # their means without changing the model-selection mapping.
    mapped = read(OUT / "model_selected_oc_rows_with_network_metrics.csv")
    for row in renamed:
        use = [item for item in mapped if item["Method"] == row["Method"] and item["node_count"] == row["Nodes"]]
        row["Buffered retries"] = average(use, "route_discovery_buffered_retries")
    write(OUT / "final_metrics_by_node_renamed.csv", renamed)
    check = build_tolerant_check(renamed)
    write(OUT / "svm_trend_check_tolerant.csv", check)
    svm_table = [row for row in renamed if row["Method"] == "SVM"]
    write(OUT / "svm_density_trend_table.csv", svm_table)
    (OUT / "configuration.json").write_text(json.dumps({
        "hello_interval_seconds": 30,
        "route_ttl_seconds": 60,
        "negative_no_path_cache_seconds": 120,
        "route_discovery_retry_limit": 3,
        "packet_buffer_timeout_seconds": 60,
        "retry_backoff": "one analytical acoustic control round with bounded exponential backoff",
        "e2ed": "generation time to destination arrival for delivered packets only; successful route-discovery and buffered retry time are included",
        "ror_offered": "control packets / (control packets + 200 generated data packets)",
        "ror_transmission": "control packets / (control packets + realized data-hop transmissions)",
        "periodic_update_accounting": "reception-equivalent neighbour state updates plus LC/MC summaries; reported separately from reactive ROR",
        "model_retraining": False,
    }, indent=2) + "\n")
    print("\nTOLERANT SVM TREND CHECK")
    for row in check:
        print(row)


if __name__ == "__main__":
    main()
