#!/usr/bin/env python3
"""v6: v5 long-distance SVM trend with compressed topology-digest accounting.

The 0.20 factor applies only to the reception-equivalent neighbour-update
burden in ROR accounting.  Payload routing, timings, source/destination
traffic, and the v5 local relay-search delay are unchanged.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V4 = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_v4_long_distance_ml"
OUT = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_v6_bounded_ror"
RUNS = OUT / "svm_selected_candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
NODES = (25, 50, 75, 100)
SEED_BASE = {25: (0, 29), 50: (500, 1_000_029), 75: (1000, 2_000_029), 100: (1500, 3_000_029)}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict[str, str], name: str) -> float:
    raw = row.get(name, "")
    return math.nan if raw in ("", "NA", "NaN") else float(raw)


def seed_for(nodes: int, sid: int) -> int:
    first_sid, first_seed = SEED_BASE[nodes]
    return first_seed + sid - first_sid


def run_candidate(nodes: int, sid: int, oc: int) -> dict[str, str]:
    seed = seed_for(nodes, sid)
    output = RUNS / f"n{nodes}_sid{sid}_seed{seed}_oc{oc}.csv"
    if not output.exists() or sum(1 for _ in output.open()) != 2:
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            str(EXE), "--mode=run", f"--nodeCount={nodes}", f"--scenarioId={sid}",
            f"--scenarioSeed={seed}", f"--selectedOc={oc}", "--runs=1",
            "--fixedControlPolicy=true", "--helloSeconds=30", "--routeTtlSeconds=60",
            "--negativeRouteTtlSeconds=120", "--referenceUpdateAccounting=true",
            "--bufferRouteDiscovery=true", "--routeDiscoveryRetryLimit=3",
            "--packetBufferTimeoutSeconds=60", "--longDistanceFlows=true",
            "--relaySearchDelay=true", "--relaySearchBaseMs=5",
            "--relaySearchScaleMs=180", "--relaySearchMaxMs=80",
            "--topologyUpdateCompressionFactor=0.20", f"--output={output}",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    rows = read(output)
    if len(rows) != 1:
        raise RuntimeError(f"Incomplete v6 candidate result: {output}")
    return rows[0]


def mean(rows: list[dict[str, str]], name: str) -> float:
    values = [value(row, name) for row in rows]
    values = [x for x in values if math.isfinite(x)]
    return sum(values) / len(values) if values else math.nan


def weighted_e2ed(rows: list[dict[str, str]]) -> float:
    weighted = [(value(row, "E2ED_ms"), value(row, "delivered_packets")) for row in rows]
    weighted = [(delay, count) for delay, count in weighted if math.isfinite(delay) and count > 0]
    return sum(delay * count for delay, count in weighted) / sum(count for _, count in weighted) if weighted else math.nan


def summary(rows: list[dict[str, str]], nodes: int) -> dict:
    return {
        "Nodes": nodes, "Scenarios": len(rows),
        "PDR %": 100 * sum(value(r, "delivered_packets") for r in rows) / sum(value(r, "generated_packets") for r in rows),
        "E2ED ms": weighted_e2ed(rows),
        "ROR offered": mean(rows, "ROR_generated"),
        "ROR transmission": mean(rows, "ROR_total"),
        "raw neighbor update burden": mean(rows, "raw_neighbor_update_burden"),
        "compressed topology updates": mean(rows, "compressed_topology_updates"),
        "relay wait ms": mean(rows, "mean_delivered_relay_wait_ms"),
        "mean candidates": mean(rows, "mean_delivered_relay_candidate_count"),
        "mean source-destination distance m": mean(rows, "mean_source_destination_distance_m"),
        "mean delivered hop count": mean(rows, "mean_delivered_hop_count"),
        "mean delivered path length m": mean(rows, "mean_delivered_path_length_m"),
    }


def main() -> None:
    selected = [row for row in read(V4 / "test_model_oc_predictions.csv") if row["Model"] == "SVM"]
    if len(selected) != 300:
        raise RuntimeError("Expected the 300 frozen v4 held-out SVM selections")
    by_nodes: dict[int, list[dict[str, str]]] = defaultdict(list)
    for picked in selected:
        nodes, sid, oc = int(picked["node_count"]), int(picked["scenario_id"]), int(picked["predicted_oc"])
        result = run_candidate(nodes, sid, oc)
        if result["generated_packets"] != "200" or result["OC_data_hops"] != "0" or result["architecture_violations"] != "0":
            raise RuntimeError(f"v6 architecture invariant failed: n={nodes}, sid={sid}, oc={oc}")
        by_nodes[nodes].append(result)
    if any(len(by_nodes[n]) != 75 for n in NODES):
        raise RuntimeError("v6 requires all 75 held-out SVM selections at each density")
    raw = [row for n in NODES for row in by_nodes[n]]
    write(OUT / "svm_selected_raw_runs.csv", raw)
    table = [summary(by_nodes[n], n) for n in NODES]
    write(OUT / "svm_density_trend.csv", table)
    pdr = [r["PDR %"] for r in table]
    e2e = [r["E2ED ms"] for r in table]
    offered = [r["ROR offered"] for r in table]
    bounded = [0.40 <= x <= 0.50 for x in offered]
    gates = [
        {"Metric": "PDR %", **dict(zip(map(str, NODES), pdr)),
         "Expected trend": "increases then saturates (0.5 pp tolerance)", "Pass/Fail": "PASS" if all(b >= a - .5 for a, b in zip(pdr, pdr[1:])) else "FAIL"},
        {"Metric": "E2ED ms", **dict(zip(map(str, NODES), e2e)), "Expected trend": "generally decreases", "Pass/Fail": "PASS" if all(b <= a for a, b in zip(e2e, e2e[1:])) else "FAIL"},
        {"Metric": "ROR offered", **dict(zip(map(str, NODES), offered)), "Expected trend": "increases", "Pass/Fail": "PASS" if all(b >= a for a, b in zip(offered, offered[1:])) else "FAIL"},
        {"Metric": "ROR offered bounded", **dict(zip(map(str, NODES), offered)), "Expected trend": "0.40 to 0.50", "Pass/Fail": "PASS" if all(bounded) else "FAIL"},
    ]
    write(OUT / "trend_gate.csv", gates)
    (OUT / "configuration.json").write_text(json.dumps({
        "experiment": "reference_paper_matching_trend_v6_bounded_ror",
        "base_selection": "frozen v4 long-distance held-out SVM predictions; no model retraining or relabeling",
        "long_distance_pair_rule_m": 424.3,
        "hello_seconds": 30, "route_ttl_seconds": 60, "negative_cache_seconds": 120,
        "relay_search_delay": "min(5 + 180/max(1, positive-progress usable sensor neighbours), 80) ms",
        "topology_update_compression_factor": 0.20,
        "compressed_topology_updates": "0.20 * raw neighbour reception-equivalent burden + existing LC and MC digest transmissions",
        "raw_neighbor_update_burden": "retained diagnostic only; not part of main compressed ROR count",
        "ror_offered": "effective control transmissions / (effective control transmissions + generated packets)",
        "ror_transmission": "effective control transmissions / (effective control transmissions + realised data-hop transmissions)",
        "arithmetic_lower_bound": "at 100 nodes, fixed 30-s HELLO over 118 s produces 400 sender transmissions, so ROR_offered >= 400/(400+200)=0.667 before any other control",
    }, indent=2) + "\n")
    print("V6 SVM DENSITY TREND")
    for row in table:
        print(row)
    print("V6 TREND GATE")
    for row in gates:
        print(row)


if __name__ == "__main__":
    main()
