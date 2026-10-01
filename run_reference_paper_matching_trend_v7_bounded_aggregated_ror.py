#!/usr/bin/env python3
"""v7: LC-region topology digests for bounded reference-style offered ROR.

The v4 held-out SVM OC choices, long-distance traffic, and v5 relay-search
delay are frozen.  This script changes only protocol control accounting from
per-sensor HELLO/update traffic to fixed-capacity LC/gateway topology digests.
It first evaluates the requested 25-sensor digest capacity.  A documented
40-sensor capacity is evaluated only if the 25-capacity result exceeds 0.50.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V4 = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_v4_long_distance_ml"
OUT = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_v7_bounded_aggregated_ror"
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


def value(row: dict[str, str], column: str) -> float:
    raw = row.get(column, "")
    return math.nan if raw in ("", "NA", "NaN") else float(raw)


def scenario_seed(nodes: int, sid: int) -> int:
    first_sid, first_seed = SEED_BASE[nodes]
    return first_seed + sid - first_sid


def run_trial(nodes: int, sid: int, oc: int, capacity: int, directory: Path) -> dict[str, str]:
    seed = scenario_seed(nodes, sid)
    output = directory / "candidate_runs" / f"n{nodes}_sid{sid}_seed{seed}_oc{oc}.csv"
    if not output.exists() or sum(1 for _ in output.open()) != 2:
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            str(EXE), "--mode=run", f"--nodeCount={nodes}", f"--scenarioId={sid}",
            f"--scenarioSeed={seed}", f"--selectedOc={oc}", "--runs=1",
            "--fixedControlPolicy=true", "--helloSeconds=30", "--routeTtlSeconds=60",
            "--negativeRouteTtlSeconds=120", "--referenceUpdateAccounting=true",
            "--aggregatedTopologyDigestAccounting=true", f"--topologyDigestCapacityNodes={capacity}",
            "--bufferRouteDiscovery=true", "--routeDiscoveryRetryLimit=3",
            "--packetBufferTimeoutSeconds=60", "--longDistanceFlows=true",
            "--relaySearchDelay=true", "--relaySearchBaseMs=5",
            "--relaySearchScaleMs=180", "--relaySearchMaxMs=80", f"--output={output}",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    rows = read(output)
    if len(rows) != 1:
        raise RuntimeError(f"Incomplete v7 run: {output}")
    return rows[0]


def mean(rows: list[dict[str, str]], column: str) -> float:
    values = [value(row, column) for row in rows]
    values = [x for x in values if math.isfinite(x)]
    return sum(values) / len(values) if values else math.nan


def weighted_e2ed(rows: list[dict[str, str]]) -> float:
    pairs = [(value(r, "E2ED_ms"), value(r, "delivered_packets")) for r in rows]
    pairs = [(delay, count) for delay, count in pairs if math.isfinite(delay) and count > 0]
    return sum(delay * count for delay, count in pairs) / sum(count for _, count in pairs) if pairs else math.nan


def summarize(rows: list[dict[str, str]], nodes: int) -> dict:
    return {
        "Nodes": nodes, "Scenarios": len(rows),
        "PDR %": 100 * sum(value(r, "delivered_packets") for r in rows) / sum(value(r, "generated_packets") for r in rows),
        "E2ED ms": weighted_e2ed(rows),
        "ROR offered": mean(rows, "ROR_generated"),
        "ROR transmission": mean(rows, "ROR_total"),
        "topology digest packets": mean(rows, "topology_digest_packets"),
        "route control packets": mean(rows, "route_request_tx") + mean(rows, "route_reply_tx") + mean(rows, "mc_control_tx"),
        "raw neighbor burden": mean(rows, "raw_neighbor_update_burden"),
        "relay wait ms": mean(rows, "mean_delivered_relay_wait_ms"),
        "mean candidates": mean(rows, "mean_delivered_relay_candidate_count"),
        "mean source-destination distance m": mean(rows, "mean_source_destination_distance_m"),
        "mean delivered hop count": mean(rows, "mean_delivered_hop_count"),
        "mean delivered path length m": mean(rows, "mean_delivered_path_length_m"),
    }


def evaluate_capacity(capacity: int) -> tuple[list[dict], list[dict[str, str]], Path]:
    directory = OUT / f"digest_capacity_{capacity}"
    selections = [r for r in read(V4 / "test_model_oc_predictions.csv") if r["Model"] == "SVM"]
    if len(selections) != 300:
        raise RuntimeError("Expected 300 frozen v4 held-out SVM selections")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in selections:
        n, sid, oc = int(row["node_count"]), int(row["scenario_id"]), int(row["predicted_oc"])
        result = run_trial(n, sid, oc, capacity, directory)
        if result["generated_packets"] != "200" or result["OC_data_hops"] != "0" or result["architecture_violations"] != "0":
            raise RuntimeError(f"Architecture invariant failure at n={n}, scenario={sid}, OC={oc}")
        grouped[n].append(result)
    if any(len(grouped[n]) != 75 for n in NODES):
        raise RuntimeError("Every density must retain its 75 held-out scenarios")
    raw = [r for n in NODES for r in grouped[n]]
    table = [summarize(grouped[n], n) for n in NODES]
    write(directory / "svm_selected_raw_runs.csv", raw)
    write(directory / "svm_density_trend.csv", table)
    return table, raw, directory


def trend_gate(table: list[dict]) -> list[dict]:
    pdr = [r["PDR %"] for r in table]
    e2e = [r["E2ED ms"] for r in table]
    offered = [r["ROR offered"] for r in table]
    return [
        {"Metric": "PDR %", **dict(zip(map(str, NODES), pdr)),
         "Expected trend": "increases then saturates (0.5 pp tolerance)", "Pass/Fail": "PASS" if all(b >= a-.5 for a, b in zip(pdr, pdr[1:])) else "FAIL"},
        {"Metric": "E2ED ms", **dict(zip(map(str, NODES), e2e)), "Expected trend": "generally decreases", "Pass/Fail": "PASS" if all(b <= a for a, b in zip(e2e, e2e[1:])) else "FAIL"},
        {"Metric": "ROR offered", **dict(zip(map(str, NODES), offered)), "Expected trend": "increases", "Pass/Fail": "PASS" if all(b >= a for a, b in zip(offered, offered[1:])) else "FAIL"},
        {"Metric": "ROR offered bounded", **dict(zip(map(str, NODES), offered)), "Expected trend": "0.40 to 0.50", "Pass/Fail": "PASS" if all(.40 <= x <= .50 for x in offered) else "FAIL"},
    ]


def main() -> None:
    table, raw, used = evaluate_capacity(25)
    # The only contingent parameter permitted by the v7 protocol request.
    if any(row["ROR offered"] > .50 for row in table):
        table, raw, used = evaluate_capacity(40)
    gates = trend_gate(table)
    write(OUT / "svm_selected_raw_runs.csv", raw)
    write(OUT / "svm_density_trend.csv", table)
    write(OUT / "trend_gate.csv", gates)
    (OUT / "configuration.json").write_text(json.dumps({
        "experiment": "reference_paper_matching_trend_v7_bounded_aggregated_ror",
        "selection": "frozen v4 long-distance held-out SVM selections; no retraining/relabeling",
        "long_distance_pair_rule_m": 424.3,
        "fixed_protocol": {"update_interval_seconds": 30, "route_ttl_seconds": 60,
                            "negative_cache_seconds": 120, "bounded_route_discovery": True},
        "relay_search_wait": "min(5 + 180/max(1, usable positive-progress neighbours), 80) ms",
        "topology_accounting": "sensors are partitioned into nearest nonselected-LC service regions; every LC emits ceil(region_sensor_count/digest_capacity_nodes) compressed topology digests per update interval",
        "requested_digest_capacity_nodes": 25,
        "used_digest_capacity_nodes": int(used.name.split("_")[-1]),
        "capacity_40_trial": (OUT / "digest_capacity_40").exists(),
        "raw_neighbor_update_burden": "diagnostic only; excluded from main ROR",
        "ror_offered": "control packets / (control packets + 200 generated packets)",
        "ror_transmission": "control packets / (control packets + realised data-hop transmissions)",
    }, indent=2) + "\n")
    print(f"V7 used digest capacity {used.name.split('_')[-1]}")
    print("V7 SVM DENSITY TREND")
    for row in table:
        print(row)
    print("V7 TREND GATE")
    for row in gates:
        print(row)


if __name__ == "__main__":
    main()
