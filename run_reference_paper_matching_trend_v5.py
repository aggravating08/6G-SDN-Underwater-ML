#!/usr/bin/env python3
"""v5: long-distance SVM test trend with local relay-search waiting.

This is a *network protocol* sensitivity run on the frozen v4 held-out SVM
choices.  It does not retrain a model or relabel a scenario.  Each selected
test candidate is rerun once under the same long-distance traffic population,
with only the explicitly configured local per-hop discovery wait enabled.
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
OUT = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_v5"
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return math.nan if value in ("", "NA", "NaN") else float(value)


def scenario_seed(nodes: int, sid: int) -> int:
    first_id, first_seed = SEED_BASE[nodes]
    return first_seed + sid - first_id


def run_candidate(nodes: int, sid: int, oc: int) -> dict[str, str]:
    seed = scenario_seed(nodes, sid)
    path = RUNS / f"n{nodes}_sid{sid}_seed{seed}_oc{oc}.csv"
    # A finished single-candidate CSV has one header and one result row.
    if not path.exists() or sum(1 for _ in path.open()) != 2:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            str(EXE), "--mode=run", f"--nodeCount={nodes}", f"--scenarioId={sid}",
            f"--scenarioSeed={seed}", f"--selectedOc={oc}", "--runs=1",
            "--fixedControlPolicy=true", "--helloSeconds=30", "--routeTtlSeconds=60",
            "--negativeRouteTtlSeconds=120", "--referenceUpdateAccounting=true",
            "--bufferRouteDiscovery=true", "--routeDiscoveryRetryLimit=3",
            "--packetBufferTimeoutSeconds=60", "--longDistanceFlows=true",
            "--relaySearchDelay=true", "--relaySearchBaseMs=5",
            "--relaySearchScaleMs=180", "--relaySearchMaxMs=80", f"--output={path}",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    rows = read(path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one completed result: {path}")
    return rows[0]


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = [number(row, key) for row in rows]
    values = [x for x in values if math.isfinite(x)]
    return sum(values) / len(values) if values else math.nan


def weighted_delay(rows: list[dict[str, str]]) -> float:
    numerator = denominator = 0.0
    for row in rows:
        e2e, delivered = number(row, "E2ED_ms"), number(row, "delivered_packets")
        if math.isfinite(e2e) and delivered > 0:
            numerator += e2e * delivered
            denominator += delivered
    return numerator / denominator if denominator else math.nan


def aggregate(rows: list[dict[str, str]], nodes: int) -> dict:
    return {
        "Nodes": nodes,
        "Scenarios": len(rows),
        "PDR %": 100.0 * sum(number(r, "delivered_packets") for r in rows) / sum(number(r, "generated_packets") for r in rows),
        # Standard E2ED: weighted average across actually delivered packets.
        "E2ED ms": weighted_delay(rows),
        "ROR offered": mean(rows, "ROR_generated"),
        "ROR transmission": mean(rows, "ROR_total"),
        "mean source-destination distance m": mean(rows, "mean_source_destination_distance_m"),
        "mean delivered hop count": mean(rows, "mean_delivered_hop_count"),
        "mean delivered path length m": mean(rows, "mean_delivered_path_length_m"),
        # Both are per realised delivered hop, not density inputs or labels.
        "mean candidate count": mean(rows, "mean_delivered_relay_candidate_count"),
        "mean relay wait ms": mean(rows, "mean_delivered_relay_wait_ms"),
        "delivered packets": int(sum(number(r, "delivered_packets") for r in rows)),
    }


def main() -> None:
    selections = [r for r in read(V4 / "test_model_oc_predictions.csv") if r["Model"] == "SVM"]
    if len(selections) != 300:
        raise RuntimeError(f"Expected 300 v4 held-out SVM selections, found {len(selections)}")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for selection in selections:
        nodes, sid, oc = int(selection["node_count"]), int(selection["scenario_id"]), int(selection["predicted_oc"])
        result = run_candidate(nodes, sid, oc)
        if (result["generated_packets"] != "200" or result["OC_data_hops"] != "0" or
                result["architecture_violations"] != "0"):
            raise RuntimeError(f"Invariant violation at nodes={nodes}, scenario={sid}, OC={oc}")
        result = dict(result)
        result["Model"] = "SVM"
        result["v4_predicted_oc"] = oc
        grouped[nodes].append(result)
    raw = [row for n in NODES for row in grouped[n]]
    if any(len(grouped[n]) != 75 for n in NODES):
        raise RuntimeError("v5 did not retain 75 held-out scenarios per density")
    write(OUT / "svm_selected_raw_runs.csv", raw)
    table = [aggregate(grouped[n], n) for n in NODES]
    write(OUT / "svm_density_trend.csv", table)

    pdr = [row["PDR %"] for row in table]
    e2e = [row["E2ED ms"] for row in table]
    offered = [row["ROR offered"] for row in table]
    checks = [
        {"Metric": "PDR %", **dict(zip(map(str, NODES), pdr)),
         "Expected trend": "increases then saturates (0.5 pp tolerance)",
         "Pass/Fail": "PASS" if all(b >= a - .5 for a, b in zip(pdr, pdr[1:])) else "FAIL"},
        {"Metric": "E2ED ms", **dict(zip(map(str, NODES), e2e)), "Expected trend": "generally decreases",
         "Pass/Fail": "PASS" if all(b <= a for a, b in zip(e2e, e2e[1:])) else "FAIL"},
        {"Metric": "ROR offered", **dict(zip(map(str, NODES), offered)), "Expected trend": "increases",
         "Pass/Fail": "PASS" if all(b >= a for a, b in zip(offered, offered[1:])) else "FAIL"},
    ]
    write(OUT / "trend_gate.csv", checks)
    (OUT / "configuration.json").write_text(json.dumps({
        "experiment": "reference_paper_matching_trend_v5",
        "base_population": "v4 long-distance held-out test scenarios and their frozen SVM selections",
        "model_retraining_or_relabeling": False,
        "minimum_source_destination_distance_m": 424.3,
        "fixed_control_policy": {"hello_seconds": 30, "route_ttl_seconds": 60,
                                  "negative_cache_seconds": 120, "reference_update_accounting": True},
        "relay_search_delay": {
            "enabled": True,
            "candidate_definition": "usable sensor neighbours with strictly positive geometric progress to destination",
            "formula_ms": "min(5 + 180 / max(1, candidate_count), 80)",
            "density_independent": True,
            "e2ed_application": "scheduled before every eligible payload hop; naturally included only if packet reaches destination",
        },
        "e2ed": "destination arrival minus source generation for delivered packets only, weighted over delivered packets",
        "ror_offered": "control packets / (control packets + generated source packets)",
        "ror_transmission": "control packets / (control packets + realised data-hop transmissions)",
    }, indent=2) + "\n")
    print("V5 SVM DENSITY TREND")
    for row in table:
        print(row)
    print("V5 TREND GATE")
    for row in checks:
        print(row)


if __name__ == "__main__":
    main()
