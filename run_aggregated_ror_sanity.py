#!/usr/bin/env python3
"""Small matched validation of LC topology-digest control accounting.

This does not overwrite any previous ledger.  It runs five held-out scenario
records at each density and all four controller candidates under a single,
fixed aggregation policy.
"""
from __future__ import annotations

import csv
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "underwater_partner_style_equivalent_2000"
OUT = INPUT / "AGGREGATED_ROR_SANITY_5_PER_NODE"
NODES = (25, 50, 75, 100)
RUN_HEADER = None


def test_records() -> dict[int, list[tuple[int, int]]]:
    test_ids = set(json.loads((INPUT / "split_manifest.json").read_text())["test"])
    seen: dict[int, tuple[int, int]] = {}
    with (INPUT / "underwater_feature_rule_2000_scenarios_8000_candidates.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            scenario_id = int(row["scenario_id"])
            if scenario_id in test_ids and scenario_id not in seen:
                seen[scenario_id] = (int(row["node_count"]), int(row["scenario_seed"]))
    records: dict[int, list[tuple[int, int]]] = {node: [] for node in NODES}
    for scenario_id, (node, seed) in sorted(seen.items()):
        if node in records and len(records[node]) < 5:
            records[node].append((scenario_id, seed))
    if any(len(records[node]) != 5 for node in NODES):
        raise RuntimeError(f"Could not obtain five test scenarios per density: {records}")
    return records


def run_one(node: int, scenario_id: int, seed: int, oc: int, out: Path, digest_capacity: int) -> dict[str, str]:
    target = out / "runs" / f"n{node}_s{scenario_id}_oc{oc}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "scratch/randy --mode=run "
        f"--nodeCount={node} --scenarioId={scenario_id} --scenarioSeed={seed} "
        f"--selectedOc={oc} --runs=1 --fixedControlPolicy=true "
        "--helloSeconds=30 --routeTtlSeconds=60 --negativeRouteTtlSeconds=120 "
        "--referenceUpdateAccounting=true --aggregatedTopologyDigestAccounting=true "
        f"--topologyDigestCapacityNodes={digest_capacity} "
        f"--output={target}"
    )
    completed = subprocess.run(["./ns3", "run", command], cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError(f"run failed n={node} scenario={scenario_id} oc={oc}: {completed.stderr}")
    with target.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    if row["generated_packets"] != "200" or row["OC_data_hops"] != "0" or row["architecture_violations"] != "0":
        raise RuntimeError(f"Invariant failure for n={node} scenario={scenario_id} oc={oc}: {row}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digest-capacity", type=int, default=25)
    parser.add_argument("--outdir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.digest_capacity <= 0:
        raise ValueError("--digest-capacity must be positive")
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    records = test_records()
    jobs = [(node, scenario_id, seed, oc) for node, pairs in records.items()
            for scenario_id, seed in pairs for oc in range(4)]
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_one, *job, out, args.digest_capacity) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda r: (int(r["node_count"]), int(r["scenario_id"]), int(r["selected_oc"])))
    with (out / "matched_candidate_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for node in NODES:
        group = [r for r in rows if int(r["node_count"]) == node]
        summary.append({
            "Nodes": node,
            "Candidate runs": len(group),
            "Mean PDR (%)": sum(float(r["PDR"]) for r in group) / len(group),
            "Mean E2ED (ms)": sum(float(r["E2ED_ms"]) for r in group if r["E2ED_ms"] != "NA") /
                              sum(r["E2ED_ms"] != "NA" for r in group),
            "Mean ROR": sum(float(r["ROR_total"]) for r in group) / len(group),
            "Mean control transmissions": sum(float(r["control_transmissions"]) for r in group) / len(group),
            "Mean data hops": sum(float(r["data_hops"]) for r in group) / len(group),
        })
    with (out / "summary_by_node.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print("Nodes | PDR | E2ED_ms | ROR")
    for row in summary:
        print(f"{row['Nodes']:>5} | {row['Mean PDR (%)']:.2f} | {row['Mean E2ED (ms)']:.2f} | {row['Mean ROR']:.3f}")


if __name__ == "__main__":
    main()
