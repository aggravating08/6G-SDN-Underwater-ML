#!/usr/bin/env python3
"""Reference-style fixed-control density sensitivity experiment.

The base runner supplies frozen test selections, fair candidate reuse and
invariant checks.  This wrapper applies one fixed protocol at every density:
30-s HELLO, 60-s route cache, 120-s per-flow no-path cache, and explicit
periodic neighbour/LC/MC topology-update accounting.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import run_reference_style_fixed_control_evaluation as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_corrected"
RUNS = OUT / "candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
NODES = (25, 50, 75, 100)
METHODS = ("SVM", "DTC", "RF", "Baseline OC0")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    return math.nan if raw in ("", "NA", "NaN") else float(raw)


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = [value(row, key) for row in rows]
    values = [item for item in values if math.isfinite(item)]
    return sum(values) / len(values) if values else math.nan


def summary(rows: list[dict[str, str]], method: str, nodes: int) -> dict:
    use = [row for row in rows if row["Method"] == method and int(row["node_count"]) == nodes]
    return {
        "Nodes": nodes, "Method": method,
        "PDR %": mean(use, "PDR"), "E2ED ms": mean(use, "E2ED_ms"),
        "ROR total": mean(use, "ROR_total"), "ROR generated": mean(use, "ROR_generated"),
        "ROR reactive": mean(use, "ROR_reactive"),
        "HELLO/update packets": mean(use, "control_transmissions") - mean(use, "route_request_tx")
                                  - mean(use, "route_reply_tx") - mean(use, "mc_control_tx"),
        "HELLO broadcasts": mean(use, "hello_tx"),
        "Neighbour update receptions": mean(use, "neighbor_update_tx"),
        "LC topology summaries": mean(use, "lc_topology_update_tx"),
        "MC topology summaries": mean(use, "mc_topology_update_tx"),
        "Route requests": mean(use, "route_request_tx"), "MC fallbacks": mean(use, "mc_fallbacks"),
        "Failed attempts": mean(use, "failed_route_attempts"), "Negative-cache hits": mean(use, "negative_cache_hits"),
        "Scenarios": len(use),
    }


def trend(rows: list[dict]) -> list[dict]:
    svm = {int(row["Nodes"]): row for row in rows if row["Method"] == "SVM"}
    definitions = (("PDR %", "increasing"), ("E2ED ms", "decreasing"), ("ROR total", "increasing"))
    result = []
    for metric, expected in definitions:
        series = [svm[n][metric] for n in NODES]
        ok = all(b >= a for a, b in zip(series, series[1:])) if expected == "increasing" else \
             all(b <= a for a, b in zip(series, series[1:]))
        result.append({"Metric": metric, **{str(n): x for n, x in zip(NODES, series)},
                       "Expected trend": expected, "Pass/Fail": "PASS" if ok else "FAIL"})
    return result


def run_candidate(key: tuple[int, int, int, int]) -> dict[str, str]:
    nodes, scenario_id, seed, oc = key
    output = RUNS / f"n{nodes}_sid{scenario_id}_seed{seed}_oc{oc}.csv"
    if not output.exists():
        subprocess.run(
            [str(EXE), "--mode=run", f"--nodeCount={nodes}", f"--scenarioId={scenario_id}",
             f"--scenarioSeed={seed}", f"--selectedOc={oc}",
             "--fixedControlPolicy=true", "--helloSeconds=30", "--routeTtlSeconds=60",
             "--negativeRouteTtlSeconds=120", "--referenceUpdateAccounting=true", f"--output={output}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
    rows = read(output)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one candidate result in {output}")
    return rows[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    base.OUT = OUT
    base.RUNS = RUNS
    base.run_candidate = run_candidate
    base.main()

    mapped = read(OUT / "model_selected_oc_rows_with_network_metrics.csv")
    by_node = [summary(mapped, method, nodes) for nodes in NODES for method in METHODS]
    checks = trend(by_node)
    write(OUT / "final_metrics_by_node.csv", by_node)
    write(OUT / "svm_trend_check.csv", checks)
    (OUT / "configuration.json").write_text(json.dumps({
        "purpose": "separate reference-style fixed-control density sensitivity experiment",
        "hello_interval_seconds": 30,
        "route_ttl_seconds": 60,
        "negative_no_path_cache_seconds": 120,
        "density_adaptation": False,
        "delta_or_summary_suppression": False,
        "periodic_update_accounting": "one sensor broadcast per sender plus one reception-equivalent neighbour-state update per usable directed sensor edge; one LC summary per nonempty LC view; one MC summary per reachable LC--MC link",
        "ror_total": "all control packets / (all control packets + realized sensor data-hop transmissions)",
        "ror_generated": "all control packets / (all control packets + 200 generated source packets)",
        "e2ed": "generation time to destination arrival for delivered packets only; route-control wait is already included",
        "model_retraining": False,
        "frozen_model_selection_source": str(base.SOURCE.relative_to(ROOT)),
    }, indent=2) + "\n")
    print("\nREFERENCE-STYLE BY NODE")
    for row in by_node:
        print(row)
    print("\nSVM TREND CHECK")
    for row in checks:
        print(row)


if __name__ == "__main__":
    main()
