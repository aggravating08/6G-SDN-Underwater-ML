#!/usr/bin/env python3
"""Evaluate frozen v4 SVM/DTC/RF selections under the completed v7 protocol."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import run_reference_paper_matching_trend_v7_bounded_aggregated_ror as v7


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_v7_bounded_aggregated_ror"
V4 = ROOT / "results/underwater_rebuild/current/reference_paper_matching_trend_v4_long_distance_ml"
NODES = (25, 50, 75, 100)
MODELS = ("SVM", "DTC", "RF")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    predictions = read(V4 / "test_model_oc_predictions.csv")
    svm_existing = read(OUT / "svm_selected_raw_runs.csv")
    by_model_node: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in svm_existing:
        row = dict(row); row["Model"] = "SVM"; by_model_node[("SVM", int(row["node_count"]))].append(row)
    # Each DTC/RF choice is rerun once with the same v7 capacity-25 protocol.
    for model in ("DTC", "RF"):
        picks = [row for row in predictions if row["Model"] == model]
        if len(picks) != 300:
            raise RuntimeError(f"Expected 300 frozen {model} test selections")
        for pick in picks:
            nodes, sid, oc = int(pick["node_count"]), int(pick["scenario_id"]), int(pick["predicted_oc"])
            result = v7.run_trial(nodes, sid, oc, 25, OUT / "all_models")
            if result["generated_packets"] != "200" or result["OC_data_hops"] != "0" or result["architecture_violations"] != "0":
                raise RuntimeError(f"Invariant violation: {model}, n={nodes}, sid={sid}, oc={oc}")
            result = dict(result); result["Model"] = model
            by_model_node[(model, nodes)].append(result)
    if any(len(by_model_node[(m, n)]) != 75 for m in MODELS for n in NODES):
        raise RuntimeError("Expected 75 held-out results for every model/density cell")
    raw = [row for m in MODELS for n in NODES for row in by_model_node[(m, n)]]
    table = []
    for n in NODES:
        for model in MODELS:
            result = v7.summarize(by_model_node[(model, n)], n)
            result["Model"] = model
            table.append(result)
    overall = []
    for model in MODELS:
        rows = [r for n in NODES for r in by_model_node[(model, n)]]
        result = v7.summarize(rows, "All")
        result["Model"] = model
        overall.append(result)
    write(OUT / "all_models_selected_raw_runs.csv", raw)
    write(OUT / "all_models_network_metrics_by_node.csv", table)
    write(OUT / "all_models_network_metrics_overall.csv", overall)
    # ML metrics are frozen from the v4 long-distance ML experiment; they are
    # included for complete reporting, not recalculated or tuned here.
    ml = read(V4 / "final_ml_metrics.csv")
    write(OUT / "all_models_frozen_ml_metrics.csv", [row for row in ml if row["Model"] in MODELS])
    print("V7 NETWORK METRICS BY NODE")
    for row in table:
        print(row)
    print("V7 NETWORK METRICS OVERALL")
    for row in overall:
        print(row)


if __name__ == "__main__":
    main()
