#!/usr/bin/env python3
"""Repair v10's reporting-only duplicate node_count merge without rerunning ML/ns-3."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/underwater_rebuild/current/final_aligned_v10_paper_trend_clean"
NODES = (25, 50, 75, 100)
METHODS = ("SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, na_values=["NA", "NaN", ""])


def write(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUT / name, index=False)


def weighted_e2ed(rows: pd.DataFrame) -> float:
    valid = rows.dropna(subset=["E2ED_ms"])
    delivered = valid.delivered_packets.sum()
    return float((valid.E2ED_ms * valid.delivered_packets).sum() / delivered) if delivered else math.nan


def ror(rows: pd.DataFrame) -> float:
    control = rows.control_transmissions.sum()
    generated = rows.generated_packets.sum()
    return float(control / (control + generated)) if control + generated else math.nan


def summary(rows: pd.DataFrame, method: str) -> dict:
    return {"Model": method,
            "Mean PDR (%)": float(100 * rows.delivered_packets.sum() / rows.generated_packets.sum()),
            "Mean E2ED (ms)": weighted_e2ed(rows), "Mean ROR": ror(rows)}


def diagnostic(rows: pd.DataFrame, method: str, nodes: int) -> dict:
    use = rows[rows.node_count.eq(nodes)]
    delivered = use.delivered_packets.sum()
    def packet_weighted(column: str) -> float:
        return float((use[column] * use.delivered_packets).sum() / delivered) if delivered else math.nan
    return {"Nodes": nodes, "Model": method,
            "Mean S-D distance (m)": float(use.mean_source_destination_distance_m.mean()),
            "Shortest path hops": float(use.mean_static_shortest_hops.mean()),
            "Delivered hops": packet_weighted("mean_delivered_hop_count"),
            "Path length (m)": packet_weighted("mean_delivered_path_length_m"),
            "Relay wait (ms)": packet_weighted("mean_delivered_relay_wait_ms"),
            "Candidate count": packet_weighted("mean_delivered_relay_candidate_count"),
            "Connected-pair ratio": float(use.connected_flow_pair_ratio.mean()),
            "Delivered scenarios": int(use.E2ED_ms.notna().sum()),
            "Delivered packets": int(delivered)}


def main() -> None:
    selected = read("selected_oc_per_test_scenario.csv")
    candidates = read("candidate_outcomes_with_labels.csv")
    split = json.loads((OUT / "split_manifest.json").read_text())
    test_ids = set(split["test"])
    candidate_map = candidates.rename(columns={"selected_oc": "candidate_oc"})
    result_parts = []
    for method in METHODS[:3]:
        picks = selected[selected.Method.eq(method)][["scenario_id", "selected_oc", "true_oc", "correct", "score"]]
        mapped = picks.merge(candidate_map, left_on=["scenario_id", "selected_oc"],
                             right_on=["scenario_id", "candidate_oc"], how="left", validate="one_to_one")
        if mapped.PDR.isna().any() or mapped.node_count.isna().any():
            raise RuntimeError(f"Missing ledger metric/node count after {method} mapping")
        mapped["Model"] = method
        result_parts.append(mapped)
    baseline = candidate_map[candidate_map.candidate_oc.eq(0) & candidate_map.scenario_id.isin(test_ids)].copy()
    baseline["Model"] = "Baseline OC0"
    baseline["selected_oc"] = 0
    result_parts.append(baseline)
    all_rows = pd.concat(result_parts, ignore_index=True)
    if len(all_rows) != 1200 or not all_rows.scenario_id.isin(test_ids).all():
        raise RuntimeError("Expected exactly 4 x 300 test-only selected candidate rows")
    if all_rows.groupby(["Model", "scenario_id"]).size().ne(1).any():
        raise RuntimeError("Selected candidate mapping is not one result per model/scenario")

    overall = pd.DataFrame([summary(all_rows[all_rows.Model.eq(method)], method) for method in METHODS])
    bynode = pd.DataFrame([
        {"Nodes": nodes, **summary(all_rows[(all_rows.Model.eq(method)) & (all_rows.node_count.eq(nodes))], method)}
        for nodes in NODES for method in METHODS
    ])[["Nodes", "Model", "Mean PDR (%)", "Mean E2ED (ms)", "Mean ROR"]]
    diagnostics = pd.DataFrame([
        diagnostic(all_rows[all_rows.Model.eq(method)], method, nodes)
        for nodes in NODES for method in METHODS
    ])
    write("model_selected_oc_rows_with_network_metrics.csv", all_rows)
    write("final_network_metrics_overall.csv", overall)
    write("final_network_metrics_by_node.csv", bynode)
    write("final_density_diagnostics.csv", diagnostics)

    audit_path = OUT / "final_alignment_audit.txt"
    audit = audit_path.read_text().rstrip().splitlines()
    audit.append("reporting repair: rebuilt model-to-ledger joins from saved test predictions and the same candidate ledger; no ns-3 run, label, split, model, or metric was changed")
    audit.append(f"reporting repair test-only rows: {len(all_rows)}; missing node_count/PDR/E2ED/ROR joins: {int(all_rows[['node_count','PDR','ROR_generated']].isna().any(axis=1).sum())}")
    audit_path.write_text("\n".join(audit) + "\n")
    print(overall.to_string(index=False))
    print(bynode.to_string(index=False))


if __name__ == "__main__":
    main()
