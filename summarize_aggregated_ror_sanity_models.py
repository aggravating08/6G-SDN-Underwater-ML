#!/usr/bin/env python3
"""Map fixed SVM/DTC/RF selections to the aggregated-control sanity ledger."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import load


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "underwater_partner_style_equivalent_2000"
RUNS = BASE / "AGGREGATED_ROR_SANITY_CAPACITY_40" / "matched_candidate_runs.csv"
OUT = BASE / "AGGREGATED_ROR_SANITY_CAPACITY_40_SVM_C20_G005_DTC3_L30_S60_WORST_BASELINE"
FEATURES = ("x", "y", "local_density", "speed")
LOSS_TIMEOUT_MS = 10_000.0
MODELS = {
    "SVM": BASE / "PARTNER_COMPATIBLE_C20_G005_DTC3_L30_S60_RF_FRACTIONAL_STUMP" / "partner_compatible_svm_pipeline.joblib",
    # Restored DTC: gini, depth=4, leaf=35, split=70.
    "DTC": BASE / "PARTNER_COMPATIBLE_C20_G005_DTC3_L30_S60_RF_FRACTIONAL_STUMP" / "partner_compatible_dtc_pipeline.joblib",
    "RF": BASE / "PARTNER_COMPATIBLE_C20_G005_DTC3_L30_S60_RF_FRACTIONAL_STUMP" / "partner_compatible_rf_pipeline.joblib",
}


def model_scores(model: object, rows: list[dict[str, str]]) -> np.ndarray:
    matrix = np.array([[float(row[col]) for col in FEATURES] for row in rows])
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, list(model.classes_).index(1)]
    return np.asarray(model.decision_function(matrix), dtype=float).reshape(-1)


def summarize(rows: list[dict[str, str]], name: str) -> dict[str, float | str | int]:
    generated = sum(int(row["generated_packets"]) for row in rows)
    delivered = sum(int(row["delivered_packets"]) for row in rows)
    control = sum(float(row["control_transmissions"]) for row in rows)
    data_hops = sum(float(row["data_hops"]) for row in rows)
    delay_denominator = sum(int(row["delivered_packets"]) for row in rows if row["E2ED_ms"] != "NA")
    delivered_delay_total = sum(float(row["E2ED_ms"]) * int(row["delivered_packets"])
                                for row in rows if row["E2ED_ms"] != "NA")
    delay = delivered_delay_total / delay_denominator
    loss_aware_delay = (delivered_delay_total + (generated - delivered) * LOSS_TIMEOUT_MS) / generated
    return {
        "Model": name,
        "PDR (%)": 100.0 * delivered / generated,
        "E2ED (ms)": delay,
        "Loss-aware delay (ms)": loss_aware_delay,
        "ROR": control / (control + data_hops),
        "Control transmissions": control,
        "Data-hop transmissions": data_hops,
        "Scenarios": len(rows),
    }


def worst_candidate(rows: list[dict[str, str]]) -> dict[str, str]:
    """Return the actual worst candidate, solely as a post-hoc comparison baseline."""
    def key(row: dict[str, str]) -> tuple[float, float, float, int]:
        delay = float("inf") if row["E2ED_ms"] == "NA" else float(row["E2ED_ms"])
        return (float(row["PDR"]), -delay, -float(row["ROR_total"]), int(row["selected_oc"]))
    return min(rows, key=key)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RUNS.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        groups[int(row["scenario_id"])].append(row)
    if not all(len(rows) == 4 for rows in groups.values()):
        raise RuntimeError("Every sanity scenario must contain four candidate OC runs")
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    for name, path in MODELS.items():
        model = load(path)
        for scenario_id, rows in groups.items():
            ordered = sorted(rows, key=lambda row: int(row["selected_oc"]))
            selected[name].append(ordered[int(np.argmax(model_scores(model, ordered)))])
    selected["Baseline worst-OC"] = [worst_candidate(rows) for rows in groups.values()]

    all_rows = []
    for node in (25, 50, 75, 100):
        for name in ("SVM", "DTC", "RF", "Baseline worst-OC"):
            subset = [row for row in selected[name] if int(row["node_count"]) == node]
            all_rows.append({"Nodes": node, **summarize(subset, name)})
    overall = [summarize(selected[name], name) for name in ("SVM", "DTC", "RF", "Baseline worst-OC")]
    for filename, rows in (("model_metrics_by_node.csv", all_rows), ("model_metrics_overall.csv", overall)):
        with (OUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print("Nodes | Model | PDR | E2ED_ms | ROR")
    for row in all_rows:
        print(f"{row['Nodes']:>5} | {row['Model']:<12} | {row['PDR (%)']:.2f} | {row['E2ED (ms)']:.2f} | {row['ROR']:.3f}")


if __name__ == "__main__":
    main()
