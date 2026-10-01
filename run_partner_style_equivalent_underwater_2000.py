#!/usr/bin/env python3
"""Transparent 2,000-scenario underwater counterpart to the partner workflow.

This is deliberately *not* a copy of the partner notebook's post-processing.
It retains a simple feature-derived OC label, but evaluates a real four-AUV
top-1 decision against matched ns-3 analytical runs.  PDR, delivered-packet
E2ED, and ROR are read unchanged from the candidate ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import run_final_aligned_v10_paper_trend_clean as core
import run_final_underwater_ppt_noncognitive_clean as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "underwater_partner_style_equivalent_2000"
FEATURES = ["x", "y", "local_density", "speed"]
NODES = (25, 50, 75, 100)
METHODS = ("SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0")


def write(name: str, frame: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def unit_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    lo, hi = values.min(), values.max()
    if np.isclose(lo, hi):
        return pd.Series(1.0, index=values.index)
    score = (values - lo) / (hi - lo)
    return score if higher_is_better else 1.0 - score


def feature_rule_dataset(features: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen density/centrality/stability OC rule per scenario."""
    rows = []
    required = {"scenario_id", "scenario_seed", "node_count", "auv_id", *FEATURES}
    missing = required - set(features.columns)
    if missing:
        raise RuntimeError(f"Feature ledger missing {sorted(missing)}")
    for scenario_id, group in features.groupby("scenario_id", sort=True):
        g = group.copy().sort_values("auv_id")
        if len(g) != 4 or set(g.auv_id) != {0, 1, 2, 3}:
            raise RuntimeError(f"Scenario {scenario_id} does not contain OC0--OC3")
        centre_distance = np.hypot(g.x - 250.0, g.y - 250.0)
        g["distance_to_center_m"] = centre_distance
        g["density_score"] = unit_score(g.local_density, True)
        g["centrality_score"] = unit_score(g.distance_to_center_m, False)
        g["stability_score"] = unit_score(g.speed, False)
        g["oc_suitability_score"] = (
            g.density_score + g.centrality_score + g.stability_score
        ) / 3.0
        # Deterministic tie order: density, centrality, stability, then OC ID.
        best = g.sort_values(
            ["oc_suitability_score", "local_density", "distance_to_center_m", "speed", "auv_id"],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        ).iloc[0].auv_id
        g["is_oc"] = g.auv_id.eq(best).astype(int)
        rows.append(g)
    data = pd.concat(rows, ignore_index=True)
    if len(data) != 8000 or not data.groupby("scenario_id").size().eq(4).all():
        raise RuntimeError("Expected 2,000 scenarios and 8,000 candidate rows")
    if not data.groupby("scenario_id").is_oc.sum().eq(1).all():
        raise RuntimeError("Each scenario must have exactly one OC")
    return data


def evaluate_models(dataset: pd.DataFrame, candidates: pd.DataFrame) -> None:
    core.OUT = OUT
    core.FEATURES = FEATURES
    split = core.grouped_split(dataset)
    (OUT / "split_manifest.json").write_text(json.dumps(split, indent=2) + "\n")
    train = dataset[dataset.scenario_id.isin(split["train"])].copy()
    validation = dataset[dataset.scenario_id.isin(split["validation"])].copy()
    test = dataset[dataset.scenario_id.isin(split["test"])].copy()
    if (len(train), len(validation), len(test)) != (5600, 1200, 1200):
        raise RuntimeError("Expected grouped 1400/300/300 scenario split")

    params, _ = core.choose_models(train, validation)
    trainval = pd.concat([train, validation], ignore_index=True)
    model_rows, prediction_rows = [], []
    for name, chosen_params in params.items():
        model = core.make_model(name, chosen_params)
        model.fit(trainval[FEATURES], trainval.is_oc)
        dump(model, OUT / f"{name.lower()}_pipeline.joblib")
        binary = model.predict(test[FEATURES])
        precision, recall, _, _ = precision_recall_fscore_support(
            test.is_oc, binary, average="binary", zero_division=0
        )
        picked = core.select_oc(model, test, name)
        picked["Model"] = f"{name}-selected OC"
        prediction_rows.append(picked)
        model_rows.append({
            "Model": name,
            "Binary accuracy": accuracy_score(test.is_oc, binary),
            "Binary precision": precision,
            "Binary recall": recall,
            "Binary F1": f1_score(test.is_oc, binary, zero_division=0),
            "Top-1 OC accuracy": picked.correct.mean(),
            "Top-1 macro F1": core.top1_macro_f1(picked),
            "Validation-selected hyperparameters": json.dumps(chosen_params, sort_keys=True),
        })
    predictions = pd.concat(prediction_rows, ignore_index=True)
    write("model_selected_oc_per_test_scenario.csv", predictions)
    write("final_ml_accuracy_comparison.csv", pd.DataFrame(model_rows))

    candidate_map = candidates.rename(columns={"selected_oc": "candidate_oc"})
    if not candidate_map.groupby("scenario_id").size().eq(4).all():
        raise RuntimeError("Candidate ledger is not matched OC0--OC3 data")
    selected_parts = []
    for method in METHODS[:3]:
        picks = predictions[predictions.Model.eq(method)][
            ["scenario_id", "selected_oc", "true_oc", "correct", "score"]
        ]
        mapped = picks.merge(
            candidate_map,
            left_on=["scenario_id", "selected_oc"],
            right_on=["scenario_id", "candidate_oc"],
            how="left",
            validate="one_to_one",
        )
        if len(mapped) != 300 or mapped.PDR.isna().any():
            raise RuntimeError(f"Missing metric mapping for {method}")
        mapped["Model"] = method
        selected_parts.append(mapped)
    baseline = candidate_map[
        candidate_map.scenario_id.isin(split["test"]) & candidate_map.candidate_oc.eq(0)
    ].copy()
    baseline["selected_oc"] = 0
    baseline["Model"] = "Baseline OC0"
    selected_parts.append(baseline)
    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) != 1200:
        raise RuntimeError("Expected 1,200 test model-to-candidate mappings")
    write("model_selected_oc_rows_with_actual_network_metrics.csv", selected)

    def summary(rows: pd.DataFrame, method: str) -> dict:
        delivered = rows.delivered_packets.sum()
        e2ed_rows = rows.dropna(subset=["E2ED_ms"])
        e2ed_delivered = e2ed_rows.delivered_packets.sum()
        mean_e2ed = (
            float((e2ed_rows.E2ED_ms * e2ed_rows.delivered_packets).sum() / e2ed_delivered)
            if e2ed_delivered else float("nan")
        )
        control = rows.control_transmissions.sum()
        data_hops = rows.data_hops.sum()
        return {
            "Model": method,
            "Mean PDR (%)": float(100.0 * delivered / rows.generated_packets.sum()),
            "Mean E2ED (ms)": mean_e2ed,
            "Mean ROR": float(control / (control + data_hops)) if control + data_hops else float("nan"),
            "Test scenarios": int(rows.scenario_id.nunique()),
        }

    overall = pd.DataFrame([summary(selected[selected.Model.eq(m)], m) for m in METHODS])
    per_node = pd.DataFrame([
        {"Nodes": n, **summary(selected[(selected.Model.eq(m)) & (selected.node_count.eq(n))], m)}
        for n in NODES for m in METHODS
    ])
    write("final_network_metrics_overall.csv", overall)
    write("final_network_metrics_by_node.csv", per_node)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # This invokes the non-cognitive, full MC fallback profile fixed in the
    # existing clean runner: 500x500, 100-m sensor range, 300-m controller
    # range, four AUVs, 200 packets/scenario, and matched OC0--OC3 trials.
    base.OUT = OUT
    base.RUNS = OUT / "candidate_runs"
    features, candidates = base.generate_ledgers()
    base.validate(features, candidates)
    dataset = feature_rule_dataset(features)
    write("underwater_feature_rule_2000_scenarios_8000_candidates.csv", dataset)
    evaluate_models(dataset, candidates)
    (OUT / "partner_side_by_side_audit.txt").write_text(
        "Partner-style comparison implemented safely\n"
        "- 2,000 underwater scenarios / 8,000 OC candidate rows: PASS\n"
        "- Feature-rule label: density + centrality + lower speed: PASS\n"
        "- ML inputs restricted to x,y,local_density,speed: PASS\n"
        "- Grouped 1400/300/300 scenario split: PASS\n"
        "- Scenario Top-1 accuracy is primary; row-level leakage is not used: PASS\n"
        "- PDR, E2ED, and ROR are actual candidate-run outputs; no post-processing: PASS\n"
        "- No E2ED division by sqrt(node_count), no synthetic baseline degradation,\n"
        "  no forced SVM/DTC/RF order, and no top-2 accuracy substitution: PASS\n"
    )
    print(f"Complete: {OUT}")


if __name__ == "__main__":
    main()
