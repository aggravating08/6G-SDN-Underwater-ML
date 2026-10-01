#!/usr/bin/env python3
"""Run reproducible, feature-only ML-selected underwater network evaluations.

The script deliberately never trains a model and never reads network outcomes
when selecting an OC.  It generates unseen candidate features with randy.cc,
uses saved SVM/DTC/RF artifacts, executes only unique selected candidates, and
maps the resulting matched candidate row back to each model.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from underwater_ml_pipeline import FEATURES, validate_long_dataset, scenario_predictions

ROOT = Path(__file__).resolve().parent


def invoke(expression: str) -> None:
    subprocess.run(["./ns3", "run", expression], cwd=ROOT, check=True)


def build_feature_rows(nodes: List[int], scenarios: int, base_seed: int, outdir: Path) -> pd.DataFrame:
    chunks = []
    for index, count in enumerate(nodes):
        # This range is disjoint from the 2,000 label-training seeds.
        seed = base_seed + index * 1_000_000
        scenario_id = 9_000_000 + index * 10_000
        csv = outdir / f"features_{count}.csv"
        invoke(f"scratch/randy --mode=labels --nodeCount={count} --runs={scenarios} "
               f"--scenarioId={scenario_id} --baseSeed={seed} --output={csv}")
        chunks.append(pd.read_csv(csv))
    features = pd.concat(chunks, ignore_index=True)
    validate_long_dataset(features)
    features.to_csv(outdir / "unseen_features.csv", index=False)
    return features


def selected_models(features: pd.DataFrame, modeldir: Path) -> pd.DataFrame:
    keys = features[["scenario_id", "scenario_seed", "node_count"]].drop_duplicates().sort_values("scenario_id")
    for name in ("SVM", "DTC", "RF"):
        model = joblib.load(modeldir / f"{name.lower()}_oc_pipeline.joblib")
        predicted = scenario_predictions(model, features).set_index("scenario_id")
        keys[name + "_selected_oc"] = keys.scenario_id.map(predicted.predicted_oc).astype(int)
    return keys


def run_candidates(selections: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    candidates: List[dict] = []
    seen = set()
    for row in selections.itertuples(index=False):
        requested = {int(row.SVM_selected_oc), int(row.DTC_selected_oc), int(row.RF_selected_oc)}
        for oc in sorted(requested):
            key = (int(row.node_count), int(row.scenario_id), oc)
            if key in seen:
                continue
            seen.add(key)
            csv = outdir / f"candidate_n{row.node_count}_s{row.scenario_id}_oc{oc}.csv"
            invoke(f"scratch/randy --mode=run --nodeCount={row.node_count} --scenarioSeed={row.scenario_seed} "
                   f"--scenarioId={row.scenario_id} --selectedOc={oc} --output={csv}")
            result = pd.read_csv(csv, na_values=["NA"]).iloc[0].to_dict()
            candidates.append(result)
    raw = pd.DataFrame(candidates)
    if not raw.generated_packets.eq(200).all():
        raise RuntimeError("Invariant failure: a candidate did not generate 200 packets.")
    if not raw.OC_data_hops.eq(0).all() or not raw.architecture_violations.eq(0).all():
        raise RuntimeError("Architecture invariant failure: AUV payload forwarding was observed.")
    # Candidate identity must never alter the generated physical scenario.
    for sid, group in raw.groupby(["node_count", "scenario_id"]):
        if group.topology_hash.nunique() != 1 or group.cognitive_state_hash.nunique() != 1:
            raise RuntimeError(f"Matched scenario hash failure: {sid}")
    raw.to_csv(outdir / "candidate_network_runs.csv", index=False)
    return raw


def map_models(selections: pd.DataFrame, raw: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for item in selections.itertuples(index=False):
        for model in ("SVM", "DTC", "RF"):
            oc = int(getattr(item, model + "_selected_oc"))
            match = raw[(raw.node_count == item.node_count) & (raw.scenario_id == item.scenario_id) &
                        (raw.selected_oc == oc)]
            if len(match) != 1:
                raise RuntimeError(f"Missing candidate row for {model}, scenario {item.scenario_id}, OC{oc}")
            row = match.iloc[0].to_dict()
            row["model"] = model
            rows.append(row)
    outcome = pd.DataFrame(rows)
    outcome.to_csv(outdir / "model_network_results.csv", index=False)
    return outcome


def ci95(values: pd.Series) -> float:
    a = values.dropna().to_numpy(dtype=float)
    if len(a) < 2:
        return float("nan")
    return float(1.96 * np.std(a, ddof=1) / np.sqrt(len(a)))


def summary(outcome: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for (nodes, model), group in outcome.groupby(["node_count", "model"], sort=True):
        for metric in ("PDR", "E2ED_ms", "ROR_total", "ROR_reactive"):
            vals = group[metric].dropna()
            rows.append({"node_count": nodes, "model": model, "metric": metric,
                         "mean": vals.mean(), "std": vals.std(ddof=1), "ci95_half_width": ci95(vals),
                         "scenarios": len(vals), "undefined": len(group) - len(vals)})
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "network_summary.csv", index=False)
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", nargs="+", type=int, default=[25, 50, 75, 100])
    p.add_argument("--scenarios", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=9_000_029)
    p.add_argument("--models", type=Path, default=ROOT / "results/underwater_rebuild/current/models")
    p.add_argument("--outdir", type=Path, default=ROOT / "results/underwater_rebuild/current/network_smoke")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    features = build_feature_rows(args.nodes, args.scenarios, args.base_seed, args.outdir)
    selections = selected_models(features, args.models)
    selections.to_csv(args.outdir / "model_selections.csv", index=False)
    raw = run_candidates(selections, args.outdir)
    results = map_models(selections, raw, args.outdir)
    print(summary(results, args.outdir).to_string(index=False))


if __name__ == "__main__":
    main()
