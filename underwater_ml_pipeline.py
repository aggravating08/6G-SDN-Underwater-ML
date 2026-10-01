#!/usr/bin/env python3
"""Feature-only OC-selection pipeline for the underwater SDN study.

The four ML inputs are deliberately and exclusively: ``x``, ``y``,
``local_density``, and ``speed``.  Training labels are created before any
network evaluation using the documented equal-weight controller-suitability
rule: high local density, central position relative to (250, 250), and low
mobility.  PDR, E2ED, routing overhead, channel state, and topology outcomes
are not used to generate these labels and are not ML input features.

The legacy ``generate-network-labels`` command remains only so old
experiments can be reproduced.  Its output is not a valid training target for
this feature-only ML methodology.  Evaluation is scenario-level: the model
scores all four candidates and selects exactly one OC.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["x", "y", "local_density", "speed"]
ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results" / "underwater_rebuild"
PROJECT_DATASET = ROOT / "results" / "underwater_partner_style_equivalent_2000" / "underwater_feature_rule_2000_scenarios_8000_candidates.csv"
PROJECT_RESULTS = ROOT / "results" / "final_three_file_project"
PROJECT_NODES = (25, 50, 75, 100)
LOSS_TIMEOUT_MS = 10_000.0


def validate_long_dataset(df: pd.DataFrame) -> None:
    required = {"scenario_id", "scenario_seed", "node_count", "auv_id", *FEATURES, "is_oc"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    counts = df.groupby("scenario_id").size()
    if not counts.eq(4).all():
        raise ValueError("Each scenario must contain exactly four AUV rows.")
    positives = df.groupby("scenario_id")["is_oc"].sum()
    if not positives.eq(1).all():
        raise ValueError("Each scenario must have exactly one is_oc=1 label.")
    if set(df["auv_id"].unique()) - {0, 1, 2, 3}:
        raise ValueError("auv_id must be in 0..3.")


def diagnostic_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the documented feature-only label rule for auditability."""
    rows: List[pd.DataFrame] = []
    for sid, group in df.groupby("scenario_id", sort=False):
        g = group.copy()
        g["distance_to_center"] = np.hypot(g.x - 250.0, g.y - 250.0)

        def high(values: pd.Series) -> np.ndarray:
            lo, hi = values.min(), values.max()
            return np.ones(len(values)) if hi == lo else ((values - lo) / (hi - lo)).to_numpy()

        g["density_score"] = high(g.local_density)
        g["centrality_score"] = high(-g.distance_to_center)
        g["stability_score"] = high(-g.speed)
        g["oc_score"] = (g.density_score + g.centrality_score + g.stability_score) / 3.0
        vals = np.sort(g.oc_score.to_numpy())
        g["winning_margin"] = vals[-1] - vals[-2]
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def generate_dataset(scenarios_per_density: int, output: Path, base_seed: int) -> pd.DataFrame:
    """Generate feature-only suitability labels without network simulation metrics."""
    output.parent.mkdir(parents=True, exist_ok=True)
    pieces: List[pd.DataFrame] = []
    for density_index, node_count in enumerate((25, 50, 75, 100)):
        tmp = output.parent / f".labels_{node_count}.csv"
        # Distinct deterministic seeds and globally unique scenario IDs.
        first_id = density_index * scenarios_per_density
        first_seed = base_seed + density_index * 1_000_000
        command = ["./ns3", "run", f"scratch/randy --mode=labels --nodeCount={node_count}"
                   f" --runs={scenarios_per_density} --scenarioId={first_id}"
                   f" --baseSeed={first_seed} --output={tmp}"]
        subprocess.run(command, cwd=ROOT, check=True)
        pieces.append(pd.read_csv(tmp))
        tmp.unlink()
    data = pd.concat(pieces, ignore_index=True)
    validate_long_dataset(data)
    data.to_csv(output, index=False)
    diagnostic_scores(data).to_csv(output.with_name(output.stem + "_label_diagnostics.csv"), index=False)
    return data


def generate_network_label_dataset(scenarios_per_density: int, output: Path, base_seed: int) -> pd.DataFrame:
    """Generate MC-free performance labels while preserving four ML inputs only.

    randy.cc evaluates all four matched candidate OCs, chooses exactly one by
    lexicographic highest PDR / lowest defined E2ED / lowest ROR / OC ID, and
    writes the outcome counters to a companion diagnostic CSV.  The returned
    training CSV never includes those outcome columns.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    pieces: List[pd.DataFrame] = []
    diagnostics: List[pd.DataFrame] = []
    for density_index, node_count in enumerate((25, 50, 75, 100)):
        tmp = output.parent / f".network_labels_{node_count}.csv"
        diag = output.parent / f".network_label_diagnostics_{node_count}.csv"
        first_id = density_index * scenarios_per_density
        first_seed = base_seed + density_index * 1_000_000
        command = ["./ns3", "run", f"scratch/randy --mode=network-labels --nodeCount={node_count}"
                   f" --runs={scenarios_per_density} --scenarioId={first_id}"
                   f" --baseSeed={first_seed} --output={tmp} --diagnosticOutput={diag}"]
        subprocess.run(command, cwd=ROOT, check=True)
        pieces.append(pd.read_csv(tmp))
        diagnostics.append(pd.read_csv(diag, na_values=["NA"]))
        tmp.unlink()
        diag.unlink()
    data = pd.concat(pieces, ignore_index=True)
    validate_long_dataset(data)
    data.to_csv(output, index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        output.with_name(output.stem + "_network_label_diagnostics.csv"), index=False
    )
    return data


def _scenario_frame(df: pd.DataFrame) -> pd.DataFrame:
    winner = df.loc[df.is_oc.eq(1), ["scenario_id", "node_count", "auv_id"]].rename(columns={"auv_id": "winner"})
    return winner.sort_values("scenario_id").reset_index(drop=True)


def build_split(data: pd.DataFrame, seed: int = 2026) -> Dict[str, List[int]]:
    """70/15/15 grouped split; stratify scenario labels by node count + winning AUV."""
    scenario = _scenario_frame(data)
    strata = scenario.node_count.astype(str) + "_" + scenario.winner.astype(str)
    first = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_i, hold_i = next(first.split(scenario, strata))
    train = scenario.iloc[train_i]
    hold = scenario.iloc[hold_i]
    hold_strata = hold.node_count.astype(str) + "_" + hold.winner.astype(str)
    second = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_i, test_i = next(second.split(hold, hold_strata))
    return {"train": train.scenario_id.astype(int).tolist(),
            "validation": hold.iloc[val_i].scenario_id.astype(int).tolist(),
            "test": hold.iloc[test_i].scenario_id.astype(int).tolist()}


def models() -> Dict[str, object]:
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=100.0, gamma=0.02, class_weight="balanced",
                               probability=False, random_state=42)),
        ]),
        "DTC": DecisionTreeClassifier(criterion="gini", max_depth=None,
                                       min_samples_leaf=10, min_samples_split=80,
                                       max_features=1, splitter="best",
                                       class_weight=None, random_state=42),
        "RF": RandomForestClassifier(n_estimators=15, criterion="gini", max_depth=1,
                                      min_samples_leaf=2, min_samples_split=20,
                                      max_features=4, max_samples=0.35,
                                      bootstrap=True, class_weight="balanced",
                                      random_state=42, n_jobs=-1),
    }


def positive_scores(model: object, rows: pd.DataFrame) -> np.ndarray:
    try:
        probs = model.predict_proba(rows[FEATURES])
        classes = list(model.classes_)
        return probs[:, classes.index(1)]
    except (AttributeError, NotImplementedError):
        # ``probability=False`` is one of the requested original-SVM settings.
        # Its decision score preserves the candidate ordering needed for Top-1.
        return np.asarray(model.decision_function(rows[FEATURES]), dtype=float)


def scenario_predictions(model: object, rows: pd.DataFrame) -> pd.DataFrame:
    answer: List[dict] = []
    for sid, group in rows.groupby("scenario_id", sort=True):
        group = group.sort_values("auv_id")
        scores = positive_scores(model, group)
        predicted = int(group.iloc[int(np.argmax(scores))].auv_id)
        truth = int(group.loc[group.is_oc.eq(1), "auv_id"].iloc[0])
        answer.append({"scenario_id": int(sid), "node_count": int(group.node_count.iloc[0]),
                       "true_oc": truth, "predicted_oc": predicted,
                       "correct": int(predicted == truth), "best_score": float(scores.max())})
    return pd.DataFrame(answer)


def summarize_predictions(pred: pd.DataFrame) -> Tuple[float, float, np.ndarray, pd.DataFrame]:
    accuracy = accuracy_score(pred.true_oc, pred.predicted_oc)
    macro_f1 = f1_score(pred.true_oc, pred.predicted_oc, labels=[0,1,2,3], average="macro", zero_division=0)
    cm = confusion_matrix(pred.true_oc, pred.predicted_oc, labels=[0,1,2,3])
    by_density = pred.groupby("node_count").correct.agg(["mean", "count"]).reset_index().rename(columns={"mean": "top1_accuracy", "count": "scenarios"})
    return accuracy, macro_f1, cm, by_density


def train(data: pd.DataFrame, outdir: Path, manifest: Path | None, seed: int) -> pd.DataFrame:
    validate_long_dataset(data)
    outdir.mkdir(parents=True, exist_ok=True)
    split = json.loads(manifest.read_text()) if manifest else build_split(data, seed)
    if manifest is None:
        (outdir / "split_manifest.json").write_text(json.dumps(split, indent=2))
    else:
        (outdir / "split_manifest.json").write_text(json.dumps(split, indent=2))
    result_rows: List[dict] = []
    for name, model in models().items():
        train_rows = data[data.scenario_id.isin(split["train"])]
        model.fit(train_rows[FEATURES], train_rows.is_oc)
        joblib.dump(model, outdir / f"{name.lower()}_oc_pipeline.joblib")
        for part in ("validation", "test"):
            prediction = scenario_predictions(model, data[data.scenario_id.isin(split[part])])
            prediction.to_csv(outdir / f"{name.lower()}_{part}_predictions.csv", index=False)
            accuracy, macro_f1, cm, density = summarize_predictions(prediction)
            pd.DataFrame(cm, index=["true_OC0","true_OC1","true_OC2","true_OC3"],
                         columns=["pred_OC0","pred_OC1","pred_OC2","pred_OC3"]).to_csv(outdir / f"{name.lower()}_{part}_confusion_matrix.csv")
            density.to_csv(outdir / f"{name.lower()}_{part}_by_node_count.csv", index=False)
            result_rows.append({"model": name, "split": part, "top1_accuracy": accuracy,
                                "macro_f1": macro_f1, "scenarios": len(prediction),
                                "features": "|".join(FEATURES)})
    results = pd.DataFrame(result_rows)
    results.to_csv(outdir / "ml_results.csv", index=False)
    return results


def sweep_original_svm(data: pd.DataFrame, outdir: Path, manifest: Path,
                       seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exhaustively validate only C, gamma, class_weight and probability.

    The SVM architecture, four raw input features, scaler, scenario-level
    Top-1 evaluation, and supplied scenario split are otherwise unchanged.
    The held-out test partition is not accessed until one configuration has
    been selected entirely by validation accuracy.
    """
    validate_long_dataset(data)
    outdir.mkdir(parents=True, exist_ok=True)
    split = json.loads(manifest.read_text())
    (outdir / "split_manifest.json").write_text(json.dumps(split, indent=2))
    train_rows = data[data.scenario_id.isin(split["train"])]
    validation_rows = data[data.scenario_id.isin(split["validation"])]

    values_c = [1, 5, 10, 20, 30, 50, 100]
    values_gamma: List[object] = ["scale", 0.001, 0.003, 0.005, 0.01, 0.03]
    values_weight: List[object] = [None, "balanced"]
    values_probability = [True, False]
    rows: List[dict] = []
    combo_index = 0
    for c in values_c:
        for gamma in values_gamma:
            for class_weight in values_weight:
                for probability in values_probability:
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("classifier", SVC(kernel="rbf", C=c, gamma=gamma,
                                           class_weight=class_weight, probability=probability,
                                           random_state=seed)),
                    ])
                    model.fit(train_rows[FEATURES], train_rows.is_oc)
                    prediction = scenario_predictions(model, validation_rows)
                    accuracy, macro_f1, _, _ = summarize_predictions(prediction)
                    rows.append({
                        "combination": combo_index,
                        "C": c,
                        "gamma": gamma,
                        "class_weight": "None" if class_weight is None else class_weight,
                        "probability": probability,
                        "validation_top1_accuracy": accuracy,
                        "validation_macro_f1": macro_f1,
                    })
                    combo_index += 1

    sweep = pd.DataFrame(rows).sort_values(
        ["validation_top1_accuracy", "validation_macro_f1", "combination"],
        ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)
    sweep.to_csv(outdir / "svm_validation_sweep.csv", index=False)
    best = sweep.iloc[0]

    # Only after validation selection do we form the final train+validation fit
    # and access the held-out test partition exactly once.
    test_rows = data[data.scenario_id.isin(split["test"])]
    train_validation = pd.concat([train_rows, validation_rows], ignore_index=True)
    selected_weight = None if best.class_weight == "None" else str(best.class_weight)
    selected_gamma: object = "scale" if best.gamma == "scale" else float(best.gamma)
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=float(best.C), gamma=selected_gamma,
                           class_weight=selected_weight, probability=bool(best.probability),
                           random_state=seed)),
    ])
    final_model.fit(train_validation[FEATURES], train_validation.is_oc)
    test_prediction = scenario_predictions(final_model, test_rows)
    test_prediction.to_csv(outdir / "svm_test_predictions.csv", index=False)
    test_accuracy, test_f1, test_cm, test_density = summarize_predictions(test_prediction)
    pd.DataFrame(test_cm, index=["true_OC0", "true_OC1", "true_OC2", "true_OC3"],
                 columns=["pred_OC0", "pred_OC1", "pred_OC2", "pred_OC3"]).to_csv(
                     outdir / "svm_test_confusion_matrix.csv"
                 )
    test_density.to_csv(outdir / "svm_test_by_node_count.csv", index=False)
    joblib.dump(final_model, outdir / "svm_oc_pipeline.joblib")
    summary = pd.DataFrame([
        {"model": "SVM", "split": "validation", "top1_accuracy": float(best.validation_top1_accuracy),
         "macro_f1": float(best.validation_macro_f1), "scenarios": int(validation_rows.scenario_id.nunique())},
        {"model": "SVM", "split": "test", "top1_accuracy": test_accuracy,
         "macro_f1": test_f1, "scenarios": int(test_rows.scenario_id.nunique())},
    ])
    summary.to_csv(outdir / "ml_results.csv", index=False)
    return sweep, summary


def predict(models_dir: Path, feature_csv: Path, output: Path) -> pd.DataFrame:
    rows = pd.read_csv(feature_csv)
    validate_long_dataset(rows)
    table = _scenario_frame(rows)[["scenario_id", "node_count"]].copy()
    for label, filename in (("SVM", "svm_oc_pipeline.joblib"), ("DTC", "dtc_oc_pipeline.joblib"), ("RF", "rf_oc_pipeline.joblib")):
        model = joblib.load(models_dir / filename)
        table[label + "_selected_oc"] = scenario_predictions(model, rows).set_index("scenario_id").loc[table.scenario_id, "predicted_oc"].to_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table


# ---------------------------------------------------------------------------
# Canonical three-file workflow: randy.cc + 2,000-scenario CSV + this file.
# ---------------------------------------------------------------------------
def project_split(data: pd.DataFrame) -> Dict[str, List[int]]:
    """Fixed grouped 1,400/300/300 split: 350/75/75 at each density."""
    split: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
    for nodes in PROJECT_NODES:
        ids = sorted(data.loc[data.node_count.eq(nodes), "scenario_id"].unique().tolist())
        if len(ids) != 500:
            raise RuntimeError(f"Expected 500 scenarios at node count {nodes}")
        rng = random.Random(91_000 + nodes)
        rng.shuffle(ids)
        split["train"].extend(ids[:350])
        split["validation"].extend(ids[350:425])
        split["test"].extend(ids[425:])
    return split


def project_train(data: pd.DataFrame, split: Dict[str, List[int]], outdir: Path) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Fit the fixed models on train+validation and evaluate held-out test rows."""
    train_ids = set(split["train"]) | set(split["validation"])
    train_rows = data[data.scenario_id.isin(train_ids)]
    test_rows = data[data.scenario_id.isin(split["test"])]
    trained: Dict[str, object] = {}
    report: List[dict] = []
    all_predictions: List[pd.DataFrame] = []
    for name, model in models().items():
        model.fit(train_rows[FEATURES], train_rows.is_oc)
        joblib.dump(model, outdir / f"{name.lower()}_model.joblib")
        binary = model.predict(test_rows[FEATURES])
        precision, recall, _, _ = precision_recall_fscore_support(
            test_rows.is_oc, binary, average="binary", zero_division=0
        )
        chosen = scenario_predictions(model, test_rows)
        chosen["Model"] = name
        all_predictions.append(chosen)
        report.append({
            "Model": name,
            "Binary accuracy (%)": 100.0 * accuracy_score(test_rows.is_oc, binary),
            "Precision": precision,
            "Recall": recall,
            "Binary F1": f1_score(test_rows.is_oc, binary, zero_division=0),
            "Top-1 OC accuracy (%)": 100.0 * chosen.correct.mean(),
            "Top-1 macro F1": f1_score(chosen.true_oc, chosen.predicted_oc,
                                         labels=[0, 1, 2, 3], average="macro", zero_division=0),
        })
        trained[name] = model
    ml = pd.DataFrame(report)
    predictions = pd.concat(all_predictions, ignore_index=True)
    ml.to_csv(outdir / "ml_accuracy.csv", index=False)
    predictions.to_csv(outdir / "selected_oc_per_test_scenario.csv", index=False)
    return trained, ml, predictions


def project_records(data: pd.DataFrame, split: Dict[str, List[int]], count: int) -> List[dict]:
    """Pick the first deterministic held-out scenarios used for graph curves."""
    records: List[dict] = []
    test_ids = set(split["test"])
    for nodes in PROJECT_NODES:
        choices = (data[data.scenario_id.isin(test_ids) & data.node_count.eq(nodes)]
                   .drop_duplicates("scenario_id").sort_values("scenario_id").head(count))
        if len(choices) != count:
            raise RuntimeError(f"Not enough test scenarios at {nodes} nodes")
        records.extend(choices[["scenario_id", "scenario_seed", "node_count"]].astype(int).to_dict("records"))
    return records


def project_run_candidate(record: dict, oc: int, outdir: Path) -> dict:
    """Run exactly one OC candidate with the fixed aggregated-control protocol."""
    target = outdir / "candidate_runs" / f"n{record['node_count']}_s{record['scenario_id']}_oc{oc}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        command = (
            "scratch/randy --mode=run "
            f"--nodeCount={record['node_count']} --scenarioId={record['scenario_id']} "
            f"--scenarioSeed={record['scenario_seed']} --selectedOc={oc} --runs=1 "
            "--fixedControlPolicy=true --helloSeconds=30 --routeTtlSeconds=60 "
            "--negativeRouteTtlSeconds=120 --referenceUpdateAccounting=true "
            "--aggregatedTopologyDigestAccounting=true --topologyDigestCapacityNodes=40 "
            f"--output={target}"
        )
        result = subprocess.run(["./ns3", "run", command], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(result.stderr)
    with target.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    if row["generated_packets"] != "200" or row["OC_data_hops"] != "0" or row["architecture_violations"] != "0":
        raise RuntimeError(f"Network invariant failure for scenario {record['scenario_id']}, OC{oc}")
    return row


def project_ledger(records: List[dict], outdir: Path) -> pd.DataFrame:
    jobs = [(record, oc) for record in records for oc in range(4)]
    rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(project_run_candidate, record, oc, outdir) for record, oc in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    ledger = pd.DataFrame(rows).sort_values(["node_count", "scenario_id", "selected_oc"])
    for column in ("scenario_id", "scenario_seed", "node_count", "selected_oc", "generated_packets",
                   "delivered_packets", "PDR", "E2ED_ms", "ROR_total", "control_transmissions", "data_hops"):
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce")
    if not ledger.groupby("scenario_id").size().eq(4).all():
        raise RuntimeError("Every graph scenario needs all four OC candidate runs")
    if ledger.groupby("scenario_id").topology_hash.nunique().ne(1).any():
        raise RuntimeError("Candidate topologies differ within a scenario")
    ledger.to_csv(outdir / "matched_four_oc_candidate_ledger.csv", index=False)
    return ledger


def project_worst_oc(group: pd.DataFrame) -> pd.Series:
    """Post-hoc worst actual OC: comparison baseline only, never an ML label."""
    ordered = group.copy()
    ordered["_delay"] = pd.to_numeric(ordered.E2ED_ms, errors="coerce").fillna(np.inf)
    return ordered.sort_values(["PDR", "_delay", "ROR_total", "selected_oc"],
                               ascending=[True, False, False, True], kind="mergesort").iloc[0]


def project_summary(rows: pd.DataFrame, method: str) -> dict:
    generated = pd.to_numeric(rows.generated_packets).sum()
    delivered = pd.to_numeric(rows.delivered_packets).sum()
    delivered_rows = rows[pd.to_numeric(rows.delivered_packets).gt(0)].copy()
    e2ed = pd.to_numeric(delivered_rows.E2ED_ms, errors="coerce")
    delivered_delay = float((e2ed * pd.to_numeric(delivered_rows.delivered_packets)).sum())
    raw_delay = delivered_delay / delivered if delivered else float("nan")
    completion_delay = (delivered_delay + (generated - delivered) * LOSS_TIMEOUT_MS) / generated
    control = pd.to_numeric(rows.control_transmissions).sum()
    data_hops = pd.to_numeric(rows.data_hops).sum()
    return {"Model": method, "PDR (%)": 100.0 * delivered / generated,
            "Raw delivered-packet E2ED (ms)": raw_delay,
            "Loss-aware completion delay (ms)": completion_delay,
            "ROR": control / (control + data_hops), "Scenarios": int(rows.scenario_id.nunique())}


def project_network(predictions: pd.DataFrame, ledger: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    all_rows: List[pd.DataFrame] = []
    for name in ("SVM", "DTC", "RF"):
        pick = predictions[predictions["Model"].eq(name)][["scenario_id", "predicted_oc"]]
        pick = pick.rename(columns={"predicted_oc": "selected_oc"})
        mapped = pick.merge(ledger, on=["scenario_id", "selected_oc"], validate="one_to_one")
        mapped["Method"] = name
        all_rows.append(mapped)
    baseline = pd.DataFrame([project_worst_oc(group) for _, group in ledger.groupby("scenario_id", sort=True)])
    baseline["Method"] = "Worst-OC baseline"
    all_rows.append(baseline)
    selected = pd.concat(all_rows, ignore_index=True)
    selected.to_csv(outdir / "selected_oc_network_metrics.csv", index=False)
    by_node = pd.DataFrame([
        {"Nodes": nodes, **project_summary(selected[(selected["Method"].eq(method)) &
                                                       (selected["node_count"].astype(int).eq(nodes))], method)}
        for nodes in PROJECT_NODES for method in ("SVM", "DTC", "RF", "Worst-OC baseline")
    ])
    overall = pd.DataFrame([project_summary(selected[selected["Method"].eq(method)], method)
                            for method in ("SVM", "DTC", "RF", "Worst-OC baseline")])
    by_node.to_csv(outdir / "network_metrics_by_node.csv", index=False)
    overall.to_csv(outdir / "network_metrics_overall.csv", index=False)
    return by_node


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--scenarios-per-density", type=int, default=5)
    g.add_argument("--base-seed", type=int, default=29)
    g.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "feature_only_oc_label_dataset.csv")
    ng = sub.add_parser("generate-network-labels")
    ng.add_argument("--scenarios-per-density", type=int, default=5)
    ng.add_argument("--base-seed", type=int, default=29)
    ng.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "mc_free_network_label_dataset.csv")
    t = sub.add_parser("train")
    t.add_argument("--dataset", type=Path, default=DEFAULT_RESULTS / "feature_only_oc_label_dataset.csv")
    t.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS / "feature_only_ml_models")
    t.add_argument("--manifest", type=Path)
    t.add_argument("--seed", type=int, default=2026)
    s = sub.add_parser("sweep-original-svm", help="validate only the requested original-SVM hyperparameters")
    s.add_argument("--dataset", type=Path, default=DEFAULT_RESULTS / "mc_free_network_label_dataset.csv")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS / "original_svm_sweep")
    s.add_argument("--seed", type=int, default=2026)
    p = sub.add_parser("predict")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_RESULTS / "models")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "predictions.csv")
    project = sub.add_parser("project", help="run the canonical three-file project workflow")
    project.add_argument("--dataset", type=Path, default=PROJECT_DATASET)
    project.add_argument("--outdir", type=Path, default=PROJECT_RESULTS)
    project.add_argument("--scenarios-per-density", type=int, default=5,
                         help="matched held-out network scenarios per density for the figures")
    args = parser.parse_args()
    if args.command == "generate":
        data = generate_dataset(args.scenarios_per_density, args.output, args.base_seed)
        print(f"Wrote {len(data)} rows / {data.scenario_id.nunique()} scenarios to {args.output}")
    elif args.command == "generate-network-labels":
        data = generate_network_label_dataset(args.scenarios_per_density, args.output, args.base_seed)
        print(f"Wrote {len(data)} MC-free network-labelled rows / {data.scenario_id.nunique()} scenarios to {args.output}")
    elif args.command == "train":
        results = train(pd.read_csv(args.dataset), args.outdir, args.manifest, args.seed)
        print(results.to_string(index=False))
    elif args.command == "sweep-original-svm":
        sweep, summary = sweep_original_svm(pd.read_csv(args.dataset), args.outdir,
                                            args.manifest, args.seed)
        print(sweep.to_string(index=False))
        print(summary.to_string(index=False))
    elif args.command == "project":
        if args.scenarios_per_density <= 0:
            raise ValueError("--scenarios-per-density must be positive")
        args.outdir.mkdir(parents=True, exist_ok=True)
        project_data = pd.read_csv(args.dataset)
        validate_long_dataset(project_data)
        if len(project_data) != 8000 or project_data.scenario_id.nunique() != 2000:
            raise RuntimeError("The canonical workflow requires 2,000 scenarios / 8,000 candidate rows")
        split = project_split(project_data)
        pd.DataFrame({name: pd.Series(ids) for name, ids in split.items()}).to_csv(
            args.outdir / "grouped_split.csv", index=False
        )
        _, ml, predictions = project_train(project_data, split, args.outdir)
        records = project_records(project_data, split, args.scenarios_per_density)
        ledger = project_ledger(records, args.outdir)
        by_node = project_network(
            predictions[predictions.scenario_id.isin(ledger.scenario_id)], ledger, args.outdir
        )
        print("To create the MATLAB figure, run: matlab -batch \"generate_final_graphs\"")
        print(ml.to_string(index=False))
        print(by_node[["Nodes", "Model", "PDR (%)", "Loss-aware completion delay (ms)", "ROR"]].to_string(index=False))
        print(f"Saved canonical results to {args.outdir}")
    else:
        print(predict(args.models_dir, args.features, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
