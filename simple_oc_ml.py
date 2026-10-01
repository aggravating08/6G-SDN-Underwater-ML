#!/usr/bin/env python3
"""Small, leakage-safe OC-selection trainer.

The model sees only x, y, local_density and speed.  It predicts a score for
each of the four AUV candidates in one scenario, then selects the AUV with the
highest positive-class score.  The reported accuracy is therefore scenario-
level Top-1 OC accuracy, not misleading binary row accuracy.

This file does not predict or tune PDR, E2ED, or routing overhead.  Those are
network-simulation outcomes measured only after a model selects an OC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["x", "y", "local_density", "speed"]


def check_dataset(data: pd.DataFrame) -> None:
    needed = {"scenario_id", "auv_id", "is_oc", *FEATURES}
    missing = needed.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not data.groupby("scenario_id").size().eq(4).all():
        raise ValueError("Every scenario must contain exactly four AUV rows.")
    if not data.groupby("scenario_id").is_oc.sum().eq(1).all():
        raise ValueError("Every scenario must contain exactly one OC label.")


def top1_predictions(model: object, rows: pd.DataFrame) -> pd.DataFrame:
    """Score four candidates together and select exactly one AUV per scenario."""
    output = []
    for scenario_id, group in rows.groupby("scenario_id", sort=True):
        group = group.sort_values("auv_id")
        scores = model.predict_proba(group[FEATURES])[:, list(model.classes_).index(1)]
        predicted = int(group.iloc[int(np.argmax(scores))].auv_id)
        actual = int(group.loc[group.is_oc.eq(1), "auv_id"].iloc[0])
        output.append({"scenario_id": int(scenario_id), "true_oc": actual,
                       "predicted_oc": predicted, "correct": int(actual == predicted)})
    return pd.DataFrame(output)


def evaluate(model: object, rows: pd.DataFrame, name: str, split: str, output_dir: Path) -> dict:
    pred = top1_predictions(model, rows)
    accuracy = accuracy_score(pred.true_oc, pred.predicted_oc)
    macro_f1 = f1_score(pred.true_oc, pred.predicted_oc, labels=[0, 1, 2, 3],
                        average="macro", zero_division=0)
    pred.to_csv(output_dir / f"{name.lower()}_{split}_predictions.csv", index=False)
    pd.DataFrame(
        confusion_matrix(pred.true_oc, pred.predicted_oc, labels=[0, 1, 2, 3]),
        index=["true_OC0", "true_OC1", "true_OC2", "true_OC3"],
        columns=["pred_OC0", "pred_OC1", "pred_OC2", "pred_OC3"],
    ).to_csv(output_dir / f"{name.lower()}_{split}_confusion_matrix.csv")
    return {"model": name, "split": split, "top1_accuracy": accuracy,
            "macro_f1": macro_f1, "scenarios": len(pred)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Existing scenario-level train/validation/test split manifest")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)
    check_dataset(data)
    split = json.loads(args.manifest.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "split_manifest.json").write_text(json.dumps(split, indent=2))

    train = data[data.scenario_id.isin(split["train"])]
    validation = data[data.scenario_id.isin(split["validation"])]
    test = data[data.scenario_id.isin(split["test"])]
    models = {
        "SVM": Pipeline([("scaler", StandardScaler()),
                         ("classifier", SVC(kernel="rbf", C=10.0, gamma="scale",
                                            class_weight="balanced", probability=True,
                                            random_state=2026))]),
        "DTC": DecisionTreeClassifier(max_depth=8, min_samples_leaf=8,
                                       class_weight="balanced", random_state=2026),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=12,
                                      min_samples_leaf=5, max_features="sqrt",
                                      class_weight="balanced", random_state=2026,
                                      n_jobs=-1),
    }

    report = []
    for name, model in models.items():
        model.fit(train[FEATURES], train.is_oc)
        report.append(evaluate(model, validation, name, "validation", args.outdir))
        report.append(evaluate(model, test, name, "test", args.outdir))
        joblib.dump(model, args.outdir / f"{name.lower()}_oc_pipeline.joblib")
    result = pd.DataFrame(report)
    result.to_csv(args.outdir / "ml_results.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
