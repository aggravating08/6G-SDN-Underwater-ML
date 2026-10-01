#!/usr/bin/env python3
"""Fresh, locked-model evaluation for the underwater controller-selection study.

This script deliberately generates scenarios that are not present in the
2,000-scenario development dataset.  The three model configurations are fixed
before the new scenarios are generated and are not tuned on these outcomes.
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
EVALUATION_DIRECTORY = os.environ.get(
    "FRESH_EVALUATION_DIRECTORY", "fresh_locked_evaluation_30_per_node_1s"
)
OUT = ROOT / "results" / EVALUATION_DIRECTORY
DEVELOPMENT_DATA = (
    ROOT
    / "results"
    / "underwater_partner_style_equivalent_2000"
    / "underwater_feature_rule_2000_scenarios_8000_candidates.csv"
)
FEATURES = ["x", "y", "local_density", "speed"]
NODES = [25, 50, 75, 100]
SCENARIOS_PER_NODE = int(os.environ.get("FRESH_SCENARIOS_PER_NODE", "30"))
FIRST_SCENARIO_ID = int(os.environ.get("FRESH_FIRST_SCENARIO_ID", "2000"))
FIRST_SCENARIO_SEED = int(os.environ.get("FRESH_FIRST_SCENARIO_SEED", "22000029"))
LOSS_PENALTY_MS = 1000.0
SVM_C = float(os.environ.get("FRESH_SVM_C", "1.0"))
SVM_GAMMA = float(os.environ.get("FRESH_SVM_GAMMA", "0.0"))


def locked_models() -> dict[str, object]:
    svm_gamma: object = "scale" if SVM_GAMMA == 0.0 else SVM_GAMMA
    return {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(C=SVM_C, gamma=svm_gamma)),
            ]
        ),
        "DTC": DecisionTreeClassifier(
            criterion="gini",
            max_depth=8,
            min_samples_leaf=2,
            min_samples_split=20,
            max_features=3,
            random_state=42,
        ),
        "RF-15": RandomForestClassifier(
            n_estimators=15,
            criterion="gini",
            max_depth=2,
            min_samples_split=20,
            min_samples_leaf=2,
            max_features=1,
            bootstrap=True,
            max_samples=0.35,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def development_split(data: pd.DataFrame) -> dict[str, list[int]]:
    split = {"train": [], "validation": [], "test": []}
    for node_count in NODES:
        scenario_ids = sorted(
            data.loc[data.node_count.eq(node_count), "scenario_id"].unique().tolist()
        )
        rng = random.Random(91_000 + node_count)
        rng.shuffle(scenario_ids)
        split["train"].extend(scenario_ids[:350])
        split["validation"].extend(scenario_ids[350:425])
        split["test"].extend(scenario_ids[425:])
    return split


def generate_fresh_features() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    total_scenarios = SCENARIOS_PER_NODE * len(NODES)
    total_candidates = total_scenarios * 4
    destination = OUT / (
        f"fresh_features_{total_scenarios}_scenarios_{total_candidates}_candidates.csv"
    )
    if destination.exists():
        return pd.read_csv(destination)

    parts = []
    for index, node_count in enumerate(NODES):
        temporary = OUT / f".fresh_labels_{node_count}.csv"
        first_scenario_id = FIRST_SCENARIO_ID + index * SCENARIOS_PER_NODE
        base_seed = FIRST_SCENARIO_SEED + index * 1_000_000
        command = (
            "scratch/randy --mode=labels "
            f"--nodeCount={node_count} --runs={SCENARIOS_PER_NODE} "
            f"--scenarioId={first_scenario_id} --baseSeed={base_seed} "
            f"--output={temporary}"
        )
        subprocess.run(["./ns3", "run", command], cwd=ROOT, check=True)
        parts.append(pd.read_csv(temporary))
        temporary.unlink()

    fresh = pd.concat(parts, ignore_index=True)
    if fresh.scenario_id.nunique() != SCENARIOS_PER_NODE * len(NODES):
        raise RuntimeError("Fresh feature generation produced the wrong scenario count")
    if fresh.groupby("scenario_id").size().ne(4).any():
        raise RuntimeError("Each fresh scenario must contain four candidates")
    if fresh.groupby("scenario_id").is_oc.sum().ne(1).any():
        raise RuntimeError("Each fresh scenario must contain one positive label")
    fresh.to_csv(destination, index=False)
    return fresh


def candidate_run(record: dict, selected_oc: int) -> dict:
    run_dir = OUT / "candidate_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    target = (
        run_dir
        / f"n{record['node_count']}_s{record['scenario_id']}_oc{selected_oc}.csv"
    )
    if not target.exists():
        command = (
            "scratch/randy --mode=run "
            f"--nodeCount={record['node_count']} "
            f"--scenarioId={record['scenario_id']} "
            f"--scenarioSeed={record['scenario_seed']} "
            f"--selectedOc={selected_oc} --runs=1 "
            "--fixedControlPolicy=true --helloSeconds=30 --routeTtlSeconds=60 "
            "--negativeRouteTtlSeconds=120 --referenceUpdateAccounting=true "
            "--aggregatedTopologyDigestAccounting=true "
            "--topologyDigestCapacityNodes=40 "
            f"--output={target}"
        )
        result = subprocess.run(
            ["./ns3", "run", command],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
    with target.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return row


def generate_ledger(fresh: pd.DataFrame) -> pd.DataFrame:
    destination = OUT / "fresh_all_four_oc_ledger.csv"
    if destination.exists():
        return pd.read_csv(destination, na_values=["NA", "NaN", ""])

    records = (
        fresh.sort_values(["node_count", "scenario_id"])
        .drop_duplicates("scenario_id")
        [["scenario_id", "scenario_seed", "node_count"]]
        .astype(int)
        .to_dict("records")
    )
    jobs = [(record, selected_oc) for record in records for selected_oc in range(4)]
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(candidate_run, record, selected_oc) for record, selected_oc in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if completed % 40 == 0:
                print(f"Completed {completed}/{len(jobs)} candidate simulations", flush=True)

    ledger = pd.DataFrame(rows).sort_values(
        ["node_count", "scenario_id", "selected_oc"]
    )
    numeric = [
        "scenario_id", "scenario_seed", "node_count", "selected_oc",
        "generated_packets", "delivered_packets", "PDR", "E2ED_ms",
        "ROR_total", "control_transmissions", "data_hops",
    ]
    for column in numeric:
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce")
    if ledger.groupby("scenario_id").size().ne(4).any():
        raise RuntimeError("The fresh ledger does not contain all four candidates")
    ledger.to_csv(destination, index=False)
    return ledger


def scenario_predictions(model: object, rows: pd.DataFrame) -> pd.DataFrame:
    try:
        scores = model.predict_proba(rows[FEATURES])[:, 1]
    except AttributeError:
        scores = model.decision_function(rows[FEATURES])
    scored = rows[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    scored["score"] = scores
    selected = scored.loc[
        scored.groupby("scenario_id").score.idxmax()
    ].copy()
    truth = rows.loc[rows.is_oc.eq(1)].set_index("scenario_id").auv_id
    selected["true_oc"] = truth.loc[selected.scenario_id].to_numpy()
    selected["correct"] = selected.auv_id.eq(selected.true_oc).astype(int)
    return selected.rename(columns={"auv_id": "selected_oc"})


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    generated = rows.generated_packets.sum()
    delivered = rows.delivered_packets.sum()
    delivered_delay = (rows.E2ED_ms.fillna(0) * rows.delivered_packets).sum()
    loss_aware_delay = (
        delivered_delay + (generated - delivered) * LOSS_PENALTY_MS
    ) / generated
    control = rows.control_transmissions.sum()
    data_hops = rows.data_hops.sum()
    return {
        "pdr_percent": 100.0 * delivered / generated,
        "loss_aware_delay_ms": loss_aware_delay,
        "routing_overhead_ratio": control / (control + data_hops),
    }


def worst_candidate(group: pd.DataFrame) -> pd.Series:
    ordered = group.copy()
    ordered["loss_aware_delay"] = (
        ordered.E2ED_ms.fillna(0) * ordered.delivered_packets
        + (ordered.generated_packets - ordered.delivered_packets) * LOSS_PENALTY_MS
    ) / ordered.generated_packets
    return ordered.sort_values(
        ["PDR", "loss_aware_delay", "ROR_total", "selected_oc"],
        ascending=[True, False, False, True],
        kind="stable",
    ).iloc[0]


def biased_bound_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    weighted_rows = []
    for _, group in ledger.groupby("scenario_id", sort=True):
        worst = int(worst_candidate(group).selected_oc)
        for _, row in group.iterrows():
            copy = row.copy()
            weight = 0.70 if int(row.selected_oc) == worst else 0.10
            for column in (
                "generated_packets", "delivered_packets", "control_transmissions", "data_hops"
            ):
                copy[column] = row[column] * weight
            copy["delivered_delay_total"] = (
                0.0 if pd.isna(row.E2ED_ms) else row.E2ED_ms * row.delivered_packets * weight
            )
            weighted_rows.append(copy)
    result = pd.DataFrame(weighted_rows)
    result["E2ED_ms"] = np.where(
        result.delivered_packets.gt(0),
        result.delivered_delay_total / result.delivered_packets,
        np.nan,
    )
    return result


def save_figure(fig: plt.Figure, stem: str) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_figure(
    accuracy: pd.DataFrame, by_node: pd.DataFrame
) -> None:
    methods = ("RF-15", "DTC", "SVM")
    colors = {"RF-15": "#4169E1", "DTC": "#FF7F0E", "SVM": "#FF3030"}
    markers = {"RF-15": "P", "DTC": "o", "SVM": "s"}
    bound_name = "70% worst-biased random bound"
    specs = [
        ("pdr_percent", "Packet delivery ratio (%)", "(a) Packet Delivery Ratio", (20, 97), "lower right"),
        ("loss_aware_delay_ms", "Loss-aware delay (ms)", "(b) Loss-Aware Delay", (280, 820), "upper right"),
        ("routing_overhead_ratio", "Routing overhead ratio", "(c) Routing Overhead", (0.14, 0.58), "upper right"),
    ]
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif", "font.size": 8.5,
            "axes.titlesize": 9.5, "axes.labelsize": 9,
            "legend.fontsize": 7.5, "xtick.labelsize": 8,
            "ytick.labelsize": 8, "axes.linewidth": 0.9,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3), constrained_layout=True)
        for ax, (metric, ylabel, title, ylim, legend_location) in zip(
            (axes[0, 0], axes[0, 1], axes[1, 0]), specs
        ):
            for method in methods:
                values = (
                    by_node[by_node.model.eq(method)]
                    .set_index("node_count")
                    .loc[NODES, metric]
                    .to_numpy()
                )
                ax.plot(
                    NODES, values, color=colors[method], marker=markers[method],
                    linestyle="-", linewidth=1.7, markersize=5.2,
                    markerfacecolor="white" if method == "RF-15" else colors[method],
                    markeredgewidth=0.9, label=method,
                )
            bound_values = (
                by_node[by_node.model.eq(bound_name)]
                .set_index("node_count")
                .loc[NODES, metric]
                .to_numpy()
            )
            ax.plot(
                NODES, bound_values, color="#B52B65", marker="D",
                linestyle="-", linewidth=1.7, markersize=5.0,
                label="70% lower bound",
            )
            ax.set_xlim(20, 105)
            ax.set_ylim(*ylim)
            ax.set_xticks(NODES)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, color="#B0B0B0", linewidth=0.55, alpha=0.4)
            ax.set_axisbelow(True)
            ax.legend(
                loc=legend_location, frameon=True, fancybox=False,
                edgecolor="#333333",
            )

        accuracy_ax = axes[1, 1]
        values = [
            float(accuracy.loc[accuracy.model.eq(method), "binary_accuracy_percent"].iloc[0])
            for method in methods
        ]
        bars = accuracy_ax.bar(
            methods, values, color=[colors[method] for method in methods],
            edgecolor="black", linewidth=0.55, width=0.62,
        )
        accuracy_ax.axhline(
            75, color="#B52B65", linestyle="-", linewidth=1.7,
            label="Majority baseline",
        )
        for bar, value in zip(bars, values):
            accuracy_ax.text(
                bar.get_x() + bar.get_width() / 2, value + 1.1,
                f"{value:.1f}%", ha="center", va="bottom", fontsize=7.5,
            )
        accuracy_ax.set_ylim(0, 100)
        accuracy_ax.set_ylabel("Binary accuracy (%)")
        accuracy_ax.set_title("(d) Fresh Binary Accuracy")
        accuracy_ax.grid(axis="y", color="#B0B0B0", linewidth=0.55, alpha=0.4)
        accuracy_ax.set_axisbelow(True)
        accuracy_ax.legend(
            loc="lower right", frameon=True, fancybox=False,
            edgecolor="#333333",
        )
        save_figure(fig, "fresh_combined_four_metrics_locked_models")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(DEVELOPMENT_DATA)
    split = development_split(development)
    training_ids = set(split["train"]) | set(split["validation"])
    training = development[development.scenario_id.isin(training_ids)]

    fresh = generate_fresh_features()
    ledger = generate_ledger(fresh)
    predictions = []
    accuracy_rows = []
    selected_network_rows = []

    for name, model in locked_models().items():
        model.fit(training[FEATURES], training.is_oc)
        binary_prediction = model.predict(fresh[FEATURES])
        chosen = scenario_predictions(model, fresh)
        chosen["model"] = name
        predictions.append(chosen)
        accuracy_rows.append(
            {
                "model": name,
                "binary_accuracy_percent": 100.0
                * accuracy_score(fresh.is_oc, binary_prediction),
                "top1_accuracy_percent": 100.0 * chosen.correct.mean(),
            }
        )
        mapped = chosen[["scenario_id", "selected_oc"]].merge(
            ledger, on=["scenario_id", "selected_oc"], validate="one_to_one"
        )
        mapped["model"] = name
        selected_network_rows.append(mapped)

    prediction_table = pd.concat(predictions, ignore_index=True)
    selected = pd.concat(selected_network_rows, ignore_index=True)
    bound = biased_bound_rows(ledger)
    bound["model"] = "70% worst-biased random bound"

    by_node_rows = []
    overall_rows = []
    for method, rows in list(selected.groupby("model")) + [
        ("70% worst-biased random bound", bound)
    ]:
        overall_rows.append({"model": method, **summarize(rows)})
        for node_count in NODES:
            by_node_rows.append(
                {
                    "model": method,
                    "node_count": node_count,
                    **summarize(rows[rows.node_count.eq(node_count)]),
                }
            )

    accuracy_table = pd.DataFrame(accuracy_rows)
    by_node_table = pd.DataFrame(by_node_rows)
    accuracy_table.to_csv(OUT / "fresh_accuracy.csv", index=False)
    prediction_table.to_csv(OUT / "fresh_controller_predictions.csv", index=False)
    selected.to_csv(OUT / "fresh_selected_network_rows.csv", index=False)
    by_node_table.to_csv(OUT / "fresh_network_metrics_by_node.csv", index=False)
    pd.DataFrame(overall_rows).to_csv(OUT / "fresh_network_metrics_overall.csv", index=False)
    (OUT / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "status": "fresh_locked_confirmation_evaluation",
                "scenarios_per_node": SCENARIOS_PER_NODE,
                "total_scenarios": SCENARIOS_PER_NODE * len(NODES),
                "candidate_runs": SCENARIOS_PER_NODE * len(NODES) * 4,
                "loss_penalty_ms": LOSS_PENALTY_MS,
                "first_scenario_id": FIRST_SCENARIO_ID,
                "first_scenario_seed": FIRST_SCENARIO_SEED,
                "model_tuning_after_generation": False,
                "svm_c": SVM_C,
                "svm_gamma": "scale" if SVM_GAMMA == 0.0 else SVM_GAMMA,
            },
            indent=2,
        )
    )
    make_combined_figure(accuracy_table, by_node_table)
    print(f"Fresh evaluation written to {OUT}")


if __name__ == "__main__":
    main()
