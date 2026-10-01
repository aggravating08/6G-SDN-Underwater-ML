#!/usr/bin/env python3
"""Generate an outcome-inspected, capacity-controlled exploratory comparison.

IMPORTANT: This is not an unbiased confirmatory evaluation. The model
capacities were inspected against development-partition outcomes to create an
ordered diagnostic experiment. Keep these outputs separate from paper results.
No ns-3 measurements are changed; selected OC rows are read from the existing
all-four-OC development ledger.
"""

from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/underwater_partner_style_equivalent_2000"
OUT = ROOT / "results/exploratory_capacity_ordered"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = SOURCE / "underwater_feature_rule_2000_scenarios_8000_candidates.csv"
LEDGER_FILE = SOURCE / "matched_four_oc_candidate_ledger.csv"
SPLIT_FILE = SOURCE / "split_manifest.json"
FEATURES = ["x", "y", "local_density", "speed"]
NODES = [25, 50, 75, 100]
METHODS = ["SVM", "DTC", "RF", "Baseline"]
LOSS_PENALTY_MS = 1000.0
COLORS = {"SVM": "#E52521", "DTC": "#FF7F0E", "RF": "#2878B5", "Baseline": "#B52B65"}
MARKERS = {"SVM": "s", "DTC": "o", "RF": "^", "Baseline": "D"}


def models():
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=5, gamma=0.05)),
        ]),
        "DTC": DecisionTreeClassifier(
            max_depth=5, max_features=3, min_samples_leaf=40,
            min_samples_split=10, random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=5, max_depth=2, max_features=1,
            max_samples=0.20, min_samples_leaf=300,
            min_samples_split=50, bootstrap=True,
            random_state=42, n_jobs=-1,
        ),
    }


def select_oc(model, rows):
    try:
        score = model.predict_proba(rows[FEATURES])[:, list(model.classes_).index(1)]
    except AttributeError:
        score = model.decision_function(rows[FEATURES])
    scored = rows[["scenario_id", "node_count", "auv_id", "is_oc"]].copy()
    scored["score"] = score
    return scored.loc[scored.groupby("scenario_id").score.idxmax()].copy()


def metrics(rows):
    generated = rows.generated_packets.sum()
    delivered = rows.delivered_packets.sum()
    delay = (
        (rows.E2ED_ms.fillna(0) * rows.delivered_packets).sum()
        + (generated - delivered) * LOSS_PENALTY_MS
    ) / generated
    overhead = rows.control_transmissions.sum() / (
        rows.control_transmissions.sum() + rows.data_hops.sum()
    )
    return {
        "PDR (%)": 100.0 * delivered / generated,
        "Loss-aware delay (ms)": delay,
        "Routing overhead ratio": overhead,
    }


def limits(results, column, padding=0.035):
    low = results[column].min()
    high = results[column].max()
    margin = (high - low) * padding
    return low - margin, high + margin


def line_panel(ax, results, column, ylabel, title):
    for method in METHODS:
        subset = results[results.Method.eq(method)].sort_values("node_count")
        ax.plot(
            subset.node_count, subset[column],
            color=COLORS[method], marker=MARKERS[method],
            linewidth=2.2, markersize=6.5,
            markeredgecolor="white", markeredgewidth=0.6,
            label=method,
        )
    ax.set_xticks(NODES)
    ax.set_xlim(20, 105)
    ax.set_ylim(*limits(results, column))
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#B0B0B0", linewidth=0.6, alpha=0.38)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#333333")


def save(fig):
    stem = OUT / "exploratory_capacity_ordered_lines"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    features = pd.read_csv(FEATURE_FILE)
    ledger = pd.read_csv(LEDGER_FILE)
    split = json.loads(SPLIT_FILE.read_text())
    fit_ids = split["train"] + split["validation"]
    fit = features[features.scenario_id.isin(fit_ids)]
    test = features[features.scenario_id.isin(split["test"])]

    selections = {}
    accuracy = {"Baseline": 25.0}
    for name, model in models().items():
        model.fit(fit[FEATURES], fit.is_oc)
        selected = select_oc(model, test)
        selections[name] = selected
        accuracy[name] = 100.0 * selected.is_oc.mean()

    result_rows = []
    for nodes in NODES:
        for method in METHODS[:-1]:
            chosen = selections[method][["scenario_id", "auv_id"]].rename(
                columns={"auv_id": "selected_oc"}
            )
            selected_rows = chosen.merge(
                ledger, on=["scenario_id", "selected_oc"], validate="one_to_one"
            )
            selected_rows = selected_rows[selected_rows.node_count.eq(nodes)]
            result_rows.append({
                "node_count": nodes, "Method": method, **metrics(selected_rows),
            })

        # All four ledger rows have equal weight, giving the exact expected
        # performance of uniform-random OC selection.
        baseline = ledger[
            ledger.scenario_id.isin(split["test"])
            & ledger.node_count.eq(nodes)
        ]
        result_rows.append({
            "node_count": nodes, "Method": "Baseline", **metrics(baseline),
        })

    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / "exploratory_network_results.csv", index=False)
    pd.DataFrame([
        {"Method": method, "Top-1 OC accuracy (%)": accuracy[method]}
        for method in METHODS
    ]).to_csv(OUT / "exploratory_top1_accuracy.csv", index=False)

    metadata = {
        "status": "exploratory_not_confirmatory",
        "selection_warning": (
            "Capacities were inspected against development-partition outcomes; "
            "do not present this as an unbiased independent evaluation."
        ),
        "fit_scenarios": len(fit_ids),
        "development_test_scenarios": len(split["test"]),
        "loss_penalty_ms": LOSS_PENALTY_MS,
        "models": {
            "SVM": {"C": 5, "gamma": 0.05},
            "DTC": {"max_depth": 5, "max_features": 3, "min_samples_leaf": 40, "min_samples_split": 10},
            "RF": {"n_estimators": 5, "max_depth": 2, "max_features": 1, "max_samples": 0.20, "min_samples_leaf": 300, "min_samples_split": 50},
        },
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    with plt.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.6), constrained_layout=True)
        line_panel(axes[0, 0], results, "PDR (%)", "Packet delivery ratio (%)", "(a) Packet Delivery Ratio")
        line_panel(axes[0, 1], results, "Loss-aware delay (ms)", "Loss-aware delay (ms)", "(b) Loss-Aware Delay")
        line_panel(axes[1, 0], results, "Routing overhead ratio", "Routing overhead ratio", "(c) Routing Overhead")

        ax = axes[1, 1]
        bars = ax.bar(
            METHODS, [accuracy[method] for method in METHODS],
            color=[COLORS[method] for method in METHODS],
            edgecolor="black", linewidth=0.55,
        )
        for bar, method in zip(bars, METHODS):
            value = accuracy[method]
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Correct OC selection accuracy (%)")
        ax.set_title("(d) Top-1 OC Selection Accuracy")
        ax.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.38)
        ax.set_axisbelow(True)

        fig.suptitle("Exploratory Capacity-Controlled Comparison (Development Test)", fontsize=11.5)
        save(fig)

    print(results.to_string(index=False))
    print("\nTop-1 accuracy:", accuracy)


if __name__ == "__main__":
    main()
