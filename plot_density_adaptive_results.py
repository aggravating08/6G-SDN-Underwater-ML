#!/usr/bin/env python3
"""Create paper-style plots from saved density-adaptive experiment CSVs.

This script only reads aggregated results.  It never runs ns-3 or alters
metrics, models, labels, or protocol parameters.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results/underwater_rebuild/current/final_density_adaptive_ror_monotonic/final_metrics_by_node.csv"
OUT = INPUT.parent / "figures"
METHODS = ("SVM-selected OC", "DTC-selected OC", "RF-selected OC", "Baseline OC0")
LABELS = {"SVM-selected OC": "SVM", "DTC-selected OC": "DTC", "RF-selected OC": "RF", "Baseline OC0": "Baseline"}
STYLES = {
    "SVM-selected OC": {"color": "#d62728", "marker": "s"},
    "DTC-selected OC": {"color": "#ff7f0e", "marker": "o"},
    "RF-selected OC": {"color": "#1f77b4", "marker": "o"},
    "Baseline OC0": {"color": "#a23b72", "marker": "D"},
}


def read_rows() -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    with INPUT.open(newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["Method"]].append({
                "nodes": float(row["Nodes"]), "pdr": float(row["PDR %"]),
                "delay": float(row["E2ED ms"]), "ror_total": float(row["ROR total"]),
                "ror_reactive": float(row["ROR reactive"]),
            })
    for values in grouped.values():
        values.sort(key=lambda row: row["nodes"])
    return grouped


def plot(grouped: dict[str, list[dict[str, float]]], metric: str, ylabel: str, filename: str, ymax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    for method in METHODS:
        values = grouped[method]
        ax.plot([row["nodes"] for row in values], [row[metric] for row in values], linewidth=2,
                markersize=6, label=LABELS[method], **STYLES[method])
    ax.set_title(ylabel + " vs Number of Sensor Nodes")
    ax.set_xlabel("Number of sensor nodes")
    ax.set_ylabel(ylabel)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xlim(20, 105)
    ax.set_ylim(bottom=0, top=ymax)
    ax.grid(True, alpha=.3)
    ax.legend(frameon=True, loc="best")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{filename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing saved experiment table: {INPUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = read_rows()
    missing = [method for method in METHODS if method not in grouped]
    if missing:
        raise ValueError(f"Missing methods in metrics CSV: {missing}")
    plot(grouped, "pdr", "Packet delivery ratio (%)", "pdr_vs_nodes", 100)
    plot(grouped, "delay", "End-to-end delay (ms)", "e2ed_vs_nodes")
    plot(grouped, "ror_total", "Routing overhead ratio (total)", "ror_total_vs_nodes", 1)
    plot(grouped, "ror_reactive", "Routing overhead ratio (reactive)", "ror_reactive_vs_nodes", 1)
    print(f"Saved PNG and PDF figures to {OUT}")


if __name__ == "__main__":
    main()
