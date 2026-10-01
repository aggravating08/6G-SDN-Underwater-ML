#!/usr/bin/env python3
"""Plot the locked experimental models over validation+test scenarios.

This combined 600-scenario view is descriptive and exploratory.  Validation
was used to select the constrained settings, so these figures are not an
independent held-out evaluation and must not be represented as one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import evaluate_experimental_model_separation as evaluation
from underwater_ml_pipeline import build_split
from tune_models_fair_network_objective import (
    FEATURES,
    NODES,
    expected_baseline,
    selections,
    summarize,
)


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "results/underwater_partner_style_equivalent_2000/underwater_feature_rule_2000_scenarios_8000_candidates.csv"
VALIDATION_LEDGER = ROOT / "results/original_2000_split_current_accounting_300_validation/matched_four_oc_candidate_ledger.csv"
TEST_LEDGER = ROOT / "results/original_2000_split_current_accounting_300_test/matched_four_oc_candidate_ledger.csv"
SEARCH_RESULT = ROOT / "results/experimental_validation_separation_search/selected_validation_result.json"
OUT = ROOT / "results/exploratory_combined_600_separated_models"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation.OUT = OUT
    settings = json.loads(SEARCH_RESULT.read_text())
    data = pd.read_csv(DATA_FILE)
    ledger = pd.concat(
        [pd.read_csv(VALIDATION_LEDGER), pd.read_csv(TEST_LEDGER)],
        ignore_index=True,
    )
    split = build_split(data, seed=2026)
    fit_rows = data[data.scenario_id.isin(split["train"])]
    evaluation_ids = split["validation"] + split["test"]
    evaluation_rows = data[data.scenario_id.isin(evaluation_ids)]

    network_rows = []
    accuracy = {}
    prediction_rows = []
    for name, model in evaluation.make_models(settings).items():
        model.fit(fit_rows[FEATURES], fit_rows.is_oc)
        chosen = selections(model, evaluation_rows)
        truth = evaluation_rows.loc[evaluation_rows.is_oc.eq(1)].set_index("scenario_id").auv_id
        chosen["true_oc"] = truth.loc[chosen.scenario_id].to_numpy()
        chosen["correct"] = chosen.selected_oc.eq(chosen.true_oc)
        accuracy[name] = 100.0 * chosen.correct.mean()
        chosen["Method"] = name
        prediction_rows.append(chosen)
        mapped = chosen[["scenario_id", "selected_oc"]].merge(
            ledger, on=["scenario_id", "selected_oc"], validate="one_to_one"
        )
        for nodes in NODES:
            network_rows.append({"Nodes": nodes, "Method": name,
                                 **summarize(mapped[mapped.node_count.eq(nodes)])})

    baseline = expected_baseline(ledger)
    for nodes in NODES:
        network_rows.append({"Nodes": nodes, "Method": "Baseline",
                             **summarize(baseline[baseline.node_count.eq(nodes)])})

    network = pd.DataFrame(network_rows)
    network.to_csv(OUT / "network_metrics_by_node.csv", index=False)
    pd.DataFrame([{"Method": method, "Correct OC selection accuracy (%)": value}
                  for method, value in accuracy.items()]).to_csv(
                      OUT / "correct_oc_accuracy.csv", index=False
                  )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        OUT / "scenario_predictions.csv", index=False
    )
    metadata = {
        "status": "Exploratory combined validation+test descriptive result; not independent held-out evidence.",
        "training_scenarios": len(split["train"]),
        "reported_scenarios": len(evaluation_ids),
        "reported_scenarios_per_node": 150,
        "settings": {name: settings[name]["parameters"] for name in ("SVM", "DTC", "RF")},
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    evaluation.plot(network, accuracy)
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame([{"Method": key, "Accuracy": value}
                        for key, value in accuracy.items()]).to_string(index=False))
    print(network.to_string(index=False))


if __name__ == "__main__":
    main()
