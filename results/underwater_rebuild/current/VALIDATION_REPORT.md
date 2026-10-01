# Rebuilt underwater CA-SDUN validation

## Scope

This validation used 20 independent unseen scenarios at each of 25, 50, 75,
and 100 sensors.  The scenario seeds are disjoint from the 2,000-scenario ML
dataset.  Saved SVM, Decision Tree (DTC), and Random Forest (RF) models each
selected one OC from the same four feature rows.  Only unique `(scenario, OC)`
network candidates were executed; shared selections reused the exact same row.

There were 97 unique matched candidate runs and 240 model-scenario outcomes.
Every candidate generated 200 packets.  All candidates passed:

- `OC_data_hops = 0`
- `architecture_violations = 0`
- same topology and cognitive-state hashes for candidates of one scenario.

## Scenario-level ML test result

| Model | Held-out top-1 accuracy | Macro F1 |
|---|---:|---:|
| SVM | 79.67% | 79.56% |
| DTC | 78.33% | 78.33% |
| RF | 83.33% | 83.29% |

These are scenario-level four-candidate results.  They are not row-level binary
classification accuracy.

## Network results (20 scenarios per density)

Values are mean; `±` is a normal-approximation 95% CI half-width across
scenarios.  E2ED is averaged over scenarios with at least one delivered packet.

| Nodes | Model | PDR (%) | E2ED (ms) | ROR total | ROR reactive |
|---:|---|---:|---:|---:|---:|
| 25 | SVM | 36.73 ± 9.72 | 265.23 ± 37.76 | 0.804 ± 0.060 | 0.723 ± 0.084 |
| 25 | DTC | 36.63 ± 9.72 | 267.73 ± 36.66 | 0.805 ± 0.060 | 0.725 ± 0.084 |
| 25 | RF | 36.73 ± 9.72 | 265.23 ± 37.76 | 0.804 ± 0.060 | 0.723 ± 0.084 |
| 50 | SVM | 82.00 ± 10.41 | 373.00 ± 49.40 | 0.596 ± 0.063 | 0.351 ± 0.092 |
| 50 | DTC | 81.35 ± 10.26 | 366.75 ± 43.43 | 0.596 ± 0.063 | 0.348 ± 0.093 |
| 50 | RF | 81.35 ± 10.26 | 366.75 ± 43.43 | 0.596 ± 0.063 | 0.348 ± 0.093 |
| 75 | SVM | 97.50 ± 3.45 | 359.87 ± 45.12 | 0.599 ± 0.025 | 0.226 ± 0.033 |
| 75 | DTC | 97.50 ± 3.45 | 364.36 ± 45.48 | 0.600 ± 0.025 | 0.230 ± 0.035 |
| 75 | RF | 97.50 ± 3.45 | 364.77 ± 46.17 | 0.600 ± 0.024 | 0.230 ± 0.032 |
| 100 | SVM | 100.00 ± 0.00 | 355.97 ± 31.43 | 0.641 ± 0.014 | 0.198 ± 0.010 |
| 100 | DTC | 100.00 ± 0.00 | 370.89 ± 32.76 | 0.644 ± 0.015 | 0.212 ± 0.020 |
| 100 | RF | 100.00 ± 0.00 | 362.59 ± 32.87 | 0.643 ± 0.015 | 0.205 ± 0.016 |

## Interpretation

PDR rises naturally with density because more sensor-only paths and common idle
channel opportunities exist.  The 25-node E2ED is not comparable as a
monotonic-density curve: one scenario had no delivered packet and the remaining
low-density deliveries are survivorship-biased toward short routes.  At higher
densities, E2ED depends on the selected stable route and controller-control
delay; it should not be artificially forced to decline in every sample.

Total ROR falls from sparse to medium density but rises from 75 to 100 nodes
because all sensors periodically beacon and the paper's total ROR includes this
control traffic.  Reactive ROR falls substantially, from approximately 0.72 at
25 nodes to 0.20–0.21 at 100 nodes.  No value is clamped or scaled.

Model selections are often shared: all three models selected the same OC in
16/20, 17/20, 16/20, and 14/20 scenarios at 25/50/75/100 nodes respectively.
Therefore similar network metrics for some model pairs are a valid consequence
of making the same decision, not duplicated simulations or an outcome leak.

## Raw evidence

- `oc_feature_dataset.csv`: 2,000 scenarios / 8,000 feature-only rows.
- `models/ml_results.csv`: held-out ML results.
- `network_validation_20/model_selections.csv`: selected OCs.
- `network_validation_20/candidate_network_runs.csv`: 97 executed candidates.
- `network_validation_20/model_network_results.csv`: all 240 mapped outcomes.
- `network_validation_20/network_summary.csv`: machine-readable table.
