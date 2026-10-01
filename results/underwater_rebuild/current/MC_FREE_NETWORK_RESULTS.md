# MC-free network evaluation: 20 unseen scenarios per density

The MC was removed. The selected OC is the highest controller and uses only its
localized gateway/LC view. There is no global route fallback. This evaluation
used the existing saved SVM/DTC/RF models and 80 new scenario seeds, disjoint
from the ML data and earlier MC-backed evaluations.

107 unique `(scenario, selected OC)` candidates cover all 240 model-scenario
outcomes. Every candidate generated 200 packets; all have `OC_data_hops = 0`
and `architecture_violations = 0`.

| Nodes | Model | Mean PDR % (95% CI) | Mean E2ED ms (95% CI) | Mean ROR total (95% CI) | Mean ROR reactive (95% CI) |
|---:|---|---:|---:|---:|---:|
| 25 | SVM | 27.35 ± 8.56 | 223.57 ± 32.43 | 0.848 ± 0.050 | 0.757 ± 0.077 |
| 25 | DTC | 26.85 ± 7.83 | 231.60 ± 29.44 | 0.853 ± 0.049 | 0.765 ± 0.075 |
| 25 | RF | 26.10 ± 7.71 | 222.56 ± 33.25 | 0.858 ± 0.047 | 0.775 ± 0.069 |
| 50 | SVM | 67.13 ± 10.21 | 331.30 ± 27.24 | 0.641 ± 0.058 | 0.390 ± 0.090 |
| 50 | DTC | 67.15 ± 10.03 | 322.58 ± 27.83 | 0.638 ± 0.059 | 0.387 ± 0.091 |
| 50 | RF | 65.40 ± 9.96 | 324.46 ± 27.33 | 0.650 ± 0.057 | 0.405 ± 0.091 |
| 75 | SVM | 84.75 ± 6.05 | 339.01 ± 29.11 | 0.627 ± 0.027 | 0.271 ± 0.046 |
| 75 | DTC | 83.90 ± 7.06 | 341.80 ± 29.47 | 0.629 ± 0.034 | 0.279 ± 0.056 |
| 75 | RF | 83.98 ± 6.10 | 345.45 ± 27.41 | 0.629 ± 0.029 | 0.279 ± 0.046 |
| 100 | SVM | 91.68 ± 3.96 | 330.65 ± 17.82 | 0.677 ± 0.022 | 0.250 ± 0.042 |
| 100 | DTC | 92.65 ± 3.54 | 330.63 ± 18.93 | 0.676 ± 0.017 | 0.244 ± 0.033 |
| 100 | RF | 90.05 ± 5.42 | 332.96 ± 18.28 | 0.685 ± 0.025 | 0.268 ± 0.051 |

The same OC was selected by all three models in 14/20, 15/20, 14/20 and 13/20
scenarios at 25/50/75/100 nodes. Similar model-level outcomes in those cases
are exact reuse of the same candidate result, not copied statistics.

Raw evidence: `model_selections.csv`, `candidate_network_runs.csv`,
`model_network_results.csv`, and `network_summary.csv` in this directory.
