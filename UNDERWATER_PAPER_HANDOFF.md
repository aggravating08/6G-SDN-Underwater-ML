# Underwater Hierarchical SDN + ML — Paper Handoff

## Purpose

This project evaluates ML-based selection of one **Optimal Controller (OC)**
among four mobile AUV candidates in an analytical underwater hierarchical SDN.
The work compares SVM, Decision Tree (DTC), and Random Forest (RF) OC
selection using packet delivery ratio (PDR), delay, and routing overhead ratio
(ROR). SVM is the proposed model.

## Architecture

- Study area: **500 m x 500 m**, using **2D Euclidean distance**.
- Sensor counts: **25, 50, 75, 100**.
- Four mobile AUV candidates. Exactly one is selected as the OC.
- The selected OC and any local controllers (LCs) are **control-plane only**.
- Payload/data plane is strictly sensor-to-sensor. AUVs never forward payload.
- A surface/master controller (MC), where active, is a **control-plane fallback
  only**. It never appears in a data path.
- Required invariant for every simulation result:
  `OC_data_hops = 0` and `architecture_violations = 0`.
- Traffic: 10 flows x 20 packets = **200 generated 64-byte packets per run**.

## Simulator and physical-model scope

The C++ simulator is an **ns-3 event-driven analytical model**, not a complete
underwater acoustic modem PHY/MAC implementation. It includes the current
project sound-speed equation, propagation delay, Thorp attenuation, ambient
noise, SNR, deterministic capacity calculation, TD/LDP-style routing costs,
and sensor-link eligibility. It does not claim collision, BER/PER, fading,
Bellhop/WOSS, ns-MIRACLE, or packet-level modem behavior.

Key frozen physical configuration:

- Sensor range: 100 m.
- AUV/control range: 300 m.
- AUV speed range in scenarios: 1.5–3.0 m/s.
- Transmit power: 150 dB re 1 uPa.
- Fixed acoustic operating frequency: 25 kHz.
- Acoustic band used in the analytical capacity model: 1–40 kHz.
- Link SNR eligibility threshold: 3 dB.

## Canonical files on the Linux workspace

1. `scratch/randy.cc` — ns-3 simulator and feature/dataset generator.
2. `results/underwater_partner_style_equivalent_2000/underwater_feature_rule_2000_scenarios_8000_candidates.csv`
   — 2,000 scenarios / 8,000 AUV-candidate rows.
3. `underwater_ml_pipeline.py` — ML training, grouped split, candidate ranking,
   matched candidate evaluation, and CSV results.
4. `generate_final_graphs.m` — MATLAB reader/plotter for the final CSVs.

The user may provide only this handoff document to a new ChatGPT/Codex chat for
paper-writing context. Code execution or modification requires the relevant
local files to be attached separately.

## Dataset and ML methodology

- 2,000 total scenarios: 500 at each node count.
- Each scenario contains four AUV-candidate rows: 8,000 rows total.
- The **only ML inputs** are:
  `x`, `y`, `local_density`, `speed`.
- The dataset includes additional audit fields, but `underwater_ml_pipeline.py`
  must keep its `FEATURES` list restricted to those four permitted inputs.
- Label rule: one OC per scenario, selected using the documented equal-weight
  feature-only suitability rule: higher local density, closer to area centre
  (250,250), and lower speed. The rule does not use PDR, delay, ROR, route
  outcome, topology outcome, or an ID as ML input.
- Grouped split: 1,400 train, 300 validation, 300 held-out test scenarios;
  350/75/75 scenarios at each node count. All four AUV rows remain in the same
  split.

This means the ML result measures how well each model reproduces the
predefined feature-only controller-suitability label. It is not a claim that
the label is a direct global network-optimal OC oracle.

## Current fixed models

### SVM — proposed model

`StandardScaler + SVC(kernel='rbf', C=20, gamma=0.005,
class_weight=None, probability=False, random_state=42)`.

Use `decision_function` to score and rank the four candidates in a scenario.

### DTC — lightweight baseline

`DecisionTreeClassifier(criterion='gini', max_depth=3,
min_samples_leaf=30, min_samples_split=60, random_state=42)`.

### RF — constrained baseline

`RandomForestClassifier(n_estimators=1, max_depth=1, max_leaf_nodes=2,
min_samples_leaf=0.45, min_samples_split=0.90, max_features=1,
max_samples=0.05, bootstrap=True, random_state=42, n_jobs=-1)`.

The DTC/RF settings are intentionally lightweight baseline configurations. Do
not describe them as fully optimized state-of-the-art tree models.

## Canonical figure-run protocol

`python3 underwater_ml_pipeline.py project` does the following:

1. Checks the 2,000-scenario / 8,000-row feature-only dataset.
2. Creates the fixed grouped split.
3. Fits each model on train + validation scenarios.
4. Evaluates ML accuracy on the 300 held-out test scenarios.
5. Selects the first five deterministic held-out scenarios at each density for
   the **graph network evaluation**.
6. Runs all four matched OC candidates for those 20 scenarios = 80 candidate
   simulations using:
   - fixed HELLO interval: 30 s;
   - route TTL: 60 s;
   - negative route TTL: 120 s;
   - reference topology/update accounting;
   - aggregated topology digest capacity: 40 nodes.
7. Maps the OC selected by each model to that candidate's actual PDR, delay,
   and ROR results.

The five-scenarios-per-density network curves are illustrative matched
evaluation curves. They are **not** a 300-scenario-per-density final
confidence-interval study and must not be described as such.

## Metrics and terminology

### PDR

`PDR (%) = delivered data packets / generated data packets x 100`.

### Raw delivered-packet E2ED

For delivered packets only:

`destination receive time - source generation time`.

This is available in the result CSV as `Raw delivered-packet E2ED (ms)`.

### Displayed delay curve

The paper-style graph displays a **packet-completion delay**, labelled on the
figure as “End-to-end Delay” / “Average completion delay”. It is computed as:

`(sum delivered-packet delays + 10,000 ms x dropped packets) / generated packets`.

It is deliberately not raw delivered-only E2ED; the timeout term prevents the
survivor bias where sparse topologies deliver only easy packets. A paper must
define this formula explicitly rather than call it standard delivered-packet
E2ED.

### ROR

`ROR = control transmissions / (control transmissions + data-hop transmissions)`.

Control includes HELLO/update, topology digest/update, route request/reply,
and controller-control transmissions. Data-hop transmissions include every
sensor-to-sensor payload hop, not only destination arrivals.

### Baseline

The graph label is “Baseline”. Internally it is selected after the fact as the
worst candidate OC per scenario (lowest PDR, then highest delay, then highest
ROR). It is a **comparison bound**, not a deployable fixed-OC protocol. Do not
claim it is a normal fixed-OC baseline in a paper without changing the design.

## Current held-out ML results

| Model | Binary accuracy | Precision | Recall | Binary F1 | Top-1 OC accuracy | Top-1 macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| SVM | 84.42% | 0.851 | 0.457 | 0.594 | 87.33% | 0.870 |
| DTC | 78.83% | 0.713 | 0.257 | 0.377 | 48.67% | 0.477 |
| RF | 75.00% | 0.000 | 0.000 | 0.000 | 22.00% | 0.090 |

Binary accuracy is not the primary OC-selection metric because three of four
candidates are normally negative. Scenario-level Top-1 OC accuracy is the
stricter selection metric.

## Current matched network-curve results

| Nodes | Model | PDR (%) | Raw delivered E2ED (ms) | Packet-completion delay (ms) | ROR |
|---:|---|---:|---:|---:|---:|
| 25 | SVM | 25.6 | 215.11 | 7495.07 | 0.429 |
| 25 | DTC | 16.1 | 197.16 | 8421.74 | 0.680 |
| 25 | RF | 25.3 | 228.72 | 7527.87 | 0.496 |
| 25 | Baseline | 13.4 | 219.72 | 8689.44 | 0.695 |
| 50 | SVM | 72.0 | 320.88 | 3031.03 | 0.156 |
| 50 | DTC | 61.3 | 325.72 | 4069.67 | 0.212 |
| 50 | RF | 58.1 | 318.52 | 4375.06 | 0.240 |
| 50 | Baseline | 53.8 | 330.28 | 4797.69 | 0.277 |
| 75 | SVM | 90.9 | 258.58 | 1145.05 | 0.162 |
| 75 | DTC | 86.7 | 258.42 | 1554.05 | 0.182 |
| 75 | RF | 88.8 | 255.54 | 1346.92 | 0.174 |
| 75 | Baseline | 72.6 | 260.64 | 2929.23 | 0.264 |
| 100 | SVM | 89.6 | 238.41 | 1253.62 | 0.184 |
| 100 | DTC | 83.7 | 244.75 | 1834.85 | 0.214 |
| 100 | RF | 84.3 | 237.45 | 1770.17 | 0.206 |
| 100 | Baseline | 65.1 | 247.80 | 3651.32 | 0.316 |

## Existing output files

Under `results/final_three_file_project/`:

- `ml_accuracy.csv`
- `network_metrics_by_node.csv`
- `network_metrics_overall.csv`
- `matched_four_oc_candidate_ledger.csv`
- `selected_oc_per_test_scenario.csv`
- `selected_oc_network_metrics.csv`
- `grouped_split.csv`
- model artifacts: `svm_model.joblib`, `dtc_model.joblib`, `rf_model.joblib`
- `model_comparison_matlab_style.png` and `.pdf`

## Commands on the Linux machine

```bash
cd /path/to/ns-3.38
./ns3 build
python3 underwater_ml_pipeline.py project
matlab -batch "generate_final_graphs"
```

## Interpretation and scope

The project is a controlled analytical evaluation of hierarchical underwater
SDN controller selection. Its central ML finding is that, on the held-out
grouped split, the SVM most accurately reproduces the predefined feature-only
controller-suitability label. The network curves then map each model's selected
OC to an actual matched candidate simulation outcome.

The plotted delay is packet-completion delay rather than conventional
delivered-only E2ED: each undelivered packet contributes a 10,000 ms timeout.
This was chosen because delivered-only E2ED can make sparse networks appear
artificially fast when only easy packets survive. The normal delivered-only
delay remains available in the result tables as a separate raw value.

The graph's “Baseline” curve is internally the worst actual OC candidate in
each matched scenario. It therefore acts as a comparison bound, rather than a
fixed controller that would be deployed in a live protocol. The current graph
ledger evaluates five deterministic held-out scenarios at each node count; the
ML accuracy values themselves use all 300 held-out scenarios. The simulator
models acoustic propagation and routing analytically within ns-3 and does not
represent a full packet-level underwater modem PHY/MAC.
