# Toward Self-Governing Underwater Networks

[![Project CI](https://github.com/aggravating08/6G-SDN-Underwater-ML/actions/workflows/per_commit.yml/badge.svg)](https://github.com/aggravating08/6G-SDN-Underwater-ML/actions/workflows/per_commit.yml)

Learning-assisted controller selection for autonomous-underwater-vehicle (AUV)
networks, implemented with ns-3.38 and a reproducible Python machine-learning
pipeline.

## Author and affiliation

**Asiya Ali Khan**<br>
School of Electrical Engineering and Computer Science (SEECS)<br>
National University of Sciences and Technology (NUST), Islamabad, Pakistan<br>
Email: asiyaaaaa8@gmail.com

## What the project does

The simulator models a hierarchical software-defined underwater network and
selects one optimal controller from four mobile AUV candidates. A candidate is
described only by position, local sensor density, and speed. SVM, Decision Tree,
and Random Forest models are trained with scenario-level splits, then evaluated
using packet delivery ratio, end-to-end completion delay, and routing overhead.

The repository contains the simulator, canonical 2,000-scenario dataset,
trained-model workflow, matched network-evaluation ledgers, and plotting code.
The bundled network model is an event-driven analytical simulation rather than
a full acoustic-modem PHY/MAC implementation.

## Main files

- `scratch/randy.cc` — deterministic ns-3 scenario generator and network evaluator.
- `underwater_ml_pipeline.py` — dataset validation, grouped ML training,
  controller selection, and matched evaluation.
- `generate_final_graphs.m` — MATLAB plotting script for the canonical results.
- `UNDERWATER_PAPER_HANDOFF.md` — methodology, assumptions, metrics, and result notes.
- `results/final_three_file_project/` — canonical tables, models, ledgers, and figures.

The canonical input is
`results/underwater_partner_style_equivalent_2000/underwater_feature_rule_2000_scenarios_8000_candidates.csv`.
It contains four controller candidates for each of 2,000 scenarios.

## Requirements

- Linux with a C++ compiler, CMake, and Ninja
- Python 3.10 or 3.11
- Python packages listed in `requirements.txt`
- MATLAB only when regenerating the final MATLAB figure

## Reproduce the workflow

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

./ns3 configure --build-profile=release \
  --enable-modules="core;network;mobility" \
  --disable-examples --disable-tests --disable-precompiled-headers \
  --disable-werror -G Ninja
./ns3 build randy

python underwater_ml_pipeline.py project
matlab -batch "generate_final_graphs"
```

For a quick deterministic simulator check:

```shell
./ns3 run "scratch/randy --mode=labels --nodeCount=25 --runs=2 --baseSeed=29 --output=/tmp/labels.csv"
```

## Reproducibility safeguards

- All four AUV candidates from one scenario remain in the same data split.
- Labels are created from the documented feature-only suitability rule before
  network evaluation.
- Network metrics are not used as ML input features.
- Matched candidate evaluation verifies topology identity within each scenario.
- The continuous-integration check validates the canonical dataset and runs a
  deterministic simulator smoke test on every change to `main`.

## Research context

This implementation accompanies the submitted study *Toward Self-Governing
Underwater Networks: Learning-Assisted Controller Selection with Autonomous
Underwater Vehicles*.

## License and upstream software

The simulator is built on ns-3.38, which is distributed under the GNU General
Public License v2. See `LICENSE` and `NOTICE` for the bundled upstream project
terms.
