# Toward Self-Governing Underwater Networks

## Learning-Assisted Controller Selection with Autonomous Underwater Vehicles

## Author and affiliation

**Asiya Ali Khan**

School of Electrical Engineering and Computer Science (SEECS), National
University of Sciences and Technology (NUST), Islamabad, Pakistan.

This repository contains an ns-3.38 analytical simulation of hierarchical
software-defined underwater networking and a machine-learning pipeline for
selecting one optimal controller from four mobile AUV candidates. The study
compares SVM, Decision Tree, and Random Forest selection using packet delivery
ratio, completion delay, and routing overhead ratio.

The project investigates how compact situational features—candidate position,
local sensor density, and vehicle speed—can support context-aware controller
selection in dynamic AUV-assisted underwater networks. It connects
controller-selection accuracy to network-level outcomes and provides the code,
generated dataset, model configurations, evaluation ledgers, and figures needed
to reproduce the reported analytical study.

The research is associated with the submitted manuscript *Toward Self-Governing
Underwater Networks: Learning-Assisted Controller Selection with Autonomous
Underwater Vehicles*.

## Project entry points

- `scratch/randy.cc`: ns-3 simulator and dataset generator.
- `underwater_ml_pipeline.py`: grouped ML training, controller selection, and
  matched network evaluation.
- `generate_final_graphs.m`: MATLAB plotting script for the canonical results.
- `UNDERWATER_PAPER_HANDOFF.md`: methodology, assumptions, metrics, and current
  result summary.
- `results/final_three_file_project/`: canonical output tables, model artifacts,
  candidate ledgers, and figures.

The canonical input dataset is
`results/underwater_partner_style_equivalent_2000/underwater_feature_rule_2000_scenarios_8000_candidates.csv`.
It contains 2,000 scenarios and 8,000 AUV-candidate rows.

## Reproduce the canonical workflow

Requirements include a C++ toolchain supported by ns-3.38, Python 3 with
NumPy, pandas, scikit-learn, joblib, and Matplotlib, plus MATLAB for the final
plotting step.

```shell
./ns3 configure
./ns3 build
python3 underwater_ml_pipeline.py project
matlab -batch "generate_final_graphs"
```

Generated `build/` and `cmake-cache/` directories are intentionally excluded
from version control. See `UNDERWATER_PAPER_HANDOFF.md` before interpreting the
results: the simulator is an event-driven analytical model, not a full acoustic
modem PHY/MAC implementation.

---

# The Network Simulator, Version 3


[![codecov](https://codecov.io/gh/nsnam/ns-3-dev-git/branch/master/graph/badge.svg)](https://codecov.io/gh/nsnam/ns-3-dev-git/branch/master/)
[![Gitlab CI](https://gitlab.com/nsnam/ns-3-dev/badges/master/pipeline.svg)](https://gitlab.com/nsnam/ns-3-dev/-/pipelines)
[![Github CI](https://github.com/nsnam/ns-3-dev-git/actions/workflows/per_commit.yml/badge.svg)](https://github.com/nsnam/ns-3-dev-git/actions)


## Table of Contents

1) [An overview](#an-open-source-project)
2) [Building ns-3](#building-ns-3)
3) [Running ns-3](#running-ns-3)
4) [Getting access to the ns-3 documentation](#getting-access-to-the-ns-3-documentation)
5) [Working with the development version of ns-3](#working-with-the-development-version-of-ns-3)

> **NOTE**: Much more substantial information about ns-3 can be found at
<https://www.nsnam.org>

## An Open Source project

ns-3 is a free open source project aiming to build a discrete-event
network simulator targeted for simulation research and education.
This is a collaborative project; we hope that
the missing pieces of the models we have not yet implemented
will be contributed by the community in an open collaboration
process.

The process of contributing to the ns-3 project varies with
the people involved, the amount of time they can invest
and the type of model they want to work on, but the current
process that the project tries to follow is described here:
<https://www.nsnam.org/developers/contributing-code/>

This README excerpts some details from a more extensive
tutorial that is maintained at:
<https://www.nsnam.org/documentation/latest/>

## Building ns-3

The code for the framework and the default models provided
by ns-3 is built as a set of libraries. User simulations
are expected to be written as simple programs that make
use of these ns-3 libraries.

To build the set of default libraries and the example
programs included in this package, you need to use the
tool 'ns3'. Detailed information on how to use ns3 is
included in the file doc/build.txt

However, the real quick and dirty way to get started is to
type the command

```shell
./ns3 configure --enable-examples
```

followed by

```shell
./ns3
```

in the directory which contains this README file. The files
built will be copied in the build/ directory.

The current codebase is expected to build and run on the
set of platforms listed in the [release notes](RELEASE_NOTES.md)
file.

Other platforms may or may not work: we welcome patches to
improve the portability of the code to these other platforms.

## Running ns-3

On recent Linux systems, once you have built ns-3 (with examples
enabled), it should be easy to run the sample programs with the
following command, such as:

```shell
./ns3 run simple-global-routing
```

That program should generate a `simple-global-routing.tr` text
trace file and a set of `simple-global-routing-xx-xx.pcap` binary
pcap trace files, which can be read by `tcpdump -tt -r filename.pcap`
The program source can be found in the examples/routing directory.

## Getting access to the ns-3 documentation

Once you have verified that your build of ns-3 works by running
the simple-point-to-point example as outlined in 3) above, it is
quite likely that you will want to get started on reading
some ns-3 documentation.

All of that documentation should always be available from
the ns-3 website: <https://www.nsnam.org/documentation/>.

This documentation includes:

- a tutorial
- a reference manual
- models in the ns-3 model library
- a wiki for user-contributed tips: <https://www.nsnam.org/wiki/>
- API documentation generated using doxygen: this is
  a reference manual, most likely not very well suited
  as introductory text:
  <https://www.nsnam.org/doxygen/index.html>

## Working with the development version of ns-3

If you want to download and use the development version of ns-3, you
need to use the tool `git`. A quick and dirty cheat sheet is included
in the manual, but reading through the git
tutorials found in the Internet is usually a good idea if you are not
familiar with it.

If you have successfully installed git, you can get
a copy of the development version with the following command:

```shell
git clone https://gitlab.com/nsnam/ns-3-dev.git
```

However, we recommend to follow the Gitlab guidelines for starters,
that includes creating a Gitlab account, forking the ns-3-dev project
under the new account's name, and then cloning the forked repository.
You can find more information in the [manual](https://www.nsnam.org/docs/manual/html/working-with-git.html).
