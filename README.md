harbor
======
A home for docking evaluation and assessment


[![GitHub Actions Build Status](https://github.com/choderalab/harbor/actions/workflows/harbor-ci.yaml/badge.svg)](https://github.com/choderalab/harbor/actions?query=workflow%3ACI+branch%3Amain)
[![codecov](https://codecov.io/gh/choderalab/harbor/graph/badge.svg?token=V6EZKD9L2F)](https://codecov.io/gh/choderalab/harbor)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/choderalab/harbor/main.svg)](https://results.pre-commit.ci/latest/github/choderalab/harbor/main)
[![Documentation Status](https://readthedocs.org/projects/harbor/badge/?version=latest)](https://harbor.readthedocs.io/en/latest/?badge=latest)

# What should harbor help me do?
* Simplify the conversion of docking results and experimental data into an easily comparable format
* Evaluate the performance of a docking method with standardized metrics
* Visualize the performance of a docking method with standardized plots
* Perform cheminformatics analysis on the docked compounds

# Informed by the following resources:
https://projects.volkamerlab.org/teachopencadd/talktorials/T007_compound_activity_machine_learning.html

# Dependencies and Installation
This package is basically a wrapper / guide for using various packages such as SciKit-Learn, Plotly, Pandas, and more...

```bash
git clone git@github.com:choderalab/harbor.git
cd harbor
mamba env create -f devtools/harbor.yaml
pip install .
```
