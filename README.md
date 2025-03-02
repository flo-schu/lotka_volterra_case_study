![testing](https://github.com/flo-schu/lotka_volterra_case_study/actions/workflows/test.yml/badge.svg)

# lotka_volterra_case_study: 

Short summary of the case study

## Instructions

### Prerequisites

Install git (https://git-scm.com/downloads), conda (https://docs.anaconda.com/free/miniconda/) and datalad (https://handbook.datalad.org/en/latest/intro/installation.html)

### Installation

**Prerequisites**

**If the case study was not already installed as a submodule** with a meta package, you can install the package as follows.

Open a command line utility (cmd, bash, ...) and execute

```bash
git clone git@github.com:flo-schu/lotka_volterra_case_study/1.0.0
cd lotka_volterra_case_study
```

Create environment, activate it and install model package. 
```bash
conda create -n lotka_volterra_case_study python=3.11
conda activate lotka_volterra_case_study
conda install 
pip install -e .
```

In order to install additional inference backends, use:

```bash
pip install pymob[numpyro]
```

For the available backends see https://pymob.readthedocs.io/en/latest/

### Case study layout

This is the layout of your folder. Files like README, requirements, LICENSE and .gitignore are not shown, because they are not strictly necessary for the case study but contain important metadata
```
└─ lotka_volterra_case_study
    ├─ lotka_volterra_case_study
    |   ├─ __init__.py
    |   ├─ data.py
    |   ├─ mod.py
    |   ├─ plot.py
    |   ├─ prob.py
    |   └─ sim.py
    ├─ data
    |   └─ observations.nc 
    ├─ results
    │   └─ ...
    ├─ scenarios
    │   ├─ test_scenario
    │   │   └─ settings.cfg
    │   └─ test_scenario_2
    │       └─ ...
    ├─ scripts
    │   └─ ...
    ├─ pyproject.toml
    ...
```
## Usage

The case studies should now be ready to use.
To get started, see: https://pymob.readthedocs.io/en/latest/

### Command line

there is a command line script provided by the `pymob` package which will directly
run inference accroding to the scenario settings.cfg file provided in the scenario
folder of the respective case study. For details see https://pymob.readthedocs.io/en/latest/case_studies.html

`pymob-infer --case_study lotka_volterra_case_study --scenario SCENARIO --inference_backend numpyro`

The options in the `settings.cfg` are the same as used when preparing the publication

The results will be stored in the results directory of the respective case study 
unless otherwise specified in the settings.cfg


## Tracking results and data with datalad and gin

1. create a new `results` dataset in the root of the repository

`datalad create -c text2git results`

if the directory already exists, use the `-f` flag:

`datalad create -f -c text2git results`

2. save the results as they come in in order to keep a version history.

`datalad save -m "Computed posterior with nuts"`


## Using a case study as a foundation for building new models

You may have noticed an `__init__.py` file. This file makes the case study a package. This way, the package can be installed as a submodule in any other package. 

This means, case-studies can be stacked on top of each other. This way, only the changes need to be captured in the 
new case study and can be transferred into the next project.

The downside is that pymob versions need to be compatible for this use case.