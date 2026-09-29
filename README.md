# Cancer Incidence Dynamics

Reproducible computational analysis accompanying a master's dissertation on age-specific cancer incidence dynamics, multistage carcinogenesis, and cohort effects.

## Overview

The repository contains the final Python analysis pipeline used to reproduce the computational results reported in the dissertation.

The analysis includes:

- age-specific cancer incidence analysis;
- multistage carcinogenesis modelling;
- colorectal and pan-cancer analyses;
- cohort-based dynamic modelling;
- screening and detection-related analyses;
- GLS estimation;
- sensitivity analyses;
- internal consistency checks;
- validation of reproduced dissertation values.

The final pipeline successfully reproduces all 121 dissertation values included in the validation framework and passes all 39 internal consistency checks.

## Repository structure

```text
.
├── dissertation_analysis_final.py
├── requirements.txt
├── outputs/
│   ├── figures/
│   └── tables/
├── README.md
└── .gitignore
