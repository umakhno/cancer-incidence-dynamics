# Cancer Incidence Dynamics

Code, source data and generated outputs for the MSc dissertation

**Age-Specific Patterns in Early-Onset Colorectal Cancer Incidence: Separating Carcinogenesis from Detection**
Uliana Makhno, MSc Mathematics (MAM410), City St George's, University of London, 2026.

## Overview

This repository contains the final Python analysis pipeline, the source data it reads, and the figures and tables it generates. Running the pipeline reproduces the computed results, figures and tables reported in the dissertation.

The pipeline covers:

- Weibull fits and log-log slope analysis of DevCan cumulative incidence (Chapter 3);
- the cohort matrix model fitted to incidence and mortality, 1999–2020 (Chapter 4);
- robustness analyses, including the parametric bootstrap (Chapter 5);
- the GLS comparison under dependent slope estimates (Section 7.3);
- the multistage matrix calculations of Appendices A and B and the indices of Appendix C;
- validation of the numerical values quoted in the dissertation.

A complete run reproduces all 119 dissertation values registered in the validation table and passes all 39 internal consistency checks. Two further values, the GLS ΔAICc differences reported in Table 7.1, are generated as a separate diagnostic and are not counted among the 119.

## Repository structure

```text
.
├── data/
│   ├── Colorectal_Devcan.xlsx
│   ├── Pan_Devcan.xlsx
│   ├── Lifetables.zip
│   ├── Multiple Cause of Death, 1999-2020 CRC.xls
│   └── United States and Puerto Rico Cancer Statistics, 1999-2022 Incidence.xls
├── outputs/
│   ├── figures/
│   └── tables/
├── dissertation_analysis_final.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Data sources

All input files are included in `data/` so that the analysis can be rerun without further downloads.

### DevCan cumulative incidence

- `Colorectal_Devcan.xlsx`
- `Pan_Devcan.xlsx`

Cumulative probabilities of diagnosis by age, F(x), for colorectal cancer and for all cancer sites combined, in 16 diagnosis periods from 1975–1977 to 2018–2021 (2020 excluded). Both files were taken from Blair Colyer's CancerIncidenceStan repository, <https://github.com/BlairColyer/CancerIncidenceStan> (accessed 30 September 2026), where they were extracted from SEER data with the NCI DevCan software.

A diagnosis period is the calendar interval represented by one DevCan observation. Where the dissertation refers to a DevCan release, it means that observation, not the calendar interval.

**Age convention.** The colorectal file labels each row by the start of a five-year age interval, whereas F(x) refers to the end of that interval. The script therefore uses age = label + 5 for the colorectal series (`CRC_AGE_SHIFT = 5.0`). The pan-cancer file already uses interval endpoints and is read unchanged.

### United States Cancer Statistics (incidence)

`United States and Puerto Rico Cancer Statistics, 1999-2022 Incidence.xls`

Annual colorectal cancer incidence (site: Colon and Rectum) by five-year age group. Source: United States Cancer Statistics Working Group, 2025 release, accessed through CDC WONDER, <https://wonder.cdc.gov/cancer-v2022.html> (accessed 18 September 2026). The analysis uses 1999–2020 and the age groups 25–29 to 60–64.

### Multiple Cause of Death (mortality)

`Multiple Cause of Death, 1999-2020 CRC.xls`

Annual colorectal cancer deaths by five-year age group, underlying cause ICD-10 C18.0–C18.9, C19 and C20. Source: National Center for Health Statistics, Multiple Cause of Death 1999–2020, accessed through CDC WONDER, <https://wonder.cdc.gov/mcd-icd10.html> (accessed 18 September 2026).

Both CDC WONDER files are tab-separated text exports despite the `.xls` extension, and the script reads them as such.

### Life tables (background mortality)

`Lifetables.zip`

United States Mortality Database period life tables. Source: Winant, C. (2026), *US State Life Tables by State, Age, Sex, Year: 1959–2023*, Harvard Dataverse, V1, <https://doi.org/10.7910/DVN/ZSHJEK> (CC0 1.0; accessed 30 September 2026). The script uses the national table for both sexes combined, `USA_bltper_5x1.txt`. The tables cover 1959–2023. The cohort model projects each cohort from birth, so it also needs calendar years before 1959; for those years the script uses the 1959 table.

## Reproducing the analysis

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline from the repository root:

```bash
python dissertation_analysis_final.py
```

A complete run typically takes about one minute on a standard laptop. Figures are written to `outputs/figures/` and tables to `outputs/tables/`. All simulations use fixed random seeds, so repeated runs give identical output.

The script was tested with Python 3.11, NumPy 2.4, pandas 3.0, SciPy 1.17 and Matplotlib 3.10. `openpyxl` is required to read the DevCan workbooks.

The input and output locations can be changed with the environment variables `DISS_ROOT` (the folder containing `data/`) and `DISS_OUT` (the output folder).

## Outputs and their numbering

Output file names retain their development numbering. The dissertation was later shortened to meet the page limit, so file numbers do not always match those in the final text, and the repository contains supplementary outputs not included in the dissertation.

### Figures in the dissertation

| Dissertation | File in `outputs/figures/` |
|---|---|
| Figure 2.1 | `Fig_2_1.png` |
| Figure 3.1 (a), (b) | `Fig_3_1.png`, `Fig_3_2.png` |
| Figure 3.2 (a), (b) | `Fig_3_3.png`, `Fig_3_4.png` |
| Figure 3.3 (a), (b) | `Fig_3_5.png`, `Fig_3_6.png` |
| Figure 3.4 | `Fig_3_8.png` |
| Figure 3.5 | `Fig_3_9.png` |
| Figure 4.1 | `Fig_4_1.png` |
| Figure 4.2 | `Fig_4_2.png` |
| Figure 4.3 | `Fig_4_3.png` |
| Figure 4.4 | `Fig_4_4.png` |
| Figure 4.5 | `Fig_4_5.png` |
| Figure 5.1 | `Fig_5_7.png` |
| Figure 5.2 | `Fig_5_11.png` |

All other files in `outputs/figures/` are supplementary.

### Tables in the dissertation

| Dissertation | File in `outputs/tables/` |
|---|---|
| Table 2.1 | `Table_2_1.csv` |
| Table 3.1 | `Table_3_1.csv` |
| Table 3.2 | `Table_3_2.csv` |
| Table 3.3 | `Table_3_3.csv` |
| Table 3.4 | `Table_3_3a.csv` |
| Table 3.5 | `Table_3_4.csv` |
| Table 4.1 | `Table_4_1.csv` |
| Table 4.2 | `Table_4_2.csv` |
| Table 5.1 | `Table_5_1.csv` |
| Table 5.2 | `Table_5_2.csv` |
| Table 6.1 | `Table_6_2.csv` |
| Table 7.1 | `Table_7_1.csv` |
| Table A.1 | `Table_A_2.csv` |
| Table B.1 | `Table_B_2.csv` |
| Tables C.1, C.2 | `Table_E_1.csv`, `Table_E_2.csv` |

Files named `Table_S…` and `Diagnostic_…` contain supplementary results quoted in the text, including the fixed-k sensitivity of Section 4.2, the age-band and grid sensitivity of Section 5.4, and the bootstrap selection frequencies of Section 5.5.

### Validation files

- `Validation_dissertation_vs_reproduced.csv` lists each registered numerical value quoted in the dissertation alongside the value computed by the pipeline. Agreement is judged at the precision printed in the dissertation. The `Where` column uses the development numbering described above.
- `Validation_internal_consistency.csv` records the 39 internal checks, including checks that tables and figures are generated from the same underlying arrays.
- `Output_manifest.csv` lists every file written by a complete run.
