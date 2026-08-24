# wg-ultrasound / DICOM and Direct Data

<!-- This repository holds the work-in-progress material for the ultrasound
working group's DICOM and Direct Data subgroup.

Join our email list:
- https://groups.google.com/g/monai-wg-ultrasound

Join our slack channel:
- https://join.slack.com/t/projectmonai/shared_invite/zt-3hucgm02q-i8Bn9XofDZs2UGOH4jUl4w

Visit our website:
- https://project-monai.github.io/wg_ultrasound.html -->



Utilities and evaluation scripts for reading, validating, converting, and benchmarking ultrasound DICOM data, built for the **MONAI Ultrasound Working Group** (DICOM & Direct Data subgroup). The pipeline runs across a multi-vendor, multi-dataset ultrasound corpus (Aliza, AIIMS, Breast Lesions USG, MONAI, Heartcycle, UTA4/7/10, ReMIND, TCIA IDC Prostate MRI-US Biopsy) and answers one core question: **why do some ultrasound DICOM files behave differently across viewers, PyDICOM, SimpleITK, and MONAI — and how do you get them all reliably into a MONAI-ready tensor?**

## General info

- **Email list** — [monai-wg-ultrasound](https://groups.google.com/g/monai-wg-ultrasound)
- **Slack** — [Join the channel](https://join.slack.com/t/projectmonai/shared_invite/zt-3hucgm02q-i8Bn9XofDZs2UGOH4jUl4w)
- **Website** — [project-monai.github.io/wg_ultrasound](https://project-monai.github.io/wg_ultrasound.html)
- **Subgroup Meeting Notes** — [Subgroup meeting notes (A+A & D3)](https://docs.google.com/document/d/1iKNylVIwgFbmesYX5Ux2O5cx_NsBu0G_eX6k1e6c_aA/edit?usp=sharing)
- **US-WG Meeting Notes** — [ultrasound Working Group meeting notes](https://docs.google.com/document/d/1iKNylVIwgFbmesYX5Ux2O5cx_NsBu0G_eX6k1e6c_aA/edit?usp=sharing)

> **Note** From June 1, 2026 onward, the 'DICOM and Direct Data' subgroup has been merged with the 'Annotation and Anonymization' subgroup.


## Repository contents

| Resource | Description |
| -------- | ----------- |
| [scripts/Evaluation_US_DICOM_Data_Heterogeneity.ipynb](scripts/Evaluation_US_DICOM_Data_Heterogeneity.ipynb) | Core interoperability evaluation pipeline. Discovers and strictly validates DICOM files across all listed datasets, reads each with PyDICOM and SimpleITK, classifies read outcomes and failure root causes, flags Doppler/colour-flow evidence, and estimates the MONAI channel-first tensor shape per file. Produces `data/interoperability_evaluation_us_dicom.csv`. |
| [scripts/Compare_pydicom_simpleitk_monai.ipynb](scripts/Compare_pydicom_simpleitk_monai.ipynb) | Head-to-head benchmark of PyDICOM, SimpleITK, and MONAI's `PydicomReader` — read success, runtime, pixel/metadata agreement (including RMSE), multi-frame handling, with plotting utilities for cross-reader comparison. |
| [scripts/Metadata_States.ipynb](scripts/Metadata_States.ipynb) | Reads the evaluation CSV and reports distribution summaries: dataset, manufacturer/model, Doppler-candidate flag, photometric interpretation, SOP class, transfer syntax, acquisition type, raw/MONAI array shapes, and failure root cause. |
| [scripts/US_DICOM_to_PNG.ipynb](scripts/US_DICOM_to_PNG.ipynb) | Converts discovered DICOM files to PNG (with VOI LUT windowing, `MONOCHROME1` inversion, YBR→RGB conversion, bit-depth preservation, multi-frame → per-frame export), preserving the source dataset folder hierarchy. |
| [scripts/US_DICOM_to_MONAI_TENSORS.ipynb](scripts/US_DICOM_to_MONAI_TENSORS.ipynb) | Extends the evaluation pipeline into an actual MONAI data-loading step: classifies ultrasound semantics per file, builds a channel-first `torch.Tensor`, and validates it through a MONAI dictionary transform into a `MetaTensor` with metadata preserved. |
| [data/interoperability_evaluation_us_dicom.csv](data/interoperability_evaluation_us_dicom.csv) | Per-file evaluation results — dataset origin, manufacturer/model, SOP class, transfer syntax, photometric interpretation, acquisition type, Doppler-candidate flag, raw/MONAI pixel shapes, and (for failures) root cause. |
| [data/README.md](data/README.md) | Local dataset folder layout, with a link to the DICOM Dataset spreadsheet for sources/access details. |

---

## Shared reference material

| Resource | Description |
| -------- | ----------- |
| [US_DICOM_Heterogeneity_Presentation](https://docs.google.com/presentation/d/1albyeijl_JkiDlU2covn2FH1Nac_HGBT) | Presentation: MONAI Ultrasound Working Group — US DICOM Data Heterogeneity Evaluation, Metadata States, PNG pipeline (DICOM & Direct Data subgroup). |
| [MONAI US meeting — Annotation+Anonymization+DICOM+Direct Data (7-15-26) Presentation](https://docs.google.com/presentation/d/1sDs8gfaeCIddfZUEoa42PAtZjXSzmpJ2CLDzeaLt6fA/edit?usp=sharing) | Joint meeting slides covering both the Annotation/Anonymization and DICOM/Direct Data subgroups. |
| [DICOM Dataset (spreadsheet)](https://docs.google.com/spreadsheets/d/1F3uH8QNcErDbMDN4_2Eiuxj_50H3MhxS5Hkk25kl6mg/edit?gid=0#gid=0) | Master list of candidate/used ultrasound DICOM datasets — repo/source links, anatomy, 2D/3D/4D mode, raw-DICOM availability, vendor(s), access type, and notes on value/limitations for each. |

---



## Repository structure

```
monai-us-dicom-direct-data/
├── README.md
├── requirements.txt
├── scripts/
│   ├── Evaluation_US_DICOM_Data_Heterogeneity.ipynb
│   ├── Compare_pydicom_simpleitk_monai.ipynb
│   ├── Metadata_States.ipynb
│   ├── US_DICOM_to_PNG.ipynb
│   └── US_DICOM_to_MONAI_TENSORS.ipynb
└── data/
    └── interoperability_evaluation_us_dicom.csv
└── Datasets/
    ├── README.md
```
<!-- 
## Contents

### `scripts/Evaluation_US_DICOM_Data_Heterogeneity.py`
The core interoperability evaluation pipeline. Scans a set of dataset folders, strictly validates each file as real DICOM (magic-byte + required-tag check, not just file extension), then runs each file through **PyDICOM** and **SimpleITK** in parallel — reading metadata and pixel data separately so a metadata-only failure doesn't hide a pixel-decode success (or vice versa). It classifies every file's outcome (`both_ok`, partial failure, root cause of failure), flags likely Doppler/colour-flow studies from metadata evidence, and estimates the MONAI channel-first tensor shape each file would map to. On the evaluated 1,414-file corpus it reports a 99.4% standard-read success rate, with the 8 failures traced to empty (0-byte) files and one proprietary Philips 3D/4D container. This is the script that produces `interoperability_evaluation_us_dicom.csv`.

### `scripts/Compare_pydicom_simpleitk_monai.py`
A head-to-head benchmark of three DICOM readers — **PyDICOM**, **SimpleITK**, and **MONAI's `PydicomReader`** — on the same files. For each file it times each reader, compares the decoded pixel arrays (shape, dtype, numeric agreement/RMSE), checks metadata consistency, and separately handles multi-frame files. Includes plotting utilities (success rate, runtime, transfer-syntax/photometric/manufacturer breakdowns, agreement heatmap, throughput vs. file size) to visualize where the three readers agree or diverge, and can save side-by-side / difference images for visual inspection of mismatches.

### `scripts/Metadata_States.py`
A reporting/analysis script that reads the evaluation CSV (output of `Evaluation_US_DICOM_Data_Heterogeneity.py`) and prints distribution summaries: files by dataset, manufacturer, manufacturer model, Doppler-candidate flag, photometric interpretation, SOP class, transfer syntax, acquisition type, raw array shape, MONAI-mapped shape, and root cause of failure. Includes a manufacturer → model breakdown table and (commented-out) `matplotlib` bar-chart versions of each distribution for a visual pass.

### `scripts/US_DICOM_to_PNG.py`
Converts every discovered ultrasound DICOM file into PNG images, preserving the original dataset folder hierarchy under `post_processed_data/`. Applies the same display transformations a DICOM viewer would (VOI LUT windowing, `MONOCHROME1` inversion, YBR→RGB colour conversion), preserves bit depth where possible (falls back to normalized 16-bit for float pixel data), and handles multi-frame files by exporting one PNG per frame into a per-study subfolder.

### `scripts/US_DICOM_to_MONAI_TENSORS.py`
Extends the interoperability evaluation into an actual MONAI data-loading pipeline. After the same discovery/validation/read steps as `Evaluation_US_DICOM_Data_Heterogeneity.py`, it goes further: classifies each file's ultrasound semantics (e.g. single-frame vs. multi-frame, volumetric properties), decodes pixel data into a channel-first array, builds a real `torch.Tensor`, and passes it through a MONAI dictionary transform (`Compose`/`EnsureTyped`) to produce a validated `MetaTensor` with metadata preserved — i.e. confirms the file is not just readable but actually consumable by a MONAI training pipeline.

### `data/interoperability_evaluation_us_dicom.csv`
Per-file evaluation results produced by `Evaluation_US_DICOM_Data_Heterogeneity.py` — one row per DICOM file, with columns covering dataset origin, manufacturer/model, SOP class, transfer syntax, photometric interpretation, acquisition type, Doppler-candidate flag, raw pixel array shape, MONAI channel-first shape, and (for failures) root cause. -->

## Setup

1. Create and register a dedicated Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=monai_d3 --display-name "Python (monai_d3)"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Installs: `pydicom`, `pylibjpeg` (+ `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg` for compressed transfer syntaxes), `SimpleITK`, `gdcm`, `openpyxl`, `matplotlib`, `monai`, `torch`.
3. Edit `DATASET_ROOTS` (and `OUT_CSV` / `OUTPUT_DIR` where relevant) at the top of each script to point at your local dataset folders.
4. Launch Jupyter and select the **Python (monai_d3)** kernel, or run the notebook directly.



## Suggested pipeline order

1. **`Evaluation_US_DICOM_Data_Heterogeneity.py`** — discover, validate, and evaluate all DICOM files → produces `interoperability_evaluation_us_dicom.csv`.
2. **`Metadata_States.py`** — summarize and inspect the evaluation results (distributions, manufacturer breakdown, failure causes).
3. **`Compare_pydicom_simpleitk_monai.py`** — benchmark reader-level agreement between PyDICOM, SimpleITK, and MONAI.
4. **`US_DICOM_to_PNG.py`** — export human-viewable PNGs for visual QA of any flagged files.
5. **`US_DICOM_to_MONAI_TENSORS.py`** — confirm end-to-end MONAI tensor loading for files that passed evaluation.