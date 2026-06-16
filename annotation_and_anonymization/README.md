# wg-ultrasound / Annotation and Anonymization

> Work-in-progress material for the ultrasound working group's **Annotation and Anonymization** subgroup.

---

## Repository contents

| Resource | Description |
| -------- | ----------- |
| [deid_tools.md](deid_tools.md) | Comparison of de-identification tools for DICOM metadata anonymization and burned-in pixel PHI removal, with pros, cons, and subgroup evaluation notes. |
| [deid_datasets.md](deid_datasets.md) | Catalog of datasets with synthetic PHI and clean ultrasound sources suitable for PHI injection, with links and usage notes. |
| [phase1_synthetic_phi.ipynb](phase1_synthetic_phi.ipynb) | Phase 1 synthetic PHI pipeline notebook: burns known synthetic identifiers into lung POCUS images, writes multi-frame DICOMs, and produces ground-truth overlay manifests for testing de-id tools. |
| [ED01_z07_Lung_zone_7.dcm.zip](ED01_z07_Lung_zone_7.dcm.zip) | Sample output from the Phase 1 pipeline: an 87-frame lung ultrasound DICOM with planted synthetic PHI in pixels and headers, for quick inspection or de-id tool benchmarking. |

---

## General info

- **Email list** — [monai-wg-ultrasound](https://groups.google.com/g/monai-wg-ultrasound)
- **Slack** — [Join the channel](https://join.slack.com/t/projectmonai/shared_invite/zt-3hucgm02q-i8Bn9XofDZs2UGOH4jUl4w)
- **Website** — [project-monai.github.io/wg_ultrasound](https://project-monai.github.io/wg_ultrasound.html)

---

## Contributors

| | |
| --- | --- |
| Deepa Krishnaswamy | Lakshmi Mahabaleshwara |
| Tamas Ungi | Dave Dinh |
| Joel Jacob Tomson | Luv Kohli |
| Sharanya Balachandran | Shuhei Misawa |
