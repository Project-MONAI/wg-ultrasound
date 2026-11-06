# Phase 2: de-identification of temporal ultrasound (2D+t / 3D+t)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Project-MONAI/wg-ultrasound/blob/main/annotation_and_anonymization/temporal-deid-US/temporal-deid-US.ipynb)

**Question:** does de-identification remove identifiable information without altering the scientific information needed to reuse a temporal ultrasound DICOM?

Phase 1 ([`phase1_synthetic_phi.ipynb`](../phase1_synthetic_phi.ipynb)) planted synthetic PHI in lung POCUS. Phase 2 plants synthetic identifiers in public cines and 3D+t volumes and runs several de-identification tools. Every output is then checked for:
- **E1, privacy:** are the identifiers gone?
- **E2, preservation:** are timing, 3D+t organisation, calibration and pixels unchanged?
- **E3, burned-in text:** is the text removed, and is the image left undamaged?
- **E4, reuse:** do the time-intensity curve, duration and distances still give the same values?

Three deliberately broken "tools" are used as controls, to show that the checks can fail.

## Run

Open the notebook in Colab (badge above), choose `MODE`, then *Runtime → Run all*.

| `MODE` | Data | Time |
|---|---|---|
| `quick` | small synthetic files, no download | ~5 min |
| `full` | public data below | ~3.5 h on a T4 GPU (*Runtime → Change runtime type*), almost all of it RSNA OCR; untick `INSTALL_RSNA` for ~30 min |

The first cell downloads this folder from GitHub and installs the tools, so nothing has to be uploaded. To test an unmerged pull request, set `GITHUB_REPO` to the fork and `GITHUB_BRANCH` to the PR branch. Results (figures, tables, `answer.md`) are saved to your own Drive under `phase2_results/`.

To run it locally instead:
1. `pip install -r requirements.txt`
2. Make sure `ffmpeg` is on the PATH.
3. Run the notebook. RSNA and CTP are optional; see the notes in `requirements.txt`.

## Data (all public; synthetic PHI only)

Each file is downloaded from its original host and checked against the SHA-256 in `data_manifest.json`. No data are stored in this repository.

| Source | Content | Licence |
|---|---|---|
| [Aliza sample DICOMs](https://www.aliza-dicom-viewer.com/download/datasets) | vendor 2D+t cines (Philips, Acuson, GE) and a GE 3D+t volume | publisher's sample data |
| TCIA B-mode-and-CEUS-Liver, [10.7937/TCIA.2021.v4z7-tc39](https://doi.org/10.7937/TCIA.2021.v4z7-tc39) | clinical contrast cines | CC BY 4.0 |
| TCIA MIDI-B Synthetic Validation, [10.7937/cf2p-aw56](https://doi.org/10.7937/cf2p-aw56) | ultrasound files with realistic synthetic PHI | CC BY 4.0 |
| Phase 1 sample `ED01_z07_Lung_zone_7` | Phase 1 output (OpenPOCUS images) | this repository |

OpenPOCUS itself is not used as a separate source: it is distributed as JPEG frames, without DICOM timing or calibration.

## Files

- `temporal-deid-US.ipynb`: the notebook.
- `deid_eval/`: the code it calls.
  - `pipeline.py`: the steps.
  - `figures.py`: the plots.
  - `validators.py`: the checks.
  - `adapters.py`: one function per tool; adding a tool takes one function here.
- `data_manifest.json`: sources, licences and checksums.

## Notes and limitations

- **pydicom-deid** must be called with `save=True`. Otherwise the returned path is the *input* file.
- **RSNA Anonymizer** needs `controller.stop()`, or its worker threads never exit.
- Identifiers are synthetic and placed in known fields. Real exports may hide PHI elsewhere.
- The MIDI-B score uses an answer key derived from the files, not TCIA's official key.
- The H.264 file is a re-encoded copy of a public cine, not a vendor export.
- Burned-in text uses a single font.
- Sample sizes are small.
