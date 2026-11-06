"""Reader for the subgroup's Phase 1 synthetic-PHI outputs (`phase1_synthetic_phi.ipynb`).

Phase 1 writes, per de-identified patient folder:
    dicom/*.dcm          multi-frame US DICOMs with burned-in PHI and PHI in the header
    run_manifest.csv     one row per zone
    phi_overlays.csv     one row per burned overlay: phi_category, phi_value, rendered_text,
                         bbox_x/bbox_y/bbox_w/bbox_h (or bbox=[x, y, w, h]), dicom_tag, sop_instance_uid, is_phi

Phase 2 consumes that ground truth directly - no canaries are injected into Phase 1 files.
A lone Phase 1 DICOM without its CSV (e.g. the repository sample ED01_z07_Lung_zone_7.dcm) is also
accepted: the header values become the metadata ground truth and the Phase 1 banner geometry
(top-left, 5 px margin, 250 x 100 px) becomes the pixel ground truth.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from pydicom import dcmread

from .dicom_utils import is_dicom, sstr

HEADER_TRUTH = [  # (keyword, category) of the identifiers Phase 1 writes into the header
    ("PatientName", "patient_direct"), ("PatientID", "patient_direct"), ("PatientBirthDate", "patient_direct"),
    ("IssuerOfPatientID", "patient_direct"), ("AccessionNumber", "care_context"), ("InstitutionName", "care_context"),
    ("StudyDate", "dates"), ("StudyInstanceUID", "uids"), ("SeriesInstanceUID", "uids"), ("SOPInstanceUID", "uids"),
]
PHASE1_BANNER = (5, 5, 255, 105)   # x0, y0, x1, y1 of the reserved burn region in Phase 1


def header_truth(ds) -> list[dict]:
    can = []
    for kw, cat in HEADER_TRUTH:
        v = sstr(getattr(ds, kw, ""))
        if len(v) >= 4:
            can.append({"category": cat, "field": kw, "value": v, "needle": v})
    comments = sstr(getattr(ds, "ImageComments", ""))
    for part in comments.split("|"):
        part = part.strip()
        if part.startswith("case="):
            can.append({"category": "free_text", "field": "ImageComments (case uuid)", "value": part[5:], "needle": part[5:]})
    for c in can:
        c["needle_b"] = c["needle"].encode("ascii", "ignore")
    return can


def _bbox(row):
    if {"bbox_x", "bbox_y", "bbox_w", "bbox_h"} <= set(row.index) and pd.notna(row.get("bbox_x")):
        x, y, w, h = (int(row[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
    else:
        x, y, w, h = (int(v) for v in ast.literal_eval(str(row["bbox"])))
    return (x - 2, y - 2, x + w + 2, y + h + 2)


def _tokens(text):
    return [t for t in str(text).replace(",", " ").replace(".", " ").replace("/", " ").split() if len(t) >= 3]


def load_phase1(path) -> list[dict]:
    """Cases from a Phase 1 run folder or a single Phase 1 DICOM.

    Each case: {case_id, path, name, canaries (header truth), boxes [(x0,y0,x1,y1)], tokens [str], truth_source}
    """
    path = Path(path)
    dicoms = [path] if path.is_file() else sorted(p for p in path.rglob("*.dcm") if is_dicom(p))
    overlays = pd.DataFrame()
    if path.is_dir():
        csvs = list(path.rglob("phi_overlays.csv"))
        if csvs:
            overlays = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    cases = []
    for i, p in enumerate(dicoms):
        ds = dcmread(p, stop_before_pixels=True)
        sop = sstr(ds.SOPInstanceUID)
        ov = overlays[overlays.sop_instance_uid.astype(str) == sop] if len(overlays) else overlays
        if len(ov):
            if "is_phi" in ov:
                ov = ov[ov.is_phi.astype(str).str.lower() == "true"]
            boxes = [_bbox(r) for _, r in ov.iterrows()]
            tokens = sum((_tokens(t) for t in ov.rendered_text), [])
            src = "phi_overlays.csv"
        else:
            pn = sstr(getattr(ds, "PatientName", "")).split("^")
            boxes = [PHASE1_BANNER]
            tokens = [t for t in pn[:2] if len(t) >= 3]
            src = "header + Phase 1 banner geometry (no phi_overlays.csv)"
        cases.append({"case_id": f"P1_{i:02d}", "path": p, "name": p.stem[:40], "canaries": header_truth(ds),
                      "boxes": boxes, "tokens": tokens, "truth_source": src})
    return cases
