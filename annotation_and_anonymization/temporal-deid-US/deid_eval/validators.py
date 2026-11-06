"""Independent validators. They never trust a tool's success message: they read the output file and
compare it with the input.

E1 privacy      canary byte scan per PHI category; date policy (+ interval); UID policy
E2 preservation per-layer structural signatures: timing, 3D+t temporal/dimensions, calibration, bitstream, region pixels
E4 reuse        duration, calibrated width (mm), time-intensity curve - recomputed from each file's own metadata
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom import dcmread
from pydicom.encaps import generate_fragments
from pydicom.uid import UID

from .canaries import PRIVACY_CATS, SEEDED_DATE, SEEDED_INTERVAL_S, scan_needles
from .dicom_utils import (MPEG_TS, g, luma, main_region, nframes, parse_dt, per_frame_table, read_frames, region_mask,
                          relative_seconds, sample_indices, sstr, to_display, ts_of)

IMAGE_PIXEL = ["Rows", "Columns", "SamplesPerPixel", "PhotometricInterpretation", "BitsAllocated", "BitsStored",
               "HighBit", "PixelRepresentation", "PlanarConfiguration"]
TEMPORAL_2D = ["NumberOfFrames", "FrameTime", "FrameTimeVector", "FrameIncrementPointer", "CineRate",
               "RecommendedDisplayFrameRate", "ActualFrameDuration", "FrameDelay", "StartTrim", "StopTrim",
               "PreferredPlaybackSequencing", "HeartRate"]
REGION_KEYS = ["RegionLocationMinX0", "RegionLocationMinY0", "RegionLocationMaxX1", "RegionLocationMaxY1",
               "PhysicalUnitsXDirection", "PhysicalUnitsYDirection", "PhysicalDeltaX", "PhysicalDeltaY",
               "ReferencePixelX0", "ReferencePixelY0", "ReferencePixelPhysicalValueX", "ReferencePixelPhysicalValueY",
               "RegionSpatialFormat", "RegionDataType", "RegionFlags"]
TOL_TIME_S, TOL_INTENSITY, TOL_MM = 1e-3, 1.0, 1e-6
REGION_MAE_MAX, REGION_CHANGED_MAX = 0.5, 0.001


def wilson(k, n, z=1.96):
    """Wilson 95 % interval -> (p, lo, hi)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- E1 privacy
def privacy_eval(out_path, canaries, cats=PRIVACY_CATS) -> dict:
    hits = scan_needles(out_path, canaries)
    res = {"n_canaries": len(canaries), "n_leaked": int(sum(hits.values()))}
    vals = []
    for cat in cats:
        fields = [c["field"] for c in canaries if c["category"] == cat and c["field"] in hits]
        if not fields:
            res[f"priv_{cat}"], res[f"leaked_{cat}"] = np.nan, ""      # not applicable, never PASS
            continue
        leaked = [f for f in fields if hits[f]]
        res[f"priv_{cat}"], res[f"leaked_{cat}"] = float(not leaked), "; ".join(leaked)
        vals.append(res[f"priv_{cat}"])
    res["metadata_privacy_pass"] = bool(vals) and all(v == 1.0 for v in vals)
    return res


def date_policy(out_ds) -> dict:
    sd, st, adt = sstr(g(out_ds, "StudyDate")), sstr(g(out_ds, "StudyTime")), sstr(g(out_ds, "AcquisitionDateTime"))
    if sd == SEEDED_DATE:
        pol = "retained"
    elif not sd:
        pol = "removed / emptied"
    elif sd in ("00010101", "19000101", "18991230"):
        pol = "dummy value"
    else:
        pol = "shifted / replaced"
    interval = np.nan
    t1, t2 = (parse_dt(sd + (st or "000000")[:6]) if sd else None), parse_dt(adt)
    if t1 is not None and t2 is not None:
        interval = float(abs((t2 - t1).total_seconds() - SEEDED_INTERVAL_S) < 1.0)
    return {"date_policy": pol, "date_interval_preserved": interval}


def uid_policy(out_ds, canaries) -> dict:
    seeded = {c["field"]: c["value"] for c in canaries if c["category"] == "uids"}
    out = {k: sstr(g(out_ds, k)) for k in seeded}
    return {"uids_replaced": all(out[k] and out[k] != seeded[k] for k in seeded),
            "uids_valid": all(UID(v).is_valid for v in out.values() if v),
            "uid_meta_consistent": sstr(getattr(out_ds.file_meta, "MediaStorageSOPInstanceUID", "")) == out.get("SOPInstanceUID", ""),
            "out_study_uid": out.get("StudyInstanceUID", "")}


def study_linkage(results: pd.DataFrame) -> pd.DataFrame:
    """Files that shared a seeded StudyInstanceUID must still share one after de-identification."""
    rows = []
    ok = results[results.status == "PASS"]
    for (tool, sk), gp in ok.groupby(["tool", "study_key"]):
        if len(gp) >= 2:
            rows.append({"tool": tool, "study_key": sk, "files": len(gp), "linkage_preserved": gp.out_study_uid.nunique() == 1})
    return pd.DataFrame(rows, columns=["tool", "study_key", "files", "linkage_preserved"])


# ---------------------------------------------------------------- E2 structure signatures
def sig_fields(ds, fields):
    return {f: sstr(g(ds, f)) for f in fields if f in ds}


def spatial_sig(ds):
    s = {}
    if "SequenceOfUltrasoundRegions" in ds:
        s["regions"] = [tuple(sstr(g(it, k, "")) for k in REGION_KEYS) for it in ds.SequenceOfUltrasoundRegions]
    if "PixelSpacing" in ds:
        s["PixelSpacing"] = sstr(ds.PixelSpacing)
    sh = g(ds, "SharedFunctionalGroupsSequence", None)
    if sh:
        sh = sh[0]
        if "PixelMeasuresSequence" in sh:
            pm = sh.PixelMeasuresSequence[0]
            s["PixelMeasures"] = (sstr(g(pm, "PixelSpacing")), sstr(g(pm, "SliceThickness")), sstr(g(pm, "SpacingBetweenSlices")))
        if "PlaneOrientationVolumeSequence" in sh:
            s["PlaneOrientationVolume"] = sstr(g(sh.PlaneOrientationVolumeSequence[0], "ImageOrientationVolume"))
    pf = per_frame_table(ds)
    if len(pf) and pf.ImagePositionVolume.notna().any():
        s["ImagePositionVolume"] = tuple(pf.ImagePositionVolume)
    return s


def temporal3d_sig(ds):
    pf = per_frame_table(ds)
    s = {}
    if not len(pf):
        return s
    if pf.TemporalPositionIndex.notna().any():
        s["TemporalPositionIndex"] = tuple(pf.TemporalPositionIndex)
    for col in ("FrameReferenceDateTime", "FrameAcquisitionDateTime"):
        if (pf[col] != "").any():
            s[col + " (relative s)"] = tuple(relative_seconds(pf[col]) or [])
    if pf.TemporalPositionTimeOffset.notna().any():
        s["TemporalPositionTimeOffset"] = tuple(pf.TemporalPositionTimeOffset)
    if pf.FrameAcquisitionDuration.notna().any():
        s["FrameAcquisitionDuration"] = tuple(sstr(x) for x in pf.FrameAcquisitionDuration)
    return s


def dimension_sig(ds):
    s = {}
    if "DimensionOrganizationType" in ds:
        s["DimensionOrganizationType"] = sstr(ds.DimensionOrganizationType)
    if "DimensionIndexSequence" in ds:
        s["DimensionIndexSequence"] = tuple((sstr(g(it, "DimensionIndexPointer")), sstr(g(it, "FunctionalGroupPointer")))
                                            for it in ds.DimensionIndexSequence)
    pf = per_frame_table(ds)
    if len(pf) and pf.DimensionIndexValues.notna().any():
        s["DimensionIndexValues"] = tuple(pf.DimensionIndexValues)
    return s


def compare_sig(before, after):
    """(1.0 equal | 0.0 changed or missing | NaN not applicable, changed keys)."""
    if not before:
        return np.nan, ""
    changed = [k for k in before if after.get(k) != before[k]]
    return float(not changed), "; ".join(changed)


# ---------------------------------------------------------------- bitstream / PTS
def ffprobe_payload(ds) -> dict:
    if not shutil.which("ffprobe"):
        return {"status": "FFPROBE_NOT_FOUND"}
    payload = b"".join(generate_fragments(ds.PixelData))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "stream.mp4"
        p.write_bytes(payload)
        cp = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=codec_name,profile,width,height,avg_frame_rate,r_frame_rate:frame=pts_time", "-of", "json",
                             str(p)], capture_output=True, text=True, timeout=300)
    if cp.returncode != 0:
        return {"status": "ERROR"}
    d = json.loads(cp.stdout or "{}")
    st = (d.get("streams") or [{}])[0]
    return {"status": "PASS", "stream": {k: st.get(k) for k in ("codec_name", "profile", "width", "height", "avg_frame_rate", "r_frame_rate")},
            "pts": [float(f["pts_time"]) for f in d.get("frames", []) if "pts_time" in f]}


# ---------------------------------------------------------------- E4 reuse quantities
def time_axis(ds, idx):
    """Seconds for frame indices from the file's OWN metadata (None when it cannot be derived)."""
    if "FrameTimeVector" in ds:
        try:
            t = np.cumsum([float(x) for x in ds.FrameTimeVector]) / 1000.0
            return [float(t[i] - t[0]) for i in idx]
        except Exception:
            pass
    if "FrameTime" in ds:
        try:
            return [i * float(ds.FrameTime) / 1000.0 for i in idx]
        except Exception:
            pass
    pf = per_frame_table(ds)
    if len(pf):
        rel = relative_seconds(pf.FrameReferenceDateTime) if (pf.FrameReferenceDateTime != "").all() else None
        if rel:
            return [rel[i] for i in idx]
        if pf.TemporalPositionIndex.notna().all():
            return [float(pf.TemporalPositionIndex[i]) for i in idx]
    return None


def duration_s(ds):
    try:
        if "FrameTimeVector" in ds:
            return float(np.sum([float(x) for x in ds.FrameTimeVector])) / 1000.0
        if "FrameTime" in ds:
            return nframes(ds) * float(ds.FrameTime) / 1000.0
    except Exception:
        return None
    t = time_axis(ds, list(range(nframes(ds))))
    return (max(t) - min(t)) if t else None


def width_mm(ds):
    r = main_region(ds)
    if r is not None and r["ux"] == 3 and r["dx"] > 0:
        return (r["x1"] - r["x0"] + 1) * r["dx"] * 10.0
    try:
        ps = g(ds, "SharedFunctionalGroupsSequence")[0].PixelMeasuresSequence[0].PixelSpacing
        return float(ps[1]) * int(ds.Columns)
    except Exception:
        return None


def roi_box(ds, rows, cols):
    """Fixed analysis ROI: central box of the INPUT's calibrated region (or of the image)."""
    r = main_region(ds, rows, cols)
    x0, y0, x1, y1 = (r["x0"], r["y0"], r["x1"], r["y1"]) if r else (0, 0, cols - 1, rows - 1)
    cx, cy, hw, hh = (x0 + x1) // 2, (y0 + y1) // 2, max(2, (x1 - x0) // 7), max(2, (y1 - y0) // 7)
    return cx - hw, cy - hh, cx + hw, cy + hh


def pixel_and_reuse(src, out, ds_in, ds_out, max_frames=80) -> dict:
    res = {}
    b_in, b_out = bytes(g(ds_in, "PixelData", b"") or b""), bytes(g(ds_out, "PixelData", b"") or b"")
    res["pixeldata_bytes_same"] = bool(b_in) and b_in == b_out
    res["transfer_syntax_same"] = ts_of(ds_in) == ts_of(ds_out)
    res["frames_same"] = nframes(ds_in) == nframes(ds_out)
    if ts_of(ds_in) in MPEG_TS:
        if res["pixeldata_bytes_same"]:
            res["h264_pts_preserved"] = 1.0
        elif ts_of(ds_out) in MPEG_TS:
            p1, p2 = ffprobe_payload(ds_in), ffprobe_payload(ds_out)
            res["h264_pts_preserved"] = float(p1.get("status") == p2.get("status") == "PASS" and p1["stream"] == p2["stream"]
                                              and len(p1["pts"]) == len(p2["pts"]) and np.allclose(p1["pts"], p2["pts"], atol=1e-6))
        else:
            res["h264_pts_preserved"] = 0.0
    else:
        res["h264_pts_preserved"] = np.nan
    rows, cols = int(ds_in.Rows), int(ds_in.Columns)
    idx = sample_indices(min(nframes(ds_in), nframes(ds_out)), max_frames)
    mask = region_mask(ds_in, rows, cols)
    rx0, ry0, rx1, ry1 = roi_box(ds_in, rows, cols)
    try:
        A = read_frames(src, idx)
        B = A if res["pixeldata_bytes_same"] else read_frames(out, idx)
        if len(A) != len(B) or A[0].shape[:2] != B[0].shape[:2]:
            raise ValueError(f"decoded geometry differs: {len(A)}x{A[0].shape} vs {len(B)}x{B[0].shape}")
        m = mask if mask is not None else np.ones((rows, cols), bool)
        acc = {k: [] for k in ("mae_in", "mae_out", "ch_in", "ch_out", "mse", "tic_a", "tic_b")}
        for a, b in zip(A, B):
            la, lb = luma(to_display(a)), luma(to_display(b))
            d = np.abs(la - lb)
            acc["mae_in"].append(d[m].mean())
            acc["mae_out"].append(d[~m].mean() if (~m).any() else 0.0)
            acc["ch_in"].append((d[m] > 2).mean())
            acc["ch_out"].append((d[~m] > 2).mean() if (~m).any() else 0.0)
            acc["mse"].append((d ** 2).mean())
            acc["tic_a"].append(float(la[ry0:ry1, rx0:rx1].mean()))
            acc["tic_b"].append(float(lb[ry0:ry1, rx0:rx1].mean()))
        res.update(decode_ok=True, has_region=mask is not None,
                   mae_in_region=float(np.mean(acc["mae_in"])), mae_outside_region=float(np.mean(acc["mae_out"])),
                   changed_frac_in_region=float(np.mean(acc["ch_in"])), changed_frac_outside=float(np.mean(acc["ch_out"])),
                   psnr_db=float(10 * np.log10(255 ** 2 / max(np.mean(acc["mse"]), 1e-12))))
        ta, tb = time_axis(ds_in, idx), time_axis(ds_out, idx)
        res["tic"] = {"t_in": ta, "t_out": tb, "i_in": acc["tic_a"], "i_out": acc["tic_b"]}
        res["reuse_tic_equal"] = float(ta is not None and tb is not None and len(ta) == len(tb)
                                       and np.allclose(ta, tb, atol=TOL_TIME_S)
                                       and np.max(np.abs(np.array(acc["tic_a"]) - np.array(acc["tic_b"]))) <= TOL_INTENSITY)
    except Exception as e:
        res.update(decode_ok=False, decode_error=f"{type(e).__name__}: {e}"[:200], reuse_tic_equal=0.0)
    res["region_pixels_preserved"] = bool(res["pixeldata_bytes_same"] or (
        res.get("decode_ok") and res["frames_same"] and res["mae_in_region"] <= REGION_MAE_MAX
        and res["changed_frac_in_region"] <= REGION_CHANGED_MAX))
    d1, d2 = duration_s(ds_in), duration_s(ds_out)
    res.update(duration_in_s=d1, duration_out_s=d2,
               reuse_duration_equal=np.nan if d1 is None else float(d2 is not None and abs(d1 - d2) <= TOL_TIME_S))
    w1, w2 = width_mm(ds_in), width_mm(ds_out)
    res.update(width_in_mm=w1, width_out_mm=w2,
               reuse_mm_equal=np.nan if w1 is None else float(w2 is not None and abs(w1 - w2) <= TOL_MM))
    return res


LAYERS = ("sci_timing_2d", "sci_temporal_3d", "sci_dimensions_3d", "sci_spatial", "h264_pts_preserved",
          "reuse_duration_equal", "reuse_mm_equal", "reuse_tic_equal")


def scientific_eval(src, out, max_frames=80) -> dict:
    ds_in, ds_out = dcmread(src), dcmread(out)
    res = {"dicom_valid": bool(sstr(g(ds_out, "SOPClassUID")) == sstr(g(ds_in, "SOPClassUID")) and "PixelData" in ds_out)}
    for name, fn in [("image_pixel", lambda d: sig_fields(d, IMAGE_PIXEL)), ("timing_2d", lambda d: sig_fields(d, TEMPORAL_2D)),
                     ("temporal_3d", temporal3d_sig), ("dimensions_3d", dimension_sig), ("spatial", spatial_sig)]:
        res[f"sci_{name}"], res[f"changed_{name}"] = compare_sig(fn(ds_in), fn(ds_out))
    res.update(pixel_and_reuse(src, out, ds_in, ds_out, max_frames))
    res["reuse_pass"] = all(res[k] == 1.0 for k in ("reuse_duration_equal", "reuse_mm_equal", "reuse_tic_equal") if not pd.isna(res[k]))
    layers = [res[k] for k in LAYERS if not pd.isna(res[k])]
    res["scientific_usability_pass"] = bool(res["dicom_valid"] and res["region_pixels_preserved"] and all(v == 1.0 for v in layers))
    return res


def evaluate(rec, case, canaries, max_frames=80) -> dict:
    """Full E1 + E2 + E4 row for one tool output."""
    row = {**rec, **{k: case[k] for k in ("kind", "tier", "name", "study_key") if k in case}}
    if rec["status"] != "PASS" or not rec.get("output") or not Path(rec["output"]).exists():
        return row
    try:
        out_ds = dcmread(rec["output"], stop_before_pixels=True)
        row.update(privacy_eval(rec["output"], canaries))
        row.update(date_policy(out_ds))
        row.update(uid_policy(out_ds, canaries))
        row.update(scientific_eval(case["input"], rec["output"], max_frames))
        row["question_endpoint_pass"] = bool(row["metadata_privacy_pass"] and row["scientific_usability_pass"])
    except Exception as e:
        row.update(status="ERROR", error=f"validator: {type(e).__name__}: {e}"[:400])
    return row


def failure_stage(r) -> str:
    """First failing stage of the privacy -> usability chain."""
    get = lambda k: r.get(k, np.nan) if isinstance(r, dict) else getattr(r, k, np.nan)
    if get("status") != "PASS":
        return "0 did not complete"
    if not get("metadata_privacy_pass"):
        return "1 identifiers leaked"
    if not get("dicom_valid"):
        return "2 invalid DICOM"
    if any(get(c) == 0 for c in ("sci_timing_2d", "sci_temporal_3d", "sci_dimensions_3d")):
        return "3 timing / temporal organisation"
    if get("sci_spatial") == 0:
        return "4 calibration"
    if not get("region_pixels_preserved") or get("h264_pts_preserved") == 0:
        return "5 image content / bitstream"
    if not get("reuse_pass"):
        return "6 reuse quantities"
    return "7 privacy + usability PASS"


# ---------------------------------------------------------------- controls
CONTROL_EXPECTATIONS = {
    "ctrl_identity": {"metadata_privacy_pass": False},
    "ctrl_reencode": {"pixeldata_bytes_same": False, "region_pixels_preserved": False, "scientific_usability_pass": False},
    "ctrl_tamper": {"reuse_pass": False, "scientific_usability_pass": False},
}


def check_controls(results: pd.DataFrame) -> pd.DataFrame:
    """Share of files in which each deliberately broken output was caught (must be 100 %)."""
    rows = []
    ok = results[results.status == "PASS"]
    for tool, exp in CONTROL_EXPECTATIONS.items():
        gp = ok[ok.tool == tool]
        for col, want in exp.items():
            v = gp[col].dropna() if col in gp else pd.Series(dtype=float)
            rows.append({"control": tool, "check": f"{col} == {want}", "files": len(v),
                         "caught": (v.astype(float) == float(want)).mean() if len(v) else np.nan})
    tg = ok[ok.tool == "ctrl_tamper"]
    for kind, col in (("2D+t", "sci_timing_2d"), ("3D+t", "sci_temporal_3d"), ("calibrated", "sci_spatial")):
        sub = tg if kind == "calibrated" else tg[tg.kind == kind]
        v = sub[col].dropna() if col in sub else pd.Series(dtype=float)
        if len(v):
            rows.append({"control": "ctrl_tamper", "check": f"{col} == 0 ({kind})", "files": len(v), "caught": (v == 0).mean()})
    return pd.DataFrame(rows)
