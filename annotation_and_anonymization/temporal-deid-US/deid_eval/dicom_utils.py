"""DICOM helpers shared by every module: frames, ultrasound regions, per-frame functional groups."""
from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pydicom.pixels as pxl
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.encaps import generate_fragments

MPEG_TS = {f"1.2.840.10008.1.2.4.{n}" for n in range(100, 109)}
TS_NAMES = {"1.2.840.10008.1.2": "Implicit LE", "1.2.840.10008.1.2.1": "Explicit LE",
            "1.2.840.10008.1.2.4.50": "JPEG Baseline", "1.2.840.10008.1.2.4.70": "JPEG Lossless",
            "1.2.840.10008.1.2.5": "RLE Lossless", "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
            "1.2.840.10008.1.2.4.91": "JPEG 2000", "1.2.840.10008.1.2.4.102": "MPEG-4 AVC/H.264 HP4.1",
            "1.2.840.10008.1.2.4.100": "MPEG-2"}
SOP_NAMES = {"1.2.840.10008.5.1.4.1.1.3.1": "US Multi-frame", "1.2.840.10008.5.1.4.1.1.6.1": "US Image",
             "1.2.840.10008.5.1.4.1.1.6.2": "Enhanced US Volume"}


def is_dicom(path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def g(ds, kw, default=None):
    try:
        return getattr(ds, kw, default)
    except Exception:
        return default


def sstr(v) -> str:
    if v is None:
        return ""
    try:
        return str(v).strip()
    except Exception:
        return repr(v)


def nframes(ds) -> int:
    try:
        return int(g(ds, "NumberOfFrames", 1) or 1)
    except Exception:
        return 1


def ts_of(ds) -> str:
    return sstr(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))


def object_kind(ds) -> str:
    if sstr(g(ds, "SOPClassUID")) == "1.2.840.10008.5.1.4.1.1.6.2" or sstr(g(ds, "DimensionOrganizationType")) == "3D_TEMPORAL":
        return "3D+t"
    return "2D+t" if nframes(ds) > 1 else "2D still"


def us_regions(ds) -> list[dict]:
    out = []
    for it in (g(ds, "SequenceOfUltrasoundRegions", None) or []):
        try:
            out.append(dict(
                x0=int(it.RegionLocationMinX0), y0=int(it.RegionLocationMinY0),
                x1=int(it.RegionLocationMaxX1), y1=int(it.RegionLocationMaxY1),
                fmt=int(g(it, "RegionSpatialFormat", 0) or 0), dtype=int(g(it, "RegionDataType", 0) or 0),
                ux=int(g(it, "PhysicalUnitsXDirection", 0) or 0), uy=int(g(it, "PhysicalUnitsYDirection", 0) or 0),
                dx=float(g(it, "PhysicalDeltaX", 0) or 0), dy=float(g(it, "PhysicalDeltaY", 0) or 0)))
        except Exception:
            pass
    return out


def main_region(ds, rows=None, cols=None) -> dict | None:
    """Largest 2D-tissue region (RegionSpatialFormat 1), else largest region; None without calibration."""
    regs = us_regions(ds)
    pref = [r for r in regs if r["fmt"] == 1] or regs
    if not pref:
        return None
    r = max(pref, key=lambda r: (r["x1"] - r["x0"]) * (r["y1"] - r["y0"]))
    if rows and cols:
        r = dict(r, x0=max(0, r["x0"]), y0=max(0, r["y0"]), x1=min(cols - 1, r["x1"]), y1=min(rows - 1, r["y1"]))
    return r


def region_mask(ds, rows, cols):
    r = main_region(ds, rows, cols)
    if r is None:
        return None
    m = np.zeros((rows, cols), bool)
    m[r["y0"]:r["y1"] + 1, r["x0"]:r["x1"] + 1] = True
    return m


def ffmpeg_decode(ds, max_frames=None) -> np.ndarray:
    """Decode an MPEG-family encapsulated stream with ffmpeg -> uint8 RGB frames (N, H, W, 3)."""
    payload = b"".join(generate_fragments(ds.PixelData))
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "stream.mp4"
        src.write_bytes(payload)
        cmd = ["ffmpeg", "-v", "error", "-i", str(src)]
        if max_frames:
            cmd += ["-frames:v", str(max_frames)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        cp = subprocess.run(cmd, capture_output=True, timeout=600)
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr.decode(errors="replace")[-300:])
        return np.frombuffer(cp.stdout, np.uint8).reshape(-1, int(ds.Rows), int(ds.Columns), 3)


def sample_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def read_frames(path, indices) -> list[np.ndarray]:
    """Decoded frames (RGB for colour) for the given indices; MPEG via ffmpeg, everything else via pydicom."""
    indices = list(indices)
    ds = dcmread(path)
    if ts_of(ds) in MPEG_TS:
        allf = ffmpeg_decode(ds, max_frames=max(indices) + 1)
        return [allf[i] for i in indices if i < len(allf)]
    if nframes(ds) == 1:
        return [np.asarray(ds.pixel_array)]
    return [np.asarray(f) for f in pxl.iter_pixels(path, indices=indices)]


def to_display(frame) -> np.ndarray:
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        a = a.astype(float)
        a = (255 * (a - a.min()) / max(1e-9, a.max() - a.min())).astype(np.uint8)
    return a


def luma(f) -> np.ndarray:
    f = np.asarray(f, np.float32)
    return f if f.ndim == 2 else (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])


def parse_dt(v):
    """DICOM DT -> naive datetime (offset-aware values are converted to UTC first, so they can be subtracted)."""
    v = sstr(v)
    if not v:
        return None
    try:
        d = pydicom.valuerep.DT(v)
    except Exception:
        try:
            d = datetime.strptime(v[:14], "%Y%m%d%H%M%S")
        except Exception:
            return None
    if d.tzinfo is not None:
        d = d.astimezone(timezone.utc).replace(tzinfo=None)
    return datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond)


PER_FRAME_COLUMNS = ["frame", "DimensionIndexValues", "TemporalPositionIndex", "FrameReferenceDateTime",
                     "FrameAcquisitionDateTime", "FrameAcquisitionDuration", "TemporalPositionTimeOffset", "ImagePositionVolume"]


def per_frame_table(ds) -> pd.DataFrame:
    """Per-frame functional-group content of Enhanced / 3D+t objects (empty for classic multi-frame)."""
    rows = []
    for i, fg in enumerate(g(ds, "PerFrameFunctionalGroupsSequence", None) or []):
        fc = fg.FrameContentSequence[0] if "FrameContentSequence" in fg else Dataset()
        div = g(fc, "DimensionIndexValues", None)
        if div is not None and not isinstance(div, (list, tuple, pydicom.multival.MultiValue)):
            div = [div]
        pos = None
        if "PlanePositionVolumeSequence" in fg:
            pos = tuple(float(x) for x in fg.PlanePositionVolumeSequence[0].get("ImagePositionVolume", []))
        rows.append({
            "frame": i,
            "DimensionIndexValues": tuple(int(x) for x in div) if div is not None else None,
            "TemporalPositionIndex": int(fc.TemporalPositionIndex) if "TemporalPositionIndex" in fc else None,
            "FrameReferenceDateTime": sstr(g(fc, "FrameReferenceDateTime")),
            "FrameAcquisitionDateTime": sstr(g(fc, "FrameAcquisitionDateTime")),
            "FrameAcquisitionDuration": g(fc, "FrameAcquisitionDuration", None),
            "TemporalPositionTimeOffset": (float(fg.TemporalPositionSequence[0].TemporalPositionTimeOffset)
                                           if "TemporalPositionSequence" in fg
                                           and "TemporalPositionTimeOffset" in fg.TemporalPositionSequence[0] else None),
            "ImagePositionVolume": pos,
        })
    return pd.DataFrame(rows, columns=PER_FRAME_COLUMNS)


def frame_time_axis(pf):
    """(values, label) describing when each frame of a 3D+t object was acquired, whichever encoding the vendor used."""
    if len(pf) and pf.TemporalPositionIndex.notna().all():
        return list(pf.TemporalPositionIndex), "TemporalPositionIndex"
    if len(pf) and pf.TemporalPositionTimeOffset.notna().all():
        return list(pf.TemporalPositionTimeOffset), "TemporalPositionTimeOffset (s)"
    if len(pf) and (pf.FrameReferenceDateTime != "").all():
        rel = relative_seconds(pf.FrameReferenceDateTime)
        if rel:
            return rel, "FrameReferenceDateTime (s from first frame)"
    return None, "frame time"


def relative_seconds(values):
    dts = [parse_dt(v) for v in values]
    if not dts or any(d is None for d in dts):
        return None
    return [round((d - dts[0]).total_seconds(), 6) for d in dts]
