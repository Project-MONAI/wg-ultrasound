"""Public data acquisition with a pinned, checksummed manifest.

`data_manifest.json` lists every input. For each source either:
  * `files` is filled (pinned)  -> exactly those objects are downloaded and their SHA-256 verified
                                   (a mismatch stops the run), or
  * `files` is empty (unpinned) -> a deterministic selection rule is applied; what was used, with
                                   SHA-256, is written to results/resolved_manifest.json.
                                   it can be copied into data_manifest.json to pin the selection.
No data are redistributed with this code; everything is fetched from the original public host.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom.pixels as pxl
from pydicom import dcmread
from pydicom.encaps import encapsulate
from pydicom.uid import UID, generate_uid

from .dicom_utils import (SOP_NAMES, TS_NAMES, g, is_dicom, nframes, object_kind, sstr, to_display, ts_of, us_regions)

NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
USER_AGENT = "wg-ultrasound-deid-eval/0.1"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dst)
    return dst


def _verify(path, expected, label):
    got = sha256_file(path)
    if expected and got != expected:
        raise RuntimeError(f"SHA-256 mismatch for {label}: expected {expected[:12]}..., got {got[:12]}... "
                           "(the public object changed - re-freeze the manifest deliberately)")
    return got


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Tier A: Aliza (Google Drive)
def fetch_gdrive(file_id, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    try:
        import gdown
        gdown.download(id=file_id, output=str(dst), quiet=True)
    except Exception:
        pass
    if not dst.exists() or dst.stat().st_size == 0:
        _download(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t", dst)
    return dst


def fetch_aliza(cfg, spec):
    items, resolved = [], []
    root = cfg.data / "aliza"
    for f in spec["files"]:
        raw = fetch_gdrive(f["gdrive_id"], root / f"{f['name']}.bin")
        sha = _verify(raw, f.get("sha256"), f["name"])
        paths = [raw]
        if zipfile.is_zipfile(raw):
            ex = root / f["name"]
            if not ex.exists():
                with zipfile.ZipFile(raw) as z:
                    z.extractall(ex)
            paths = sorted(p for p in ex.rglob("*") if p.is_file() and is_dicom(p))
        for k, p in enumerate(p for p in paths if is_dicom(p)):
            items.append({"tier": "A", "source": spec["title"], "name": f["name"] if k == 0 else f"{f['name']}_{k}", "path": p})
        resolved.append({**f, "sha256": sha})
    return items, resolved


# ---------------------------------------------------------------- Tiers B / C: TCIA via the public NBIA REST API
def nbia_json(cfg, endpoint, **params):
    url = f"{NBIA}/{endpoint}?" + urllib.parse.urlencode({**params, "format": "json"})
    cache = cfg.data / "_nbia_cache" / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    cache.write_text(json.dumps(data))
    return data


def nbia_instance(series_uid, sop_uid, dst):
    if not (dst.exists() and is_dicom(dst)):
        _download(f"{NBIA}/getSingleImage?" + urllib.parse.urlencode({"SeriesInstanceUID": series_uid, "SOPInstanceUID": sop_uid}), dst)
    return dst


def _series_table(cfg, collection, modality="US"):
    sel = pd.DataFrame(nbia_json(cfg, "getSeries", Collection=collection, Modality=modality))
    sel["mean_instance_MB"] = sel.FileSize.astype(float) / 1e6 / sel.ImageCount.astype(int)
    return sel.sort_values(["mean_instance_MB", "SeriesInstanceUID"]).reset_index(drop=True)


def fetch_tcia(cfg, spec, tier, max_files, multiframe_only):
    root = cfg.data / spec["collection"]
    items, resolved = [], []
    if spec.get("files"):                                   # pinned
        for f in spec["files"][:max_files]:
            p = nbia_instance(f["series_uid"], f["sop_uid"], root / f["series_uid"][-12:] / f"{f['sop_uid'][-12:]}.dcm")
            resolved.append({**f, "sha256": _verify(p, f.get("sha256"), f["name"])})
            items.append({"tier": tier, "source": spec["title"], "name": f["name"], "path": p})
        return items, resolved
    rule = spec.get("selection", {})                        # unpinned: deterministic selection rule
    sel = _series_table(cfg, spec["collection"])
    if rule.get("max_mean_instance_MB"):
        sel = sel[sel.mean_instance_MB <= rule["max_mean_instance_MB"]]
    if rule.get("one_series_per_patient"):
        sel = sel.drop_duplicates("PatientID")
    for r in sel.itertuples():
        kept = 0
        sops = sorted(x["SOPInstanceUID"] for x in nbia_json(cfg, "getSOPInstanceUIDs", SeriesInstanceUID=r.SeriesInstanceUID))
        for sop in sops[: rule.get("max_checked_per_series", 3)]:
            if len(items) >= max_files:
                break
            p = nbia_instance(r.SeriesInstanceUID, sop, root / r.SeriesInstanceUID[-12:] / f"{sop[-12:]}.dcm")
            if multiframe_only and nframes(dcmread(p, stop_before_pixels=True)) <= 1:
                continue
            name = f"{r.PatientID}_{sop[-6:]}"
            items.append({"tier": tier, "source": spec["title"], "name": name, "path": p})
            resolved.append({"name": name, "series_uid": r.SeriesInstanceUID, "sop_uid": sop, "sha256": sha256_file(p)})
            kept += 1
            if kept >= rule.get("max_per_series", 2):
                break
        if len(items) >= max_files:
            break
    return items, resolved


# ---------------------------------------------------------------- Phase 1 sample (wg-ultrasound repository)
def fetch_phase1_sample(cfg, spec):
    root = cfg.data / "phase1_sample"
    zp = root / Path(urllib.parse.urlparse(spec["url"]).path).name
    if not zp.exists():
        _download(spec["url"], zp)
    sha = _verify(zp, spec.get("sha256"), zp.name)
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".dcm") and "__MACOSX" not in n]
        for n in names:
            if not (root / n).exists():
                z.extract(n, root)
    return [root / n for n in names], {**spec, "sha256": sha}


def fetch_all(cfg, manifest) -> tuple[list[dict], list[Path], dict]:
    """-> (temporal/public items, phase1 dicoms, resolved manifest). Failures are reported, not fatal."""
    src = manifest["sources"]
    items, p1, resolved = [], [], json.loads(json.dumps(manifest))
    steps = [("A", lambda: fetch_aliza(cfg, src["A"])),
             ("B", lambda: fetch_tcia(cfg, src["B"], "B", cfg.max_tcia_cines, True)),
             ("C", lambda: fetch_tcia(cfg, src["C"], "C", cfg.max_midi_files, False))]
    for tier, fn in steps:
        if tier in ("B", "C") and (cfg.max_tcia_cines if tier == "B" else cfg.max_midi_files) == 0:
            continue
        try:
            got, res = fn()
            items += got
            resolved["sources"][tier]["files"] = res
            print(f"Tier {tier}: {len(got)} files ({'pinned' if src[tier].get('files') else 'selection rule - unpinned'})")
        except Exception as e:
            print(f"Tier {tier} unavailable: {type(e).__name__}: {str(e)[:200]}")
    try:
        p1, res = fetch_phase1_sample(cfg, src["P1"])
        resolved["sources"]["P1"] = res
        print(f"Phase 1 sample: {len(p1)} DICOM")
    except Exception as e:
        print(f"Phase 1 sample unavailable: {type(e).__name__}: {str(e)[:200]}")
    (cfg.results / "resolved_manifest.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return items, p1, resolved


# ---------------------------------------------------------------- inventory and derived H.264 fixture
def inventory_row(item) -> dict:
    p = Path(item["path"])
    ds = dcmread(p, stop_before_pixels=True)
    try:
        ft = float(g(ds, "FrameTime", None))
    except Exception:
        ft = np.nan
    regs = us_regions(ds)
    return {**{k: item[k] for k in ("tier", "source", "name")}, "path": str(p), "size_MB": round(p.stat().st_size / 2**20, 2),
            "kind": object_kind(ds), "sop_class": SOP_NAMES.get(sstr(g(ds, "SOPClassUID")), sstr(g(ds, "SOPClassUID"))),
            "transfer_syntax": TS_NAMES.get(ts_of(ds), ts_of(ds)), "manufacturer": sstr(g(ds, "Manufacturer")),
            "frames": nframes(ds), "rows": int(g(ds, "Rows", 0) or 0), "cols": int(g(ds, "Columns", 0) or 0),
            "photometric": sstr(g(ds, "PhotometricInterpretation")), "frame_time_ms": ft,
            "has_us_regions": bool(regs), "calibrated_cm": any(r["ux"] == 3 and r["dx"] > 0 for r in regs),
            "orig_study_uid": sstr(g(ds, "StudyInstanceUID"))}


def build_inventory(items) -> pd.DataFrame:
    rows = []
    for it in items:
        try:
            rows.append(inventory_row(it))
        except Exception as e:
            print("  skipped", it.get("name"), type(e).__name__, str(e)[:120])
    return pd.DataFrame(rows)


def make_h264_fixture(src_path, dst, max_frames=150):
    """Re-encode a public cine as MPEG-4 AVC/H.264 (MP4 bitstream) DICOM keeping header, timing, calibration."""
    ds = dcmread(src_path)
    n = min(nframes(ds), max_frames)
    frames = [to_display(f) for f in pxl.iter_pixels(src_path, indices=range(n))]
    arr = np.stack([f if f.ndim == 3 else np.repeat(f[..., None], 3, -1) for f in frames])
    h, w = arr.shape[1] // 2 * 2, arr.shape[2] // 2 * 2
    arr = np.ascontiguousarray(arr[:, :h, :w, :3])
    ft = float(g(ds, "FrameTime", 0) or 0) or 40.0
    with tempfile.TemporaryDirectory() as td:
        mp4 = Path(td) / "x.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{1000 / ft:.6f}",
               "-i", "pipe:0", "-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p", "-crf", "18",
               "-movflags", "+faststart", str(mp4)]
        cp = subprocess.run(cmd, input=arr.tobytes(), capture_output=True, timeout=600)
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr.decode(errors="replace")[-300:])
        payload = mp4.read_bytes()
    for kw in ("FrameTimeVector", "LargestImagePixelValue", "SmallestImagePixelValue"):
        if kw in ds:
            delattr(ds, kw)
    ds.file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.4.102")
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.NumberOfFrames, ds.Rows, ds.Columns = len(arr), h, w
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "YBR_PARTIAL_420", 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.FrameTime, ds.LossyImageCompression = f"{ft:.4f}", "01"
    ds.DerivationDescription = "Public cine re-encoded to H.264 for a de-identification benchmark"
    ds.PixelData = encapsulate([payload], has_bot=False)
    ds["PixelData"].VR = "OB"
    ds.save_as(dst, enforce_file_format=True)
    return Path(dst)
