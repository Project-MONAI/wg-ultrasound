"""De-identification strategies and negative controls behind one batch interface.

Every adapter takes `(jobs, cfg)` where each job is {"case_id", "src", "dst"} and returns one record per job:
{"case_id", "status" (PASS | ERROR | TIMEOUT | NOT_EVALUABLE), "runtime_s", "error", "detail", "output"}.
A batch is one tool *session* (dicognito / RSNA keep pseudonym and UID mappings within a session),
which is how study linkage across files is tested.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pydicom.pixels as pxl
from PIL import Image
from pydicom import dcmread
from pydicom.encaps import encapsulate
from pydicom.uid import JPEGBaseline8Bit

from .dicom_utils import MPEG_TS, ffmpeg_decode, g, nframes, to_display, ts_of

TOOLS = {
    "kitware":   {"label": "Kitware dicom-anonymizer", "scope": "metadata"},
    "deid":      {"label": "pydicom-deid", "scope": "metadata"},
    "dicognito": {"label": "dicognito", "scope": "metadata"},
    "rsna":      {"label": "RSNA Anonymizer", "scope": "metadata"},
    "deid_us":   {"label": "pydicom-deid + US pixel recipe", "scope": "metadata+pixels"},
    "rsna_ocr":  {"label": "RSNA Anonymizer + OCR pixel PHI", "scope": "metadata+pixels"},
    "ctp":       {"label": "CTP DicomAnonymizerTool -dpa", "scope": "metadata+pixels"},
}
CONTROLS = {
    "ctrl_identity": {"label": "Control: identity copy", "scope": "control"},
    "ctrl_reencode": {"label": "Control: lossy re-encode", "scope": "control"},
    "ctrl_tamper":   {"label": "Control: timing/calibration tamper", "scope": "control"},
}
ALL = {**TOOLS, **CONTROLS}
LABEL = {k: v["label"] for k, v in ALL.items()}


NOISE = ("Using CPU", "UserWarning", "warn(", "quantize_per_tensor", "super().__init__", "FutureWarning", "PixelSpacing missing")


def stderr_tail(text, n=400):
    """Last meaningful stderr lines (library warnings removed) - so a crash is diagnosable from the results table."""
    lines = [l for l in (text or "").splitlines() if l.strip() and not any(k in l for k in NOISE)]
    return " | ".join(lines)[-n:] or "(no error text; check free disk space and memory)"


def _rec(job, status, t0, error="", detail=""):
    return {"case_id": job["case_id"], "status": status, "runtime_s": round(time.perf_counter() - t0, 3),
            "error": str(error)[:400], "detail": detail, "output": str(job["dst"]) if status == "PASS" else ""}


def _not_evaluable(jobs, why):
    return [{"case_id": j["case_id"], "status": "NOT_EVALUABLE", "runtime_s": 0.0, "error": why, "detail": "", "output": ""}
            for j in jobs]


def _guard(job):
    if Path(job["dst"]).resolve() == Path(job["src"]).resolve():
        raise RuntimeError("output path resolves to the input")


def _each(jobs, fn):
    res = []
    for j in jobs:
        t0 = time.perf_counter()
        try:
            _guard(j)
            detail = fn(j) or ""
            res.append(_rec(j, "PASS", t0, detail=detail))
        except Exception as e:
            res.append(_rec(j, "ERROR", t0, f"{type(e).__name__}: {e}"))
    return res


# ---------------------------------------------------------------- metadata tools
def run_kitware(jobs, cfg):
    from dicomanonymizer.simpledicomanonymizer import anonymize_dicom_file
    return _each(jobs, lambda j: anonymize_dicom_file(str(j["src"]), str(j["dst"]), delete_private_tags=True))


def deid_metadata(src, dst, workdir):
    """pydicom-deid default recipe. NB: replace_identifiers() defaults to save=False; then `.filename` is the SOURCE."""
    from deid.config import DeidRecipe
    from deid.dicom import get_identifiers, replace_identifiers
    tmp = Path(workdir)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    out = replace_identifiers(dicom_files=[str(src)], deid=DeidRecipe(), ids=get_identifiers([str(src)]), save=True,
                              output_folder=str(tmp), overwrite=True, remove_private=True)
    produced = Path(out[0] if isinstance(out[0], (str, os.PathLike)) else out[0].filename)
    if produced.resolve() == Path(src).resolve() or not produced.exists():
        raise RuntimeError("pydicom-deid did not write a new file")
    shutil.move(str(produced), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)


def run_deid(jobs, cfg):
    return _each(jobs, lambda j: deid_metadata(j["src"], j["dst"], Path(j["dst"]).parent / f"_tmp_{j['case_id']}"))


def run_dicognito(jobs, cfg):
    from dicognito.anonymizer import Anonymizer
    anon = Anonymizer(seed=str(cfg.seed))      # one session per batch -> consistent pseudonyms and date shift

    def one(j):
        ds = dcmread(j["src"])
        anon.anonymize(ds)
        ds.save_as(j["dst"], enforce_file_format=True)
    return _each(jobs, one)


# ---------------------------------------------------------------- RSNA (own interpreter)
RSNA_RUNNER = r'''
import argparse, json, os, shutil, time
from pathlib import Path
os.environ["ANONYMIZER_ASCII_STORAGE_DIRS"] = "1"
ap = argparse.ArgumentParser()
for k in ("--jobs", "--mode", "--workspace", "--results"):
    ap.add_argument(k, required=True)
a = ap.parse_args()
jobs = json.loads(Path(a.jobs).read_text(encoding="utf-8"))
results = []
def flush():
    Path(a.results).write_text(json.dumps(results), encoding="utf-8")
import anonymizer
from pydicom import dcmread
from anonymizer.model.project import ProjectModel
from anonymizer.controller.anonymizer import AnonymizerController
ws = Path(a.workspace)
shutil.rmtree(ws, ignore_errors=True)
ws.mkdir(parents=True)
script = Path(anonymizer.__file__).resolve().parent / "assets" / "scripts" / "default-anonymizer.script"
project = ProjectModel(project_name="DEIDBENCH", storage_dir=ws, modalities=["US"], anonymizer_script_path=script)
project.private_dir().mkdir(parents=True, exist_ok=True)
project.images_dir().mkdir(parents=True, exist_ok=True)
reader = None
if a.mode == "ocr":
    from anonymizer.controller.runner import RemovePixelPhiRunner
    from anonymizer.controller.ai.remove_pixel_phi import remove_pixel_phi, PixelPhiRemovalMode
    reader = RemovePixelPhiRunner().enter_models().reader
controller = AnonymizerController(project)
try:
    for j in jobs:
        t0 = time.perf_counter()
        rec = {"case_id": j["case_id"], "status": "ERROR", "error": "", "detail": ""}
        try:
            project.add_storage_class(str(dcmread(j["src"], stop_before_pixels=True).SOPClassUID))
            err, ds = controller.anonymize_file(Path(j["src"]))
            if err is not None or ds is None:
                rec["error"] = str(err)[:400]
            else:
                dst = Path(j["dst"])
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    ds.save_as(dst, enforce_file_format=True)      # pydicom 3
                except TypeError:
                    ds.save_as(dst, write_like_original=False)     # pydicom 2.4 (what rsna-anonymizer declares)
                if reader is not None:
                    modified, texts, changed = remove_pixel_phi(dst, reader, removal_mode=PixelPhiRemovalMode.BLACKOUT)
                    rec["detail"] = json.dumps({"pixels_modified": bool(modified), "ocr_texts": list(texts)[:40],
                                                "pixels_changed": int(changed)})
                rec["status"] = "PASS"
        except Exception as e:
            rec["error"] = (type(e).__name__ + ": " + str(e))[:400]
        rec["runtime_s"] = round(time.perf_counter() - t0, 3)
        results.append(rec)
        flush()
finally:
    controller.stop()          # RSNA worker threads are non-daemon: without stop() the process never exits
    flush()
'''


def run_rsna(jobs, cfg, mode="meta"):
    if cfg.rsna_python is None:
        return _not_evaluable(jobs, "RSNA interpreter not configured (set RSNA_PYTHON)")
    tag = f"{mode}_{abs(hash(tuple(j['case_id'] for j in jobs))) % 10**8:08d}"
    tmp = cfg.work / "_rsna" / tag
    tmp.mkdir(parents=True, exist_ok=True)
    ws = cfg.rsna_workspace / tag[-8:]            # short path: RSNA nests four UID-named levels
    runner, jf, rf = tmp / "runner.py", tmp / "jobs.json", tmp / "results.json"
    runner.write_text(RSNA_RUNNER, encoding="utf-8")
    jf.write_text(json.dumps([{k: str(v) for k, v in j.items()} for j in jobs]), encoding="utf-8")
    rf.unlink(missing_ok=True)
    cache = cfg.data / "_rsna_cache"              # RSNA resolves its OCR model folder relative to the working dir
    cache.mkdir(parents=True, exist_ok=True)
    timeout = cfg.rsna_timeout_per_file * len(jobs) + (600 if mode == "ocr" else 120)
    status_all, err_all = None, ""
    try:
        cp = subprocess.run([str(cfg.rsna_python), str(runner), "--jobs", str(jf), "--mode", mode, "--workspace", str(ws),
                             "--results", str(rf)], capture_output=True, text=True, timeout=timeout, cwd=str(cache))
        if cp.returncode != 0:
            err_all = f"exit code {cp.returncode}: " + stderr_tail(cp.stderr)
    except subprocess.TimeoutExpired:
        status_all, err_all = "TIMEOUT", f"batch exceeded {timeout}s"
    done = {r["case_id"]: r for r in (json.loads(rf.read_text(encoding="utf-8")) if rf.exists() else [])}
    res = []
    for j in jobs:
        r = done.get(j["case_id"])
        if r is None:
            res.append({"case_id": j["case_id"], "status": status_all or "ERROR", "runtime_s": np.nan,
                        "error": err_all or "no result", "detail": "", "output": ""})
        else:
            r["output"] = str(j["dst"]) if r["status"] == "PASS" else ""
            res.append(r)
    shutil.rmtree(ws, ignore_errors=True)
    return res


# ---------------------------------------------------------------- pixel strategies
def run_deid_us(jobs, cfg):
    from deid.data import data_base
    from deid.dicom import DicomCleaner
    us_recipe = os.path.join(data_base, "deid.dicom.ultrasound")

    def one(j):
        tmp = Path(j["dst"]).parent / f"_tmp_{j['case_id']}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            cleaner = DicomCleaner(output_folder=str(tmp), deid=us_recipe)
            det = cleaner.detect(str(j["src"]))
            src, action = j["src"], "not flagged -> pixels untouched"
            if det and det.get("flagged"):
                cleaner.clean()
                src, action = cleaner.save_dicom(output_folder=str(tmp)), "flagged -> outside-region pixels blanked"
            deid_metadata(src, j["dst"], tmp / "meta")
            return json.dumps({"pixel_action": action})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return _each(jobs, one)


def run_ctp(jobs, cfg):
    jar = cfg.ctp_dir / "DAT.jar" if cfg.ctp_dir else None
    if jar is None or not jar.exists():
        return _not_evaluable(jobs, "CTP DicomAnonymizerTool not installed (set CTP_DIR)")
    java = os.environ.get("CTP_JAVA") or shutil.which("java")      # current DAT.jar needs Java 25+
    if not java:
        return _not_evaluable(jobs, "java not found")

    def one(j):
        cmd = [java, "-jar", str(jar), "-in", str(j["src"]), "-out", str(j["dst"]),
               "-da", str(cfg.ctp_dir / "dicom-anonymizer.script"), "-dpa", str(cfg.ctp_dir / "dicom-pixel-anonymizer.script"),
               "-dec", "-rec"]
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.tool_timeout_per_file, cwd=str(cfg.ctp_dir))
        if not (Path(j["dst"]).exists() and Path(j["dst"]).stat().st_size > 0):
            raise RuntimeError((cp.stdout + cp.stderr)[-300:])
    return _each(jobs, one)


# ---------------------------------------------------------------- controls (must be caught by the validators)
def run_identity(jobs, cfg):
    return _each(jobs, lambda j: shutil.copy2(j["src"], j["dst"]) and None)


def jpeg_reencode(path):
    """Naive cleaner: decode, blank the top band, re-encode every frame as lossy JPEG (quality 75)."""
    ds = dcmread(path)
    if ts_of(ds) in MPEG_TS:
        frames = list(ffmpeg_decode(ds))
    elif nframes(ds) == 1:
        frames = [ds.pixel_array]
    else:
        frames = pxl.iter_pixels(path)
    enc, color = [], False
    for f in frames:
        f = to_display(f).copy()
        f[: max(4, f.shape[0] // 12)] = 0
        color = f.ndim == 3
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=75)
        enc.append(buf.getvalue())
    ds.PixelData = encapsulate(enc)
    ds["PixelData"].VR = "OB"
    ds.file_meta.TransferSyntaxUID = JPEGBaseline8Bit
    ds.SamplesPerPixel = 3 if color else 1
    ds.PhotometricInterpretation = "YBR_FULL_422" if color else "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    if color:
        ds.PlanarConfiguration = 0
    ds.LossyImageCompression = "01"
    ds.save_as(path, enforce_file_format=True)


def tamper(path):
    """Break timing (FrameTime x1.1, or reversed 3D+t temporal indices) and delete calibration."""
    ds = dcmread(path)
    if "FrameTime" in ds:
        ds.FrameTime = f"{float(ds.FrameTime) * 1.1:.4f}"
    if "FrameTimeVector" in ds:
        ds.FrameTimeVector = [float(x) * 1.1 for x in ds.FrameTimeVector]
    if "CineRate" in ds:
        ds.CineRate = int(round(int(ds.CineRate) / 1.1))
    if "SequenceOfUltrasoundRegions" in ds:
        del ds.SequenceOfUltrasoundRegions
    pf = g(ds, "PerFrameFunctionalGroupsSequence", None)
    if pf:
        tps = [int(fg.FrameContentSequence[0].TemporalPositionIndex) for fg in pf
               if "FrameContentSequence" in fg and "TemporalPositionIndex" in fg.FrameContentSequence[0]]
        if tps:
            hi = max(tps) + 1
            for fg in pf:
                fc = fg.FrameContentSequence[0]
                if "TemporalPositionIndex" in fc:
                    fc.TemporalPositionIndex = hi - int(fc.TemporalPositionIndex)
        # vendors that encode time as offsets / per-frame datetimes instead of indices: stretch those by 10 %
        from datetime import timedelta
        from .dicom_utils import parse_dt
        t0 = None
        for fg in pf:
            if "TemporalPositionSequence" in fg and "TemporalPositionTimeOffset" in fg.TemporalPositionSequence[0]:
                tp = fg.TemporalPositionSequence[0]
                tp.TemporalPositionTimeOffset = float(tp.TemporalPositionTimeOffset) * 1.1
            fc = fg.FrameContentSequence[0] if "FrameContentSequence" in fg else None
            for kw in ("FrameReferenceDateTime", "FrameAcquisitionDateTime"):
                d = parse_dt(fc.get(kw)) if fc is not None and kw in fc else None
                if d is not None:
                    t0 = t0 or d
                    setattr(fc, kw, (t0 + (d - t0) * 1.1 + timedelta(microseconds=1)).strftime("%Y%m%d%H%M%S.%f"))
    sh = g(ds, "SharedFunctionalGroupsSequence", None)
    if sh and "PixelMeasuresSequence" in sh[0]:
        del sh[0].PixelMeasuresSequence
    ds.save_as(path, enforce_file_format=True)


def _kitware_then(post):
    def run(jobs, cfg):
        res = run_kitware(jobs, cfg)
        for j, r in zip(jobs, res):
            if r["status"] == "PASS":
                t0 = time.perf_counter()
                try:
                    post(j["dst"])
                    r["runtime_s"] = round(r["runtime_s"] + time.perf_counter() - t0, 3)
                except Exception as e:
                    r.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:400], output="")
        return res
    return run


RUNNERS = {
    "kitware": run_kitware, "deid": run_deid, "dicognito": run_dicognito,
    "rsna": lambda jobs, cfg: run_rsna(jobs, cfg, "meta"), "deid_us": run_deid_us,
    "rsna_ocr": lambda jobs, cfg: run_rsna(jobs, cfg, "ocr"), "ctp": run_ctp,
    "ctrl_identity": run_identity, "ctrl_reencode": _kitware_then(jpeg_reencode), "ctrl_tamper": _kitware_then(tamper),
}


def run_batch(tool, cases, input_col, variant, cfg, verbose=True) -> list[dict]:
    """Run `tool` on every row of `cases` (DataFrame with case_id and `input_col`) as one session."""
    outdir = cfg.work / "outputs" / variant / tool
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [{"case_id": r.case_id, "src": Path(getattr(r, input_col)), "dst": outdir / f"{r.case_id}.dcm"}
            for r in cases.itertuples()]
    for j in jobs:
        Path(j["dst"]).unlink(missing_ok=True)
    t0 = time.perf_counter()
    res = RUNNERS[tool](jobs, cfg) if jobs else []
    for r in res:
        r.update(tool=tool, variant=variant)
    if verbose:
        n_ok = sum(r["status"] == "PASS" for r in res)
        print(f"  {variant:<5} | {LABEL[tool]:<36} {n_ok}/{len(res)} PASS  ({time.perf_counter() - t0:7.1f}s)")
        err = next((r for r in res if r["status"] != "PASS"), None)
        if err:
            print(f"          first problem: {err['status']} {str(err.get('error', ''))[:200]}")
    return res
