"""Burned-in pixel PHI: challenge construction, OCR and objective text-residual metrics."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydicom import dcmread
from pydicom.uid import ExplicitVRLittleEndian

from .canaries import tok
from .dicom_utils import luma, main_region, nframes, read_frames, sstr, to_display

# matplotlib ships DejaVu fonts on every platform -> identical rendering everywhere.
FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"


def get_font(px: int):
    return ImageFont.truetype(str(FONT_PATH), px)


def place_header_box(rows, cols, reg, bw, bh):
    """Put header text outside the calibrated region when there is room for it."""
    if reg is None:
        return (8, 8)
    if reg["y0"] >= bh + 8:
        return (max(8, reg["x0"]), max(4, (reg["y0"] - bh) // 2))
    if reg["x0"] >= bw + 8:
        return (max(4, (reg["x0"] - bw) // 2), max(8, reg["y0"]))
    if cols - reg["x1"] >= bw + 8:
        return (reg["x1"] + 4, max(8, reg["y0"]))
    return (8, 8)


def make_burned_in(case_id, src, out_path, gt_path, seed, max_frames=16) -> dict:
    """Copy the first frames of `src`, draw header text H and in-region text R on every frame.

    Writes an uncompressed DICOM (same header, canaries, regions, timing) and an .npz ground truth with the
    text-free frames, the challenge frames, exact glyph masks and the text boxes.
    """
    ds = dcmread(src)
    n = min(nframes(ds), max_frames)
    base = np.stack([to_display(f) for f in read_frames(src, range(n))])
    rows, cols = base.shape[1], base.shape[2]
    color = base.ndim == 4
    reg = main_region(ds, rows, cols)
    fs = max(14, cols // 38)
    font_h, font_r = get_font(fs), get_font(max(13, int(fs * 0.9)))
    th, tr = tok(seed, case_id, "burnH", n=6), tok(seed, case_id, "burnR", n=6)
    lines = ["NAME DOE^JANE", f"ID {th}", "DOB 1971-02-03"]
    step = int(fs * 1.35)
    bw, bh = int(fs * 11), len(lines) * step + 10
    hx, hy = place_header_box(rows, cols, reg, bw, bh)
    box_h = (hx - 4, hy - 4, min(cols - 1, hx + bw), min(rows - 1, hy + bh))
    if reg is not None:
        rx = int(reg["x0"] + 0.30 * (reg["x1"] - reg["x0"]))
        ry = int(reg["y0"] + 0.62 * (reg["y1"] - reg["y0"]))
    else:
        rx, ry = int(cols * 0.35), int(rows * 0.6)
    text_r = f"MRN {tr}"
    box_r = (rx - 4, ry - 4, min(cols - 1, rx + int(fs * 6.5)), min(rows - 1, ry + int(fs * 1.4)))

    def glyphs(draw_fn):
        m = Image.new("L", (cols, rows), 0)
        draw_fn(ImageDraw.Draw(m), 255)
        return np.array(m) > 0

    draw_h = lambda d, fill: [d.text((hx, hy + k * step), line, fill=fill, font=font_h) for k, line in enumerate(lines)]
    draw_r = lambda d, fill: d.text((rx, ry), text_r, fill=fill, font=font_r)
    mask_h, mask_r = glyphs(draw_h), glyphs(draw_r)

    chal, clean = base.copy(), base.copy()     # clean = challenge without glyphs (keeps the black header box)
    white, black = ((255, 255, 255), (0, 0, 0)) if color else (255, 0)
    for f in range(n):
        im = Image.fromarray(chal[f])
        d = ImageDraw.Draw(im)
        d.rectangle(box_h, fill=black)
        clean[f] = np.array(im)
        draw_h(d, white)
        draw_r(d, white)
        chal[f] = np.array(im)

    if "FrameTimeVector" in ds:
        ds.FrameTimeVector = list(ds.FrameTimeVector)[:n]
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.NumberOfFrames, ds.Rows, ds.Columns = n, rows, cols
    ds.SamplesPerPixel = 3 if color else 1
    ds.PhotometricInterpretation = "RGB" if color else "MONOCHROME2"
    if color:
        ds.PlanarConfiguration = 0
    elif "PlanarConfiguration" in ds:
        del ds.PlanarConfiguration
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    for kw in ("LargestImagePixelValue", "SmallestImagePixelValue", "PixelPaddingValue"):
        if kw in ds:
            delattr(ds, kw)
    ds.SOPInstanceUID = ds.SOPInstanceUID + ".1"
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.BurnedInAnnotation = "YES"
    ds.PixelData = chal.tobytes()
    ds["PixelData"].VR = "OB"
    ds.save_as(out_path, enforce_file_format=True)

    region = np.array([reg["x0"], reg["y0"], reg["x1"], reg["y1"]]) if reg else np.array([-1] * 4)
    np.savez_compressed(gt_path, base=base, clean=clean, chal=chal, maskH=mask_h, maskR=mask_r,
                        boxH=np.array(box_h), boxR=np.array(box_r), region=region)
    overlap = bool(reg is not None and mask_h[reg["y0"]:reg["y1"] + 1, reg["x0"]:reg["x1"] + 1].any())
    return {"input_P": str(out_path), "gt_path": str(gt_path), "frames_P": n,
            "tokens_H": ["DOE", "JANE", th, "1971"], "tokens_R": ["MRN", tr],
            "header_overlaps_region": overlap, "has_region": reg is not None}


# ---------------------------------------------------------------- metrics
def text_residual(out, chal, box, base=None) -> float:
    """How much of the text pattern survives inside `box` (1 = intact, ~0 = removed).

    With a text-free reference `base` (the challenge without glyphs), the correlation is computed on the text signal
    (image - base), so blacking out or inpainting the text scores ~0; without one
    (e.g. Phase 1 outputs, where only the burned image exists) it is computed on the raw box content.
    """
    x0, y0, x1, y1 = [int(v) for v in box]
    sl = (slice(max(0, y0), y1), slice(max(0, x0), x1))
    o, c = luma(out)[sl], luma(chal)[sl]
    if base is not None:
        b = luma(base)[sl]
        o, c = o - b, c - b
    u, v = o.ravel(), c.ravel()
    if u.size == 0 or u.std() < 1e-6 or v.std() < 1e-6:
        return 0.0
    return float(max(0.0, np.corrcoef(u, v)[0, 1]))


def dilate(mask, r=6):
    m = mask.copy()
    for _ in range(r):
        m[1:] |= m[:-1]; m[:-1] |= m[1:]; m[:, 1:] |= m[:, :-1]; m[:, :-1] |= m[:, 1:]
    return m


def norm_text(s: str) -> str:
    """Upper-case alphanumerics; map OCR-ambiguous glyphs O->0, I/L->1."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper().replace("O", "0").replace("I", "1").replace("L", "1"))


def lev1(a: str, b: str) -> bool:
    """True if `b` contains a substring within edit distance 1 of `a`."""
    if not a:
        return False
    if a in b:
        return True
    n = len(a)
    for i in range(len(b)):
        for w in (n - 1, n, n + 1):
            s = b[i:i + w]
            if len(s) != w or w == 0:
                continue
            if w == n and sum(x != y for x, y in zip(s, a)) <= 1:
                return True
            if w == n + 1 and any(s[:k] + s[k + 1:] == a for k in range(w)):
                return True
            if w == n - 1 and any(a[:k] + a[k + 1:] == s for k in range(n)):
                return True
    return False


def ocr_recall(text, tokens) -> float:
    t = norm_text(text)
    return float(np.mean([lev1(norm_text(k), t) for k in tokens])) if tokens else float("nan")


OCR_SCRIPT = r'''
import json, sys
import numpy as np
import easyocr
from PIL import Image
import torch
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), model_storage_directory=sys.argv[2], verbose=False)
jobs = json.load(open(sys.argv[1]))
out = {k: " ".join(reader.readtext(np.array(Image.open(p).convert("RGB")), detail=0)) for k, p in jobs.items()}
json.dump(out, open(sys.argv[3], "w"))
'''


def run_easyocr(jobs: dict, python, model_dir, workdir, timeout=3600) -> dict:
    """OCR {key: png_path} with EasyOCR in `python` (the RSNA interpreter ships it). Returns {key: text}."""
    if not jobs or python is None:
        return {}
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    jf, of, sf = workdir / "ocr_jobs.json", workdir / "ocr_out.json", workdir / "ocr_runner.py"
    jf.write_text(json.dumps(jobs))
    sf.write_text(OCR_SCRIPT)
    of.unlink(missing_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    cp = subprocess.run([str(python), str(sf), str(jf), str(model_dir), str(of)], capture_output=True, text=True, timeout=timeout)
    if not of.exists():
        from .adapters import stderr_tail
        raise RuntimeError(f"OCR failed (exit code {cp.returncode}): " + stderr_tail(cp.stderr))
    return json.loads(of.read_text())


def save_png(frame, path) -> str:
    Image.fromarray(to_display(frame)).save(path)
    return str(path)


def _same_ndim(a, ref):
    if a.ndim == ref.ndim:
        return a
    return np.repeat(a[..., None], 3, -1) if a.ndim == 2 else a.mean(-1).astype(np.uint8)


def eval_challenge_output(out_path, gt_path, input_frame_time) -> dict:
    """Challenge outputs (with text-free ground truth): text residual H/R, collateral inside/outside region."""
    gt = np.load(gt_path)
    ds = dcmread(out_path)
    n = min(nframes(ds), len(gt["base"]))
    frs = [_same_ndim(to_display(f), gt["base"][0]) for f in read_frames(out_path, range(n))]
    rows, cols = gt["base"][0].shape[:2]
    if frs[0].shape[:2] != (rows, cols):
        raise ValueError(f"output geometry {frs[0].shape} differs from input {gt['base'][0].shape}")
    reg = gt["region"]
    inreg = np.zeros((rows, cols), bool)
    if reg[0] >= 0:
        inreg[reg[1]:reg[3] + 1, reg[0]:reg[2] + 1] = True
    textzone = dilate(gt["maskH"] | gt["maskR"], 8)
    for b in (gt["boxH"], gt["boxR"]):
        textzone[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1] = True
    d = np.mean([np.abs(luma(f) - luma(c)) for f, c in zip(frs, gt["chal"][:n])], axis=0)
    return {
        "frames_preserved": nframes(ds) == len(gt["base"]),
        "frame_time_preserved": sstr(getattr(ds, "FrameTime", "")) == sstr(input_frame_time),
        "residual_H": text_residual(frs[0], gt["chal"][0], gt["boxH"], gt["clean"][0]),
        "residual_R": text_residual(frs[0], gt["chal"][0], gt["boxR"], gt["clean"][0]),
        "collateral_mae_in_region": float(d[inreg & ~textzone].mean()) if (inreg & ~textzone).any() else float("nan"),
        "collateral_mae_outside_region": float(d[~inreg & ~textzone].mean()) if (~inreg & ~textzone).any() else float("nan"),
        "_frame0": frs[0],
    }


def eval_boxes_output(out_path, in_path, boxes) -> dict:
    """Phase 1 outputs (no text-free base): residual inside the ground-truth boxes, collateral elsewhere."""
    a = to_display(read_frames(in_path, [0])[0])
    b = _same_ndim(to_display(read_frames(out_path, [0])[0]), a)
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(f"output geometry {b.shape} differs from input {a.shape}")
    zone = np.zeros(a.shape[:2], bool)
    for x0, y0, x1, y1 in boxes:
        zone[max(0, y0):y1, max(0, x0):x1] = True
    zone = dilate(zone, 6)
    d = np.abs(luma(a) - luma(b))
    return {"residual_boxes": float(np.mean([text_residual(b, a, bx) for bx in boxes])) if boxes else float("nan"),
            "collateral_mae_outside_text": float(d[~zone].mean()), "_frame0": b}
