"""Small, deterministic synthetic temporal ultrasound DICOMs.

Used by quick mode (offline, minutes) and by the unit tests. They are *structurally* realistic for what
the validators look at - multi-frame timing, SequenceOfUltrasoundRegions calibration, Enhanced US Volume
per-frame functional groups - and carry a contrast-like wash-in so a time-intensity curve is meaningful.
They are not claimed to be fully IOD-conformant clinical objects.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

T0 = datetime(2000, 1, 1, 12, 0, 0)
US_MULTIFRAME = "1.2.840.10008.5.1.4.1.1.3.1"
ENHANCED_US_VOLUME = "1.2.840.10008.5.1.4.1.1.6.2"


def _base(path, sop_class, rng):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = generate_uid(), generate_uid(), generate_uid()
    ds.Modality, ds.Manufacturer = "US", "SYNTHETIC"
    ds.PatientName, ds.PatientID, ds.PatientSex = "SYNTHETIC^BASE", "SYN-0000", "O"
    ds.StudyDate, ds.StudyTime, ds.StudyID, ds.AccessionNumber = "20000101", "120000", "1", ""
    ds.ReferringPhysicianName, ds.SeriesNumber, ds.InstanceNumber = "", 1, 1
    ds.BurnedInAnnotation = "NO"
    return ds


def _region(x0, y0, x1, y1, delta_cm=0.02):
    r = Dataset()
    r.RegionSpatialFormat, r.RegionDataType, r.RegionFlags = 1, 1, 0
    r.RegionLocationMinX0, r.RegionLocationMinY0, r.RegionLocationMaxX1, r.RegionLocationMaxY1 = x0, y0, x1, y1
    r.PhysicalUnitsXDirection = r.PhysicalUnitsYDirection = 3     # cm
    r.PhysicalDeltaX = r.PhysicalDeltaY = delta_cm
    r.ReferencePixelX0, r.ReferencePixelY0 = (x0 + x1) // 2, y0
    return r


def _speckle_frames(n, rows, cols, rng, region):
    """Rayleigh speckle + an ellipse whose brightness washes in over time (CEUS-like)."""
    x0, y0, x1, y1 = region
    yy, xx = np.mgrid[0:rows, 0:cols]
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 5, (y1 - y0) / 5
    lesion = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    inside = np.zeros((rows, cols), bool)
    inside[y0:y1 + 1, x0:x1 + 1] = True
    t = np.arange(n)
    wash = 90 * (1 - np.exp(-t / max(1, n / 4)))
    out = np.zeros((n, rows, cols), np.uint8)
    for k in range(n):
        f = rng.rayleigh(28, (rows, cols))
        f[lesion] += wash[k]
        f[~inside] = 0
        out[k] = np.clip(f, 0, 255)
    return out


def make_cine(path, n_frames=12, rows=240, cols=320, color=False, frame_time_ms=33.3, with_vector=False, seed=0):
    """Classic US Multi-frame (2D+t) cine, uncompressed, with one calibrated region."""
    rng = np.random.default_rng(seed)
    ds = _base(path, US_MULTIFRAME, rng)
    region = (int(cols * 0.12), int(rows * 0.15), int(cols * 0.94), int(rows * 0.95))
    frames = _speckle_frames(n_frames, rows, cols, rng, region)
    if color:
        frames = np.repeat(frames[..., None], 3, axis=-1)
        frames[:, :12, :, 1] = 200                       # coloured vendor band (outside region)
    ds.Rows, ds.Columns, ds.NumberOfFrames = rows, cols, n_frames
    ds.FrameTime = f"{frame_time_ms}"
    ds.FrameIncrementPointer = (0x0018, 0x1063)
    if with_vector:
        ds.FrameTimeVector = [0.0] + [frame_time_ms] * (n_frames - 1)
        ds.FrameIncrementPointer = (0x0018, 0x1065)
    ds.CineRate = int(round(1000 / frame_time_ms))
    ds.SamplesPerPixel = 3 if color else 1
    ds.PhotometricInterpretation = "RGB" if color else "MONOCHROME2"
    if color:
        ds.PlanarConfiguration = 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.SequenceOfUltrasoundRegions = Sequence([_region(*region)])
    ds.PixelData = frames.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return Path(path)


def make_enhanced_volume(path, n_times=4, n_slices=6, rows=96, cols=128, seed=0, dt_s=0.5):
    """Enhanced US Volume (3D+t): per-frame TemporalPositionIndex, FrameReferenceDateTime, DimensionIndexValues."""
    rng = np.random.default_rng(seed)
    ds = _base(path, ENHANCED_US_VOLUME, rng)
    region = (0, 0, cols - 1, rows - 1)
    vol = []
    for t in range(n_times):
        f = _speckle_frames(n_slices, rows, cols, rng, region)
        vol.append(np.clip(f.astype(int) + 15 * t, 0, 255).astype(np.uint8))
    frames = np.concatenate(vol)
    n = len(frames)
    ds.Rows, ds.Columns, ds.NumberOfFrames = rows, cols, n
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.ImageType = ["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]
    ds.DimensionOrganizationType = "3D_TEMPORAL"
    ds.DimensionOrganizationSequence = Sequence([Dataset()])
    ds.DimensionOrganizationSequence[0].DimensionOrganizationUID = generate_uid()
    d1, d2 = Dataset(), Dataset()
    d1.DimensionIndexPointer, d1.FunctionalGroupPointer = (0x0020, 0x9128), (0x0020, 0x9111)   # TemporalPositionIndex
    d2.DimensionIndexPointer, d2.FunctionalGroupPointer = (0x0020, 0x9301), (0x0020, 0x9310)   # ImagePositionVolume
    for d in (d1, d2):
        d.DimensionOrganizationUID = ds.DimensionOrganizationSequence[0].DimensionOrganizationUID
    ds.DimensionIndexSequence = Sequence([d1, d2])
    sh = Dataset()
    pm = Dataset(); pm.PixelSpacing = [0.3, 0.3]; pm.SliceThickness = 0.5
    po = Dataset(); po.ImageOrientationVolume = [1, 0, 0, 0, 1, 0]
    sh.PixelMeasuresSequence, sh.PlaneOrientationVolumeSequence = Sequence([pm]), Sequence([po])
    ds.SharedFunctionalGroupsSequence = Sequence([sh])
    pf = []
    for t in range(n_times):
        for z in range(n_slices):
            fg = Dataset()
            fc = Dataset()
            fc.TemporalPositionIndex = t + 1
            fc.DimensionIndexValues = [t + 1, z + 1]
            fc.FrameReferenceDateTime = (T0 + timedelta(seconds=t * dt_s)).strftime("%Y%m%d%H%M%S.%f")
            fc.FrameAcquisitionDuration = dt_s * 1000 / n_slices
            pp = Dataset(); pp.ImagePositionVolume = [0.0, 0.0, z * 0.5]
            fg.FrameContentSequence, fg.PlanePositionVolumeSequence = Sequence([fc]), Sequence([pp])
            pf.append(fg)
    ds.PerFrameFunctionalGroupsSequence = Sequence(pf)
    ds.PixelData = frames.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return Path(path)


def make_quick_dataset(root) -> list[dict]:
    """The quick-mode cohort: grey and colour 2D+t cines, one with FrameTimeVector, and one 3D+t volume."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    items = [
        ("SYN_2Dt_grey", lambda p: make_cine(p, seed=1)),
        ("SYN_2Dt_rgb", lambda p: make_cine(p, color=True, rows=300, cols=400, seed=2)),
        ("SYN_2Dt_vector", lambda p: make_cine(p, with_vector=True, n_frames=20, frame_time_ms=25.0, seed=3)),
        ("SYN_3Dt_volume", lambda p: make_enhanced_volume(p, seed=4)),
    ]
    out = []
    for name, fn in items:
        p = root / f"{name}.dcm"
        if not p.exists():
            fn(p)
        out.append({"tier": "S", "source": "Synthetic (quick mode)", "name": name, "path": p})
    return out
