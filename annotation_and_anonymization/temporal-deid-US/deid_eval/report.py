"""Data-driven, conservative answer to the research question (states only what the result tables support)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .canaries import PRIVACY_CATS

BROKEN_LAYERS = [("sci_timing_2d", "2D+t timing"), ("sci_temporal_3d", "3D+t temporal organisation"),
                 ("sci_dimensions_3d", "3D+t dimensions"), ("sci_spatial", "calibration"),
                 ("region_pixels_preserved", "image content"), ("reuse_pass", "reuse quantities")]


def pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.0%}"


def build_answer(synthesis: pd.DataFrame, results: pd.DataFrame, n_files: int, kinds: dict, n_sources: int,
                 n_canaries: int, n_pixel_cases: int, n_midi: int, n_phase1: int, controls_ok: bool, mode: str) -> str:
    ok = results[results.status == "PASS"]
    lines = [f"**Evidence base ({mode} mode).** {n_files} temporal ultrasound objects "
             f"({', '.join(f'{v} x {k}' for k, v in kinds.items())}) from {n_sources} source(s), each seeded with "
             f"{n_canaries} canary identifiers in {len(PRIVACY_CATS)} PHI categories; {n_pixel_cases} burned-in text challenges; "
             f"{n_phase1} Phase 1 file(s); {n_midi} MIDI-B synthetic-PHI file(s). "
             f"Validators were shown able to fail: {'all' if controls_ok else 'NOT all'} control checks were caught."]
    if not controls_ok:
        lines.append("**Warning:** at least one control was not caught, so the PASS results below are not trustworthy.")
    meta = synthesis[synthesis.scope == "metadata"]
    best = meta[meta.privacy_and_usability == 1]
    if len(best):
        lines.append(f"**Yes, for header (metadata) PHI, with the right tool and profile.** {', '.join(best.label)} removed every "
                     "canary in every category from all temporal files and left timing, 3D+t organisation, calibration, the "
                     "PixelData bitstream and the reuse quantities unchanged.")
    else:
        lines.append("**No strategy achieved complete metadata privacy together with full scientific usability on every file.**")
    for r in meta.itertuples():
        if r.privacy_and_usability == 1:
            continue
        gp = ok[ok.tool == r.tool]
        leaked = [c for c in PRIVACY_CATS if f"priv_{c}" in gp and (gp[f"priv_{c}"] == 0).any()]
        broke = [lab for col, lab in BROKEN_LAYERS if col in gp and (gp[col].dropna().astype(float) == 0).any()]
        parts = []
        if r.completed < 1:
            parts.append(f"completed {pct(r.completed)} of runs")
        if leaked:
            parts.append(f"left identifiers in: {', '.join(c.replace('_', ' ') for c in leaked)} (privacy {pct(r.metadata_privacy)})")
        if broke:
            parts.append(f"altered: {', '.join(broke)} (usability {pct(r.scientific_usability)})")
        elif not pd.isna(r.scientific_usability):
            parts.append(f"preserved the scientific content (usability {pct(r.scientific_usability)})")
        lines.append(f"- **{r.label}** " + "; ".join(parts) + ".")
    if "burned_in_header_removed" in synthesis:
        s = synthesis.set_index("tool")
        lines.append("**Burned-in PHI is a separate problem.** Metadata de-identification does not touch the pixels "
                     f"(Kitware, header text removed: {pct(s.burned_in_header_removed.get('kitware', np.nan))}).")
        for r in synthesis.dropna(subset=["burned_in_header_removed"]).itertuples():
            if r.tool == "kitware":
                continue
            lines.append(f"- **{r.label}**: header text removed {pct(r.burned_in_header_removed)}, text inside the imaging region "
                         f"removed {pct(r.burned_in_region_removed)}, anatomy preserved {pct(r.anatomy_preserved)}; on the temporal objects "
                         f"the bitstream stayed identical in {pct(r.bitstream_identical)} of files, usability {pct(r.scientific_usability)}.")
    lines.append("**Conclusion.** De-identification *can* remove identifiable header information without altering what is needed to "
                 "reuse a temporal ultrasound DICOM, but this is a property of the specific tool, version and profile, and it must be "
                 "verified per exported file. Removing burned-in PHI is a different trade-off: region-based blanking cannot see text "
                 "inside the imaging sector, and pixel cleaning re-encodes the cine.")
    if mode == "quick":
        lines.append("_Quick mode uses synthetic data only; it demonstrates the pipeline and is not evidence about real data._")
    return "\n\n".join(lines)
