"""The eight figures shown in the notebook (saved to results/figures as PNG)."""
from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from pydicom import dcmread

from . import adapters as A, canaries as CN, pipeline as PL
from .dicom_utils import frame_time_axis, luma, nframes, per_frame_table, read_frames, to_display
from .plots import BASE, CAT, INK2, STATUS, SURFACE, rate_heatmap, rate_matrix, savefig, tool_colors
from .report import pct

LABEL = A.LABEL
COLOR = tool_colors(list(A.TOOLS), list(A.CONTROLS))
PCT = mpl.ticker.PercentFormatter(1.0)


def controls(cfg, R):
    ctl = R["controls"]
    mat = ctl.assign(row=ctl.control + " | " + ctl.check).set_index("row")[["caught"]]
    mat.columns = ["files caught"]
    rate_heatmap(mat, "Fig 1. Controls: each broken output must be caught (100 %)", cfg.figures, "F1_controls",
                 figsize=(7, 0.42 * len(mat) + 1.4))


def privacy(cfg, R):
    labels = {f"priv_{c}": c.replace("_", " ") for c in CN.PRIVACY_CATS} | {"metadata_privacy_pass": "ALL categories"}
    m, n = rate_matrix(R["ok"], list(labels), R["order"], LABEL, labels)
    rate_heatmap(m, "Fig 2. E1 metadata privacy: files with no leaked identifier, by category", cfg.figures, "F2_privacy", counts=n)


def preservation(cfg, R):
    labels = {"dicom_valid": "DICOM valid", "sci_timing_2d": "2D+t timing", "sci_temporal_3d": "3D+t temporal",
              "sci_dimensions_3d": "3D+t dimensions", "sci_spatial": "calibration", "h264_pts_preserved": "H.264 PTS",
              "pixeldata_bytes_same": "pixels identical", "reuse_pass": "reuse quantities",
              "scientific_usability_pass": "USABLE"}
    m, n = rate_matrix(R["ok"], list(labels), R["order"], LABEL, labels)
    rate_heatmap(m, "Fig 3. E2 + E4 scientific preservation: files whose layer is unchanged", cfg.figures, "F3_preservation",
                 counts=n, note="n/a = the layer does not exist in those inputs.")


def temporal_3d(cfg, D, R):
    cases, ok = D["cases"], R["ok"]
    c3 = cases[cases.kind == "3D+t"]
    if not len(c3):
        print("No 3D+t object in this run.")
        return
    c3 = c3.iloc[0]
    y_in, ylab = frame_time_axis(per_frame_table(dcmread(c3.input, stop_before_pixels=True)))
    if y_in is None:
        print("This 3D+t object has no per-frame timing to plot.")
        return
    tools = [t for t in R["order"] if len(ok[(ok.case_id == c3.case_id) & (ok.tool == t)])]
    fig, axes = plt.subplots(1, len(tools), figsize=(2.3 * len(tools), 2.8), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], tools):
        pf = per_frame_table(dcmread(ok[(ok.case_id == c3.case_id) & (ok.tool == t)].output.iloc[0], stop_before_pixels=True))
        y_out, _ = frame_time_axis(pf)
        ax.plot(range(len(y_in)), y_in, color=BASE, lw=4, label="input")
        if y_out is not None:
            ax.plot(range(len(y_out)), y_out, color=COLOR[t], lw=1.6, label="output")
        else:
            ax.text(0.5, 0.5, "removed", transform=ax.transAxes, ha="center", color=STATUS["critical"], fontweight="bold")
        ax.set_title(LABEL[t], fontsize=8)
        ax.set_xlabel("frame")
    axes[0, 0].set_ylabel(ylab)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Fig 4. 3D+t: acquisition time of each frame (input grey, output colour) - {c3['name']}",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    savefig(fig, cfg.figures, "F4_3Dt_temporal")


def reuse_tic(cfg, D, R):
    cases, ok = D["cases"], R["ok"]
    c = cases[cases.tier == "B"]
    c = (c if len(c) else cases[cases.kind == "2D+t"]).iloc[0]
    tools = [t for t in R["order"] if len(ok[(ok.case_id == c.case_id) & (ok.tool == t)])]
    ncol = 5
    nrow = math.ceil(len(tools) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.4 * nrow), sharey=True, squeeze=False)
    for k, t in enumerate(tools):
        ax = axes[k // ncol, k % ncol]
        r = ok[(ok.case_id == c.case_id) & (ok.tool == t)].iloc[0]
        tic = r.get("tic")
        if isinstance(tic, dict) and tic.get("t_in"):
            ax.plot(tic["t_in"], tic["i_in"], color=BASE, lw=4)
            if tic.get("t_out"):
                ax.plot(tic["t_out"], tic["i_out"], color=COLOR[t], lw=1.5)
                same = r.reuse_tic_equal == 1
                ax.text(0.97, 0.05, "same curve" if same else "DIFFERENT", transform=ax.transAxes, fontsize=8, ha="right",
                        color=STATUS["good"] if same else STATUS["critical"], fontweight="bold")
            else:
                ax.text(0.5, 0.5, "time axis lost", transform=ax.transAxes, ha="center", color=STATUS["critical"])
        ax.set_title(LABEL[t], fontsize=8)
        ax.set_xlabel("time (s)")
    for k in range(len(tools), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    axes[0, 0].set_ylabel("mean ROI intensity")
    fig.suptitle(f"Fig 5. E4 reuse: time-intensity curve recomputed from each output ({c['name']}; input in grey)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    savefig(fig, cfg.figures, "F5_reuse_tic")


def real_images(cfg, D, R, n_cases=3):
    """Middle frame of real cines: input next to each tool's output, labelled identical / changed."""
    ok = R["ok"]
    tools = [t for t in ["kitware", "dicognito", "rsna", "deid_us", "rsna_ocr", "ctp"] if t in set(ok.tool)]
    cand = D["pix_subset"]
    cand = cand[(cand.kind == "2D+t") & ~cand.transfer_syntax.str.contains("H.264")]
    pick = pd.concat([cand.drop_duplicates("manufacturer"), cand]).drop_duplicates("case_id").head(n_cases)
    if not len(pick) or not tools:
        print("No real cine to show.")
        return
    fig, axes = plt.subplots(len(pick), len(tools) + 1, figsize=(2.5 * (len(tools) + 1), 2.3 * len(pick) + 0.8), squeeze=False)
    for i, c in enumerate(pick.itertuples()):
        mid = nframes(dcmread(c.input, stop_before_pixels=True)) // 2
        f_in = read_frames(c.input, [mid])[0]
        axes[i, 0].imshow(to_display(f_in), cmap="gray" if f_in.ndim == 2 else None)
        axes[i, 0].set_title(f"{c.name}\noriginal, frame {mid}", fontsize=7.5)
        for j, t in enumerate(tools, 1):
            ax, r = axes[i, j], ok[(ok.case_id == c.case_id) & (ok.tool == t)]
            if i == 0:
                ax.text(0.5, 1.32, LABEL[t], transform=ax.transAxes, ha="center", fontsize=8, fontweight="bold")
            if not len(r):
                ax.text(0.5, 0.5, "no output", transform=ax.transAxes, ha="center", va="center", color=STATUS["critical"])
                continue
            try:
                f = read_frames(r.output.iloc[0], [mid])[0]
            except Exception:
                ax.text(0.5, 0.5, "cannot decode", transform=ax.transAxes, ha="center", va="center", color=STATUS["critical"])
                continue
            ax.imshow(to_display(f), cmap="gray" if f.ndim == 2 else None)
            if f.shape[:2] == f_in.shape[:2]:
                d = float(np.abs(luma(f) - luma(f_in)).mean())
                same = d == 0
                ax.set_title("identical" if same else f"changed (mean diff {d:.1f})", fontsize=7.5,
                             color=STATUS["good"] if same else STATUS["critical"])
            else:
                ax.set_title("size changed", fontsize=7.5, color=STATUS["critical"])
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Fig 6. {'Synthetic' if cfg.quick else 'Real'} cines after each tool (middle frame)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    savefig(fig, cfg.figures, "F6_real_images")


def burned_in(cfg, D, R):
    pix, ex_imgs = R["pixel"], R["pixel_examples"]
    if not len(pix):
        print("No burned-in challenge in this run.")
        return
    ex = D["pixel_cases"].iloc[0]
    gt = np.load(ex.gt_path)
    tools = [t for t in PL.P_TOOLS if t in ex_imgs]
    imgs = [("input (synthetic text burned in)", gt["chal"][0], None)] + [(LABEL[t], ex_imgs[t], t) for t in tools]
    hb, rb = gt["boxH"].astype(int), gt["boxR"].astype(int)
    fig, axes = plt.subplots(2, len(imgs), figsize=(3.0 * len(imgs), 4.3), gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    for j, (title, im, t) in enumerate(imgs):
        cm = "gray" if im.ndim == 2 else None
        axes[0, j].imshow(im, cmap=cm)
        axes[0, j].set_title(title, fontsize=8.5)
        for b, col in ((hb, CAT[1]), (rb, CAT[4])):
            axes[0, j].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=col, lw=1.5))
        axes[1, j].imshow(im[max(0, rb[1] - 10):rb[3] + 10, max(0, rb[0] - 10):rb[2] + 10], cmap=cm)
        if t is None:
            axes[1, j].set_title("zoom: text inside the image", fontsize=7.5)
        else:
            r = pix[(pix.case_id == ex.case_id) & (pix.tool == t)].iloc[0]
            axes[1, j].set_title(f"outside {'removed' if r.text_removed_H else 'kept'} | inside {'removed' if r.text_removed_R else 'kept'}",
                                 fontsize=7.5)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.legend(handles=[Patch(edgecolor=CAT[1], facecolor="none", label="text outside the image area"),
                        Patch(edgecolor=CAT[4], facecolor="none", label="text inside the image area")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Fig 7. E3 burned-in identifiers after each pixel strategy", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    savefig(fig, cfg.figures, "F7_burned_in")


def _label_groups(ax, pts):
    """Tools at the same position share one label instead of overlapping."""
    groups = {}
    for x, y, t, extra in pts:
        groups.setdefault((round(x, 3), round(y, 3)), []).append((t, extra))
    for (x, y), items in groups.items():
        for t, _ in items:
            ax.scatter(x, y, s=150, color=COLOR[t], edgecolor=SURFACE, linewidth=2, zorder=3)
        text = "\n".join(LABEL[t] + (f" ({e})" if e else "") for t, e in items)
        right, low = x > 0.6, y < 0.3
        ax.annotate(text, (x, y), xytext=(-10 if right else 10, 8 if low else -6), textcoords="offset points", fontsize=8,
                    ha="right" if right else "left", va="bottom" if low else "top")


def tradeoffs(cfg, R):
    syn = R["synthesis"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    _label_groups(ax, [(r.metadata_privacy, r.scientific_usability, r.tool, "") for r in syn.dropna(subset=["metadata_privacy"]).itertuples()])
    ax.add_patch(Rectangle((0.95, 0.95), 0.1, 0.1, facecolor=STATUS["good"], alpha=0.12, zorder=0))
    ax.set_xlabel("E1 metadata privacy (all categories clean)")
    ax.set_ylabel("E2 + E4 scientific usability")
    ax.set_title("Privacy vs usability (goal: top right)")
    ax = axes[1]
    if "burned_in_header_removed" in syn:
        s = syn.dropna(subset=["burned_in_header_removed"])
        _label_groups(ax, [(np.nanmean([r.burned_in_header_removed, r.burned_in_region_removed]), r.anatomy_preserved, r.tool,
                            f"out {pct(r.burned_in_header_removed)}, in {pct(r.burned_in_region_removed)}") for r in s.itertuples()])
    ax.set_xlabel("E3 burned-in text removed (mean of outside / inside)")
    ax.set_ylabel("image inside the ultrasound region unchanged")
    ax.set_title("Burned-in text: removal vs image damage")
    for ax in axes:
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.08)
        ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
    fig.suptitle("Fig 8. Trade-offs", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    savefig(fig, cfg.figures, "F8_tradeoffs")


def decision_matrix(cfg, R):
    cols = {"completed": "runs completed", "metadata_privacy": "E1 privacy", "scientific_usability": "E2+E4 usability",
            "privacy_and_usability": "E1 and E2+E4", "burned_in_header_removed": "E3 text outside removed",
            "burned_in_region_removed": "E3 text inside removed", "anatomy_preserved": "E3 image unchanged",
            "phase1_header_clean": "Phase 1 header clean", "midi_b_files_clean": "MIDI-B clean"}
    syn = R["synthesis"]
    dm = syn.set_index("label")[[c for c in cols if c in syn]].rename(columns=cols)
    rate_heatmap(dm, "Fig 9. Summary: what each strategy achieves", cfg.figures, "F9_summary")
