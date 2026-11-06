"""Shared plotting style and generic charts.

Colour is assigned per entity in a fixed order (a tool keeps its colour in every figure); controls are neutral grey.
Rates use one single-hue sequential ramp and every cell / bar carries its value as text, so nothing depends on
colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def setup_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": BASE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150, "lines.linewidth": 2,
    })


def tool_colors(tool_keys, control_keys):
    col = {k: CAT[i % len(CAT)] for i, k in enumerate(tool_keys)}
    col.update({k: MUTED for k in control_keys})
    return col


def savefig(fig, folder, name, show=True):
    """Save as PNG and embed that exact file in the notebook.

    Embedding the saved file (instead of plt.show()) works whatever matplotlib backend an imported tool
    may have switched to, and guarantees the notebook shows the same image that is archived in results/.
    """
    path = Path(folder) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if show:
        try:
            from IPython.display import Image, display
            display(Image(filename=str(path)))
        except Exception:
            pass
    return path


def rate_heatmap(matrix, title, folder, name, counts=None, figsize=None, note=None):
    """matrix: rows = entities, cols = criteria, values 0..1 (NaN = not applicable)."""
    m = matrix.astype(float)
    figsize = figsize or (1.25 * m.shape[1] + 3.4, 0.48 * m.shape[0] + 1.6)
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.imshow(np.ma.masked_invalid(m.to_numpy()), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if pd.isna(v):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#f0efec", edgecolor=SURFACE, lw=2, hatch="///"))
                ax.text(j, i, "n/a", ha="center", va="center", color=MUTED, fontsize=8)
            else:
                lab = f"{v:.0%}" + (f"\n{counts.iat[i, j]}" if counts is not None else "")
                ax.text(j, i, lab, ha="center", va="center", fontsize=8.5, color="#ffffff" if v >= 0.55 else INK)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=30, ha="right")
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, m.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, m.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title)
    if note:
        fig.text(0.01, -0.02, note, fontsize=8, color=INK2, ha="left", va="top", wrap=True)
    return savefig(fig, folder, name)


def ci_bars(tab, colors, title, folder, name, xlabel="share of files (95% Wilson CI)"):
    """tab columns: tool, label, k, n, rate, lo, hi."""
    tab = tab[tab.n > 0].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(tab) + 1.4))
    y = np.arange(len(tab))
    ax.barh(y, tab.rate, color=[colors[t] for t in tab.tool], height=0.55, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(tab.rate, y, xerr=[tab.rate - tab.lo, tab.hi - tab.rate], fmt="none", ecolor=INK2, capsize=3, lw=1)
    for yi, r in zip(y, tab.itertuples()):
        ax.text(min(r.hi, 1) + 0.02, yi, f"{r.rate:.0%}  ({r.k}/{r.n})", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(tab.label)
    ax.set_xlim(0, 1.25)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return savefig(fig, folder, name)


def rate_matrix(df, cols, order, label_of, col_labels=None):
    """(rates, 'k/n' counts) per entity x criterion, from boolean / 0-1 columns."""
    mat, cnt = {}, {}
    for t in order:
        gp = df[df.tool == t]
        mat[label_of[t]], cnt[label_of[t]] = {}, {}
        for c in cols:
            z = gp[c].dropna().astype(float) if c in gp else pd.Series(dtype=float)
            name = (col_labels or {}).get(c, c)
            mat[label_of[t]][name] = z.mean() if len(z) else np.nan
            cnt[label_of[t]][name] = f"{int(z.sum())}/{len(z)}" if len(z) else ""
    return pd.DataFrame(mat).T, pd.DataFrame(cnt).T
