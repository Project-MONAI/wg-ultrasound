"""Run configuration.

Everything machine-specific comes from environment variables, so the same notebook runs on
Windows, Linux, macOS and Colab without edits:

    DEID_EVAL_HOME     workspace root (default: ./workspace next to the notebook)
    DEID_EVAL_MODE     "quick" (synthetic data only, minutes, offline) or "full" (public data)
    RSNA_PYTHON        interpreter with `rsna-anonymizer` installed (default: current interpreter if it has it)
    RSNA_WORKSPACE     short folder for RSNA's storage tree (Windows 260-character path limit)
    CTP_DIR            folder containing CTP DicomAnonymizerTool's DAT.jar (optional)
"""
from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _rsna_python() -> Path | None:
    env = os.environ.get("RSNA_PYTHON")
    if env:
        return Path(env) if Path(env).exists() else None
    if importlib.util.find_spec("anonymizer") is not None:
        return Path(sys.executable)
    return None


def _rsna_workspace() -> Path:
    env = os.environ.get("RSNA_WORKSPACE")
    if env:
        return Path(env)
    # RSNA nests <patient>/<study UID>/<series UID>/<SOP UID>.dcm (~200 characters); keep the root short.
    if platform.system() == "Windows":
        return Path.home() / ".rsw"
    return Path(tempfile.gettempdir()) / "rsw"


@dataclass
class Config:
    home: Path
    mode: str = "full"
    seed: int = 20261104
    rsna_python: Path | None = None
    rsna_workspace: Path = field(default_factory=_rsna_workspace)
    ctp_dir: Path | None = None
    # limits
    max_tcia_cines: int = 8
    max_midi_files: int = 24
    pixel_frames: int = 16            # frames kept in burned-in challenge copies
    analysis_max_frames: int = 80     # frames decoded per file for pixel comparison
    pixel_tool_max_files: int = 9     # real cines processed by pixel-cleaning tools
    rsna_timeout_per_file: int = 900
    tool_timeout_per_file: int = 1800

    @property
    def quick(self) -> bool:
        return self.mode == "quick"

    @property
    def data(self) -> Path:
        return self.home / "data"

    @property
    def results(self) -> Path:
        return self.home / "results"

    @property
    def work(self) -> Path:
        return self.results / "work"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    def makedirs(self) -> "Config":
        for p in (self.data, self.results, self.work, self.figures, self.tables):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def check_disk(self, need_gb: float) -> float:
        """Free space (GB) at the workspace; raises if below `need_gb`."""
        import shutil
        self.home.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.home).free / 2**30
        if free < need_gb:
            raise RuntimeError(f"Only {free:.1f} GB free at {self.home}; this mode needs about {need_gb:.0f} GB. "
                               "Free space or set DEID_EVAL_HOME to another drive.")
        return free


def load_config(**overrides) -> Config:
    mode = os.environ.get("DEID_EVAL_MODE", "full").lower()
    home = Path(os.environ.get("DEID_EVAL_HOME", Path.cwd() / "workspace")).resolve()
    ctp = os.environ.get("CTP_DIR")
    cfg = Config(home=home, mode=mode, rsna_python=_rsna_python(), ctp_dir=Path(ctp) if ctp else None)
    if cfg.quick:
        cfg.max_tcia_cines, cfg.max_midi_files, cfg.analysis_max_frames, cfg.pixel_tool_max_files = 0, 0, 24, 6
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg.makedirs()


def tool_versions(cfg: Config) -> dict:
    """Versions of everything that influences the results (written to environment.json)."""
    import importlib.metadata as ilm

    def ver(pkg):
        try:
            return ilm.version(pkg)
        except Exception:
            return "not installed"

    def first_line(cmd):
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out = (cp.stdout or cp.stderr or "").strip().splitlines()
            return out[0] if out else ""
        except Exception as e:
            return f"unavailable ({type(e).__name__})"

    rsna = "not configured"
    if cfg.rsna_python is not None:
        rsna = first_line([str(cfg.rsna_python), "-c",
                           "import importlib.metadata as m;print(m.version('rsna-anonymizer'), 'easyocr', m.version('easyocr'))"])
    return {
        "platform": platform.platform(), "python": sys.version.split()[0],
        "pydicom": ver("pydicom"), "numpy": ver("numpy"), "pandas": ver("pandas"), "matplotlib": ver("matplotlib"),
        "pylibjpeg": ver("pylibjpeg"), "dicom-anonymizer (Kitware)": ver("dicom-anonymizer"),
        "deid (pydicom-deid)": ver("deid"), "dicognito": ver("dicognito"),
        "rsna-anonymizer / easyocr (RSNA_PYTHON)": rsna,
        "ffmpeg": first_line(["ffmpeg", "-version"]), "java": first_line([os.environ.get("CTP_JAVA") or "java", "-version"]),
        "mode": cfg.mode, "seed": cfg.seed,
    }
