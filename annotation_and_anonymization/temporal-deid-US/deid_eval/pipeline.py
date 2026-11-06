"""End-to-end steps called by the notebook: prepare data -> run tools -> validate -> summarise."""
from __future__ import annotations

import re
import shutil
import time

import numpy as np
import pandas as pd
from pydicom import dcmread

from . import adapters as A, burned_in as BI, canaries as CN, datasets as DS, phase1 as P1, synthetic as SY, validators as V
from .dicom_utils import read_frames

M_TOOLS = ["kitware", "deid", "dicognito", "rsna"]          # metadata strategies
PIX_TOOLS = ["deid_us", "rsna_ocr", "ctp"]                  # metadata + pixel strategies
CTRL = list(A.CONTROLS)
P_TOOLS = ["kitware", "deid_us", "rsna_ocr", "ctp", "ctrl_identity"]
WILD_TOOLS = ["kitware", "deid_us", "rsna_ocr", "ctp"]
LABEL = A.LABEL


def _save(cfg, df, name):
    df.to_csv(cfg.tables / f"{name}.csv", index=False)
    return df


# ---------------------------------------------------------------- 1. data
def prepare_data(cfg, manifest_path="data_manifest.json"):
    """Download (or synthesise) the inputs and build the ground truth: canaries, burned-in text, Phase 1 truth."""
    phase1_files = []
    if cfg.quick:
        items = SY.make_quick_dataset(cfg.data / "synthetic")
    else:
        items, phase1_files, _ = DS.fetch_all(cfg, DS.load_manifest(manifest_path))
    inv = DS.build_inventory(items)
    midi_inv = inv[inv.tier == "C"].reset_index(drop=True)
    inv = inv[inv.tier != "C"].reset_index(drop=True)

    # derived H.264 fixture from the smallest real (else synthetic) 2D+t cine
    pool = inv[inv.kind == "2D+t"]
    pool = pool[pool.tier == "B"] if (pool.tier == "B").any() else pool
    if len(pool) and shutil.which("ffmpeg"):
        src = pool.sort_values("size_MB").iloc[0]
        fx = cfg.data / "derived_h264_fixture.dcm"
        try:
            if not fx.exists():
                DS.make_h264_fixture(src.path, fx)
            inv = pd.concat([inv, DS.build_inventory([{"tier": "D", "source": f"H.264 derived from {src['name']}",
                                                       "name": "H264_fixture", "path": fx}])], ignore_index=True)
        except Exception as e:
            print("H.264 fixture not created:", type(e).__name__, str(e)[:200])
    _save(cfg, inv.drop(columns=["orig_study_uid"]), "inventory")
    temporal = inv[inv.kind.isin(["2D+t", "3D+t"])].reset_index(drop=True)

    # canaries (variant M: PixelData untouched)
    seeded, pixel_in, gt_dir = cfg.work / "inputs_M", cfg.work / "inputs_P", cfg.work / "pixel_ground_truth"
    for p in (seeded, pixel_in, gt_dir):
        p.mkdir(parents=True, exist_ok=True)
    canaries, rows = {}, []
    for i, r in temporal.iterrows():
        cid = f"{r.tier}{i:02d}_{''.join(ch for ch in r['name'] if ch.isalnum())[:18]}"
        study_key = r.orig_study_uid or cid
        dst = seeded / f"{cid}.dcm"
        try:
            canaries[cid] = CN.seed_canaries(r.path, dst, cid, study_key, f"{study_key}|{r['name']}", cfg.seed)
        except Exception as e:
            print("seeding failed:", cid, type(e).__name__, str(e)[:150])
            continue
        rows.append({**r.to_dict(), "case_id": cid, "study_key": study_key, "input": str(dst)})
    cases = pd.DataFrame(rows)
    for c in cases.itertuples():   # sanity: every canary findable, pixels untouched
        found = sum(CN.scan_needles(c.input, canaries[c.case_id]).values())
        same = bytes(dcmread(c.path).PixelData) == bytes(dcmread(c.input).PixelData)
        assert found == len(canaries[c.case_id]) and same, f"seeding sanity check failed for {c.case_id}"

    # burned-in text challenge (variant P)
    prow = []
    for c in cases[(cases.kind == "2D+t") & ~cases.transfer_syntax.str.contains("H.264")].itertuples():
        try:
            info = BI.make_burned_in(c.case_id, c.input, pixel_in / f"{c.case_id}_P.dcm", gt_dir / f"{c.case_id}.npz",
                                     cfg.seed, cfg.pixel_frames)
            prow.append({"case_id": c.case_id, "name": c.name, "input_M": c.input, **info})
        except Exception as e:
            print("burned-in copy failed:", c.case_id, type(e).__name__, str(e)[:150])

    # Phase 1 files, used as produced
    p1 = []
    for p in phase1_files:
        p1 += P1.load_phase1(p)
    p1 = pd.DataFrame(p1)
    if len(p1):
        p1["input"] = p1.path.astype(str)

    D = dict(inventory=inv, midi_inv=midi_inv, cases=cases, canaries=canaries,
             n_canaries=len(next(iter(canaries.values()))) if canaries else 0,
             pixel_cases=pd.DataFrame(prow), p1_cases=p1)
    print(f"{len(cases)} temporal files ({cases.kind.value_counts().to_dict()}), {D['n_canaries']} canaries each | "
          f"{len(prow)} burned-in challenges | {len(p1)} Phase 1 | {len(midi_inv)} MIDI-B")
    return D


# ---------------------------------------------------------------- 2. tools
def run_tools(cfg, D):
    cases, pc, p1, midi = D["cases"], D["pixel_cases"], D["p1_cases"], D["midi_inv"]
    subset = pd.concat([cases[cases.kind == "3D+t"], cases[cases.transfer_syntax.str.contains("H.264")],
                        cases[cases.tier.isin(["A", "S"]) & (cases.kind == "2D+t")], cases[cases.tier == "B"]]
                       ).drop_duplicates("case_id").head(cfg.pixel_tool_max_files)
    D["pix_subset"] = subset
    t0 = time.perf_counter()
    raw = {"M": [], "P": [], "P1": [], "C": []}
    for t in M_TOOLS + CTRL:
        raw["M"] += A.run_batch(t, cases, "input", "M", cfg)
    for t in PIX_TOOLS:
        raw["M"] += A.run_batch(t, subset, "input", "M", cfg)
    if len(pc):
        for t in P_TOOLS:
            raw["P"] += A.run_batch(t, pc.assign(input=pc.input_P), "input", "P", cfg)
    if len(p1):
        for t in M_TOOLS + PIX_TOOLS + ["ctrl_identity"]:
            raw["P1"] += A.run_batch(t, p1, "input", "P1", cfg)
    if len(midi):
        midi["case_id"] = [f"C{i:02d}" for i in range(len(midi))]
        for t in M_TOOLS + ["ctrl_identity"]:
            raw["C"] += A.run_batch(t, midi.assign(input=midi.path), "input", "MIDI", cfg)
    runs = pd.DataFrame(raw["M"] + raw["P"] + raw["P1"] + raw["C"])
    print(f"{len(runs)} tool runs in {(time.perf_counter() - t0) / 60:.1f} min")
    return raw


# ---------------------------------------------------------------- 3. validation
def _ocr_words(t):
    return set(re.findall(r"[A-Z0-9]{4,}", (t or "").upper()))


def evaluate(cfg, D, raw):
    """Read every output and compare it with its input; tool messages are never trusted."""
    cases, canaries, pc, p1 = D["cases"], D["canaries"], D["pixel_cases"], D["p1_cases"]
    case_lookup = {r.case_id: r._asdict() for r in cases.itertuples()}
    R = {}

    res = pd.DataFrame([V.evaluate(r, case_lookup[r["case_id"]], canaries[r["case_id"]], cfg.analysis_max_frames) for r in raw["M"]])
    res["label"] = res.tool.map(LABEL)
    res["failure_stage"] = [V.failure_stage(r) for r in res.to_dict("records")]
    _save(cfg, res.drop(columns=["tic"], errors="ignore"), "results_temporal")
    R.update(results=res, ok=res[res.status == "PASS"], linkage=V.study_linkage(res),
             not_completed=res[res.status != "PASS"][["label", "case_id", "status", "error"]])

    midi = pd.DataFrame()
    if raw["C"]:
        keys = {r.case_id: CN.derived_answer_key(r.path) for r in D["midi_inv"].itertuples()}
        midi = pd.DataFrame([{**r, "label": LABEL[r["tool"]],
                              **(V.privacy_eval(r["output"], keys[r["case_id"]]) if r["status"] == "PASS" else {})} for r in raw["C"]])
        _save(cfg, midi, "results_midi_b")
    R["midi"] = midi

    p1_lookup = {r.case_id: r._asdict() for r in p1.itertuples()} if len(p1) else {}
    rows = []
    for r in raw["P1"]:
        c = p1_lookup[r["case_id"]]
        row = {**r, "label": LABEL[r["tool"]]}
        if r["status"] == "PASS":
            try:
                row.update(V.privacy_eval(r["output"], c["canaries"]))
                row.update(V.scientific_eval(c["input"], r["output"], cfg.analysis_max_frames))
                row.update({k: v for k, v in BI.eval_boxes_output(r["output"], c["input"], c["boxes"]).items() if not k.startswith("_")})
            except Exception as e:
                row.update(status="ERROR", error=f"validator: {type(e).__name__}: {e}"[:300])
        rows.append(row)
    p1r = pd.DataFrame(rows)

    # burned-in challenge + OCR
    pc_lookup = {r.case_id: r._asdict() for r in pc.itertuples()}
    png = cfg.work / "ocr_png"
    png.mkdir(exist_ok=True)
    jobs, examples, prow = {}, {}, []
    for r in raw["P"]:
        c = pc_lookup[r["case_id"]]
        gt = np.load(c["gt_path"])
        jobs.setdefault(f"{r['case_id']}__Pinput", BI.save_png(gt["chal"][0], png / f"{r['case_id']}__Pinput.png"))
        row = {k: r[k] for k in ("case_id", "tool", "status", "runtime_s", "error", "detail")}
        row["label"] = LABEL[r["tool"]]
        if r["status"] == "PASS":
            try:
                ev = BI.eval_challenge_output(r["output"], c["gt_path"], dcmread(c["input_P"], stop_before_pixels=True).get("FrameTime", ""))
                f0 = ev.pop("_frame0")
                row.update(ev)
                jobs[f"{r['case_id']}__P{r['tool']}"] = BI.save_png(f0, png / f"{r['case_id']}__P{r['tool']}.png")
                if r["case_id"] == pc.case_id.iloc[0]:
                    examples[r["tool"]] = f0
            except Exception as e:
                row.update(status="ERROR", error=f"pixel validator: {type(e).__name__}: {e}"[:300])
        prow.append(row)
    pix = pd.DataFrame(prow)

    subset = D["pix_subset"]
    for cid in subset[subset.kind == "2D+t"].case_id:      # on-screen text already present in the real cines
        jobs[f"{cid}__Minput"] = BI.save_png(read_frames(case_lookup[cid]["input"], [0])[0], png / f"{cid}__Minput.png")
        for r in raw["M"]:
            if r["case_id"] == cid and r["tool"] in WILD_TOOLS and r["status"] == "PASS":
                try:
                    jobs[f"{cid}__M{r['tool']}"] = BI.save_png(read_frames(r["output"], [0])[0], png / f"{cid}__M{r['tool']}.png")
                except Exception as e:
                    print("frame for OCR failed:", cid, r["tool"], type(e).__name__)
    for r in raw["P1"]:
        if r["status"] == "PASS":
            jobs.setdefault(f"{r['case_id']}__P1input", BI.save_png(read_frames(p1_lookup[r["case_id"]]["input"], [0])[0],
                                                                    png / f"{r['case_id']}__P1input.png"))
            jobs[f"{r['case_id']}__P1{r['tool']}"] = BI.save_png(read_frames(r["output"], [0])[0], png / f"{r['case_id']}__P1{r['tool']}.png")
    ocr = {}
    try:
        ocr = BI.run_easyocr(jobs, cfg.rsna_python, cfg.data / "_rsna_cache" / "assets" / "ai" / "ocr" / "model", cfg.work / "ocr")
    except Exception as e:
        print("OCR not available:", str(e)[:300])
    print(f"OCR: {len(ocr)}/{len(jobs)} frames read")

    if len(pix):
        for side in ("H", "R"):
            pix[f"ocr_recall_{side}"] = [BI.ocr_recall(ocr[f"{r.case_id}__P{r.tool}"], pc_lookup[r.case_id][f"tokens_{side}"])
                                         if f"{r.case_id}__P{r.tool}" in ocr else np.nan for r in pix.itertuples()]
            pix[f"ocr_recall_{side}_input"] = [BI.ocr_recall(ocr[f"{r.case_id}__Pinput"], pc_lookup[r.case_id][f"tokens_{side}"])
                                               if f"{r.case_id}__Pinput" in ocr else np.nan for r in pix.itertuples()]
            pix[f"text_removed_{side}"] = (pix.get(f"residual_{side}", np.nan) < 0.2) & (pix[f"ocr_recall_{side}"].fillna(0) < 0.5)
        pix["anatomy_preserved"] = pix.get("collateral_mae_in_region", np.nan) < 1.0
        _save(cfg, pix, "results_burned_in")
    wild = pd.DataFrame([{"case_id": cid, "name": case_lookup[cid]["name"], "tool": t, "label": LABEL[t],
                          "text_retained": (len(_ocr_words(ocr.get(f"{cid}__Minput")) & _ocr_words(ocr[f"{cid}__M{t}"]))
                                            / len(_ocr_words(ocr.get(f"{cid}__Minput")))) if _ocr_words(ocr.get(f"{cid}__Minput")) else np.nan}
                         for cid in subset.case_id for t in WILD_TOOLS if f"{cid}__M{t}" in ocr])
    if len(p1r):
        p1r["ocr_recall_boxes"] = [BI.ocr_recall(ocr.get(f"{r.case_id}__P1{r.tool}", ""), p1_lookup[r.case_id]["tokens"])
                                   if f"{r.case_id}__P1{r.tool}" in ocr else np.nan for r in p1r.itertuples()]
        _save(cfg, p1r.drop(columns=["tic"], errors="ignore"), "results_phase1")
    R.update(pixel=pix, pixel_examples=examples, wild=wild, phase1=p1r)

    ctl = V.check_controls(res)
    ctl["control"] = ctl.control.map(LABEL)
    R["controls"] = ctl
    R["controls_ok"] = bool(len(ctl) and (ctl.caught.dropna() == 1).all())
    R["order"] = [t for t in M_TOOLS + PIX_TOOLS + CTRL if t in set(res.tool)]
    R["synthesis"] = _synthesis(cfg, R)
    if len(R["not_completed"]):
        print(f"{len(R['not_completed'])} runs did not complete (kept in the tables as ERROR / NOT_EVALUABLE)")
    print("All controls caught:", R["controls_ok"])
    return R


def _synthesis(cfg, R):
    res, ok, pix, wild, p1r, midi = R["results"], R["ok"], R["pixel"], R["wild"], R["phase1"], R["midi"]
    rows = []
    for t in [x for x in R["order"] if x in A.TOOLS]:
        gp, allr = ok[ok.tool == t], res[res.tool == t]
        row = {"tool": t, "label": LABEL[t], "scope": A.TOOLS[t]["scope"], "files": len(allr), "completed": (allr.status == "PASS").mean(),
               "metadata_privacy": gp.metadata_privacy_pass.mean() if len(gp) else np.nan,
               "scientific_usability": gp.scientific_usability_pass.mean() if len(gp) else np.nan,
               "bitstream_identical": gp.pixeldata_bytes_same.mean() if len(gp) else np.nan,
               "privacy_and_usability": gp.question_endpoint_pass.mean() if len(gp) else np.nan,
               "median_runtime_s": allr.runtime_s.median()}
        if len(pix) and t in set(pix.tool):
            pp = pix[(pix.tool == t) & (pix.status == "PASS")]
            row.update(burned_in_header_removed=pp.text_removed_H.mean(), burned_in_region_removed=pp.text_removed_R.mean(),
                       anatomy_preserved=pp.anatomy_preserved.mean())
        if len(wild) and t in set(wild.tool):
            row["real_onscreen_text_removed"] = 1 - wild[wild.tool == t].text_retained.mean()
        if len(p1r) and t in set(p1r.tool):
            row["phase1_header_clean"] = p1r[(p1r.tool == t) & (p1r.status == "PASS")].metadata_privacy_pass.mean()
        if len(midi) and t in set(midi.tool):
            row["midi_b_files_clean"] = midi[(midi.tool == t) & (midi.status == "PASS")].metadata_privacy_pass.mean()
        rows.append(row)
    return _save(cfg, pd.DataFrame(rows), "synthesis")
