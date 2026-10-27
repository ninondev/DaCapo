"""E5 scoring (main environment). Per-note pitch accuracy in the gap against every hypothesis.

  python score_analogy.py gate --program P [--manifest M] [--ids a,b] [--out F]   ruler gate (K0), no DiT
  python score_analogy.py quality p0b|main [--attempt N]    integrity, then trackers + registered gate
  python score_analogy.py quality p0b|main --void-report    after the terminal outcome: a void attempt's decision
  python score_analogy.py score main [--attempt N] [--diagnostic]   accuracies, summary, verdicts
  python score_analogy.py measurement                       are the measurement packages the pinned ones?
  python score_analogy.py regress-v1                        recompute the archived v1 numbers

Each hypothesis is a note list starting at the gap (12 s). Notes inside the 0.4 s gap margins are
excluded (latent crossfade). Every hypothesis gets the same symmetric lag search (+-120 ms), so foils
and rule answers are treated the same. The quality stage checks integrity first; only then do the
trackers run, and it freezes the per-draw gate and every audio file's sha256 before any accuracy exists.
A void attempt gets no quality decision until the E5 v2 terminal outcome exists. The score stage only
reads the frozen file and the stored tracker output. Exit codes: 0 done, 2 refused, 3 terminal E5
outcome written or present, 4 incomplete (generation may still run), 5 the attempt is void (integrity
failure before any accuracy).
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import v2_config as C  # noqa: E402
from dacapo import motif_features as F  # noqa: E402
from dacapo import motif_integrity as I  # noqa: E402
from dacapo import motif_stats as S  # noqa: E402
from dacapo.analysis import _to_grid, f0_swift, notes_basic_pitch, skyline  # noqa: E402
from dacapo.metrics import note_pitch_accuracy  # noqa: E402
from dacapo.provenance import write_result  # noqa: E402

SPB = 0.5            # 60 / 120 BPM, as in every E5 manifest (checked against the manifest in use)
G0, G1 = 12.0, 16.0
MARGIN = 0.4
LAGS = np.arange(-0.12, 0.1201, 0.02)
SETS = S.CLASS_SETS
TAU = 0.7
EXIT_OK, EXIT_REFUSED, EXIT_TERMINAL, EXIT_INCOMPLETE, EXIT_VOID = 0, 2, 3, 4, 5
TRACKER_PACKAGES = ("swift-f0", "basic-pitch", "onnxruntime", "librosa", "soxr", "numpy", "scipy", "soundfile",
                    "mir_eval", "pretty_midi")


def notes_in_gap(note_dicts: list[dict]) -> list[tuple[float, float, int]]:
    out = []
    for n in note_dicts:
        s, e = G0 + n["start"] * SPB, G0 + (n["start"] + n["dur"]) * SPB
        if s >= G0 + MARGIN and e <= G1 - MARGIN + 0.25:
            out.append((s, min(e, G1 - MARGIN), n["pitch"]))
    return out


def acc(et, ef, notes, octave_tolerant=False) -> float:
    return max(note_pitch_accuracy(notes, et, ef, lag=lag, octave_tolerant=octave_tolerant)["npa"] for lag in LAGS)


def check_layout(man: dict) -> None:
    if man["bpm"] != 120.0 or list(man["gap_s"]) != [G0, G1]:
        raise ValueError("notes_in_gap assumes 120 BPM and a 12-16 s gap")


def validate_manifest(man: dict) -> list[str]:
    """Every hypothesis and B2 must have at least one scored note (onset in the window, length >= 20 ms
    after clipping), so note-F1 and onset-F1 are defined for every row. Checked by K0."""
    bad = []
    for it in man["items"]:
        for h, notes in list(it["hyps"].items()) + [("B2", it["B2"])]:
            if not [n for n in notes_in_gap(notes) if n[1] - n[0] >= F.MIN_REF_NOTE_S]:
                bad.append(f"{it['id']}:{h}")
    return bad


def tracker_provenance(packages=TRACKER_PACKAGES) -> dict:
    vers = {}
    for p in packages:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    import basic_pitch
    import swift_f0
    models = {"swift_f0/model.onnx": I.sha256_file(Path(swift_f0.__file__).parent / "model.onnx"),
              "basic_pitch/nmp.onnx": I.sha256_file(Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx")}
    return {"versions": vers, "model_sha256": models}


def measurement_problems() -> dict:
    """Pinned measurement packages or ONNX models that differ from addendum 2 (section measurement).
    numpy, scipy and soundfile are recorded in every quality file but not pinned."""
    m = (C.read_json(C.addendum_path()) or {}).get("measurement") or {}
    pinned, onnx = m.get("pinned") or {}, m.get("onnx_sha256") or {}
    if not pinned or not onnx:
        return {"addendum": "no pinned measurement software registered"}
    prov = tracker_provenance(tuple(pinned))
    bad = {p: {"registered": v, "installed": prov["versions"].get(p)} for p, v in pinned.items() if prov["versions"].get(p) != v}
    bad.update({f: {"registered": h, "installed": prov["model_sha256"].get(f)} for f, h in onnx.items()
                if prov["model_sha256"].get(f) != h})
    return bad


def measurement() -> int:
    bad = measurement_problems()
    print("measurement software:", "as registered" if not bad else f"differs from addendum 2: {bad}", flush=True)
    return EXIT_OK if not bad else EXIT_REFUSED


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _pool_map(fn, jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        return pool.map(fn, jobs, chunksize=1)       # results in the order of `jobs`


# Ruler gate (K0): exact hypothesis renders must be classified as themselves.

def _gate_item(args) -> dict:
    """K0 rows and wrong-note scores for one item. Renders are made from the manifest when absent and
    compared with the content sha256 recorded in it; stream=True deletes them afterwards."""
    from dacapo.render import render
    from dacapo.symbolic import Melody, Note

    it, program, e5, wrong, stream, bpm, unit_beats = args
    import make_items
    key = make_items.hyp_key(program)
    if key not in it:                                   # a manifest that lists flute renders only (v1 layout)
        it = {**it, key: {h: f"hyp_renders_p{program}/{make_items.hyp_file(it['id'], h)}" for h in it["hyps"]}}
    rec = (it.get("hyp_render_sha256") or {}).get(str(program), {})
    made, mismatch, rows = [], [], []
    for h, rel in it[key].items():
        wav = e5 / rel
        if not wav.exists():
            wav.parent.mkdir(parents=True, exist_ok=True)
            D = make_items.melody(it["hyps"]["D"])
            render([make_items.hyp_full(make_items.melody(it["hyps"][h]), D).to_instrument(program)], bpm, wav, sr=48000)
            made.append(wav)
        if rec and I.audio_sha256(wav) != rec[h]:
            mismatch.append(rel)
        et, ef = f0_swift(wav)
        for cond, hs in SETS.items():
            if h in hs:
                rows.append({"id": it["id"], "cond": cond, "true": h,
                             "accs": {x: acc(et, ef, notes_in_gap(it["hyps"][x])) for x in hs}})
    good, bad, scored = {}, {}, {}
    B = Melody(tuple(Note(**n) for n in it["hyps"]["B"]), bpm)
    et, ef = f0_swift(e5 / it[key]["B"])
    for k, (idx, shifts) in wrong.items():
        scored[k] = all(notes_in_gap([it["hyps"]["B"][i]]) for i in idx)
        good[k] = acc(et, ef, notes_in_gap(it["hyps"]["B"]))
        W = B.with_pitch_errors(idx, shifts)
        full = Melody(tuple(W.shift(3 * unit_beats).notes), bpm)
        wav = e5 / "hyp_renders" / "_wrong" / f"{it['id']}_B_{k}wrong_p{program}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        render([full.to_instrument(program)], bpm, wav, sr=48000)
        et2, ef2 = f0_swift(wav)
        bad[k] = acc(et2, ef2, notes_in_gap(it["hyps"]["B"]))
        if stream:
            wav.unlink()
    if stream:
        for w in made:
            w.unlink(missing_ok=True)
    return {"rows": rows, "good": good, "bad": bad, "scored": scored, "mismatch": mismatch}


def gate(program: int, manifest: str = "manifest_v2.json", ids: list[str] | None = None, out: str | None = None,
         stream: bool = True, workers: int = 1) -> int:
    """K0 for one timbre (GM program): tau fixed at 0.7, ties counted as errors. Passes iff the
    classification rate is >= 0.95, the 1- and 2-wrong-note AUCs are both >= 0.9, every hypothesis has a
    scorable note, and every render has the content sha256 the manifest records."""
    t0 = time.time()
    e5 = C.e5_dir()
    man_path = e5 / manifest
    man = json.loads(man_path.read_text())
    check_layout(man)
    items = [it for it in man["items"] if not ids or it["id"] in ids]
    rng = np.random.default_rng(0)
    wrong = {it["id"]: {} for it in items}
    for k in (1, 2):                      # the v1 random stream: all items for k = 1, then for k = 2
        for it in items:
            n = len(it["hyps"]["B"])
            idx = sorted(rng.choice(n, size=k, replace=False).tolist())
            wrong[it["id"]][k] = (idx, rng.choice([-2, -1, 1, 2], size=k).tolist())
    res_items = _pool_map(_gate_item, [(it, program, e5, wrong[it["id"]], stream, man["bpm"], man["unit_beats"])
                                       for it in items], workers)
    rows = [r for x in res_items for r in x["rows"]]
    mismatch = [m for x in res_items for m in x["mismatch"]]
    res = {"per_condition": {}, "tau_candidates": {}, "program": program, "manifest": manifest,
           "manifest_sha256": I.sha256_file(man_path), "n_items": len(items), "ids": [it["id"] for it in items]}
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
        res["tau_candidates"][str(tau)] = round(float(np.mean([S.classify(r["accs"], list(r["accs"]), tau) == r["true"] for r in rows])), 4)
    for cond in SETS:
        sel = [r for r in rows if r["cond"] == cond]
        res["per_condition"][cond] = round(float(np.mean([S.classify(r["accs"], list(r["accs"]), TAU) == r["true"] for r in sel])), 4)
    tied = [(r["id"], r["cond"], r["true"]) for r in rows if S.classify(r["accs"], list(r["accs"]), TAU) == "tie"]
    n_ok = sum(S.classify(r["accs"], list(r["accs"]), TAU) == r["true"] for r in rows)
    res.update({"tau": TAU, "overall": round(n_ok / len(rows), 4), "n_correct": n_ok, "n_rows": len(rows),
                "n_tie_rows": len(tied), "tied_rows": tied,
                "self_acc_mean": round(float(np.mean([r["accs"][r["true"]] for r in rows])), 4)})
    aucs = {}
    for k in (1, 2):
        g = np.array([x["good"][k] for x in res_items])
        b = np.array([x["bad"][k] for x in res_items])
        aucs[f"{k}_wrong"] = round(float((g[:, None] > b[None]).mean() + 0.5 * (g[:, None] == b[None]).mean()), 4)
    res["wrong_note_auc"] = aucs
    # Exploratory, no decision: the registered draw also perturbs notes the ruler never scores (the first note
    # of B starts at 12.0 s, inside the 0.4 s margin), which a correct ruler cannot detect.
    res["wrong_note_detail"] = [{"id": it["id"], "k": k, "notes": wrong[it["id"]][k][0], "all_scored": x["scored"][k],
                                 "acc_correct": x["good"][k], "acc_wrong": x["bad"][k]}
                                for it, x in zip(items, res_items) for k in (1, 2)]
    aucs_scored = {}
    for k in (1, 2):
        g = np.array([x["good"][k] for x in res_items])
        b = np.array([x["bad"][k] for x in res_items if x["scored"][k]])
        aucs_scored[f"{k}_wrong"] = round(float((g[:, None] > b[None]).mean() + 0.5 * (g[:, None] == b[None]).mean()), 4) \
            if len(b) else None
    res["exploratory_wrong_note_auc_scored_notes_only"] = aucs_scored
    res["n_wrong_renders_with_unscored_notes"] = {f"{k}_wrong": sum(not x["scored"][k] for x in res_items) for k in (1, 2)}
    res["manifest_problems"] = validate_manifest({"items": items})
    res["render_mismatch"] = mismatch
    res["passed"] = bool(n_ok / len(rows) >= 0.95 and min(aucs.values()) >= 0.9 and not res["manifest_problems"]
                         and not mismatch)
    out_path = Path(out) if out else C.ruler_gate_path(program)
    write_result(out_path, {"experiment": "e5_ruler_gate", "lags": LAGS.tolist(), "margin_s": MARGIN,
                            "tie_rule": "strict", "tau_rule": "fixed at 0.7"}, {"summary": res, "rows": rows}, t0)
    print(f"K0 program {program} on {len(items)} items: {'passed' if res['passed'] else 'failed'}; classification "
          f"{res['overall']} ({n_ok}/{len(rows)}), wrong-note AUC {aucs}, manifest problems "
          f"{len(res['manifest_problems'])}, render mismatches {len(mismatch)}; details in {out_path.name}", flush=True)
    return EXIT_OK if res["passed"] else EXIT_TERMINAL


# Quality stage: hypothesis-blind features and the frozen gate.

def swift_with_conf(x: np.ndarray, sr: int, fmin: float = 65.0, fmax: float = 1500.0):
    """Same voicing rule and grid as dacapo.analysis.f0_swift (confidence >= 0.5), plus the confidence
    track. One ONNX thread gives output identical to the default thread count."""
    from swift_f0 import SwiftF0
    r = SwiftF0(threads=1).detect(x.astype(np.float32), sr, fmin=fmin, fmax=fmax)
    conf = np.asarray(r.confidence)
    f0 = np.where(conf >= 0.5, np.asarray(r.pitch_hz), 0.0)
    t, f0g = _to_grid(np.asarray(r.timestamps), f0, len(x) / sr)
    _, cg = _to_grid(np.asarray(r.timestamps), conf, len(x) / sr)
    return t, f0g, cg


def _file_job(path: str) -> dict:
    """One draw before any tracker runs: container, subtype, channel peak and sha256 of the stored file."""
    import soundfile as sf
    info = sf.info(path)
    x, _ = sf.read(path, always_2d=True)
    return {"wav_sha256": I.sha256_file(path), "peak_dbfs": F.peak_dbfs(x),
            "info": {"format": info.format, "subtype": info.subtype, "samplerate": info.samplerate, "channels": info.channels}}


def _features_job(args) -> dict:
    """One draw: read the audio as written, run both trackers once, store their output."""
    import librosa
    import soundfile as sf
    wav, cond, npz = args
    x, sr = sf.read(wav, always_2d=True)
    mono = x.mean(axis=1)
    et, ef, conf = swift_with_conf(mono, sr)
    bp = notes_basic_pitch(wav)
    g = (et >= F.W_GAP[0]) & (et < F.W_GAP[1])
    gap = F._seg(mono, sr, *F.W_GAP)
    q = {"gap_voiced": F.voiced_fraction(et, ef), "gap_polyphony": F.polyphony_v1(bp),
         "gap_polyphony_octfold": F.polyphony_octfold(bp), **F.level_features(x, sr, cond),
         "swift_conf_median": float(np.median(conf[g])) if g.any() else None,
         "flatness_median": float(np.median(librosa.feature.spectral_flatness(y=gap.astype(np.float32), n_fft=2048, hop_length=512))),
         "double_f0_frac": F.double_f0_fraction([n for n in bp if n[1] > F.W_GAP[0] and n[0] < F.W_GAP[1]])}
    tmp = Path(npz + ".tmp.npz")
    np.savez(tmp, et=et, ef=ef, conf=conf, bp=F.notes_to_array(bp))
    os.replace(tmp, npz)
    return {"quality": q, "wav_sha256": I.sha256_file(wav), "npz_sha256": I.sha256_file(npz)}


def _npz_name(r: dict) -> str:
    return Path(I.audio_name(r)).stem + ".npz"


def _load_design(phase: str, attempt: int) -> tuple[Path, dict, list]:
    od = C.phase_dir(phase, attempt)
    design = json.loads((od / "design.json").read_text())
    return od, design, I.read_ledger(od / "ledger.jsonl")


def _void(phase: str, attempt: int, failed: list[str], detail) -> int:
    return EXIT_TERMINAL if C.record_void(phase, attempt, failed, detail) else EXIT_VOID


def resumable(design: dict, events: list[dict], status: dict) -> dict:
    """Pending draws that generation may still run: a lost draw (audio gone before the freeze) always, a
    never-completed draw while it has fewer than MAX_STARTS attempts and fewer than MAX_RESUME_PASSES
    automatic resume passes were made."""
    st, n_resume = I.ledger_state(events)
    lost = [k for k, v in status.items() if v == "lost"]
    missing = [k for k, v in status.items() if v == "missing"]
    can = [k for k in missing if st.get(k, {}).get("starts", 0) < C.MAX_STARTS] if n_resume < C.MAX_RESUME_PASSES else []
    return {"lost": len(lost), "missing": len(missing), "resumable_missing": len(can), "n_resume": n_resume}


def _integrity(phase: str, design: dict, events: list, od: Path, st: dict, rows: list[dict], fmap: dict) -> dict:
    """Checks 1-7, 9 and 10 of addendum 2 (S11) on the files as stored, before any tracker runs."""
    e5 = C.e5_dir()
    src_sha = {s: I.audio_sha256(e5 / s) if (e5 / s).exists() else None for s in sorted({j["src"] for j in design["jobs"]})}
    levels = {s: I.source_levels(e5 / s) for s in src_sha if src_sha[s]}
    seeds = (C.read_json(e5 / "registration" / "e5v2_seeds.json") or {}) if design.get("scale") == "full" else {}
    checks = [I.check_lossless({k: f["info"] for k, f in fmap.items()}),
              I.check_peaks(rows),
              I.check_sources(design, levels) if len(levels) == len(src_sha) else I._res("source_levels", False, "source missing"),
              I.check_seeds(design, events, seeds.get(design["phase"])),
              I.check_files_vs_design(design, od, src_sha),
              I.check_ledger(design, events, {k: f["wav_sha256"] for k, f in fmap.items()}),
              I.check_registration(st["addendum"], st["current"], design, st["addendum_sha256"], events, st["current_data"]),
              I._res("manifest_as_designed", (e5 / design["manifest"]).exists()
                     and I.sha256_file(e5 / design["manifest"]) == design["manifest_sha256"])]
    if phase == "main":
        pilot = C.read_json(Path(C.run_dir()) / design["pilot_dir"] / "design.json")
        checks.insert(4, I.check_disjoint(design, pilot))
        checks.insert(6, I.check_ruler(C.read_json(C.ruler_gate_path(design["ruler_program"])), design))
    return I.summarize_checks(checks)


def _rows(jobs: list[dict], status: dict, fmap: dict) -> list[dict]:
    rows = []
    for j in jobs:
        k = I.job_key(j)
        r = {"id": j["id"], "split": j["split"], "cond": j["cond"], "arm": j.get("arm", "base"), "seed": j["seed"],
             "dit_seed": j["dit_seed"], "rng_seed": j["rng_seed"], "status": status[k]}
        if k in fmap:
            r.update({"wav_sha256": fmap[k]["wav_sha256"], "wav_info": fmap[k]["info"],
                      "quality": {"peak_dbfs": fmap[k]["peak_dbfs"]}})
        rows.append(r)
    return rows


def _decide(phase: str, design: dict, rows: list[dict], jobs: list[dict]) -> tuple[dict, dict]:
    planned = [I.job_key(j) for j in jobs]
    if phase == "p0b":
        dec = lambda g: S.pilot_decision(rows, planned, design["arm_order"], k1_frac=design["k1_frac"],
                                         excluded=design.get("excluded_arms", ()), missing_as_failure=True, gate_key=g)
    else:
        dec = lambda g: S.quality_decision(rows, planned, frac_draws=design["q1_frac"], frac_cells=design["q2_frac"],
                                           min_pass_per_cell=design["min_pass_per_cell"], missing_as_failure=True, gate_key=g)
    return dec("pass"), {g: dec(g) for g in ("pass_octfold", "pass_splice_safe")}


def _features(od: Path, rows: list[dict], workers: int, sub: str = "features") -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    (od / sub).mkdir(exist_ok=True)
    feats = _pool_map(_features_job, [(str(od / I.audio_name(r)), r["cond"], str(od / sub / _npz_name(r))) for r in ok],
                      workers)
    return {I.job_key(r): f for r, f in zip(ok, feats)}


def quality(phase: str, attempt: int = 1, workers: int = 4, void_report: bool = False) -> int:
    t0 = time.time()
    if void_report:
        return _void_report(phase, attempt, workers)
    if C.terminal_path().exists():
        print("terminal outcome exists:", C.terminal_path().read_text())
        return EXIT_TERMINAL
    qpath = C.quality_path(phase, attempt)
    if qpath.exists():
        print(f"{qpath.name} is frozen; the quality stage runs once per phase and attempt")
        return _outcome_from_frozen(phase, attempt, json.loads(qpath.read_text()))
    od, design, events = _load_design(phase, attempt)
    status = I.draw_status(design, events, od)
    pend = resumable(design, events, status)
    if pend["lost"] or pend["resumable_missing"]:
        C.write_json(C.results_dir() / f"e5v2_{phase}_incomplete.json", {**pend, "utc": C.utc()})
        print("incomplete: generation must run again before the quality decision", pend)
        return EXIT_INCOMPLETE
    meas = measurement_problems()
    if meas:
        print("quality stage refused: measurement software differs from addendum 2:", meas)
        return EXIT_REFUSED
    st = C.addendum_status()
    jobs = sorted(design["jobs"], key=I.job_key)
    ok_jobs = [j for j in jobs if status[I.job_key(j)] == "ok"]
    files = _pool_map(_file_job, [str(od / I.audio_name(j)) for j in ok_jobs], workers)
    fmap = {I.job_key(j): f for j, f in zip(ok_jobs, files)}
    rows = _rows(jobs, status, fmap)
    integrity = _integrity(phase, design, events, od, st, rows, fmap)
    cfg = {"experiment": "e5v2_quality", "phase": phase, "attempt": attempt, "scale": design.get("scale"),
           "gate": F.GATE_REGISTERED, "exploratory_gates": [F.GATE_OCTFOLD, F.GATE_SPLICE_SAFE],
           "windows": {"gap": F.W_GAP, "ctx": F.W_CTX, "D": F.W_D}, "design_sha256": I.sha256_file(od / "design.json"),
           "code_sha256": st["current"], "data_sha256": st["current_data"], "trackers": tracker_provenance(),
           "pending": pend}
    if integrity["ok"]:
        feats = _features(od, rows, workers)
        moved = sorted("|".join(k) for k, f in feats.items() if f["wav_sha256"] != fmap[k]["wav_sha256"])
        if moved:
            integrity["checks"].append(I._res("audio_unchanged_during_quality", False, moved[:50]))
            integrity["ok"] = False
    if not integrity["ok"]:
        write_result(qpath, cfg, {"integrity": integrity, "decision": None, "rows": rows,
                                  "note": "integrity failed before any tracker output was used: no quality decision"}, t0)
        print(json.dumps({"integrity_ok": False, "failed": [c["name"] for c in integrity["checks"] if not c["ok"]]}, indent=1))
        return _outcome(phase, attempt, integrity, None, qpath)
    for r in rows:
        f = feats.get(I.job_key(r))
        if f:
            r.update({"npz_sha256": f["npz_sha256"], "quality": f["quality"], "gate": F.gate_record(f["quality"])})
    decision, alternatives = _decide(phase, design, rows, jobs)
    null_levels = [r["quality"]["gap_rms_dbfs"] for r in rows if r["status"] == "ok" and r["cond"] == "null"]
    out = {"integrity": integrity, "decision": decision, "exploratory_decisions": alternatives,
           "null_gap_rms_dbfs": null_levels, "rows": rows}
    write_result(qpath, cfg, out, t0)
    print(json.dumps({"integrity_ok": True, "decision": decision}, indent=1, default=str))
    return _outcome(phase, attempt, integrity, decision, qpath)


def _void_report(phase: str, attempt: int, workers: int) -> int:
    """The quality decision of a void attempt, computed only once the E5 v2 terminal outcome exists and
    reported in full (exploratory); nothing reads it."""
    t0 = time.time()
    qf = C.read_json(C.quality_path(phase, attempt))
    if not C.terminal_path().exists() or qf is None or qf["results"]["integrity"]["ok"]:
        print("--void-report needs the terminal outcome and a quality file whose integrity failed")
        return EXIT_REFUSED
    od, design, _ = _load_design(phase, attempt)
    rows = [dict(r) for r in qf["results"]["rows"]]
    feats = _features(od, rows, workers, sub="features_void_report")
    for r in rows:
        f = feats.get(I.job_key(r))
        if f:
            r.update({"npz_sha256": f["npz_sha256"], "quality": f["quality"], "gate": F.gate_record(f["quality"])})
    jobs = sorted(design["jobs"], key=I.job_key)
    decision, alternatives = _decide(phase, design, rows, jobs)
    out_path = C.quality_path(phase, attempt).with_name(C.quality_path(phase, attempt).stem + "_void_report.json")
    write_result(out_path, {"experiment": "e5v2_quality_void_report", "phase": phase, "attempt": attempt,
                            "label": "EXPLORATORY: the decision a void attempt would have had"},
                 {"decision": decision, "exploratory_decisions": alternatives, "rows": rows}, t0)
    print(json.dumps({"void_attempt_decision": decision}, indent=1, default=str))
    return EXIT_OK


def _outcome_from_frozen(phase: str, attempt: int, qf: dict) -> int:
    """A frozen phase: its outcome files are written again if a reset came between the freeze and them."""
    return _outcome(phase, attempt, qf["results"]["integrity"], qf["results"]["decision"], C.quality_path(phase, attempt))


def _outcome(phase: str, attempt: int, integrity: dict, decision: dict | None, qpath: Path) -> int:
    """The files that follow from a frozen quality file: void record, selection, terminal. Idempotent."""
    if not integrity["ok"]:
        failed = [c["name"] for c in integrity["checks"] if not c["ok"]]
        return _void(phase, attempt, failed, [c for c in integrity["checks"] if not c["ok"]])
    d = decision["decision"]
    if phase == "p0b":
        sel = C.read_json(C.selection_path())
        if sel is None or sel.get("attempt") != attempt:
            C.write_json(C.selection_path(), {"decision": d, "cell": decision.get("arm"), "attempt": attempt,
                                              "per_arm": decision.get("per_arm"), "quality_file_sha256": I.sha256_file(qpath),
                                              "utc": C.utc()})
        if d == "k1_triggered":
            C.write_terminal("infill_failure_v2_K1", "K1_phase0b", 2, quality_file_sha256=I.sha256_file(qpath), attempt=attempt)
            return EXIT_TERMINAL
        return EXIT_OK
    if d == "fail":
        C.write_terminal("infill_failure_v2_gate", "main_quality", 2, quality_file_sha256=I.sha256_file(qpath), attempt=attempt)
        return EXIT_TERMINAL
    return EXIT_OK


# Score stage: accuracies from the stored tracker output, then the summary.

def _score_job(args) -> dict:
    npz, item = args
    z = np.load(npz)
    et, ef = z["et"], z["ef"]
    bp = F.array_to_notes(z["bp"])
    hyps = dict(item["hyps"]) | {"B2": item["B2"]}
    ref = {h: notes_in_gap(v) for h, v in hyps.items()}
    est = F.est_notes_in_gap(skyline(bp))
    return {"acc": {h: acc(et, ef, ref[h]) for h in item["hyps"]},
            "acc_oct": {h: acc(et, ef, ref[h], True) for h in item["hyps"]},
            "nf1": {h: F.note_f1(ref[h], est) for h in item["hyps"]},
            "of1": {h: F.note_f1(ref[h], est, onset_only=True) for h in hyps}}


def score(attempt: int = 1, diagnostic: bool = False, workers: int = 4, n_boot: int = S.N_BOOT) -> int:
    t0 = time.time()
    term = C.read_json(C.terminal_path())
    if term is not None and not diagnostic:
        print("terminal outcome exists:", term)
        return EXIT_TERMINAL
    if diagnostic and term is None:
        print("--diagnostic runs only after a terminal E5 outcome exists; run_all never reads its file")
        return EXIT_REFUSED
    od, design, events = _load_design("main", attempt)
    qpath = C.quality_path("main", attempt)
    qf = C.read_json(qpath)
    if qf is None:
        print("no frozen quality file")
        return EXIT_REFUSED
    if not qf["results"]["integrity"]["ok"] or qf["results"].get("decision") is None:
        print("the quality stage found an integrity failure: no tracker output and no decision to score")
        return EXIT_REFUSED
    meas = measurement_problems()
    if meas:
        print("score stage refused: measurement software differs from addendum 2:", meas)
        return EXIT_REFUSED
    qrows = qf["results"]["rows"]
    st = C.addendum_status()
    ok_rows = [r for r in qrows if r["status"] == "ok"]
    npz = {I.job_key(r): od / "features" / _npz_name(r) for r in ok_rows}
    man_path = C.e5_dir() / design["manifest"]
    checks = [I.check_registration(st["addendum"], st["current"], design, st["addendum_sha256"], events, st["current_data"]),
              I._res("manifest_as_designed", man_path.exists() and I.sha256_file(man_path) == design["manifest_sha256"]),
              I.check_ruler(C.read_json(C.ruler_gate_path(design["ruler_program"])), design),
              I._res("code_unchanged_since_quality", qf["config"]["code_sha256"] == st["current"]
                     and qf["config"].get("data_sha256") == st["current_data"]),
              I.check_wav_hashes(qrows, {I.job_key(r): I.sha256_file(od / I.audio_name(r)) for r in ok_rows
                                         if (od / I.audio_name(r)).exists()}),
              I._res("features_unchanged_since_quality",
                     all(npz[I.job_key(r)].exists() and I.sha256_file(npz[I.job_key(r)]) == r["npz_sha256"] for r in ok_rows))]
    out_path = C.results_dir() / ("e5v2_main_diagnostic.json" if diagnostic else "e5v2_main.json")
    if not all(c["ok"] for c in checks) and not diagnostic:
        failed = [c["name"] for c in checks if not c["ok"]]
        C.write_json(out_path, {"paper_level": "invalid_run", "stage": "score", "checks": checks})
        return _void("main", attempt, failed, checks)
    if not diagnostic and qf["results"]["decision"]["decision"] != "pass":
        print("the quality stage did not pass; nothing to score")
        return EXIT_REFUSED
    if n_boot != S.N_BOOT and design.get("scale") == "full" and not diagnostic:
        print(f"the registered estimator uses {S.N_BOOT} bootstrap resamples, not {n_boot}")
        return EXIT_REFUSED
    man = json.loads((C.e5_dir() / design["manifest"]).read_text())
    check_layout(man)
    items = {it["id"]: it for it in man["items"]}
    scored = _pool_map(_score_job, [(str(npz[I.job_key(r)]), items[r["id"]]) for r in ok_rows], workers)
    tau = C.read_json(C.ruler_gate_path(design["ruler_program"]))["results"]["summary"]["tau"]
    rows = []
    for r in qrows:
        r = dict(r)
        if r["status"] == "ok":
            r.update(scored.pop(0))
            not_scorable = sorted(h for h in r["nf1"] if r["nf1"][h] is None)
            if not_scorable:
                raise ValueError(f"{I.audio_name(r)}: hypotheses without scorable reference notes {not_scorable}")
            r["choice"] = S.classify(r["acc"], SETS[r["cond"]], tau)
        rows.append(r)
    first_ids = sorted({j["id"] for j in design["jobs"]})[:6]
    summ = S.summarize(rows, qf["results"]["decision"], tau, first_ids, n_boot=n_boot, n_boot_nested=n_boot,
                       diagnostic=diagnostic)
    cfg = {"experiment": "e5v2_main", "attempt": attempt, "scale": design.get("scale"), "tau": tau, "n_boot": n_boot,
           "diagnostic": diagnostic, "label": "EXPLORATORY: no claim" if diagnostic else "confirmatory" if design.get("scale") == "full"
           else f"{design.get('scale')} scale: a test of the code, no claim",
           "quality_file_sha256": I.sha256_file(qpath), "code_sha256": st["current"], "data_sha256": st["current_data"]}
    write_result(out_path, cfg, {"summary": summ, "rows": rows, "score_checks": checks}, t0)
    if not diagnostic:
        C.write_terminal(summ["paper_level"], "score", None, result_sha256=I.sha256_file(out_path),
                         K2=summ.get("K2", {}).get("status"), K3=summ.get("K3_status"), attempt=attempt)
    print(json.dumps({"paper_level": summ["paper_level"],
                      "verdicts": {h: v["verdict"] for h, v in summ.get("verdicts", {}).items()}}, indent=1))
    return EXIT_OK


# v1 regression: the v2 statistics reproduce every number the registered v1 scorer stored.

V1_SHA = {"e5_main.json": "63609991cfafde56ae43ae7783e814fb7a2b96df4062badf42bbb559be61f366",
          "e5_phase0.json": "e976c5fb31b512e3e7ed345aaf2700800f9325a2cb5fef3d15b39925a99e3383",
          "e5_ruler_gate.json": "26b21b2b71be808665ddbe53716e0fb62796b179bc931b1eac20378807779a6f"}


def regress_v1() -> int:
    v1 = C.repo_root() / "results" / "gpu_run1"
    for f, h in V1_SHA.items():
        if I.sha256_file(v1 / f) != h:
            print("fixture changed:", f)
            return 1
    d = json.loads((v1 / "e5_main.json").read_text())
    rows, summ = d["results"]["rows"], d["results"]["summary"]
    r4 = lambda x: round(float(x), 4)
    bad = []
    for name in ("CI_lit", "CI_ctl", "ANA", "RI+2", "RI+5", "RI-3", "LI+2", "LI+5", "LI-3"):
        ix = S.index(rows, *S.INDEX_SPEC[name], S.pop_all)
        st = summ["base"][name]
        if (r4(ix["mean"]), [r4(ix["lo"]), r4(ix["hi"])], ix["n_melodies"]) != (st["mean"], st["ci95"], st["n_melodies"]):
            bad.append(name)
    h2 = S.diff_index(rows, S.INDEX_SPEC["CI_lit"], S.INDEX_SPEC["CI_ctl"], S.pop_all)
    if (r4(h2["mean"]), r4(h2["lo"]), r4(h2["hi"])) != (0.1147, 0.0695, 0.1667):
        bad.append("H2")
    if any(S.classify(r["acc"], SETS[r["cond"]], summ["tau"], tie="first") != r["choice"] for r in rows):
        bad.append("choices")
    if sum(F.gate_flags({**r, "gap_rms_db_vs_context_v1": r["gap_rms_db_vs_context"]})["pass"]
           for r in rows if r["cond"] != "null") != 59:
        bad.append("registered gate")
    print("v1 regression:", "OK" if not bad else f"MISMATCH {bad}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--program", type=int, default=73)
    g.add_argument("--manifest", default="manifest_v2.json")
    g.add_argument("--ids", default="")
    g.add_argument("--out", default="")
    g.add_argument("--keep-renders", action="store_true")
    g.add_argument("--workers", type=int, default=1)
    q = sub.add_parser("quality")
    q.add_argument("phase", choices=["p0b", "main"])
    q.add_argument("--attempt", type=int, default=1)
    q.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    q.add_argument("--void-report", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("phase", choices=["main"])
    s.add_argument("--attempt", type=int, default=1)
    s.add_argument("--diagnostic", action="store_true")
    s.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    s.add_argument("--n-boot", type=int, default=S.N_BOOT)
    sub.add_parser("regress-v1")
    sub.add_parser("measurement")
    a = ap.parse_args(argv)
    if a.cmd == "gate":
        return gate(a.program, a.manifest, [x for x in a.ids.split(",") if x], a.out or None, not a.keep_renders, a.workers)
    if a.cmd == "quality":
        return quality(a.phase, a.attempt, a.workers, a.void_report)
    if a.cmd == "measurement":
        return measurement()
    if a.cmd == "score":
        return score(a.attempt, a.diagnostic, a.workers, a.n_boot)
    return regress_v1()


if __name__ == "__main__":
    sys.exit(main())
