"""E0: calibrate the ruler.

Question: if a model follows the target melody with accuracy Y, how much of Y is measurement
error? Here the ground truth is exact (FluidSynth renders of known MIDI), so every error comes
from the analysis chain.

Part A: accuracy of each extractor across timbres, accompaniment levels and reverb.
Part B: sensitivity. Can the metric tell a melody with k wrong notes from the correct one,
        under the same acoustic conditions?

Usage: python experiments/e0_ruler/run.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from dacapo.analysis import chroma_argmax, f0, notes_basic_pitch, skyline
from dacapo.metrics import frame_metrics, note_metrics, pitch_class_accuracy
from dacapo.provenance import write_result
from dacapo.render import GM, render_melody
from dacapo.symbolic import melody_from_spec, random_melody

OUT = Path(__file__).parent / "results"
F0 = ("swift", "fcpe")  # CREPE (full) ran at ~0.25x real time on CPU; see LOG.md (E0)

ODE = melody_from_spec([(64, 1), (64, 1), (65, 1), (67, 1), (67, 1), (65, 1), (64, 1), (62, 1),
                        (60, 1), (60, 1), (62, 1), (64, 1), (64, 1.5), (62, 0.5), (62, 2)], bpm=100).transpose(7)


def conditions(quick: bool) -> list[dict]:
    solo = ["piano", "flute", "violin", "trumpet", "saw_lead", "alto_sax", "choir_aahs"]
    accomp_leads = ["piano", "saw_lead", "choir_aahs"]
    if quick:
        solo, accomp_leads = ["piano", "choir_aahs"], ["choir_aahs"]
    c = [{"lead": t, "accomp_vel": None, "reverb": False} for t in solo]
    c += [{"lead": t, "accomp_vel": v, "reverb": False} for t in accomp_leads for v in (50, 70, 90)]
    c += [{"lead": t, "accomp_vel": v, "reverb": True} for t, v in [("piano", None), ("choir_aahs", 70)]]
    return c


def label(c: dict) -> str:
    acc = "solo" if c["accomp_vel"] is None else f"acc{c['accomp_vel']}"
    return f"{c['lead']}/{acc}{'/rev' if c['reverb'] else ''}"


def measure(m_ref, m_played, c: dict, td: Path) -> dict:
    wav = render_melody(m_played, td / "x.wav", program=GM[c["lead"]], accomp_vel=c["accomp_vel"], reverb=c["reverb"])
    rt, rf = m_ref.f0_track()
    iv, hz = m_ref.note_events_seconds()
    notes = notes_basic_pitch(wav)
    ct, cpc = chroma_argmax(wav)
    out = {f: frame_metrics(rt, rf, *f0(wav, f)) for f in F0}
    out["chroma"] = pitch_class_accuracy(rt, rf, ct, cpc)
    out["bp_all"] = note_metrics(iv, hz, notes)
    out["bp_skyline"] = note_metrics(iv, hz, skyline(notes))
    return out


def part_a(melodies, conds, td) -> list[dict]:
    rows = []
    for c in conds:
        for i, m in enumerate(melodies):
            r = measure(m, m, c, td)
            rows.append({"cond": label(c), "melody": i, **{f"{k}.{kk}": vv for k, v in r.items() for kk, vv in v.items()}})
        print(f"[A] {label(c):28s} RPA={np.mean([r['swift.RPA'] for r in rows if r['cond'] == label(c)]):.3f} "
              f"skylineF={np.mean([r['bp_skyline.F'] for r in rows if r['cond'] == label(c)]):.3f}", flush=True)
    return rows


def part_b(melodies, conds, ks, rng, td) -> list[dict]:
    rows = []
    for c in conds:
        for i, m in enumerate(melodies):
            for k in ks:
                idx = sorted(rng.choice(len(m.notes), size=k, replace=False).tolist()) if k else []
                sem = rng.choice([-1, 1], size=k).tolist() if k else []
                played = m.with_pitch_errors(idx, sem)
                r = measure(m, played, c, td)  # scored against the intended melody, not the played one
                rows.append({"cond": label(c), "melody": i, "k": k, "n_notes": len(m.notes),
                             **{f"{f}.RPA": r[f]["RPA"] for f in F0}, "chroma.pc_acc": r["chroma"]["pc_acc"],
                             "bp_skyline.F": r["bp_skyline"]["F"]})
        print(f"[B] {label(c)} done", flush=True)
    return rows


def auc(neg: list[float], pos: list[float]) -> float:
    """P(score(correct) > score(k-error)); ties count half. 1.0 = perfectly separable."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def summarize(a_rows, b_rows) -> dict:
    summ = {"A": {}, "B": {}}
    for cond in dict.fromkeys(r["cond"] for r in a_rows):
        rs = [r for r in a_rows if r["cond"] == cond]
        keys = [f"{f}.{m}" for f in F0 for m in ("RPA", "RCA", "VR", "VFA")] + ["chroma.pc_acc", "bp_all.P", "bp_all.R", "bp_skyline.F"]
        summ["A"][cond] = {k: round(float(np.mean([r[k] for r in rs])), 3) for k in keys}
    for cond in dict.fromkeys(r["cond"] for r in b_rows):
        rs = [r for r in b_rows if r["cond"] == cond]
        ks = sorted({r["k"] for r in rs})
        metrics = [f"{f}.RPA" for f in F0] + ["chroma.pc_acc", "bp_skyline.F"]
        base = [r for r in rs if r["k"] == 0]
        summ["B"][cond] = {}
        for k in ks:
            sel = [r for r in rs if r["k"] == k]
            entry = {m: round(float(np.mean([r[m] for r in sel])), 3) for m in metrics}
            if k:  # does the metric rank correct takes above k-error takes? All pairs, pooled over melodies.
                entry.update({f"AUC.{m}": round(auc([r[m] for r in sel], [r[m] for r in base]), 3) for m in metrics})
            summ["B"][cond][f"k={k}"] = entry
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = {"experiment": "e0_ruler", "quick": args.quick, "seed": 0, "n_random": 2 if args.quick else 8,
           "bars": 4, "bpm": 100, "ks": [0, 1, 2, 4], "f0_methods": list(F0), "chroma_fmin_hz": 250,
           "onset_tol_s": 0.05, "pitch_tol_cents": 50, "max_lag_s": 0.3}
    rng = np.random.default_rng(cfg["seed"])
    melodies = [ODE] + [random_melody(rng, bars=cfg["bars"], bpm=cfg["bpm"]) for _ in range(cfg["n_random"])]
    conds = conditions(args.quick)
    b_conds = [c for c in conds if label(c) in {"piano/solo", "choir_aahs/solo", "choir_aahs/acc70", "saw_lead/acc90", "choir_aahs/acc90"}]
    with tempfile.TemporaryDirectory() as td:
        a_rows = part_a(melodies, conds, Path(td))
        b_rows = part_b(melodies[: 3 if args.quick else 6], b_conds, cfg["ks"], np.random.default_rng(1), Path(td))
    summ = summarize(a_rows, b_rows)
    name = "e0_quick.json" if args.quick else "e0_full.json"
    write_result(OUT / name, cfg, {"summary": summ, "rows_A": a_rows, "rows_B": b_rows}, t0)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
