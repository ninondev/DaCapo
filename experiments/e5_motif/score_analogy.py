"""E5 scoring (main environment). Per-note pitch accuracy in the gap against every hypothesis.

  python score_analogy.py gate              # ruler gate on exact hypothesis renders (no DiT)
  python score_analogy.py score <outdir>    # score repaint outputs, write results/e5_<outdir>.json

Each hypothesis is a note list starting at the gap (12 s). Notes inside the 0.4 s gap margins are
excluded (latent crossfade). Every hypothesis gets the same symmetric lag search (+-120 ms),
so foils and rule answers are treated the same.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from dacapo.analysis import f0_swift, notes_basic_pitch
from dacapo.metrics import note_pitch_accuracy
from dacapo.provenance import write_result

HERE = Path(__file__).parent
MAN = json.loads((HERE / "manifest.json").read_text())
SPB = 60.0 / MAN["bpm"]
G0, G1 = MAN["gap_s"]
MARGIN = 0.4
LAGS = np.arange(-0.12, 0.1201, 0.02)
SETS = {
    "lit": ["B", "B~", "A", "D"],
    "ctl": ["B", "B~", "C", "D"],
    "rel+2": ["T+2B", "T+2B~", "B", "B~", "T+2A", "D", "T+1B", "T+3B"],
    "rel+5": ["T+5B", "T+5B~", "B", "B~", "T+5A", "D", "T+4B", "T+6B"],
    "rel-3": ["T-3B", "T-3B~", "B", "B~", "T-3A", "D", "T-4B", "T-2B"],
    "seq": ["T+6A", "T+4A", "T+2A", "T+5A", "T+7A", "D"],
    "ana": ["T+5B", "T+5B~", "B", "T+5A", "D"],
    "null": ["B", "B~", "A", "D"],
}


def notes_in_gap(note_dicts: list[dict]) -> list[tuple[float, float, int]]:
    out = []
    for n in note_dicts:
        s, e = G0 + n["start"] * SPB, G0 + (n["start"] + n["dur"]) * SPB
        if s >= G0 + MARGIN and e <= G1 - MARGIN + 0.25:
            out.append((s, min(e, G1 - MARGIN), n["pitch"]))
    return out


def acc(et, ef, notes, octave_tolerant=False) -> float:
    return max(note_pitch_accuracy(notes, et, ef, lag=lag, octave_tolerant=octave_tolerant)["npa"] for lag in LAGS)


def gap_quality(wav: Path, et, ef) -> dict:
    from dacapo.render import load_mono

    x, sr = load_mono(wav)
    seg = lambda a, b: x[int(a * sr):int(b * sr)]
    rms = lambda y: float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    g = (et >= G0 + MARGIN) & (et < G1 - MARGIN)
    notes = [n for n in notes_basic_pitch(wav) if G0 + MARGIN <= n[0] < G1 - MARGIN]
    grid = np.arange(G0 + MARGIN, G1 - MARGIN, 0.05)
    poly = float(np.mean([sum(s <= t < e for s, e, _ in notes) for t in grid])) if notes else 0.0
    return {"gap_voiced": float((ef[g] > 0).mean()), "gap_polyphony": poly,
            "gap_rms_db_vs_context": 20 * np.log10(rms(seg(G0 + MARGIN, G1 - MARGIN)) / rms(seg(0, G0)))}


def score_wav(wav: Path, item: dict, cond: str) -> dict:
    et, ef = f0_swift(wav)
    accs = {h: acc(et, ef, notes_in_gap(item["hyps"][h])) for h in SETS[cond]}
    accs_oct = {h: acc(et, ef, notes_in_gap(item["hyps"][h]), True) for h in SETS[cond]}
    return {"acc": accs, "acc_oct": accs_oct, **gap_quality(wav, et, ef)}


def classify(accs: dict, tau: float) -> str:
    h, a = max(accs.items(), key=lambda kv: kv[1])
    return h if a >= tau else "none"


def gate():
    """Ruler gate: each exact hypothesis render must be classified as itself (in every condition's
    hypothesis set that contains it) at >= 95%, and 1 or 2 wrong notes must be detectable."""
    from dacapo.render import render
    from dacapo.symbolic import Melody, Note

    t0 = time.time()
    res = {"per_condition": {}, "tau_candidates": {}}
    rows = []
    for it in MAN["items"]:
        for h, rel in it["hyp_renders"].items():
            et, ef = f0_swift(HERE / rel)
            for cond, hs in SETS.items():
                if h in hs:
                    accs = {x: acc(et, ef, notes_in_gap(it["hyps"][x])) for x in hs}
                    rows.append({"id": it["id"], "cond": cond, "true": h, "accs": accs})
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
        res["tau_candidates"][tau] = round(float(np.mean([classify(r["accs"], tau) == r["true"] for r in rows])), 4)
    tau = max(t for t, v in res["tau_candidates"].items() if v >= max(res["tau_candidates"].values()) - 1e-9)
    for cond in SETS:
        sel = [r for r in rows if r["cond"] == cond]
        res["per_condition"][cond] = round(float(np.mean([classify(r["accs"], tau) == r["true"] for r in sel])), 4)
    res["tau"] = tau
    res["overall"] = res["tau_candidates"][tau]
    res["self_acc_mean"] = round(float(np.mean([r["accs"][r["true"]] for r in rows])), 4)

    # Wrong-note sensitivity: AUC of acc(B) for correct vs k-wrong renders of B in the gap.
    tmp = HERE / "hyp_renders" / "_wrong"
    tmp.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    aucs = {}
    for k in (1, 2):
        good, bad = [], []
        for it in MAN["items"]:
            B = Melody(tuple(Note(**n) for n in it["hyps"]["B"]), MAN["bpm"])
            et, ef = f0_swift(HERE / it["hyp_renders"]["B"])
            good.append(acc(et, ef, notes_in_gap(it["hyps"]["B"])))
            idx = sorted(rng.choice(len(B.notes), size=k, replace=False).tolist())
            W = B.with_pitch_errors(idx, rng.choice([-2, -1, 1, 2], size=k).tolist())
            full = Melody(tuple(W.shift(3 * MAN["unit_beats"]).notes), MAN["bpm"])
            wav = tmp / f"{it['id']}_B_{k}wrong.wav"
            render([full.to_instrument(MAN["lead_program"])], MAN["bpm"], wav, sr=48000)
            et, ef = f0_swift(wav)
            bad.append(acc(et, ef, notes_in_gap(it["hyps"]["B"])))
        g, b = np.array(good), np.array(bad)
        aucs[f"{k}_wrong"] = round(float((g[:, None] > b[None]).mean() + 0.5 * (g[:, None] == b[None]).mean()), 4)
    res["wrong_note_auc"] = aucs
    res["passed"] = bool(res["overall"] >= 0.95 and min(aucs.values()) >= 0.9)
    write_result(HERE / "results" / "e5_gate.json", {"experiment": "e5_ruler_gate", "lags": LAGS.tolist(), "margin_s": MARGIN},
                 {"summary": res, "rows": rows}, t0)
    print(json.dumps(res, indent=1))


def score(outdir: str):
    t0 = time.time()
    gate_res = json.loads((HERE / "results" / "e5_gate.json").read_text())["results"]["summary"]
    tau = gate_res["tau"]
    items = {it["id"]: it for it in MAN["items"]}
    rows = []
    for wav in sorted((HERE / outdir).glob("*.wav")):
        mid, cond, arm, seed = wav.stem.split("__")
        r = {"id": mid, "split": items[mid]["split"], "cond": cond, "arm": arm, "seed": seed}
        r.update(score_wav(wav, items[mid], cond))
        r["choice"] = classify(r["acc"], tau)
        rows.append(r)
        print(f"{wav.stem:>40s} -> {r['choice']:>6s}  " + " ".join(f"{h}:{a:.2f}" for h, a in r["acc"].items()), flush=True)

    def index(cond, pos, neg, arm="base"):
        sel = [r for r in rows if r["cond"] == cond and r["arm"] == arm and r["split"] == "confirm"]
        if not sel:
            return None
        by_mel = {}
        for r in sel:
            by_mel.setdefault(r["id"], []).append(r["acc"][pos] - r["acc"][neg])
        vals = np.array([np.mean(v) for v in by_mel.values()])
        rng = np.random.default_rng(0)
        boots = [rng.choice(vals, len(vals)).mean() for _ in range(10000)]
        return {"mean": round(float(vals.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                                              round(float(np.percentile(boots, 97.5)), 4)], "n_melodies": len(vals)}

    summ = {"tau": tau}
    for arm in sorted({r["arm"] for r in rows}):
        s = {"CI_lit": index("lit", "B", "B~", arm), "CI_ctl": index("ctl", "B", "B~", arm),
             "ANA": index("ana", "T+5B", "T+5B~", arm)}
        for k in ("+2", "+5", "-3"):
            s[f"RI{k}"] = index(f"rel{k}", f"T{k}B", f"T{k}B~", arm)
            s[f"LI{k}"] = index(f"rel{k}", "B", "B~", arm)
        for cond in SETS:
            sel = [r for r in rows if r["cond"] == cond and r["arm"] == arm and r["split"] == "confirm"]
            if sel:
                s[f"choices_{cond}"] = {h: sum(r["choice"] == h for r in sel) for h in SETS[cond] + ["none"]}
                s[f"quality_{cond}"] = {q: round(float(np.mean([r[q] for r in sel])), 3)
                                        for q in ("gap_voiced", "gap_polyphony", "gap_rms_db_vs_context")}
        summ[arm] = s
    write_result(HERE / "results" / f"e5_{outdir}.json", {"experiment": "e5_analogy", "outdir": outdir, "tau": tau},
                 {"summary": summ, "rows": rows}, t0)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"gate": lambda: gate(), "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
