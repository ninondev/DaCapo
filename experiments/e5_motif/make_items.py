"""E5 stimuli: in-context motif tests for a song DiT, with exact symbolic answers.

Source layout (120 BPM, 1 unit = 2 bars = 4 s = 100 latent frames):
    [U1][U2][U3][  gap 12-16 s, silent  ][D (1 bar)]   -> 18 s, repainting_start/end = 12/16
Conditions (U1 U2 U3):
    lit      A  B  A          rule answer: B            (A B A -> B)
    ctl      A  B  C          lag control: no rule; B would be pure 8-s lag copying
    rel+2/+5/-3  A  B  T_k(A)  rule answer: T_k(B)      (transposition-relational induction)
    seq      A  T2(A)  T4(A)  rule answer: T6(A)        (sequence extrapolation)
    ana      A  T5(A)  B      rule answer: T5(B)        (analogy A : A' :: B : ?)
    null     -  -  -          no context (chance for every hypothesis)
Every hypothesis is an exact symbolic object (rhythm + pitches), also rendered for calibration.
B~ is a derangement of B's pitches (same rhythm and pitch multiset, >= 75% positions differ).
Melodies 0-1 are pilot items; 2-13 are confirmatory.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from dacapo.render import render
from dacapo.symbolic import Melody, Note, random_melody

HERE = Path(__file__).parent
BPM, UNIT, GAP_START, GAP_END = 120.0, 8.0, 12.0, 16.0  # UNIT in beats (2 bars); gap in seconds
FLUTE = 73
CONDITIONS = ["lit", "ctl", "rel+2", "rel+5", "rel-3", "seq", "ana", "null"]


def unit(rng: np.random.Generator, tonic: int) -> Melody:
    while True:
        m = random_melody(rng, bars=2, bpm=BPM, tonic=tonic)
        if 6 <= len(m.notes) <= 12 and len(set(m.pitches())) >= 4:
            return m


def derange(m: Melody, rng: np.random.Generator) -> Melody:
    p = m.pitches()
    for _ in range(2000):
        q = list(rng.permutation(p))
        if np.mean([a != b for a, b in zip(p, q)]) >= 0.75:
            return Melody(tuple(replace(n, pitch=int(x)) for n, x in zip(m.notes, q)), m.bpm)
    raise RuntimeError("no derangement")


def differ(a: Melody, b: Melody) -> bool:
    """Pitch sequences differ at >= 75% of aligned positions (or rhythms differ)."""
    pa, pb = a.pitches(), b.pitches()
    if len(pa) != len(pb) or [n.start for n in a.notes] != [n.start for n in b.notes]:
        return True
    return np.mean([x != y for x, y in zip(pa, pb)]) >= 0.75


def at(m: Melody, beats: float) -> Melody:
    return m.shift(beats)


def build(rng: np.random.Generator, idx: int) -> dict:
    tonic = int(rng.choice([67, 69, 72]))
    while True:
        A, B, C, B2 = (unit(rng, tonic) for _ in range(4))
        D = Melody(tuple(n for n in unit(rng, tonic).notes if n.start < 4.0), BPM)  # 1 bar
        Bt = derange(B, rng)
        hyps = {"B": B, "B~": Bt, "A": A, "C": C, "D": D}
        for k in (2, 5, -3, 4, 6, 7, 1, 3, -2, -4):
            hyps[f"T{k:+d}B"], hyps[f"T{k:+d}B~"], hyps[f"T{k:+d}A"] = B.transpose(k), Bt.transpose(k), A.transpose(k)
        core = [B, Bt, A, C, D, B.transpose(2), B.transpose(5), B.transpose(-3), A.transpose(6), A.transpose(4)]
        if all(differ(x, y) for i, x in enumerate(core) for y in core[i + 1:]):
            break
    contexts = {"lit": [A, B, A], "ctl": [A, B, C], "rel+2": [A, B, A.transpose(2)], "rel+5": [A, B, A.transpose(5)],
                "rel-3": [A, B, A.transpose(-3)], "seq": [A, A.transpose(2), A.transpose(4)],
                "ana": [A, A.transpose(5), B], "null": []}
    return {"A": A, "B": B, "C": C, "D": D, "B2": B2, "hyps": hyps, "contexts": contexts, "tonic": tonic}


def source(ctx: list[Melody], D: Melody) -> Melody:
    notes = []
    for i, u in enumerate(ctx):
        notes += list(at(u, i * UNIT).notes)
    notes += list(at(D, 4 * UNIT).notes)
    return Melody(tuple(notes), BPM)


def main(n: int = 14, seed: int = 101):
    rng = np.random.default_rng(seed)
    (HERE / "sources").mkdir(exist_ok=True)
    (HERE / "hyp_renders").mkdir(exist_ok=True)
    items = []
    for i in range(n):
        b = build(rng, i)
        mid = f"mel{i:02d}"
        conds = {}
        for c in CONDITIONS:
            src = source(b["contexts"][c], b["D"])
            wav = HERE / "sources" / f"{mid}_{c}.wav"
            render([src.to_instrument(FLUTE)], BPM, wav, sr=48000)
            conds[c] = f"sources/{wav.name}"
        # Calibration renders: each hypothesis sits in the gap of the 'null' layout.
        hyp_wavs = {}
        for h, m in b["hyps"].items():
            full = Melody(tuple(at(m, 3 * UNIT).notes) + tuple(at(b["D"], 4 * UNIT).notes), BPM)
            wav = HERE / "hyp_renders" / f"{mid}_{h.replace('~', 'x')}.wav"
            render([full.to_instrument(FLUTE)], BPM, wav, sr=48000)
            hyp_wavs[h] = f"hyp_renders/{wav.name}"
        items.append({"id": mid, "split": "pilot" if i < 2 else "confirm", "tonic": b["tonic"], "sources": conds,
                      "hyp_renders": hyp_wavs,
                      "hyps": {h: [asdict(x) for x in m.notes] for h, m in b["hyps"].items()},
                      "B2": [asdict(x) for x in b["B2"].notes]})
    meta = {"seed": seed, "bpm": BPM, "unit_beats": UNIT, "gap_s": [GAP_START, GAP_END], "duration_s": 18.0,
            "lead_program": FLUTE, "conditions": CONDITIONS, "items": items}
    (HERE / "manifest.json").write_text(json.dumps(meta, indent=1))
    print(n, "melodies x", len(CONDITIONS), "conditions")


if __name__ == "__main__":
    main()
