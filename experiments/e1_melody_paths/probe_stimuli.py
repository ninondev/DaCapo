"""E1b stimuli: short known-melody clips for a tokenizer-only probe of ACE-Step's 5 Hz codes.

Factorial design, to see what the codes are sensitive to:
  melody (24 random 4-bar melodies) x lead timbre (piano / flute / violin), all with the same band,
  plus changed versions of 6 melodies: +2 semitone transposition, 2 wrong notes, 200 ms delay.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dacapo.render import GM, render_melody
from dacapo.symbolic import random_melody

HERE = Path(__file__).parent
TIMBRES = ["piano", "flute", "violin"]


def main(n: int = 24, bpm: float = 120.0, seed: int = 11):
    rng = np.random.default_rng(seed)
    out = HERE / "probe_stimuli"
    out.mkdir(exist_ok=True)
    items = []

    def add(m, mid, timbre, variant):
        sid = f"p{mid:02d}_{timbre}_{variant}"
        wav = out / f"{sid}.wav"
        render_melody(m, wav, program=GM[timbre], accomp_vel=60, drums=True, sr=48000)
        items.append({"id": sid, "wav": wav.name, "melody_id": mid, "timbre": timbre, "variant": variant,
                      "melody": {"bpm": m.bpm, "notes": [asdict(x) for x in m.notes]}})

    for i in range(n):
        m = random_melody(rng, bars=4, bpm=bpm).shift(2.0)  # 2 beats of band-only lead-in
        for t in TIMBRES:
            add(m, i, t, "orig")
        if i < 6:
            add(m.transpose(2), i, "piano", "transp+2")
            idx = sorted(rng.choice(len(m.notes), size=2, replace=False).tolist())
            add(m.with_pitch_errors(idx, [1, -1]), i, "piano", "2wrong")
            add(m.shift(0.2 * bpm / 60.0), i, "piano", "delay200ms")
    (out / "manifest.json").write_text(json.dumps({"seed": seed, "bpm": bpm, "items": items}, indent=1))
    print(f"{len(items)} clips -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="number of melodies")
    a = ap.parse_args()
    main(n=a.n)
