"""Render E1 stimuli: known 8-bar melodies (lead + simple band) at several tempi (--tempi).

Tempo is the point: at 150 BPM an eighth note lasts 200 ms, exactly one 5 Hz code frame.
If the codes cannot carry pitch changes inside a frame, melody fidelity should drop at the fast tempo.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dacapo.render import GM, render_melody
from dacapo.symbolic import random_melody

HERE = Path(__file__).parent
CAPTION = {
    "piano": "instrumental pop, clear piano lead melody, soft block-chord piano, bass guitar, simple rock drums",
    "flute": "instrumental pop, clear flute lead melody, soft block-chord piano, bass guitar, simple rock drums",
}
REORCH = "cinematic instrumental, solo violin lead melody over a warm string orchestra, no drums"


def main(n_melodies: int = 4, tempi=(90, 150), seed: int = 7):
    rng = np.random.default_rng(seed)
    out = HERE / "stimuli"
    out.mkdir(exist_ok=True)
    stimuli = []
    for i in range(n_melodies):
        base = random_melody(rng, bars=8, bpm=100)
        lead = "piano" if i % 2 == 0 else "flute"
        for bpm in tempi:
            m = type(base)(base.notes, float(bpm))
            sid = f"m{i}_{lead}_{bpm}"
            wav = out / f"{sid}.wav"
            render_melody(m, wav, program=GM[lead], accomp_vel=60, drums=True, sr=48000)
            dur = float(np.ceil(m.length_beats * m.spb + 1.0))
            stimuli.append({"id": sid, "wav": wav.name, "duration": max(10.0, dur), "bpm": bpm,
                            "lead": lead, "caption": f"{CAPTION[lead]}, {bpm} bpm", "reorch_caption": f"{REORCH}, {bpm} bpm",
                            "melody": {"bpm": m.bpm, "notes": [asdict(n) for n in m.notes]},
                            "min_note_ms": round(min(n.dur for n in m.notes) * m.spb * 1000)})
    (out / "manifest.json").write_text(json.dumps({"seed": seed, "stimuli": stimuli}, indent=1))
    print(f"{len(stimuli)} stimuli -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of melodies")
    ap.add_argument("--tempi", default="90,150")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(n_melodies=a.n, tempi=tuple(int(x) for x in a.tempi.split(",")), seed=a.seed)
