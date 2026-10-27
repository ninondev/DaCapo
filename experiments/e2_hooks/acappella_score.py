"""Score the a cappella pilot: is the output monophonic singing that SwiftF0 can track?

Signals (all independent of ACE-Step):
  voiced_frac   fraction of 10 ms frames SwiftF0 calls voiced (singing should be mostly voiced)
  low_energy    fraction of spectral energy below 120 Hz (bass/kick drum indicator)
  polyphony     mean number of simultaneous Basic Pitch notes (accompaniment indicator)
  f0_range      5th-95th percentile of voiced f0 in semitones (a sung melody spans several)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from dacapo.analysis import f0_swift, notes_basic_pitch
from dacapo.render import load_mono

HERE = Path(__file__).parent


def stats(wav: Path) -> dict:
    import librosa

    x, sr = load_mono(wav)
    S = np.abs(librosa.stft(x, n_fft=4096, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    _, f0 = f0_swift(wav)
    v = f0[f0 > 0]
    notes = notes_basic_pitch(wav)
    grid = np.arange(0, len(x) / sr, 0.05)
    poly = np.mean([sum(s <= g < e for s, e, _ in notes) for g in grid]) if notes else 0.0
    semis = 12 * np.log2(v / 440.0) if len(v) else np.array([0.0])
    return {"voiced_frac": round(float((f0 > 0).mean()), 3),
            "low_energy": round(float(S[freqs < 120].sum() / S.sum()), 4),
            "polyphony": round(float(poly), 2),
            "f0_range_semitones": round(float(np.percentile(semis, 95) - np.percentile(semis, 5)), 1),
            "median_f0_hz": round(float(np.median(v)), 1) if len(v) else None}


def main(folder: str = "outputs_acappella"):
    d = HERE / folder
    log = json.loads((d / "log.json").read_text())
    rows = []
    for r in log["runs"]:
        s = stats(d / r["wav"])
        rows.append({**r, **s})
        print(r["key"], r["seed"], s, flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{folder}_stats.json").write_text(json.dumps({"settings": log["settings"], "rows": rows}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
