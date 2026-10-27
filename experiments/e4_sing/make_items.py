"""E4 items: melodies with exactly one note per (monosyllabic) word, plus wordless guide renders.

Each item has a 4-bar melody at 100 BPM after a 1-bar lead-in (A3-E5, major key, stepwise), one
lyric line whose words are all one syllable (so word = syllable = note), and FluidSynth guides of
the melody in two timbres: clarinet (no vocal quality) and Voice Oohs (vocal, single vowel).
Items 0-2 are development items (for choosing settings); 3-11 are held out.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dacapo.render import render
from dacapo.symbolic import Melody, Note

HERE = Path(__file__).parent
BPM = 100.0
LEAD_IN = 4.0  # beats
LINES = [
    "I walk down to the sea and watch the light go slow",
    "The rain falls on the road and all the town is still",
    "We ran past the old mill where the white birds sing",
    "Hold my hand and stay with me till the night is done",
    "The cold wind blows through trees and calls my name at dawn",
    "Bright stars burn in the dark and fade when we wake up",
    "She sang a song of home with a soft and clear voice",
    "Let the boat drift on the lake as the sun sinks low",
    "Our hearts beat as one when the bells ring in the square",
    "Give me one more day to see the green fields bloom",
    "The lamp glows warm and we sit and tell the tales of old",
    "Snow fell on the roofs and the whole world turned white",
]
GUIDES = {"clarinet": 71, "oohs": 53}
KEYS = {"C major": 60, "D major": 62, "F major": 65, "G major": 67}
SCALE = [0, 2, 4, 5, 7, 9, 11]


def rhythm(n: int, rng: np.random.Generator) -> list[float]:
    """n durations (beats) summing to 16: start from quarters, merge random neighbours."""
    d = [1.0] * 16
    while len(d) > n:
        i = int(rng.integers(0, len(d) - 1))
        d[i:i + 2] = [d[i] + d[i + 1]]
    return d


def pitches(n: int, tonic: int, rng: np.random.Generator) -> list[int]:
    """Stepwise walk over the in-key notes of A3-E5 (reflecting at the ends, no octave folding)."""
    lo, hi = 57, 76  # comfortable for a female voice (or a male voice an octave down)
    tones = [p for p in range(lo, hi + 1) if (p - tonic) % 12 in SCALE]
    i, out = int(np.argmin([abs(p - (tonic + 7 if tonic + 7 <= hi else tonic)) for p in tones])), []
    for _ in range(n):
        i += int(rng.choice([-2, -1, 1, 2, 0, 3, -3], p=[.15, .3, .3, .1, .05, .05, .05]))
        if i < 0:
            i = -i
        if i >= len(tones):
            i = 2 * (len(tones) - 1) - i
        out.append(int(tones[i]))
    return out


def main(seed: int = 21):
    rng = np.random.default_rng(seed)
    (HERE / "guides").mkdir(exist_ok=True)
    items = []
    for i, line in enumerate(LINES):
        words = line.split()
        key = list(KEYS)[i % len(KEYS)]
        durs = rhythm(len(words), rng)
        ps = pitches(len(words), KEYS[key], rng)
        notes, t = [], LEAD_IN
        for p, d in zip(ps, durs):
            notes.append(Note(p, t, d * 0.9 if d >= 1 else d * 0.95))
            t += d
        m = Melody(tuple(notes), BPM)
        guides = {}
        for g, prog in GUIDES.items():
            wav = HERE / "guides" / f"item{i:02d}_{g}.wav"
            render([m.to_instrument(prog)], BPM, wav, sr=48000)
            guides[g] = f"guides/{wav.name}"
        items.append({"id": f"item{i:02d}", "split": "dev" if i < 3 else "test", "key": key, "words": words,
                      "lyrics": "[Verse]\n" + line, "melody": {"bpm": BPM, "notes": [asdict(n) for n in m.notes]},
                      "guides": guides, "duration": float(np.ceil((LEAD_IN + 16 + 2) * 60 / BPM))})
    (HERE / "manifest.json").write_text(json.dumps({"seed": seed, "bpm": BPM, "items": items}, indent=1))
    print(len(items), "items")


if __name__ == "__main__":
    main()
