"""E3 stimuli: audio with exact ground truth, to look for a timbre-invariant pitch subspace in
the ACE-Step 1.5 VAE latent space (25 Hz x 64).

  * scales/: for each of 30 General MIDI timbres, chromatic notes MIDI 48-79 (0.4 s on, 0.1 s off)
    at two velocities, split into 8 s clips -> frame labels are exact.
  * melodies/: 8 random 4-bar melodies on held-out timbres (incl. three voice-like patches) plus,
    for each, a target melody with the same rhythm but new pitches (the edit task).
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from dacapo.render import render
from dacapo.symbolic import Melody, Note, random_melody

HERE = Path(__file__).parent
TIMBRES = {  # GM program -> name
    0: "grand_piano", 4: "e_piano", 6: "harpsichord", 11: "vibraphone", 16: "organ", 21: "accordion",
    22: "harmonica", 24: "nylon_guitar", 25: "steel_guitar", 26: "jazz_guitar", 40: "violin", 42: "cello",
    48: "strings", 52: "choir_aahs", 53: "voice_oohs", 54: "synth_voice", 56: "trumpet", 57: "trombone",
    60: "french_horn", 65: "alto_sax", 66: "tenor_sax", 68: "oboe", 71: "clarinet", 73: "flute",
    75: "pan_flute", 79: "ocarina", 80: "square_lead", 81: "saw_lead", 85: "voice_lead", 89: "warm_pad",
}
HELDOUT = [52, 53, 54, 71, 42, 75]  # voice-like patches + three others never used to fit the subspace
PITCHES = list(range(48, 80))
BPM = 120.0  # 1 beat = 0.5 s


def scale_melodies(vel: int) -> list[Melody]:
    notes = [Note(p, i * 1.0, 0.8, vel) for i, p in enumerate(PITCHES)]  # 0.4 s on, 0.1 s off
    half = len(notes) // 2
    first = Melody(tuple(notes[:half]), BPM)
    second = Melody(tuple(replace(n, start=n.start - half) for n in notes[half:]), BPM)
    return [first, second]


def repitch(m: Melody, rng: np.random.Generator) -> Melody:
    """Same onsets and durations, new stepwise pitches in the same register."""
    other = random_melody(rng, bars=8, bpm=m.bpm, tonic=int(rng.choice([60, 62, 65, 67])))
    pitches = [n.pitch for n in other.notes][: len(m.notes)]
    while len(pitches) < len(m.notes):
        pitches.append(pitches[-1])
    return Melody(tuple(replace(n, pitch=p) for n, p in zip(m.notes, pitches)), m.bpm)


def main(seed: int = 3):
    rng = np.random.default_rng(seed)
    (HERE / "scales").mkdir(exist_ok=True)
    (HERE / "melodies").mkdir(exist_ok=True)
    items = []
    for prog, name in TIMBRES.items():
        for vel in (80, 110):
            for k, m in enumerate(scale_melodies(vel)):
                wav = HERE / "scales" / f"{name}_v{vel}_{k}.wav"
                render([m.to_instrument(prog, gap=0.0)], m.bpm, wav, sr=48000)
                items.append({"id": wav.stem, "wav": f"scales/{wav.name}", "program": prog, "timbre": name,
                              "heldout": prog in HELDOUT, "kind": "scale",
                              "melody": {"bpm": m.bpm, "notes": [asdict(n) for n in m.notes]}})
    for i in range(8):
        prog = HELDOUT[i % len(HELDOUT)]
        src = random_melody(rng, bars=4, bpm=100.0, tonic=int(rng.choice([60, 62, 65, 67])))
        tgt = repitch(src, rng)
        wav = HERE / "melodies" / f"mel{i}_{TIMBRES[prog]}.wav"
        render([src.to_instrument(prog)], src.bpm, wav, sr=48000)
        tgt_wav = HERE / "melodies" / f"mel{i}_{TIMBRES[prog]}_target.wav"  # reference only (not used by the edit)
        render([tgt.to_instrument(prog)], tgt.bpm, tgt_wav, sr=48000)
        items.append({"id": wav.stem, "wav": f"melodies/{wav.name}", "target_wav": f"melodies/{tgt_wav.name}",
                      "program": prog, "timbre": TIMBRES[prog], "heldout": True, "kind": "melody",
                      "melody": {"bpm": src.bpm, "notes": [asdict(n) for n in src.notes]},
                      "target": {"bpm": tgt.bpm, "notes": [asdict(n) for n in tgt.notes]}})
    (HERE / "manifest.json").write_text(json.dumps({"seed": seed, "heldout_programs": HELDOUT, "items": items}, indent=1))
    print(len(items), "clips")


if __name__ == "__main__":
    main()
