"""Deterministic MIDI -> audio rendering (FluidSynth) and simple accompaniment stimuli.

We know every note in these renders exactly, so we use them to calibrate the analysis chain
before we use it on generated music.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from .symbolic import Melody

_SF2_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # Debian/Ubuntu: apt install fluid-soundfont-gm
    "/opt/homebrew/share/soundfonts/default.sf2",
    "/usr/local/share/soundfonts/default.sf2",
]


def default_sf2() -> Path:
    """$DACAPO_SF2 if set, else the first General MIDI soundfont found in common locations."""
    env = os.environ.get("DACAPO_SF2")
    for c in ([env] if env else []) + _SF2_CANDIDATES:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("No GM soundfont found; download one (e.g. FluidR3_GM.sf2) and set DACAPO_SF2=/path/to/file.sf2")


GM = {"piano": 0, "violin": 40, "trumpet": 56, "alto_sax": 65, "flute": 73, "saw_lead": 81, "choir_aahs": 52}


def accompaniment(length_beats: float, bpm: float, root: int = 55, vel: int = 70, drums: bool = True) -> list[pretty_midi.Instrument]:
    """I-V-vi-IV block chords (one per bar) + root bass + a basic rock beat."""
    spb = 60.0 / bpm
    # offsets keep the voicing compact around the root (G3 by default: chords span ~D3-G4)
    prog = [(0, [0, 4, 7]), (-5, [0, 4, 7]), (-3, [0, 3, 7]), (5, [0, 4, 7])]
    keys = pretty_midi.Instrument(program=GM["piano"])
    bass = pretty_midi.Instrument(program=33)
    kit = pretty_midi.Instrument(program=0, is_drum=True)
    bars = int(np.ceil(length_beats / 4))
    for b in range(bars):
        off, triad = prog[b % 4]
        s, e = b * 4 * spb, (b + 1) * 4 * spb - 0.02
        for iv in triad:
            keys.notes.append(pretty_midi.Note(vel, root + off + iv, s, e))
        for beat in range(4):
            bs = s + beat * spb
            bass.notes.append(pretty_midi.Note(vel, root - 12 + off - (12 if off > 0 else 0), bs, bs + spb * 0.9))
            if drums:
                kit.notes.append(pretty_midi.Note(vel, 36 if beat % 2 == 0 else 38, bs, bs + 0.1))
                kit.notes.append(pretty_midi.Note(int(vel * 0.7), 42, bs, bs + 0.05))
                kit.notes.append(pretty_midi.Note(int(vel * 0.6), 42, bs + spb / 2, bs + spb / 2 + 0.05))
    return [keys, bass] + ([kit] if drums else [])


def render(instruments: list[pretty_midi.Instrument], bpm: float, out_wav: str | Path, sr: int = 22050,
           reverb: bool = False, sf2: Path | None = None) -> Path:
    if shutil.which("fluidsynth") is None:
        raise RuntimeError("fluidsynth not found; install it (apt install fluidsynth fluid-soundfont-gm / brew install fluid-synth)")
    sf2 = sf2 or default_sf2()
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    pm.instruments.extend(instruments)
    out_wav = Path(out_wav)
    with tempfile.TemporaryDirectory() as td:
        mid = Path(td) / "x.mid"
        pm.write(str(mid))
        flags = ["-R", "1" if reverb else "0", "-C", "1" if reverb else "0"]
        subprocess.run(["fluidsynth", "-ni", *flags, "-g", "0.6", "-F", str(out_wav), "-r", str(sr), str(sf2), str(mid)],
                       check=True, capture_output=True)
    return out_wav


def render_melody(m: Melody, out_wav: str | Path, program: int = 0, accomp_vel: int | None = None,
                  drums: bool = True, reverb: bool = False, sr: int = 22050) -> Path:
    insts = [m.to_instrument(program)]
    if accomp_vel is not None:
        insts += accompaniment(m.length_beats, m.bpm, vel=accomp_vel, drums=drums)
    return render(insts, m.bpm, out_wav, sr=sr, reverb=reverb)


def mix(a: np.ndarray, b: np.ndarray, gain_b_db: float) -> np.ndarray:
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[: len(a)] += a
    out[: len(b)] += b * 10 ** (gain_b_db / 20)
    return out / max(1e-9, np.abs(out).max()) * 0.9


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True)
    return x.mean(axis=1), sr
