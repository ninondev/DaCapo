"""Symbolic melodies: a small representation we can transform and test exactly.

Times are in beats. We convert to seconds only at the edges (rendering, reference f0 tracks),
so transformations do not depend on tempo.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pretty_midi


@dataclass(frozen=True)
class Note:
    pitch: int  # MIDI pitch
    start: float  # beats
    dur: float  # beats
    vel: int = 100


@dataclass(frozen=True)
class Melody:
    notes: tuple[Note, ...]
    bpm: float = 100.0

    @property
    def spb(self) -> float:
        return 60.0 / self.bpm

    @property
    def length_beats(self) -> float:
        return max(n.start + n.dur for n in self.notes)

    def pitches(self) -> list[int]:
        return [n.pitch for n in self.notes]

    def intervals(self) -> list[int]:
        p = self.pitches()
        return [b - a for a, b in zip(p, p[1:])]

    # Transformations are exact, and invertible where that makes musical sense.
    def transpose(self, k: int) -> Melody:
        return replace(self, notes=tuple(replace(n, pitch=n.pitch + k) for n in self.notes))

    def invert(self, axis: int | None = None) -> Melody:
        axis = self.notes[0].pitch if axis is None else axis
        return replace(self, notes=tuple(replace(n, pitch=2 * axis - n.pitch) for n in self.notes))

    def retrograde(self) -> Melody:
        end = self.length_beats
        rev = [replace(n, start=end - n.start - n.dur) for n in reversed(self.notes)]
        return replace(self, notes=tuple(rev))

    def scale_time(self, factor: float) -> Melody:
        """factor=2 is augmentation, 0.5 is diminution."""
        return replace(self, notes=tuple(replace(n, start=n.start * factor, dur=n.dur * factor) for n in self.notes))

    def shift(self, beats: float) -> Melody:
        return replace(self, notes=tuple(replace(n, start=n.start + beats) for n in self.notes))

    def with_pitch_errors(self, idx: list[int], semitones: list[int]) -> Melody:
        notes = list(self.notes)
        for i, s in zip(idx, semitones):
            notes[i] = replace(notes[i], pitch=notes[i].pitch + s)
        return replace(self, notes=tuple(notes))

    def with_velocity(self, vel: int) -> Melody:
        return replace(self, notes=tuple(replace(n, vel=vel) for n in self.notes))

    def note_events_seconds(self) -> tuple[np.ndarray, np.ndarray]:
        """(intervals [N,2] in seconds, pitches in Hz) for mir_eval.transcription."""
        iv = np.array([[n.start * self.spb, (n.start + n.dur) * self.spb] for n in self.notes])
        hz = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in self.notes])
        return iv, hz

    def f0_track(self, hop: float = 0.01, total: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reference frame-level f0 (Hz, 0 = unvoiced) for mir_eval.melody."""
        total = total if total is not None else self.length_beats * self.spb + 0.5
        times = np.arange(0.0, total, hop)
        f0 = np.zeros_like(times)
        for n in self.notes:
            s, e = n.start * self.spb, (n.start + n.dur) * self.spb
            f0[(times >= s) & (times < e)] = pretty_midi.note_number_to_hz(n.pitch)
        return times, f0

    def to_instrument(self, program: int = 0, gap: float = 0.02) -> pretty_midi.Instrument:
        inst = pretty_midi.Instrument(program=program)
        for n in self.notes:
            s = n.start * self.spb
            e = max(s + 0.03, (n.start + n.dur) * self.spb - gap)
            inst.notes.append(pretty_midi.Note(velocity=n.vel, pitch=n.pitch, start=s, end=e))
        return inst

    def to_midi(self, path: str | Path, program: int = 0) -> None:
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.bpm)
        pm.instruments.append(self.to_instrument(program))
        pm.write(str(path))

    @staticmethod
    def from_midi(path: str | Path, track: int = 0, bpm: float | None = None) -> Melody:
        pm = pretty_midi.PrettyMIDI(str(path))
        if bpm is None:
            _, tempi = pm.get_tempo_changes()
            bpm = float(tempi[0]) if len(tempi) else 120.0
        spb = 60.0 / bpm
        notes = sorted(pm.instruments[track].notes, key=lambda n: (n.start, -n.pitch))
        return Melody(tuple(Note(n.pitch, n.start / spb, (n.end - n.start) / spb, n.velocity) for n in notes), bpm)


def melody_from_spec(spec: list[tuple[int, float]], bpm: float = 100.0, start: float = 0.0) -> Melody:
    """Build a melody from (pitch, duration_in_beats) pairs; pitch < 0 is a rest."""
    notes, t = [], start
    for p, d in spec:
        if p >= 0:
            notes.append(Note(p, t, d))
        t += d
    return Melody(tuple(notes), bpm)


def random_melody(rng: np.random.Generator, bars: int = 4, bpm: float = 100.0, tonic: int = 67) -> Melody:
    """A singable, mostly stepwise major-key melody in 4/4 (a test stimulus, not 'music')."""
    scale = np.array([0, 2, 4, 5, 7, 9, 11])
    degree, notes, t = 0, [], 0.0
    durations = [0.5, 1.0, 1.0, 1.5, 2.0]
    probs = [0.3, 0.4, 0.1, 0.1, 0.1]
    while t < bars * 4 - 1e-9:
        d = float(rng.choice(durations, p=probs))
        d = min(d, bars * 4 - t)
        step = int(rng.choice([-2, -1, 1, 2, 0, 3, -3], p=[0.15, 0.3, 0.3, 0.1, 0.05, 0.05, 0.05]))
        degree = int(np.clip(degree + step, -3, 9))
        octave, idx = divmod(degree, 7)
        notes.append(Note(int(tonic + 12 * octave + scale[idx]), t, d))
        t += d
    return Melody(tuple(notes), bpm)
