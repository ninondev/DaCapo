import numpy as np
import pytest

from dacapo.metrics import frame_metrics, interval_similarity, levenshtein, note_metrics
from dacapo.symbolic import Melody, melody_from_spec, random_melody

ODE = melody_from_spec([(64, 1), (64, 1), (65, 1), (67, 1), (67, 1), (65, 1), (64, 1), (62, 1),
                        (60, 1), (60, 1), (62, 1), (64, 1), (64, 1.5), (62, 0.5), (62, 2)], bpm=100)


def test_transpose_preserves_intervals():
    assert ODE.transpose(5).intervals() == ODE.intervals()
    assert ODE.transpose(5).pitches()[0] == 69


def test_inversion_negates_intervals():
    assert ODE.invert().intervals() == [-i for i in ODE.intervals()]


def test_retrograde_is_involution_and_reverses_pitches():
    r = ODE.retrograde()
    assert r.pitches() == ODE.pitches()[::-1]
    rr = r.retrograde()
    assert [(n.pitch, round(n.start, 6), n.dur) for n in rr.notes] == [(n.pitch, round(n.start, 6), n.dur) for n in ODE.notes]


def test_augmentation_doubles_length():
    assert ODE.scale_time(2).length_beats == pytest.approx(2 * ODE.length_beats)


def test_midi_roundtrip(tmp_path):
    p = tmp_path / "m.mid"
    ODE.to_midi(p)
    back = Melody.from_midi(p)
    assert back.pitches() == ODE.pitches()
    assert [round(n.start, 3) for n in back.notes] == [round(n.start, 3) for n in ODE.notes]


def test_random_melody_is_deterministic():
    a = random_melody(np.random.default_rng(1))
    b = random_melody(np.random.default_rng(1))
    assert a == b and a.length_beats == pytest.approx(16)


def test_frame_metrics_perfect_and_lagged():
    t, f0 = ODE.f0_track()
    m = frame_metrics(t, f0, t, f0)
    assert m["RPA"] == 1.0 and m["VFA"] == 0.0
    # Estimate arrives 120 ms late: lag search must recover it.
    lagged = np.concatenate([np.zeros(12), f0])[: len(f0)]
    m2 = frame_metrics(t, f0, t, lagged)
    assert m2["lag_s"] == pytest.approx(0.12, abs=0.011) and m2["RPA"] > 0.97


def test_frame_metrics_transposition():
    t, f0 = ODE.f0_track()
    _, f0_up = ODE.transpose(3).f0_track(total=t[-1] + 0.01)
    assert frame_metrics(t, f0, t, f0_up)["RPA"] < 0.05
    m = frame_metrics(t, f0, t, f0_up, transpose_invariant=True)
    assert m["RPA"] == pytest.approx(1.0) and m["shift_semitones"] == 3.0
    assert frame_metrics(t, f0, t, f0 * 2)["RCA"] == pytest.approx(1.0)  # octave error is chroma-correct


def test_note_metrics_detects_pitch_errors():
    iv, hz = ODE.note_events_seconds()
    perfect = [(s, e, n.pitch) for (s, e), n in zip(iv, ODE.notes)]
    assert note_metrics(iv, hz, perfect)["F"] == 1.0
    wrong = ODE.with_pitch_errors([2, 7], [1, -2])
    est = [(s, e, n.pitch) for (s, e), n in zip(iv, wrong.notes)]
    assert note_metrics(iv, hz, est)["F"] == pytest.approx(13 / 15)


def test_interval_similarity():
    assert interval_similarity(ODE.pitches(), ODE.transpose(7).pitches()) == 1.0
    assert levenshtein([1, 2, 3], [1, 3]) == 1
    assert interval_similarity(ODE.pitches(), ODE.invert().pitches()) < 0.5


def test_pitch_class_accuracy_perfect_and_shifted():
    from dacapo.metrics import pitch_class_accuracy
    t, f0 = ODE.f0_track()
    pc = np.where(f0 > 0, np.mod(np.round(12 * np.log2(np.where(f0 > 0, f0, 1) / 440.0)) + 9, 12), -1)
    assert pitch_class_accuracy(t, f0, t, pc)["pc_acc"] == 1.0
    assert pitch_class_accuracy(t, f0, t, np.where(pc >= 0, (pc + 1) % 12, -1))["pc_acc"] < 0.1


def test_note_pitch_accuracy():
    from dacapo.metrics import note_pitch_accuracy
    t, f0 = ODE.f0_track()
    iv, _ = ODE.note_events_seconds()
    notes = [(s, e, n.pitch) for (s, e), n in zip(iv, ODE.notes)]
    assert note_pitch_accuracy(notes, t, f0)["npa"] == 1.0
    _, f0_oct = ODE.transpose(12).f0_track(total=t[-1] + 0.01)
    assert note_pitch_accuracy(notes, t, f0_oct)["npa"] == 0.0
    assert note_pitch_accuracy(notes, t, f0_oct, octave_tolerant=True)["npa"] == 1.0
    wrong = ODE.with_pitch_errors([0, 5], [2, -2])
    _, f0_w = wrong.f0_track(total=t[-1] + 0.01)
    assert note_pitch_accuracy(notes, t, f0_w)["npa"] == pytest.approx(13 / 15)
