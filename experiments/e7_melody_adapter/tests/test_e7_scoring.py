"""Scoring contracts, including rendered audio with known note timings."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from dacapo.analysis import f0_fcpe, f0_swift
from dacapo.render import default_sf2, render_melody
from dacapo.symbolic import Melody, Note, melody_from_spec
from experiments.e7_melody_adapter import lib


NOTES = [
    (0.00, 0.40, 60),
    (0.40, 0.75, 64),
    (0.75, 1.25, 61),
    (1.25, 1.75, 67),
    (1.75, 2.10, 62),
    (2.10, 2.60, 69),
    (2.60, 3.15, 65),
    (3.15, 3.55, 72),
]


def track(notes, *, start=0.0, speed=1.0, transpose=0, tail=0.3):
    end = start + max(e for _, e, _ in notes) / speed + tail
    times = np.arange(0.0, end, 0.01)
    hz = np.zeros(len(times))
    for onset, offset, pitch in notes:
        selected = (times >= start + onset / speed) & (times < start + offset / speed)
        hz[selected] = 440.0 * 2.0 ** ((pitch + transpose - 69) / 12.0)
    return times, hz


def wrong_contour(notes):
    pitches = [72, 57, 70, 60, 68, 58, 71, 62]
    return [(s, e, p) for (s, e, _), p in zip(notes, pitches)]


def test_strict_search_finds_global_time_tempo_and_key():
    times, hz = track(NOTES, start=1.37, speed=1.03, transpose=2)
    score = lib.strict_external(NOTES, times, hz)
    assert score["npa"] == pytest.approx(1.0)
    assert score["npa_oct"] == pytest.approx(1.0)
    assert score["start_s"] == pytest.approx(1.37, abs=0.02)
    assert score["speed_ratio"] == pytest.approx(1.03, abs=0.011)
    assert score["transpose_semitones"] == 2
    assert score["frame_accuracy"] > 0.97


def test_strict_search_does_not_make_an_unrelated_melody_correct():
    times, hz = track(NOTES, start=0.73, speed=0.97, transpose=-3)
    own = lib.strict_external(NOTES, times, hz)
    mismatch = lib.strict_external(wrong_contour(NOTES), times, hz)
    assert own["npa"] >= 0.99
    assert mismatch["npa"] <= 0.5
    assert own["npa"] - mismatch["npa"] >= 0.4


def test_octave_tolerance_is_only_a_diagnostic():
    played = [(s, e, p + (12 if i == 3 else 0)) for i, (s, e, p) in enumerate(NOTES)]
    times, hz = track(played, start=0.8)
    score = lib.strict_external(NOTES, times, hz)
    assert score["npa"] == pytest.approx(7 / 8)
    assert score["npa_oct"] == pytest.approx(1.0)
    assert score["transpose_semitones"] == 0


def test_unvalidated_note_f1_reuses_the_selected_global_alignment():
    start, speed, shift = 1.37, 1.03, 2
    times, hz = track(NOTES, start=start, speed=speed, transpose=shift)
    events = [(start + s / speed, start + e / speed, p + shift) for s, e, p in NOTES]
    score = lib.strict_external(NOTES, times, hz, estimated_notes=events)
    assert score["note_f1_unvalidated"] == pytest.approx(1.0)
    shifted_events = [(s + 5.0, e + 5.0, p) for s, e, p in events]
    shifted_score = lib.strict_external(NOTES, times, hz, estimated_notes=shifted_events)
    assert shifted_score["note_f1_unvalidated"] == 0.0
    assert shifted_score["start_s"] == score["start_s"]


def test_silent_audio_cannot_pass_strict_pitch_scoring():
    times = np.arange(0.0, 5.0, 0.01)
    score = lib.strict_external(NOTES, times, np.zeros_like(times))
    assert score["npa"] == 0.0
    assert score["npa_oct"] == 0.0
    assert score["voiced_notes"] == 0.0


def test_relaxed_alignment_recovers_subsequence_and_global_transposition():
    times, hz = track(NOTES, start=1.23, transpose=2)
    score = lib.relaxed_alignment(NOTES, times, hz)
    assert score["deviation_semitones"] < 0.02
    assert score["note_accuracy"] == pytest.approx(1.0)
    assert score["transpose_semitones"] == 2
    assert 1.23 - 0.02 <= score["start_s"] <= 1.23 + NOTES[0][1]
    assert score["end_s"] > score["start_s"]


def test_relaxed_alignment_permits_bounded_tempo_change():
    times, hz = track(NOTES, start=0.52, speed=1.5, transpose=-2)
    score = lib.relaxed_alignment(NOTES, times, hz)
    assert score["deviation_semitones"] < 0.08
    assert score["note_accuracy"] >= 0.99
    assert score["transpose_semitones"] == -2
    span_ratio = (score["end_s"] - score["start_s"]) / NOTES[-1][1]
    assert 0.5 - 0.01 <= span_ratio <= 2.0 + 0.01


def test_relaxed_alignment_preserves_order_and_reports_chance_gap():
    times, hz = track(NOTES, start=0.4)
    own = lib.relaxed_alignment(NOTES, times, hz)
    mismatch = lib.relaxed_alignment(wrong_contour(NOTES), times, hz)
    assert mismatch["deviation_semitones"] > own["deviation_semitones"] + 0.5
    assert mismatch["note_accuracy"] < own["note_accuracy"]


def test_relaxed_silence_cost_is_finite_and_not_a_pitch_match():
    times = np.arange(0.0, 5.0, 0.01)
    score = lib.relaxed_alignment(NOTES, times, np.zeros_like(times))
    assert score["deviation_semitones"] == pytest.approx(3.0)
    assert score["note_accuracy"] == 0.0
    assert score["octave_note_accuracy"] == 0.0


def test_word_recall_strips_sections_and_counts_repeated_words_once_per_match():
    assert lib.word_recall("[Verse]\nGo, go home!", "go home") == pytest.approx(2 / 3)
    assert lib.word_recall("go go home", "go go home home") == 1.0
    assert lib.word_recall("go home", "home go") == 0.5
    assert lib.word_recall("[Verse] Don't go", "don't go") == 1.0
    assert lib.word_recall("", "words that were never requested") == 0.0


def test_lyric_mismatch_reveals_unrelated_hallucinated_words():
    transcript = "bright stars guide the boat through the night"
    correct = lib.word_recall("bright stars guide the boat", transcript)
    unrelated = lib.word_recall("cold rain falls on red roofs", transcript)
    assert correct == 1.0
    assert unrelated == 0.0
    assert correct - unrelated > 0.5


def test_voice_present_requires_energy_share_and_one_second_above_rms_floor():
    sample_rate = 1000
    mixture = np.ones(sample_rate)
    vocals = np.full(sample_rate, np.sqrt(0.2 + 1e-9))
    score = lib.voice_present(vocals, mixture, sample_rate)
    assert score["voice_present"]
    assert score["energy_share"] == pytest.approx(0.2 + 1e-9)
    assert score["active_seconds"] == pytest.approx(1.0)
    assert not lib.voice_present(vocals[:950], mixture[:950], sample_rate)["voice_present"]
    low_share = np.full(sample_rate, np.sqrt(0.2 - 1e-9))
    assert not lib.voice_present(low_share, mixture, sample_rate)["voice_present"]
    low_level = np.full(sample_rate, 0.009)
    assert not lib.voice_present(low_level, 2 * low_level, sample_rate)["voice_present"]


def test_voice_present_handles_stereo_and_silence():
    sample_rate = 1000
    mixture = np.ones((sample_rate, 2))
    vocals = np.full_like(mixture, 0.5)
    assert lib.voice_present(vocals, mixture, sample_rate)["voice_present"]
    score = lib.voice_present(np.zeros_like(vocals), np.zeros_like(mixture), sample_rate)
    assert not score["voice_present"]
    assert score["energy_share"] == 0.0
    assert score["active_seconds"] == 0.0


def test_bootstrap_is_seeded_and_does_not_invent_variation():
    first = lib.bootstrap_interval([0.1, 0.4, 0.7, 0.9], seed=7, draws=200, statistic="median")
    second = lib.bootstrap_interval([0.1, 0.4, 0.7, 0.9], seed=7, draws=200, statistic="median")
    assert first == second
    assert 0.1 <= first[0] <= first[1] <= 0.9
    assert lib.bootstrap_interval([0.7] * 5, seed=3, draws=100, statistic="mean") == pytest.approx([0.7, 0.7])


def test_chance_scoring_uses_the_same_search_and_correct_difference_direction():
    times, hz = track(NOTES, start=0.4, speed=1.02, transpose=3)
    score = lib.score_against_references(NOTES, [wrong_contour(NOTES)], times, hz)
    assert score["n_mismatched"] == 1
    assert score["own"]["strict"]["npa"] >= 0.99
    assert score["adjusted"]["strict"]["npa"] > 0.4
    assert score["adjusted"]["relaxed"]["deviation_semitones"] > 0.5
    assert score["chance"]["strict"]["note_f1_unvalidated"] is None


def test_label_selector_and_evaluator_are_distinct():
    lib.validate_tool_roles("swift", "fcpe", "crepe")
    for roles in [("swift", "fcpe", "fcpe"), ("swift", "swift", "crepe"),
                  ("swift", "fcpe", "swift"), ("SwiftF0", "swift", "crepe"),
                  ("swift", "torchfcpe", "FCPE"), ("", "fcpe", "crepe")]:
        with pytest.raises(ValueError):
            lib.validate_tool_roles(*roles)


@pytest.fixture(scope="module")
def rendered_performance(tmp_path_factory):
    assert shutil.which("fluidsynth"), "FluidSynth is required for the CPU audio contract"
    assert default_sf2().is_file(), "The configured soundfont is required"
    reference = melody_from_spec([
        (60, 1), (64, 1), (61, 1), (67, 1),
        (62, 1), (69, 1), (65, 1), (72, 1),
    ], bpm=80)
    iv, _ = reference.note_events_seconds()
    notes = [(float(s), float(e), n.pitch) for (s, e), n in zip(iv, reference.notes)]
    speed, start, transpose = 1.03, 1.37, 2
    bpm = reference.bpm * speed
    shifted = Melody(tuple(Note(n.pitch + transpose, n.start + start * bpm / 60, n.dur, n.vel)
                           for n in reference.notes), bpm=bpm)
    output = tmp_path_factory.mktemp("render-contract") / "voice_oohs.wav"
    render_melody(shifted, output, program=53, sr=22050)
    return notes, output, start, speed, transpose


@pytest.mark.parametrize("tracker", [f0_swift, f0_fcpe], ids=["swift", "fcpe"])
def test_rendered_oohs_recovers_melody_time_tempo_and_key(rendered_performance, tracker):
    notes, wav, start, speed, transpose = rendered_performance
    times, hz = tracker(wav)
    score = lib.strict_external(notes, times, hz)
    wrong = lib.strict_external(wrong_contour(notes), times, hz)
    assert score["npa"] >= 0.9
    assert score["start_s"] == pytest.approx(start, abs=0.1)
    assert score["speed_ratio"] == pytest.approx(speed, abs=0.011)
    assert score["transpose_semitones"] == transpose
    assert score["npa"] - wrong["npa"] > 0.4
    relaxed = lib.relaxed_alignment(notes, times, hz)
    wrong_relaxed = lib.relaxed_alignment(wrong_contour(notes), times, hz)
    assert relaxed["note_accuracy"] >= 0.9
    assert relaxed["transpose_semitones"] == transpose
    assert wrong_relaxed["deviation_semitones"] - relaxed["deviation_semitones"] > 0.5


def test_note_f1_does_not_invent_onsets_at_the_search_boundary():
    start = 1.0
    times, hz = track(NOTES, start=start)
    events = [(start + s, start + e, p) for s, e, p in NOTES]
    # This note began half a second before the performed melody. Its pitch
    # is right, but it has no onset at the requested first note.
    events[0] = (0.5, events[0][1], events[0][2])
    score = lib.strict_external(NOTES, times, hz, estimated_notes=events)
    assert score["npa"] == 1.0
    assert score["note_f1_unvalidated"] == pytest.approx(7 / 8)


def test_dtw_compound_steps_charge_every_reference_frame():
    reference = np.array([60.0, 65.0, 60.0])
    observed = np.array([60.0, 60.0])
    deviation, endpoint = lib._dtw_cost(reference, observed, 0)
    traced, traced_endpoint, ref_indices, output_indices = lib._dtw_trace(reference, observed, 0)
    assert deviation == pytest.approx(5 / 3)
    assert traced == pytest.approx(deviation)
    assert endpoint == traced_endpoint == 1
    assert sorted(ref_indices.tolist()) == [0, 1, 2]
    assert output_indices.min() == 0 and output_indices.max() == 1


def test_chance_scores_all_eleven_references_independently():
    times, hz = track(NOTES, start=0.37, transpose=2)
    others = [wrong_contour(NOTES) for _ in range(11)]
    score = lib.score_against_references(NOTES, others, times, hz)
    direct = lib.score_reference(others[0], times, hz)
    assert score["n_mismatched"] == 11
    assert score["chance"]["strict"]["npa"] == pytest.approx(direct["strict"]["npa"])
    assert score["chance"]["relaxed"]["deviation_semitones"] == pytest.approx(direct["relaxed"]["deviation_semitones"])
    assert score["adjusted"]["strict"]["npa"] > 0.4
    assert score["adjusted"]["relaxed"]["deviation_semitones"] > 0.5


def test_frozen_track_contract_rejects_irregular_times_and_nonfinite_pitch():
    times, hz = track(NOTES)
    irregular = times.copy()
    irregular[2] += 0.001
    with pytest.raises(ValueError, match="10 ms"):
        lib.strict_external(NOTES, irregular, hz)
    broken = hz.copy()
    broken[3] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        lib.relaxed_alignment(NOTES, times, broken)
