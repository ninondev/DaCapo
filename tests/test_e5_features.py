"""Unit tests for dacapo.motif_features (arrays and note lists only)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dacapo import motif_features as F
from dacapo import motif_integrity as I

SR = 48000


def _flute_like(f0, dur, amp, sr=SR, phase=0.0):
    t = np.arange(int(dur * sr)) / sr
    return amp * (np.sin(2 * np.pi * f0 * t + phase) + 0.3 * np.sin(4 * np.pi * f0 * t + 2 * phase))


def _clip(ctx_amp, gap_amp, d_amp, leak=True, stereo_phase=None):
    """18 s clip: context 0-12 s, gap 12-16 s, D 16-18 s; ACE-Step-style splice (generated audio leaks into
    11.975-12.0 s through the 25 ms ramp) and -1 dBFS peak normalisation over all channels.
    stereo_phase: phase offset of the right channel in the gap (decorrelated stereo)."""
    def chan(phase):
        x = np.zeros(18 * SR)
        x[:12 * SR] = _flute_like(440, 12, ctx_amp)
        gen = _flute_like(523.25, 18, gap_amp, phase=phase)
        x[12 * SR:16 * SR] = gen[12 * SR:16 * SR]
        if leak:
            n = int(0.025 * SR)
            ramp = np.linspace(0, 1, n + 2)[1:-1]
            x[12 * SR - n:12 * SR] = ramp * gen[12 * SR - n:12 * SR] + (1 - ramp) * x[12 * SR - n:12 * SR]
        x[16 * SR:] = _flute_like(392, 2, d_amp)
        return x
    x = chan(0.0) if stereo_phase is None else np.stack([chan(0.0), chan(stereo_phase)], axis=1)
    return x / np.abs(x).max() * 10 ** (-1 / 20)


def _q(x, cond, **kw):
    return {**F.level_features(x, SR, cond), "gap_voiced": 1.0, "gap_polyphony": 1.0, "gap_polyphony_octfold": 1.0, **kw}


def test_registered_level_term_on_null_and_the_splice_safe_alternative():
    """The registered level term (context 0-12 s) reads about +31 dB on a filled null gap, because the splice
    crossfade leaks into the silent context; the exploratory splice-safe gate uses D for null."""
    q = F.level_features(_clip(0.0, 0.1, 0.1), SR, "null")
    assert 30.0 < q["gap_rms_db_vs_context_v1"] < 33.0
    assert q["level_ref"] == "D" and abs(q["gap_rms_db_vs_ref"]) < 0.5
    assert not F.gate_flags(_q(_clip(0.0, 0.1, 0.1), "null"))["level"]
    assert F.gate_flags(_q(_clip(0.0, 0.1, 0.1), "null"), F.GATE_SPLICE_SAFE)["pass"]
    f0 = F.gate_flags(_q(_clip(0.0, 1e-4, 0.1), "null"), F.GATE_SPLICE_SAFE)       # near-silent null gap
    assert not f0["level_abs"] and not f0["level"] and not f0["pass"]


def test_nonnull_levels_and_quiet_gap():
    q = F.level_features(_clip(0.1, 0.1, 0.1), SR, "lit")
    assert q["level_ref"] == "ctx" and abs(q["gap_rms_db_vs_ref"]) < 0.5 and q["gap_rms_dbfs"] > -15
    assert abs(q["gap_rms_db_vs_ref"] - q["gap_rms_db_vs_context_v1"]) < 0.1
    assert F.gate_flags(_q(_clip(0.1, 0.1, 0.1), "lit"))["pass"]
    f = F.gate_flags(_q(_clip(0.1, 0.1 * 10 ** (-30 / 20), 0.1), "lit"))
    assert not f["level"] and not f["pass"]


def test_peak_is_channel_peak_on_decorrelated_stereo():
    """ACE-Step normalises the max over all channels. On a stereo output whose gap channels are decorrelated
    the mono-mix peak sits well below -1 dBFS; check 2 must read the channel peak."""
    x = _clip(0.02, 0.1, 0.02, stereo_phase=2 * np.pi / 3)
    q = F.level_features(x, SR, "lit")
    assert q["peak_dbfs"] == pytest.approx(-1.0, abs=1e-6) and q["peak_dbfs_mono"] < -1.5
    row = {"id": "m", "cond": "lit", "arm": "base", "seed": "d0", "status": "ok", "quality": q}
    assert I.check_peaks([row])["ok"]
    assert not I.check_peaks([{**row, "quality": {**q, "peak_dbfs": q["peak_dbfs_mono"]}}])["ok"]
    assert I.check_peaks([{**row, "quality": {**q, "peak_dbfs": -200.0}}])["ok"]     # silent: normalisation skipped


def test_note_f1_lag_symmetry_onset_only_and_short_refs():
    B = [(12.5, 13.0, 72), (13.0, 13.5, 74), (13.5, 14.0, 76), (14.0, 15.0, 77)]
    Bx = [(s, e, p) for (s, e, _), p in zip(B, [76, 77, 72, 74])]
    est = [(s + 0.1, e + 0.1, p) for s, e, p in B]
    assert F.note_f1(B, est) == 1.0 and F.note_f1(Bx, est) == 0.0
    assert F.note_f1(B, est, onset_only=True) == F.note_f1(Bx, est, onset_only=True) == 1.0
    assert F.note_f1(B, [(s + 0.2, e + 0.2, p) for s, e, p in B]) == 0.0
    assert F.note_f1(B, []) == 0.0 and F.note_f1([], est) is None
    assert F.note_f1([(15.6, 15.6, 72)], est) is None                    # clipped to zero length: not scorable
    assert F.note_f1(B + [(15.6, 15.6, 72)], est) == 1.0                 # no mir_eval crash on a clipped note


def test_polyphony_measures():
    line = [(12.4, 13.4, 72), (13.4, 14.4, 74), (14.4, 15.6, 76)]
    doubled = line + [(s, e, p + 12) for s, e, p in line]
    unison = line + [(12.4, 13.4, 72)]
    fifth = line + [(12.4, 14.0, 79)]
    assert F.polyphony_v1(doubled) == 2.0 and F.polyphony_octfold(doubled) == 1.0
    assert F.polyphony_v1(unison) == F.polyphony_octfold(unison)          # unison duplicates count as in v1
    assert F.polyphony_octfold(fifth) == F.polyphony_v1(fifth) > 1.4
    assert F.polyphony_v1([]) == F.polyphony_octfold([]) == 0.0


def test_gate_uses_registered_polyphony_and_level_and_stores_alternatives():
    q = {"gap_voiced": 1.0, "gap_polyphony": 1.4, "gap_polyphony_octfold": 1.1, "gap_rms_db_vs_context_v1": 0.0,
         "gap_rms_db_vs_ref": 0.0, "gap_rms_dbfs": -10.0}
    g = F.gate_record(q)
    assert g["pass"] is False and g["pass_octfold"] is True and g["pass_splice_safe"] is False
    q = {**q, "gap_polyphony": 1.0, "gap_rms_db_vs_context_v1": 6.0, "gap_rms_db_vs_ref": 6.5}
    g = F.gate_record(q)
    assert g["pass"] is True and g["pass_splice_safe"] is False         # |level| <= 6 dB is inclusive
    assert F.GATE_REGISTERED["voiced_min"] == 0.6 and F.GATE_REGISTERED["poly_max"] == 1.3


def test_double_f0():
    mono = [(12.4, 15.6, 72)]
    assert F.double_f0_fraction(mono) == 0.0 and F.double_f0_fraction(mono + [(12.4, 15.6, 84)]) == 0.0
    assert F.double_f0_fraction(mono + [(12.4, 14.0, 79)]) == pytest.approx(0.5, abs=0.01)


def test_gate_flags_json_and_npz_roundtrip(tmp_path):
    q = F.level_features(_clip(0.1, 0.1, 0.1), SR, "lit")
    g = F.gate_record({**q, "gap_voiced": np.float64(1.0), "gap_polyphony": np.float64(1.0), "gap_polyphony_octfold": 1.0})
    json.dumps({"q": q, "gate": g})
    assert all(type(v) is bool for v in g.values())
    for notes in ([], [(12.5, 13.0, 72), (13.0, 13.5, 74)]):
        np.savez(tmp_path / "f.npz", bp=F.notes_to_array(notes))
        back = F.array_to_notes(np.load(tmp_path / "f.npz")["bp"])
        assert back == [(float(s), float(e), int(p)) for s, e, p in notes]
        assert np.load(tmp_path / "f.npz")["bp"].shape == (len(notes), 3)


def test_registered_level_passes_a_digitally_silent_null_gap_but_voicing_fails():
    x = np.zeros(18 * SR)
    x[16 * SR:] = _flute_like(392, 2, 0.1)
    q = F.level_features(x / np.abs(x).max() * 10 ** (-1 / 20), SR, "null")
    assert abs(q["gap_rms_db_vs_context_v1"]) < 1e-6
    flags = F.gate_flags({**q, "gap_voiced": 0.0, "gap_polyphony": 0.0, "gap_polyphony_octfold": 0.0})
    assert flags["level"] and not flags["voiced"] and not flags["pass"]
    assert q["gap_rms_dbfs"] == F.FLOOR_DBFS
    assert not F.gate_flags({**q, "gap_voiced": 0.0, "gap_polyphony": 0.0, "gap_polyphony_octfold": 0.0},
                            F.GATE_SPLICE_SAFE)["level_abs"]
