"""E5 v2 per-draw quality features, the per-draw gate, and note/onset F1, from arrays and note lists.

Windows in seconds, 18 s source layout. ACE-Step splices the source waveform back outside [12, 16] s
with a 25 ms crossfade that lies outside the window, then peak-normalises the whole stereo tensor to
-1 dBFS (maximum over all channels and samples).
    gap  [12.4, 15.6)  scored window (0.4 s margins, as in v1)
    ctx  [0.5, 11.5)   splice-safe context, used only by an exploratory alternative gate
    D    [16.1, 17.9)  the copy-next bar D, present in every condition including null
Levels are RMS of the channel mean (dacapo.render.load_mono), as in v1. Peaks are taken over all
channels, as ACE-Step's normalisation does.

The governing gate is the registered Phase 0 rule of v1 run_analogy.gate_choice: voiced fraction
>= 0.6, v1 Basic Pitch polyphony <= 1.3 and |gap RMS relative to the 0-12 s context| <= 6 dB.
It decides Phase 0b (K1), the confirmatory quality rules (Q1, Q2) and the PASS population.
"""
from __future__ import annotations

import numpy as np

G0, G1, MARGIN = 12.0, 16.0, 0.4
W_GAP = (G0 + MARGIN, G1 - MARGIN)
W_CTX = (0.5, G0 - 0.5)
W_D = (16.1, 17.9)
FLOOR_DBFS = -200.0
LAGS = np.arange(-0.12, 0.1201, 0.02)
MIN_REF_NOTE_S = 0.02
REF_VALID_DBFS = -60.0

GATE_REGISTERED = {"name": "registered", "voiced_min": 0.6, "poly_max": 1.3, "poly_measure": "gap_polyphony",
                   "level_key": "gap_rms_db_vs_context_v1", "rel_db_max": 6.0, "abs_dbfs_min": None}
# Exploratory alternatives, stored per draw and never used for a decision.
GATE_OCTFOLD = GATE_REGISTERED | {"name": "octfold", "poly_measure": "gap_polyphony_octfold"}
GATE_SPLICE_SAFE = GATE_REGISTERED | {"name": "splice_safe", "level_key": "gap_rms_db_vs_ref", "abs_dbfs_min": -30.0}


def _seg(x: np.ndarray, sr: int, a: float, b: float) -> np.ndarray:
    return x[int(a * sr):int(b * sr)]          # int() truncation, as in v1 gap_quality


def rms_dbfs(y: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    return float(20 * np.log10(r)) if r > 1e-10 else FLOOR_DBFS


def _db(v: float) -> float:
    return float(20 * np.log10(v)) if v > 1e-10 else FLOOR_DBFS


def peak_dbfs(x: np.ndarray) -> float:
    """Peak over all channels and samples, as ACE-Step's normalisation measures it (integrity check 2)."""
    return _db(float(np.max(np.abs(x))) if x.size else 0.0)


def level_features(x: np.ndarray, sr: int, cond: str) -> dict:
    """x: the wav as written, (n,) or (n, channels).
    gap_rms_db_vs_context_v1 is the registered level term (v1 formula, context 0-12 s). On null draws it
    is not meaningful: the 25 ms splice crossfade leaks into the silent context (+31 dB in v1), so null
    draws never enter Q1/Q2 and their absolute gap level (gap_rms_dbfs) is reported instead."""
    x2 = x[:, None] if x.ndim == 1 else x
    mono = x2.mean(axis=1)
    gap, ctx, d = (rms_dbfs(_seg(mono, sr, *w)) for w in (W_GAP, W_CTX, W_D))
    ref_name, ref = ("D", d) if cond == "null" else ("ctx", ctx)
    rel = gap - ref if ref > REF_VALID_DBFS else None
    rms_v1 = lambda y: float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    legacy = 20 * np.log10(rms_v1(_seg(mono, sr, G0 + MARGIN, G1 - MARGIN)) / rms_v1(_seg(mono, sr, 0, G0)))
    return {"gap_rms_dbfs": gap, "ctx_rms_dbfs": ctx, "d_rms_dbfs": d, "level_ref": ref_name,
            "gap_rms_db_vs_ref": rel, "gap_rms_db_vs_D": gap - d if d > REF_VALID_DBFS else None,
            "gap_rms_db_vs_context_v1": float(legacy),
            "peak_dbfs": peak_dbfs(x2),                                               # channel peak (check 2)
            "peak_dbfs_mono": _db(float(np.max(np.abs(mono))) if mono.size else 0.0)}  # information only


def _gap_notes(notes):
    return [n for n in notes if W_GAP[0] <= n[0] < W_GAP[1]]


def polyphony_v1(notes: list[tuple[float, float, int]]) -> float:
    """Registered measure (v1 score_analogy.gap_quality): Basic Pitch notes with onset in the gap window,
    mean number sounding on a 50 ms grid over the window (0 if none)."""
    sel = _gap_notes(notes)
    if not sel:
        return 0.0
    grid = np.arange(W_GAP[0], W_GAP[1], 0.05)
    return float(np.mean([sum(s <= t < e for s, e, _ in sel) for t in grid]))


def polyphony_octfold(notes: list[tuple[float, float, int]]) -> float:
    """As polyphony_v1 (unison duplicates still count twice), except that a note 12n semitones (n >= 1)
    above another sounding note is not counted. Exploratory only."""
    sel = _gap_notes(notes)
    if not sel:
        return 0.0
    grid = np.arange(W_GAP[0], W_GAP[1], 0.05)
    counts = []
    for t in grid:
        ps = [p for s, e, p in sel if s <= t < e]
        counts.append(sum(not any(p - q >= 12 and (p - q) % 12 == 0 for q in ps) for p in ps))
    return float(np.mean(counts))


def voiced_fraction(et: np.ndarray, ef: np.ndarray) -> float:
    g = (et >= W_GAP[0]) & (et < W_GAP[1])
    return float((ef[g] > 0).mean())


def double_f0_fraction(notes: list[tuple[float, float, int]], hop: float = 0.01) -> float:
    """Registered superposition guard (reported, no threshold): share of gap frames where two or more
    Basic Pitch notes sound whose pitches are not octave-related."""
    grid = np.arange(W_GAP[0], W_GAP[1], hop)
    hit = 0
    for t in grid:
        ps = sorted({p for s, e, p in notes if s <= t < e})
        hit += any((b - a) % 12 != 0 for i, a in enumerate(ps) for b in ps[i + 1:])
    return hit / len(grid)


def gate_flags(q: dict, g: dict = GATE_REGISTERED) -> dict:
    """Per-draw gate; every flag must hold. Plain Python bools, since rows go through json.dumps."""
    lvl = q[g["level_key"]]
    f = {"voiced": bool(q["gap_voiced"] >= g["voiced_min"]),
         "poly": bool(q[g["poly_measure"]] <= g["poly_max"]),
         "level": bool(lvl is not None and abs(lvl) <= g["rel_db_max"])}
    if g.get("abs_dbfs_min") is not None:
        f["level_abs"] = bool(q["gap_rms_dbfs"] >= g["abs_dbfs_min"])
    f["pass"] = all(f.values())
    return f


def gate_record(q: dict) -> dict:
    """What a row stores: the registered flags plus the two exploratory alternatives."""
    f = gate_flags(q, GATE_REGISTERED)
    f["pass_octfold"] = gate_flags(q, GATE_OCTFOLD)["pass"]
    f["pass_splice_safe"] = gate_flags(q, GATE_SPLICE_SAFE)["pass"]
    return f


def est_notes_in_gap(notes, pad: float = 0.05 + 0.12):
    """Transcribed notes considered for note and onset F1: onset within the gap window widened by the
    onset tolerance plus the largest lag, the same for every hypothesis."""
    return [n for n in notes if W_GAP[0] - pad <= n[0] < W_GAP[1] + pad]


def note_f1(ref: list[tuple[float, float, int]], est: list[tuple[float, float, int]], onset_only: bool = False,
            onset_tol: float = 0.05, pitch_tol_cents: float = 50.0, lags=LAGS) -> float | None:
    """Best-lag note F1 (mir_eval, offsets ignored) of transcribed notes against a hypothesis;
    onset_only=True gives onset F1. Same symmetric lag grid as acc(). Reference notes shorter than
    20 ms (clipped at the window edge) are dropped; None means not scorable."""
    import mir_eval
    import pretty_midi

    ref = [n for n in ref if n[1] - n[0] >= MIN_REF_NOTE_S]
    if not ref:
        return None
    if not est:
        return 0.0
    r_iv = np.array([[s, e] for s, e, _ in ref])
    r_hz = np.array([pretty_midi.note_number_to_hz(p) for _, _, p in ref])
    e_hz = np.array([pretty_midi.note_number_to_hz(p) for _, _, p in est])
    best = 0.0
    for lag in lags:
        e_iv = np.array([[s - lag, max(e, s + 1e-3) - lag] for s, e, _ in est])
        if onset_only:
            f = mir_eval.transcription.onset_precision_recall_f1(r_iv, e_iv, onset_tolerance=onset_tol)[2]
        else:
            f = mir_eval.transcription.precision_recall_f1_overlap(
                r_iv, r_hz, e_iv, e_hz, onset_tolerance=onset_tol, pitch_tolerance=pitch_tol_cents,
                offset_ratio=None)[2]
        best = max(best, float(f))
    return best


def notes_to_array(notes) -> np.ndarray:
    """Basic Pitch notes -> (n, 3) float array, shape (0, 3) when empty, so the npz round trip is safe."""
    return np.asarray(notes, float).reshape(-1, 3)


def array_to_notes(a: np.ndarray) -> list[tuple[float, float, int]]:
    return [(float(s), float(e), int(round(p))) for s, e, p in np.asarray(a, float).reshape(-1, 3)]
