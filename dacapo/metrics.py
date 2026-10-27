"""Melody-adherence metrics.

Frame-level metrics follow mir_eval's definitions (RPA/RCA/VR/VFA/OA) but are computed here,
so the lag search and transposition handling are easy to check. Note-level metrics use
mir_eval.transcription.
"""
from __future__ import annotations

import numpy as np
import mir_eval
import pretty_midi

HOP = 0.01


def _cents(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1200.0 * np.log2(a / b)


def _on_grid(ref_t: np.ndarray, est_t: np.ndarray, est_f0: np.ndarray, lag: float) -> np.ndarray:
    """Estimate sampled on the reference grid; lag > 0 means the estimate is late."""
    idx = np.round((ref_t + lag - est_t[0]) / HOP).astype(int)
    out = np.zeros_like(ref_t)
    ok = (idx >= 0) & (idx < len(est_f0))
    out[ok] = est_f0[idx[ok]]
    return out


def frame_metrics(ref_t, ref_f0, est_t, est_f0, max_lag: float = 0.3, transpose_invariant: bool = False,
                  tol_cents: float = 50.0) -> dict:
    """Best-lag frame metrics. With transpose_invariant=True the estimate is shifted by the
    whole-semitone offset that best matches the reference (for 'same melody, other key')."""
    best = None
    for lag in np.arange(-max_lag, max_lag + 1e-9, HOP):
        est = _on_grid(ref_t, est_t, est_f0, lag)
        both = (ref_f0 > 0) & (est > 0)
        shift = 0.0
        if transpose_invariant and both.sum() > 10:
            shift = 100.0 * np.round(np.median(_cents(est[both], ref_f0[both])) / 100.0)
        d = _cents(np.where(est > 0, est, 1), np.where(ref_f0 > 0, ref_f0, 1)) - shift
        # lag search uses chroma agreement, so octave errors do not break the alignment
        score = (both & (np.abs((d + 600) % 1200 - 600) < tol_cents)).sum() + 1e-3 * (both & (np.abs(d) < tol_cents)).sum()
        if best is None or score > best[0]:
            best = (score, lag, shift, est)
    _, lag, shift, est = best
    rv, ev = ref_f0 > 0, est > 0
    both = rv & ev
    d = np.zeros_like(ref_f0)
    d[both] = _cents(est[both], ref_f0[both]) - shift
    correct = both & (np.abs(d) < tol_cents)
    chroma = both & (np.abs((d + 600) % 1200 - 600) < tol_cents)
    n_rv = max(1, rv.sum())
    return {
        "RPA": float(correct.sum() / n_rv),
        "RCA": float(chroma.sum() / n_rv),
        "VR": float(both.sum() / n_rv),
        "VFA": float((ev & ~rv).sum() / max(1, (~rv).sum())),
        "OA": float((correct.sum() + (~rv & ~ev).sum()) / len(ref_f0)),
        "median_abs_cents": float(np.median(np.abs(d[both]))) if both.any() else float("nan"),
        "lag_s": float(lag),
        "shift_semitones": float(shift / 100.0),
    }


def note_metrics(ref_intervals: np.ndarray, ref_hz: np.ndarray, est_notes: list[tuple[float, float, int]],
                 onset_tol: float = 0.05, transpose_invariant: bool = False, max_lag: float = 0.3) -> dict:
    """Onset+pitch note F1 (offsets ignored), with a global time lag search (lag > 0: estimate late)."""
    if not est_notes:
        return {"P": 0.0, "R": 0.0, "F": 0.0, "lag_s": 0.0, "shift_semitones": 0.0, "n_est": 0}
    est_iv = np.array([[s, e] for s, e, _ in est_notes])
    est_p = np.array([p for _, _, p in est_notes], dtype=float)
    ref_p = np.array([pretty_midi.hz_to_note_number(h) for h in ref_hz])
    shifts = [0.0]
    if transpose_invariant:
        shifts = list(range(-12, 13))
    best = None
    for lag in np.arange(-max_lag, max_lag + 1e-9, 0.01):
        for sh in shifts:
            # shift both sides by max_lag so no interval is negative (mir_eval rejects that)
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals + max_lag, ref_hz, np.maximum(est_iv - lag + max_lag, 0.0), pretty_midi.note_number_to_hz(est_p - sh),
                onset_tolerance=onset_tol, pitch_tolerance=50.0, offset_ratio=None)
            if best is None or f > best[2]:
                best = (p, r, f, lag, sh)
    p, r, f, lag, sh = best
    return {"P": float(p), "R": float(r), "F": float(f), "lag_s": float(lag), "shift_semitones": float(sh),
            "n_est": len(est_notes), "n_ref": int(len(ref_p))}


def levenshtein(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def interval_similarity(ref_pitches: list[int], est_pitches: list[int]) -> float:
    """Timing-free, key-free similarity: 1 - normalized edit distance of interval sequences."""
    ri = [b - a for a, b in zip(ref_pitches, ref_pitches[1:])]
    ei = [b - a for a, b in zip(est_pitches, est_pitches[1:])]
    if not ri and not ei:
        return 1.0
    return 1.0 - levenshtein(ri, ei) / max(len(ri), len(ei))


def pitch_class_accuracy(ref_t, ref_f0, est_t, est_pc, max_lag: float = 0.3) -> dict:
    """Fraction of reference-voiced frames whose dominant pitch class matches the reference
    (octave-free, best global lag). Pair with a mismatched-reference baseline."""
    ref_pc = np.full(len(ref_f0), -1.0)
    v = ref_f0 > 0
    ref_pc[v] = np.mod(np.round(12 * np.log2(ref_f0[v] / 440.0)) + 9, 12)  # A=9 so C=0
    best = (0.0, 0.0)
    for lag in np.arange(-max_lag, max_lag + 1e-9, HOP):
        idx = np.round((ref_t + lag - est_t[0]) / np.median(np.diff(est_t))).astype(int)
        ok = v & (idx >= 0) & (idx < len(est_pc))
        acc = float((est_pc[idx[ok]] == ref_pc[ok]).sum() / max(1, v.sum()))
        if acc > best[0]:
            best = (acc, float(lag))
    return {"pc_acc": best[0], "lag_s": best[1]}


def note_pitch_accuracy(notes_s: list[tuple[float, float, int]], est_t: np.ndarray, est_f0: np.ndarray, lag: float = 0.0,
                        central: float = 0.6, tol_cents: float = 50.0, octave_tolerant: bool = False,
                        min_voiced: int = 3) -> dict:
    """Per-note pitch accuracy (NPA): a note is correct if the median estimated f0 over its central
    `central` fraction is within `tol_cents` of the note's pitch. Unvoiced/missing notes count as wrong.
    `lag` > 0 means the estimate is late (as in frame_metrics)."""
    hop = float(np.median(np.diff(est_t)))
    correct, voiced = 0, 0
    for s, e, p in notes_s:
        m = (e - s) * (1 - central) / 2
        a, b = s + m + lag, e - m + lag
        sel = est_f0[(est_t >= a) & (est_t < b)]
        sel = sel[sel > 0]
        if len(sel) < min_voiced and (b - a) / hop >= min_voiced:
            continue
        if len(sel) == 0:
            continue
        voiced += 1
        d = 1200 * np.log2(np.median(sel) / pretty_midi.note_number_to_hz(p))
        if octave_tolerant:
            d = (d + 600) % 1200 - 600
        correct += abs(d) < tol_cents
    n = len(notes_s)
    return {"npa": correct / max(1, n), "voiced_notes": voiced / max(1, n), "n_notes": n}
