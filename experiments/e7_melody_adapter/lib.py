"""Frozen Stage S measurement rules. This module does not import the song model.

A speed ratio above one means a faster performance. Its note times are
``start_s + reference_time / speed_ratio``. Positive transposition means
that the performance is higher than the written melody.
"""
from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np
from numba import njit

from dacapo.metrics import note_metrics, note_pitch_accuracy

HOP = 0.01
SCORING_CONFIG = {
    "version": "stage-s-v1",
    "f0_hop_s": HOP,
    "pitch_tolerance_cents": 50.0,
    "strict_external": {
        "speed_ratios": [round(x / 100, 2) for x in range(90, 111)],
        "speed_definition": "performed tempo / reference tempo; time = start + reference_time / speed",
        "transpositions_semitones": list(range(-12, 13)),
        "coarse_start_step_s": 0.1,
        "fine_start_step_s": HOP,
        "fine_radius_s": 0.1,
        "coarse_candidates_per_speed_and_shift": 2,
        "global_candidates_refined": 16,
        "search_objective": "fraction of written voiced-frame duration matching absolute pitch within 50 cents",
        "reference_regions": "full note durations, without rests",
        "complete_reference_required": True,
        "tie_order": "higher frame accuracy, speed nearer one, smaller absolute shift, earlier start, lower speed, lower signed shift",
        "npa": {"central": 0.6, "min_voiced": 3, "tol_cents": 50.0, "tolerance_operator": "strictly less than"},
        "octave_metric": "same chosen alignment, diagnostic only",
        "note_f1": "unvalidated Basic Pitch notes; same chosen alignment; no second lag or transposition search; onset tolerance 0.05 s, offsets ignored",
    },
    "relaxed": {
        "reference": "10 ms MIDI pitch frames; remove reference rest frames only",
        "output": "10 ms MIDI pitch frames; retain unvoiced frames",
        "steps": [[1, 1], [1, 2], [2, 1]],
        "transpositions_semitones": list(range(-12, 13)),
        "silence_cost_semitones": 3.0,
        "pitch_cost": "absolute semitone error, no octave folding",
        "transition_costs": "(1,1): one cost; (1,2): mean of two output costs for one reference frame; (2,1): sum of two reference costs at one output frame",
        "normalization": "number of non-rest reference frames",
        "subsequence_start": "any output frame, all reference frames consumed",
        "duration_ratio": "0.5 to 2 times non-rest reference frame count, to frame quantization",
        "tie_order": "lower normalized cost, smaller absolute shift, earlier end, lower signed shift; path ties prefer (1,1), (1,2), (2,1)",
        "note_accuracy": "median matched voiced output pitch per reference note within 50 cents; missing notes wrong",
        "octave_metric": "same path and transposition; fold error into [-6, 6] for deviation and note accuracy; diagnostic only",
        "comparison_limit": "units match external studies; references, tools and alignment differ, so scores are not directly comparable",
    },
    "words": {
        "section_tags": "remove text enclosed in square brackets",
        "normalization": "lowercase ASCII words; retain internal apostrophes; normalize curly apostrophe",
        "matching": "longest common subsequence, preserving order and token multiplicity",
        "recall": "matched reference tokens / reference token count; empty reference returns zero",
        "chance": "mean recall against each of the other eleven lyrics",
    },
    "voice_present": {
        "vocal_energy_share_min": 0.2,
        "active_seconds_min": 1.0,
        "active_rms_dbfs_min": -40.0,
        "frame_seconds": 0.05,
        "frame_overlap": False,
        "energy_share": "sum squared vocal samples / sum squared mixture samples over all channels",
        "partial_frame": "use its actual duration and samples",
        "peak_gate": "not used for external audio",
    },
    "chance": "apply the identical full search to every other reference; average their scores",
    "adjustment": "correct minus chance for accuracies and WR; chance minus correct for deviation",
    "bootstrap": {"unit": "item", "draws": 2000, "confidence": 0.95, "seed": 20261027},
    "ruler_roles": {"label": "swift", "evaluator": "fcpe", "selector": "crepe"},
    "stage_s_f0": "swift",
}


def _notes(notes_s: Sequence[Sequence[float]]) -> np.ndarray:
    a = np.asarray(notes_s, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a):
        raise ValueError("notes must be a nonempty sequence of (start_s, end_s, MIDI pitch)")
    if not np.isfinite(a).all() or np.any(a[:, 0] < 0) or np.any(a[:, 1] <= a[:, 0]):
        raise ValueError("note values must be finite with nonnegative onset and positive duration")
    if np.any(np.diff(a[:, 0]) < 0) or np.any(a[1:, 0] < a[:-1, 1] - 1e-8):
        raise ValueError("reference notes must be ordered and monophonic")
    return a


def _track(times: Sequence[float], f0_hz: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    t, f = np.asarray(times, dtype=np.float64), np.asarray(f0_hz, dtype=np.float64)
    if t.ndim != 1 or f.shape != t.shape or len(t) < 2:
        raise ValueError("pitch times and values must be equal one-dimensional arrays with at least two frames")
    if not np.isfinite(t).all() or not np.isfinite(f).all() or np.any(f < 0):
        raise ValueError("pitch track contains nonfinite times or negative/nonfinite Hz")
    if t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("pitch times must increase from a nonnegative origin")
    if not np.allclose(np.diff(t), HOP, atol=1e-6, rtol=0):
        raise ValueError("Stage S pitch tracks must use the registered 10 ms grid")
    return t, f


def _midi(f0: np.ndarray) -> np.ndarray:
    out = np.full(f0.shape, np.nan, dtype=np.float64)
    voiced = f0 > 0
    out[voiced] = 69 + 12 * np.log2(f0[voiced] / 440)
    return out


def _rank(score: float, speed: float, shift: int, start: float) -> tuple:
    return (-round(float(score), 12), round(abs(speed - 1), 8), abs(shift), round(float(start), 8), speed, shift)


def _window_scores(notes: np.ndarray, speed: float, shift: int, starts: np.ndarray,
                   track_start: float, n: int, prefixes: dict) -> np.ndarray:
    """Equal time weighting of all written note frames; rests do not add credit."""
    hits = np.zeros(len(starts), dtype=np.float64)
    total = np.zeros(len(starts), dtype=np.float64)
    for begin, end, pitch in notes:
        # ceil chooses exactly the samples inside the half-open written interval.
        left = np.ceil((starts + begin / speed - track_start) / HOP - 1e-8).astype(np.int64)
        right = np.ceil((starts + end / speed - track_start) / HOP - 1e-8).astype(np.int64)
        a, b = np.clip(left, 0, n), np.clip(right, 0, n)
        prefix = prefixes[float(pitch + shift)]
        hits += prefix[b] - prefix[a]
        total += np.maximum(0, right - left)
    return hits / np.maximum(1, total)


def _empty_strict(estimated_notes) -> dict:
    return {"npa": 0.0, "npa_oct": 0.0, "voiced_notes": 0.0, "start_s": 0.0,
            "speed_ratio": 1.0, "transpose_semitones": 0, "frame_accuracy": 0.0,
            "note_f1_unvalidated": None if estimated_notes is None else 0.0,
            "matched": False}


def strict_external(notes_s, times, f0_hz, *, estimated_notes=None) -> dict:
    """Choose one global offset, tempo ratio and key for all strict metrics.

    Coarse search examines every 100 ms origin, all 21 tempo ratios and all
    25 transpositions. It keeps the best two origins per ratio/shift and
    refines the best 16 candidates on the 10 ms grid within +/-100 ms.
    The objective uses full-note frame accuracy; final NPA uses the existing
    central-window scorer. Diagnostic metrics never select another alignment.
    """
    notes = _notes(notes_s)
    t, f = _track(times, f0_hz)
    midi = _midi(f)
    shifts = SCORING_CONFIG["strict_external"]["transpositions_semitones"]
    speeds = SCORING_CONFIG["strict_external"]["speed_ratios"]
    pitches = {float(p + sh) for p in notes[:, 2] for sh in shifts}
    prefixes = {p: np.r_[0, np.cumsum(np.abs(midi - p) < 0.5, dtype=np.int64)] for p in pitches}
    end_s = t[-1] + HOP
    candidates = []
    for speed in speeds:
        last = end_s - notes[-1, 1] / speed
        if last < t[0] - 1e-9:
            continue
        starts = np.arange(math.ceil(t[0] / 0.1 - 1e-8), math.floor(last / 0.1 + 1e-8) + 1) * 0.1
        if not len(starts):
            continue
        for shift in shifts:
            scores = _window_scores(notes, speed, shift, starts, t[0], len(f), prefixes)
            # Stable order resolves equal origin scores to the earlier origin.
            indices = np.argsort(-scores, kind="stable")[:2]
            candidates.extend((float(scores[i]), speed, shift, float(starts[i])) for i in indices)
    if not candidates:
        return _empty_strict(estimated_notes)
    candidates.sort(key=lambda x: _rank(*x))
    refined = []
    for _, speed, shift, origin in candidates[:16]:
        upper = min(origin + 0.1, end_s - notes[-1, 1] / speed)
        lower = max(t[0], origin - 0.1)
        starts = np.arange(math.ceil(lower / HOP - 1e-8), math.floor(upper / HOP + 1e-8) + 1) * HOP
        if not len(starts):
            continue
        scores = _window_scores(notes, speed, shift, starts, t[0], len(f), prefixes)
        idx = int(np.argmax(scores))
        refined.append((float(scores[idx]), speed, shift, float(starts[idx])))
    score, speed, shift, start = min(refined, key=lambda x: _rank(*x))
    transformed = [(float(a / speed), float(b / speed), float(p + shift)) for a, b, p in notes]
    common = {"central": 0.6, "tol_cents": 50.0, "min_voiced": 3, "lag": start}
    normal = note_pitch_accuracy(transformed, t, f, **common)
    octave = note_pitch_accuracy(transformed, t, f, octave_tolerant=True, **common)
    result = {"npa": float(normal["npa"]), "npa_oct": float(octave["npa"]),
              "voiced_notes": float(normal["voiced_notes"]), "start_s": round(start, 8),
              "speed_ratio": float(speed), "transpose_semitones": int(shift),
              "frame_accuracy": score, "note_f1_unvalidated": None, "matched": bool(score > 0)}
    if estimated_notes is not None:
        # Restrict the transcription to the selected performance, then use
        # exactly the chosen transform. Do not optimize note-F1 separately.
        begin, end = start + notes[0, 0] / speed, start + notes[-1, 1] / speed
        # Keep absolute onsets. Cropping an earlier sustained note to zero
        # would invent a new onset at the chosen segment boundary.
        selected = [(float(a), float(b), float(p))
                    for a, b, p in estimated_notes if a < end and b > begin]
        intervals = notes[:, :2] / speed + start
        hz = 440 * 2 ** ((notes[:, 2] + shift - 69) / 12)
        result["note_f1_unvalidated"] = note_metrics(intervals, hz, selected, onset_tol=0.05,
                                                       transpose_invariant=False, max_lag=0.0)["F"]
    return result


def _reference_frames(notes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0, notes[-1, 1], HOP)
    pitch = np.full(len(times), np.nan)
    ids = np.full(len(times), -1, dtype=np.int64)
    for i, (a, b, p) in enumerate(notes):
        mask = (times >= a - 1e-9) & (times < b - 1e-9)
        pitch[mask], ids[mask] = p, i
    valid = ids >= 0
    if not valid.any():
        raise ValueError("reference has no frames on the registered 10 ms grid")
    return pitch[valid], ids[valid]


@njit(cache=True)
def _cost(ref: float, observed: float, shift: int) -> float:
    if np.isnan(observed):
        return 3.0
    return abs(observed - (ref + shift))


@njit(cache=True)
def _dtw_cost(ref: np.ndarray, observed: np.ndarray, shift: int):
    """Two rolling rows, exact registered transitions, no distance downsampling."""
    n, m = len(ref), len(observed)
    previous2 = np.full(m, np.inf)
    previous = np.empty(m)
    # Start anywhere. For one reference frame the single-frame start wins
    # ties; a two-output-frame start is not needed for a lower endpoint cost.
    for j in range(m):
        previous[j] = _cost(ref[0], observed[j], shift)
    for i in range(1, n):
        current = np.full(m, np.inf)
        for j in range(m):
            c = _cost(ref[i], observed[j], shift)
            if j >= 1:
                current[j] = previous[j - 1] + c
            if j >= 2:
                candidate = previous[j - 2] + (_cost(ref[i], observed[j - 1], shift) + c) / 2
                if candidate < current[j]:
                    current[j] = candidate
            if i >= 2 and j >= 1:
                candidate = previous2[j - 1] + _cost(ref[i - 1], observed[j], shift) + c
                if candidate < current[j]:
                    current[j] = candidate
        previous2, previous = previous, current
    endpoint = int(np.argmin(previous))
    return previous[endpoint] / n, endpoint


@njit(cache=True)
def _dtw_trace(ref: np.ndarray, observed: np.ndarray, shift: int):
    """Repeat only the winning key and store one byte per path decision."""
    n, m = len(ref), len(observed)
    directions = np.zeros((n, m), dtype=np.uint8)
    previous2 = np.full(m, np.inf)
    previous = np.empty(m)
    for j in range(m):
        previous[j] = _cost(ref[0], observed[j], shift)
    for i in range(1, n):
        current = np.full(m, np.inf)
        for j in range(m):
            c = _cost(ref[i], observed[j], shift)
            if j >= 1:
                current[j] = previous[j - 1] + c
                directions[i, j] = 1
            if j >= 2:
                candidate = previous[j - 2] + (_cost(ref[i], observed[j - 1], shift) + c) / 2
                if candidate < current[j]:
                    current[j] = candidate
                    directions[i, j] = 2
            if i >= 2 and j >= 1:
                candidate = previous2[j - 1] + _cost(ref[i - 1], observed[j], shift) + c
                if candidate < current[j]:
                    current[j] = candidate
                    directions[i, j] = 3
        previous2, previous = previous, current
    endpoint = int(np.argmin(previous))
    # A path maps at most two observed frames per reference frame.
    ref_indices = np.empty(n * 2, dtype=np.int64)
    out_indices = np.empty(n * 2, dtype=np.int64)
    count, i, j = 0, n - 1, endpoint
    if not np.isfinite(previous[endpoint]):
        return previous[endpoint], endpoint, ref_indices[:0], out_indices[:0]
    while i > 0:
        step = directions[i, j]
        ref_indices[count], out_indices[count] = i, j
        count += 1
        if step == 1:
            i, j = i - 1, j - 1
        elif step == 2:
            ref_indices[count], out_indices[count] = i, j - 1
            count += 1
            i, j = i - 1, j - 2
        elif step == 3:
            ref_indices[count], out_indices[count] = i - 1, j
            count += 1
            i, j = i - 2, j - 1
        else:
            break
    ref_indices[count], out_indices[count] = 0, j
    count += 1
    return previous[endpoint] / n, endpoint, ref_indices[:count], out_indices[:count]


def relaxed_alignment(notes_s, times, f0_hz) -> dict:
    """Exact subsequence DTW with reference-frame-weighted semitone cost.

    Only reference rest frames are removed. Silent output frames remain at
    cost three. Each skipped coordinate of a compound step is charged, so
    skipping every second reference frame cannot halve the reported error.
    The winning path determines the note score and its octave diagnostic.
    """
    notes = _notes(notes_s)
    t, f = _track(times, f0_hz)
    reference, note_ids = _reference_frames(notes)
    observed = _midi(f)
    candidates = []
    for shift in range(-12, 13):
        cost, endpoint = _dtw_cost(reference, observed, shift)
        candidates.append((float(cost), abs(shift), int(endpoint), shift))
    deviation, _, _, shift = min(candidates)
    if not np.isfinite(deviation):
        return {"deviation_semitones": None, "octave_deviation_semitones": None, "note_accuracy": 0.0, "octave_note_accuracy": 0.0,
                "start_s": None, "end_s": None, "transpose_semitones": 0,
                "reference_frames": len(reference), "matched": False}
    _, endpoint, ri, oi = _dtw_trace(reference, observed, shift)
    errors = observed[oi] - (reference[ri] + shift)
    folded_cost = np.where(np.isfinite(errors), np.abs((errors + 6) % 12 - 6), 3.0)
    sums = np.bincount(ri, weights=folded_cost, minlength=len(reference))
    counts = np.bincount(ri, minlength=len(reference))
    octave_deviation = float(np.mean(sums / np.maximum(1, counts)))
    correct, octave = 0, 0
    for idx, (_, _, pitch) in enumerate(notes):
        selected = observed[oi[note_ids[ri] == idx]]
        selected = selected[np.isfinite(selected)]
        if not len(selected):
            continue
        d = float(np.median(selected) - (pitch + shift))
        correct += abs(d) < 0.5
        octave += abs((d + 6) % 12 - 6) < 0.5
    return {"deviation_semitones": float(deviation), "octave_deviation_semitones": octave_deviation, "note_accuracy": float(correct / len(notes)),
            "octave_note_accuracy": float(octave / len(notes)), "start_s": float(t[int(oi.min())]),
            "end_s": float(t[endpoint] + HOP), "transpose_semitones": int(shift),
            "reference_frames": len(reference), "matched": bool(np.isfinite(observed[oi]).any())}


def lyric_words(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]", " ", str(text)).lower().replace("’", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text)


def word_recall(reference: str, hypothesis: str) -> float:
    """Ordered matches cannot reuse a recognized token for repeated lyrics."""
    ref, hyp = lyric_words(reference), lyric_words(hypothesis)
    if not ref:
        return 0.0
    previous = [0] * (len(hyp) + 1)
    for expected in ref:
        current = [0]
        for j, actual in enumerate(hyp, 1):
            current.append(previous[j - 1] + 1 if expected == actual else max(previous[j], current[-1]))
        previous = current
    return previous[-1] / len(ref)


def word_scores(reference: str, others: Sequence[str], hypothesis: str) -> dict:
    if not others:
        raise ValueError("chance scoring needs other lyrics")
    own = word_recall(reference, hypothesis)
    chance = float(np.mean([word_recall(other, hypothesis) for other in others]))
    return {"wr": own, "wr_chance": chance, "wr_adjusted": own - chance}


def voice_present(vocals, mixture, sr: int) -> dict:
    """Evaluate the registered external-audio gate on aligned audio arrays."""
    vocal, mix = np.asarray(vocals, dtype=np.float64), np.asarray(mixture, dtype=np.float64)
    if sr <= 0 or vocal.ndim not in (1, 2) or mix.ndim not in (1, 2):
        raise ValueError("audio must be samples or samples-by-channels with a positive sample rate")
    if vocal.shape != mix.shape or not len(vocal) or not np.isfinite(vocal).all() or not np.isfinite(mix).all():
        raise ValueError("vocal and mixture audio must have equal, nonempty, finite shapes")
    energy = float(np.square(mix).sum())
    share = float(np.square(vocal).sum() / energy) if energy > 0 else 0.0
    frame = max(1, round(sr * 0.05))
    active = 0.0
    for start in range(0, len(vocal), frame):
        chunk = vocal[start:start + frame]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms >= 10 ** (-40 / 20):
            active += len(chunk) / sr
    return {"energy_share": share, "active_seconds": active,
            "voice_present": bool(share >= 0.2 and active >= 1.0 - 1e-9)}


def score_reference(notes_s, times, f0_hz, *, estimated_notes=None) -> dict:
    return {"strict": strict_external(notes_s, times, f0_hz, estimated_notes=estimated_notes),
            "relaxed": relaxed_alignment(notes_s, times, f0_hz)}


def score_against_references(own_notes, other_notes, times, f0_hz, *, estimated_notes=None) -> dict:
    """Use exactly the same searches for the intended and mismatched melodies."""
    if not other_notes:
        raise ValueError("chance scoring needs at least one mismatched reference")
    own = score_reference(own_notes, times, f0_hz, estimated_notes=estimated_notes)
    other = [score_reference(notes, times, f0_hz, estimated_notes=estimated_notes) for notes in other_notes]
    keys = {"strict": ["npa", "npa_oct", "voiced_notes", "frame_accuracy", "note_f1_unvalidated"],
            "relaxed": ["deviation_semitones", "octave_deviation_semitones", "note_accuracy", "octave_note_accuracy"]}
    chance, adjusted = {}, {}
    for group, metrics in keys.items():
        chance[group], adjusted[group] = {}, {}
        for key in metrics:
            values = [r[group][key] for r in other]
            value = None if any(v is None for v in values) else float(np.mean(values))
            chance[group][key] = value
            correct = own[group][key]
            adjusted[group][key] = (None if correct is None or value is None else
                                     value - correct if key.endswith("deviation_semitones") else correct - value)
    return {"own": own, "chance": chance, "adjusted": adjusted, "n_mismatched": len(other)}


def bootstrap_interval(values, *, seed: int = 20261027, draws: int = 2000, statistic: str = "mean") -> list[float]:
    """Item-level percentile interval. Supply paired item differences for comparisons."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or draws < 1:
        raise ValueError("bootstrap needs finite item values and positive draw count")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be mean or median")
    rng = np.random.default_rng(seed)
    sample = a[rng.integers(0, len(a), size=(draws, len(a)))]
    estimates = sample.mean(axis=1) if statistic == "mean" else np.median(sample, axis=1)
    return [float(x) for x in np.percentile(estimates, [2.5, 97.5])]


def validate_tool_roles(label: str, evaluator: str, selector: str) -> None:
    aliases = {"swiftf0": "swift", "swift-f0": "swift", "torchfcpe": "fcpe", "torchcrepe": "crepe"}
    names = [str(x).strip().lower() if x is not None else "" for x in (label, evaluator, selector)]
    canonical = [aliases.get(name, name) for name in names]
    if not all(canonical) or len(set(canonical)) != 3:
        raise ValueError("label, evaluator and best-of-N selector must be different named tools")
