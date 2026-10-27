"""E5 v2 statistics and decision rules: pure functions on scored rows (no audio, no files).

Row schema (v2): id, split, cond, arm, seed, status ('ok' | 'missing' | 'lost'), acc{h}, acc_oct{h},
nf1{h}, of1{h}, quality{...}, gate{voiced, poly, level, pass, ...}.
v1 rows (results/gpu_run1/e5_main.json) carry acc/acc_oct for the v1 classification sets only and no
gate dict. Every function takes the hypothesis keys and the population predicate explicitly, so the
same code reproduces the v1 summary (population ALL, first-listed ties) and computes v2.

Populations (addendum 2): ALL, every completed draw, is primary and decides H1-H5 with the registered
estimator. PASS, the draws whose registered quality gate passed, is secondary. A clause that compares two
conditions (H2, the RI_k - RI_k(X-ctl) clause of H3, the H4 test against X-ctl, K3) must also hold in
PASS, where the comparison is paired by melody over melodies with passing draws in both conditions.

A draw is keyed by (id, cond, arm, seed) everywhere: Phase 0b has several arms per (id, cond, seed).
"""
from __future__ import annotations

import hashlib
import math
import warnings
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np
from scipy.stats import binom, binomtest
from scipy.stats import t as student_t

# Registered classification sets: identical to score_analogy.SETS (v1). Unchanged in v2.
CLASS_SETS: dict[str, list[str]] = {
    "lit": ["B", "B~", "A", "D"],
    "ctl": ["B", "B~", "C", "D"],
    "rel+2": ["T+2B", "T+2B~", "B", "B~", "T+2A", "D", "T+1B", "T+3B"],
    "rel+5": ["T+5B", "T+5B~", "B", "B~", "T+5A", "D", "T+4B", "T+6B"],
    "rel-3": ["T-3B", "T-3B~", "B", "B~", "T-3A", "D", "T-4B", "T-2B"],
    "seq": ["T+6A", "T+4A", "T+2A", "T+5A", "T+7A", "D"],
    "ana": ["T+5B", "T+5B~", "B", "T+5A", "D"],
    "null": ["B", "B~", "A", "D"],
}
RULE_ANSWER = {"lit": "B", "rel+2": "T+2B", "rel+5": "T+5B", "rel-3": "T-3B", "seq": "T+6A", "ana": "T+5B"}
KS = ("+2", "+5", "-3")
N_BOOT, BOOT_SEED, PCT = 10000, 0, (2.5, 97.5)
MIN_MELODIES = 12          # registered sample size: a quantity on fewer melodies is 'not_evaluable'
TIE_EPS = 1e-9
ALPHA_H4 = 0.01

Row = dict
Pred = Callable[[Row], bool]


def draw_key(r: Row) -> tuple[str, str, str, str]:
    return (r["id"], r["cond"], r.get("arm", "base"), r["seed"])


def _tiebreak(r: Row) -> str:
    """Deterministic, content-free order among equal values (used only by Lee trimming)."""
    return hashlib.sha256("|".join(draw_key(r)).encode()).hexdigest()


def pop_all(r: Row) -> bool:
    """ALL (primary): every completed and scored draw, whatever its gate status. Draws without audio
    are not in ALL; the quality decision counts them as gate failures."""
    return r.get("status", "ok") == "ok"


def pop_pass(r: Row) -> bool:
    """PASS (secondary): draws whose frozen, hypothesis-blind registered quality gate passed."""
    return pop_all(r) and bool(r["gate"]["pass"])


def gate_v1(r: Row) -> bool:
    """The Phase-0 rule in run_analogy.gate_choice (v1 rows only; v2 rows carry r['gate'])."""
    return r["gap_voiced"] >= 0.6 and r["gap_polyphony"] <= 1.3 and abs(r["gap_rms_db_vs_context"]) <= 6


def classify(accs: dict, hyps: list[str], tau: float, tie: str = "strict") -> str:
    """argmax over `hyps` with the render-calibrated tau.
    tie='strict' (v2): several hypotheses within TIE_EPS of the maximum -> 'tie' (credited to none).
    tie='first' (v1, regression only): the first-listed hypothesis wins."""
    sub = [(h, accs[h]) for h in hyps]
    if tie == "first":
        h, a = max(sub, key=lambda kv: kv[1])
        return h if a >= tau else "none"
    m = max(a for _, a in sub)
    if m < tau:
        return "none"
    winners = [h for h, a in sub if a >= m - TIE_EPS]
    return winners[0] if len(winners) == 1 else "tie"


def missing_hyps(rows: list[Row], hyps: Iterable[str], key: str = "acc") -> set[str]:
    have = set.intersection(*[set(r[key]) for r in rows]) if rows else set()
    return set(hyps) - have


def boot_index_matrix(n: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, n) resample indices from the exact random stream of v1 score_analogy.index():
    a fresh default_rng(seed), one rng.choice per replicate (rng.choice(vals, n) draws the same
    integers as rng.choice(n, n); test_boot_stream_matches_v1 pins this)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(n, n) for _ in range(n_boot)]) if n else np.zeros((n_boot, 0), int)


def boot_mean_ci(vals, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """v1 cluster bootstrap of a mean over clusters (loop kept verbatim for the regression)."""
    vals = np.asarray(list(vals), float)
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(vals, len(vals)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(b, PCT, method="linear")
    return float(vals.mean()), float(lo), float(hi)


def t_ci(vals) -> tuple[float, float] | None:
    """Disclosed sensitivity only: t interval over melodies."""
    v = np.asarray(list(vals), float)
    if len(v) < 2:
        return None
    h = float(student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))
    return float(v.mean() - h), float(v.mean() + h)


def boot_nested_ci(cells: dict[str, list[float]], n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Two-stage bootstrap (melodies, then draws within each resampled melody). Sensitivity only."""
    mels = sorted(cells)
    arr = [np.asarray(cells[m], float) for m in mels]
    if not arr:
        return None
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(mels), len(mels))
        b[i] = np.mean([arr[j][rng.integers(0, len(arr[j]), len(arr[j]))].mean() for j in pick])
    lo, hi = np.percentile(b, PCT, method="linear")
    return float(np.mean([a.mean() for a in arr])), float(lo), float(hi)


def boot_crossed_ci(mat: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Two-way (melody x seed label) bootstrap; NaN marks an absent draw. Primary only when
    design['ci_method'] == 'crossed' (one noise draw shared by all melodies, as in v1)."""
    mat = np.asarray(mat, float)
    I, J = mat.shape
    rng = np.random.default_rng(seed)

    def stat(m):
        rowm = np.array([np.nanmean(x) if np.isfinite(x).any() else np.nan for x in m])
        return np.nanmean(rowm)

    b = np.empty(n_boot)
    for i in range(n_boot):
        b[i] = stat(mat[np.ix_(rng.integers(0, I, I), rng.integers(0, J, J))])
    b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, PCT, method="linear")
    return float(stat(mat)), float(lo), float(hi)


def _select(rows, cond, include, arm="base", split="confirm"):
    sel = [r for r in rows if r["cond"] == cond and r.get("arm", "base") == arm
           and r.get("split", "confirm") == split and include(r)]
    return sorted(sel, key=lambda r: (r["id"], r["seed"]))


def per_melody(rows, cond, value: Callable[[Row], float], include: Pred, **kw) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in _select(rows, cond, include, **kw):
        by[r["id"]].append(value(r))
    return {m: float(np.mean(v)) for m, v in sorted(by.items())}


def per_melody_cells(rows, cond, value, include, **kw) -> dict[str, list[float]]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in _select(rows, cond, include, **kw):
        by[r["id"]].append(value(r))
    return dict(sorted(by.items()))


def contrast(pos: str, neg: str, key: str = "acc") -> Callable[[Row], float]:
    def f(r):
        a, b = r[key][pos], r[key][neg]
        if a is None or b is None:
            raise ValueError(f"{draw_key(r)}: {key}[{pos}] or {key}[{neg}] is not scorable")
        return a - b
    return f


def pack(pm: dict, n_boot: int = N_BOOT, extra: dict | None = None) -> dict | None:
    """Registered estimator on per-melody values: mean over melodies (sorted ids) and the
    percentile cluster bootstrap over melodies."""
    if not pm:
        return None
    vals = np.array(list(pm.values()), float)
    b = vals[boot_index_matrix(len(vals), n_boot)].mean(axis=1)
    lo, hi = np.percentile(b, PCT, method="linear")
    out = {"mean": float(vals.mean()), "lo": float(lo), "hi": float(hi), "n_melodies": len(pm),
           "evaluable": len(pm) >= MIN_MELODIES, "t_ci": t_ci(vals)}
    out.update(extra or {})
    return out


def index(rows, cond, pos, neg, include: Pred, key: str = "acc", n_boot: int = N_BOOT, **kw) -> dict | None:
    """Registered index: per melody, mean over its draws in the population of key[pos] - key[neg];
    mean over melodies; 95% percentile cluster bootstrap over melodies."""
    pm = per_melody(rows, cond, contrast(pos, neg, key), include, **kw)
    return pack(pm, n_boot, {"n_draws": len(_select(rows, cond, include, **kw))})


def _diff_from_pm(pa: dict, pb: dict, n_boot: int) -> dict | None:
    """Difference of two indices with a joint melody bootstrap: the point estimate is
    index_a - index_b, each over its own melodies (the registered quantity); each bootstrap
    replicate resamples melodies from the union, so the two terms stay paired where a melody has
    both. With identical melody sets this is the paired bootstrap of per-melody differences."""
    if not pa or not pb:
        return None
    mels = sorted(set(pa) | set(pb))
    va = np.array([pa.get(m, np.nan) for m in mels])
    vb = np.array([pb.get(m, np.nan) for m in mels])
    idx = boot_index_matrix(len(mels), n_boot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a replicate may hold no melody of one term
        b = np.nanmean(va[idx], axis=1) - np.nanmean(vb[idx], axis=1)
    b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, PCT, method="linear")
    common = sorted(set(pa) & set(pb))
    paired = float(np.mean([pa[m] - pb[m] for m in common])) if common else None
    return {"mean": float(np.mean(list(pa.values())) - np.mean(list(pb.values()))), "lo": float(lo), "hi": float(hi),
            "n_melodies_a": len(pa), "n_melodies_b": len(pb), "n_common": len(common),
            "paired_common_mean": paired,
            "evaluable": len(pa) >= MIN_MELODIES and len(pb) >= MIN_MELODIES}


def diff_index(rows, a: tuple, b: tuple, include: Pred, key: str = "acc", n_boot: int = N_BOOT) -> dict | None:
    """a, b = (cond, pos, neg). H2 uses (lit,B,B~) - (ctl,B,B~); H3 uses RI_k - RI_k(X-ctl)."""
    pa = per_melody(rows, a[0], contrast(a[1], a[2], key), include)
    pb = per_melody(rows, b[0], contrast(b[1], b[2], key), include)
    return _diff_from_pm(pa, pb, n_boot)


def diff_cells(rows, a: tuple, b: tuple, include: Pred, key: str = "acc", n_boot: int = N_BOOT) -> dict | None:
    """Sensitivity: difference restricted to (melody, arm, seed) draws in the population in both
    conditions (H2 'both pass')."""
    fa, fb = contrast(a[1], a[2], key), contrast(b[1], b[2], key)
    ra = {(r["id"], r.get("arm", "base"), r["seed"]): r for r in _select(rows, a[0], include)}
    rb = {(r["id"], r.get("arm", "base"), r["seed"]): r for r in _select(rows, b[0], include)}
    by: dict[str, list[float]] = defaultdict(list)
    for k in sorted(set(ra) & set(rb)):
        by[k[0]].append(fa(ra[k]) - fb(rb[k]))
    return pack({m: float(np.mean(v)) for m, v in sorted(by.items())}, n_boot)


def paired_diff(rows, a: tuple, b: tuple, include: Pred, key: str = "acc", n_boot: int = N_BOOT) -> dict | None:
    """Comparison of two conditions paired by melody: per-melody contrast in a minus per-melody contrast
    in b, over the melodies that have draws of both conditions in the population; mean over those
    melodies and the registered cluster bootstrap. Evaluable needs MIN_MELODIES common melodies. In ALL
    every melody has both conditions, and this equals diff_index."""
    pa = per_melody(rows, a[0], contrast(a[1], a[2], key), include)
    pb = per_melody(rows, b[0], contrast(b[1], b[2], key), include)
    common = sorted(set(pa) & set(pb))
    out = pack({m: pa[m] - pb[m] for m in common}, n_boot)
    if out is not None:
        out.update(n_melodies_a=len(pa), n_melodies_b=len(pb), n_common=len(common))
    return out


# Selection sensitivity of PASS contrasts (Lee bounds), exploratory.
def pass_rate(rows, cond) -> float | None:
    sel = _select(rows, cond, pop_all)
    return float(np.mean([bool(r["gate"]["pass"]) for r in sel])) if sel else None


def lee_trim(rows, cond, value: Callable[[Row], float], q: float, side: str) -> set:
    """Keys of the PASS draws of `cond` removed by trimming a share q of them from the top
    (largest values) or bottom. Ties are broken by a content-free hash of the draw key."""
    sel = _select(rows, cond, pop_pass)
    k = int(math.floor(q * len(sel) + 0.5))
    order = sorted(sel, key=lambda r: (value(r), _tiebreak(r)), reverse=(side == "top"))
    return {draw_key(r) for r in order[:k]}


def lee_bounds_diff(rows, a: tuple, b: tuple, value_a=None, value_b=None, n_boot: int = N_BOOT,
                    diff_fn=None) -> dict | None:
    """Selection-sensitivity bounds for a PASS contrast between two conditions (Lee 2009, pooled
    trimming). Assumption: a draw that would pass in the lower-pass condition would also pass in the
    higher-pass one. The higher-pass condition's PASS draws are trimmed by (p_hi - p_lo) / p_hi from
    the top (or bottom) of the contrast distribution; the lower bound is the difference computed on
    the trimmed draws whose removal favours the null."""
    value_a = value_a or contrast(a[1], a[2])
    value_b = value_b or contrast(b[1], b[2])
    diff_fn = diff_fn or (lambda rr: paired_diff(rr, a, b, pop_pass, n_boot=n_boot))
    pa, pb = pass_rate(rows, a[0]), pass_rate(rows, b[0])
    if not pa or not pb:
        return None
    if pa == pb:
        d = diff_fn(rows)
        return {"p_a": pa, "p_b": pb, "q": 0.0, "trimmed": None, "lower": d, "upper": d}
    hi_cond, val = (a[0], value_a) if pa > pb else (b[0], value_b)
    q = abs(pa - pb) / max(pa, pb)
    # lower bound of (a - b): remove a's best draws if a passes more, else b's worst draws
    low_side, up_side = ("top", "bottom") if hi_cond == a[0] else ("bottom", "top")
    drop_lo = lee_trim(rows, hi_cond, val, q, low_side)
    drop_up = lee_trim(rows, hi_cond, val, q, up_side)
    keep = lambda drop: [r for r in rows if draw_key(r) not in drop]
    return {"p_a": pa, "p_b": pb, "q": q, "trimmed": hi_cond, "n_trimmed": len(drop_lo),
            "lower": diff_fn(keep(drop_lo)), "upper": diff_fn(keep(drop_up))}


def item_rates(rows, cond, set_name, target, include: Pred, tau: float, tie: str = "strict",
               n_boot: int = N_BOOT) -> dict | None:
    """Per-melody share of `cond` draws classified as `target` with CLASS_SETS[set_name], averaged
    over melodies. set_name != cond gives coincidence rates (null or ctl draws, seq set)."""
    hyps = CLASS_SETS[set_name]
    sel = _select(rows, cond, include)
    if not sel:
        return None
    miss = missing_hyps(sel, hyps)
    if miss:
        return {"error": f"rows of {cond} lack scores for {sorted(miss)}"}
    by: dict[str, list[int]] = defaultdict(list)
    for r in sel:
        by[r["id"]].append(int(classify(r["acc"], hyps, tau, tie) == target))
    rates = {m: float(np.mean(v)) for m, v in sorted(by.items())}
    k, n = sum(map(sum, by.values())), sum(map(len, by.values()))
    p = pack(rates, n_boot)
    cp = binomtest(k, n).proportion_ci(method="exact")
    return {"rate": p["mean"], "lo": p["lo"], "hi": p["hi"], "n_items": len(rates), "k": k, "n": n,
            "cp95_unclustered": [float(cp.low), float(cp.high)], "per_item": rates,
            "evaluable": len(rates) >= MIN_MELODIES}


def sign_test_greater(d: Iterable[float], eps: float = 1e-12) -> dict:
    """One-sided exact sign test (binomial, p = 0.5) that d > 0; zeros dropped."""
    d = [x for x in d if abs(x) > eps]
    n_pos = sum(x > 0 for x in d)
    p = float(binom.sf(n_pos - 1, len(d), 0.5)) if d else 1.0
    return {"n_pos": n_pos, "n_neg": len(d) - n_pos, "n": len(d), "p": p}


def binom_vs_rate(k: int, n: int, k_c: int, n_c: int) -> dict:
    """Registered-form H4 test: one-sided exact binomial test of k seq hits in n seq draws against
    the comparator's pooled rate p0 = max(k_c, 0.5) / n_c (a zero count is floored at half a count,
    so the test stays defined when the comparator never shows T+6A). Draw-level, so it ignores
    clustering; that is why H4 also needs the melody-level sign test."""
    if n == 0 or n_c == 0:
        return {"p": 1.0, "p0": None, "k": k, "n": n, "k_c": k_c, "n_c": n_c}
    p0 = max(k_c, 0.5) / n_c
    return {"p": float(binomtest(k, n, p0, alternative="greater").pvalue), "p0": p0,
            "p0_literal": (k_c / n_c) if k_c else None, "k": k, "n": n, "k_c": k_c, "n_c": n_c}


def mcnemar_exact(pairs: Iterable[tuple[bool, bool]]) -> dict:
    pairs = list(pairs)
    b = sum(x and not y for x, y in pairs)
    c = sum(y and not x for x, y in pairs)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * float(binom.cdf(min(b, c), n, 0.5)))
    return {"only_first": b, "only_second": c, "p": p}


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k from n draws with c successes (Chen et al. 2021)."""
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_summary(rows, cond, target, include, tau, k=2, n_boot=N_BOOT) -> dict | None:
    hyps = CLASS_SETS[cond]
    by: dict[str, list[int]] = defaultdict(list)
    for r in _select(rows, cond, include):
        by[r["id"]].append(int(classify(r["acc"], hyps, tau) == target))
    vals = {m: pass_at_k(len(v), sum(v), k) for m, v in by.items()}
    vals = {m: v for m, v in sorted(vals.items()) if v is not None}
    if not vals:
        return None
    p = pack(vals, n_boot)
    return {f"pass@{k}": p["mean"], "lo": p["lo"], "hi": p["hi"],
            "pass@1": float(np.mean([np.mean(v) for _, v in sorted(by.items())])), "n_items": len(vals)}




# Quality decisions. 'missing' (never completed) and 'lost' (completed, audio gone before the freeze)
# draws are pending: they block a decision until the generation phase can no longer run them, after
# which (missing_as_failure=True) they count as gate failures.

def _pending(got: dict, k: tuple) -> bool:
    return got.get(k, {}).get("status", "missing") in ("missing", "lost")


def quality_decision(rows, planned: list[tuple], frac_draws: float = 0.75, frac_cells: float = 0.75,
                     min_pass_per_cell: int = 2, missing_as_failure: bool = False, gate_key: str = "pass") -> dict:
    """Confirmatory rules: Q1, at least frac_draws of the non-null planned draws pass the gate; Q2, at least
    frac_cells of the non-null (melody, condition, arm) cells have at least min_pass_per_cell passing
    draws. Null draws enter neither rule. planned: draw keys (id, cond, arm, seed)."""
    got = {draw_key(r): r for r in rows}
    planned = [tuple(k) for k in planned]
    keys = [k for k in planned if k[1] != "null"]
    if not keys:
        return {"decision": "incomplete", "reason": "no non-null planned draws"}
    n_pending = sum(_pending(got, k) for k in planned)
    n_missing = sum(got.get(k, {}).get("status", "missing") == "missing" for k in planned)
    n_lost = sum(got.get(k, {}).get("status") == "lost" for k in planned)

    def ok(k):
        r = got.get(k)
        return bool(r is not None and r.get("status") == "ok" and r["gate"][gate_key])

    passed = {k: ok(k) for k in keys}
    cells: dict[tuple, int] = defaultdict(int)
    for k in keys:
        cells[(k[0], k[1], k[2])] += passed[k]
    n_pass = sum(passed.values())
    n_cells_ok = sum(v >= min_pass_per_cell for v in cells.values())
    need_draws, need_cells = math.ceil(frac_draws * len(keys) - 1e-9), math.ceil(frac_cells * len(cells) - 1e-9)
    if n_pending and not missing_as_failure:
        decision = "incomplete"
    elif n_pass >= need_draws and n_cells_ok >= need_cells:
        decision = "pass"
    else:
        decision = "fail"
    return {"decision": decision, "n_pass": n_pass, "n_nonnull": len(keys), "need_pass": need_draws,
            "draw_pass_rate": n_pass / len(keys), "n_cells_ok": n_cells_ok, "n_cells": len(cells),
            "need_cells": need_cells, "cell_rate": n_cells_ok / len(cells), "min_pass_per_cell": min_pass_per_cell,
            "n_missing": n_missing, "n_lost": n_lost, "missing_as_failure": missing_as_failure, "gate_key": gate_key}


def pilot_decision(rows, planned: list[tuple], arm_order: list[str], k1_frac: float = 0.75,
                   excluded: Iterable[str] = (), per_melody_min_frac: float | None = None,
                   missing_as_failure: bool = False, gate_key: str = "pass") -> dict:
    """Phase 0b (K1): per arm (configuration cell), the passing share of its planned non-null draws. An
    arm passes K1 at >= k1_frac (84 of 112 at full scale). The selected arm is the K1-passing, non-excluded
    arm with the most passing draws; ties go to the arm listed first in arm_order. K1 triggers if no arm
    passes."""
    got = {draw_key(r): r for r in rows}
    plan = [tuple(k) for k in planned if k[1] != "null"]
    if not plan:
        return {"decision": "incomplete", "reason": "no non-null planned pilot draws"}
    n_pending = sum(_pending(got, k) for k in plan)
    if n_pending and not missing_as_failure:
        return {"decision": "incomplete", "n_pending": n_pending, "n_planned": len(plan)}
    per_arm = {}
    for arm in arm_order:
        ks = [k for k in plan if k[2] == arm]
        if not ks:
            continue
        ok = [bool(got.get(k, {}).get("status") == "ok" and got[k]["gate"][gate_key]) for k in ks]
        by_mel: dict[str, list[bool]] = defaultdict(list)
        for k, o in zip(ks, ok):
            by_mel[k[0]].append(o)
        mel_ok = per_melody_min_frac is None or all(np.mean(v) >= per_melody_min_frac for v in by_mel.values())
        need = math.ceil(k1_frac * len(ks) - 1e-9)
        per_arm[arm] = {"n_pass": sum(ok), "n_planned": len(ks), "need": need, "rate": sum(ok) / len(ks),
                        "k1_pass": bool(sum(ok) >= need and mel_ok), "excluded": arm in set(excluded)}
    elig = [a for a in arm_order if a in per_arm and per_arm[a]["k1_pass"] and not per_arm[a]["excluded"]]
    out = {"per_arm": per_arm, "n_pending": n_pending, "missing_as_failure": missing_as_failure, "gate_key": gate_key}
    if not elig:
        return {"decision": "k1_triggered", **out}
    best = max(elig, key=lambda a: (per_arm[a]["n_pass"], -arm_order.index(a)))
    return {"decision": "selected", "arm": best, **out}


# Verdicts. Values are rounded to 1e-9 before a threshold comparison: that removes floating-point noise
# (a mean of exact fractions such as 0.1 can come out as 0.09999999999999999) and nothing else.

def _r(x: float) -> float:
    return round(float(x), 9)


def _ok(ix) -> bool:
    return ix is not None and "error" not in ix and bool(ix.get("evaluable", False))


def verdict_threshold(ix, thr: float) -> str:
    """'supported' iff the mean, rounded to 1e-9, is >= thr and the rounded percentile lower bound is > 0."""
    if not _ok(ix):
        return "not_evaluable"
    return "supported" if _r(ix["mean"]) >= thr and _r(ix["lo"]) > 0 else "not_supported"


def verdict_comparison(all_d, pass_d, thr: float, need_lo: bool = True) -> str:
    """A clause that compares two conditions holds only if it holds in ALL (registered difference) and in
    PASS (paired over melodies with passing draws in both conditions). A failing ALL clause decides."""
    holds = lambda d: _r(d["mean"]) >= thr and (_r(d["lo"]) > 0 if need_lo else True)
    if not _ok(all_d):
        return "not_evaluable"
    if not holds(all_d):
        return "not_supported"
    if not _ok(pass_d):
        return "not_evaluable"
    return "supported" if holds(pass_d) else "not_supported"


def verdict_h3(ri: dict, ri_minus_li: dict, ctl_all: dict, ctl_pass: dict, li: dict) -> dict:
    """Registered: met for k if RI_k >= 0.15 with CI > 0, RI_k - RI_k(X-ctl) >= 0.10 and RI_k > LI_k;
    supported if met for at least 2 of the 3 k. RI_k and RI_k - LI_k come from ALL; the X-ctl clause must
    hold in ALL and PASS. The literal-lag copier (LI_k >= 0.15 with the RI_k CI containing 0) is reported
    per k, as registered."""
    per_k, met, unevaluable = {}, 0, 0
    for k in KS:
        if not (_ok(ri[k]) and _ok(ri_minus_li[k])):
            per_k[k] = "not_evaluable"
            unevaluable += 1
            continue
        own = _r(ri[k]["mean"]) >= 0.15 and _r(ri[k]["lo"]) > 0 and _r(ri_minus_li[k]["mean"]) > 0
        cmp_ = verdict_comparison(ctl_all[k], ctl_pass[k], 0.10, need_lo=False)
        if not own or cmp_ == "not_supported":
            per_k[k] = "not_met"
        elif cmp_ == "not_evaluable":
            per_k[k] = "not_evaluable"
            unevaluable += 1
        else:
            per_k[k] = "met"
            met += 1
    verdict = "supported" if met >= 2 else ("not_evaluable" if met + unevaluable >= 2 else "not_supported")
    lag = {k: (bool(_r(li[k]["mean"]) >= 0.15 and _r(ri[k]["lo"]) <= 0 <= _r(ri[k]["hi"])) if _ok(li[k]) and _ok(ri[k]) else None)
           for k in KS}
    return {"verdict": verdict, "per_k": per_k, "literal_lag_copier_per_k": lag}


def h4_tests(r6: dict | None, cmp_rates: dict | None) -> dict:
    """Melody-level one-sided exact sign test of the seq T+6A share minus the comparator's share (common
    melodies, ties dropped) and the registered draw-level one-sided exact binomial test against
    p0 = max(k_c, 0.5) / n_c. Evaluable needs MIN_MELODIES melodies in each and in common."""
    if not (_ok(r6) and _ok(cmp_rates)):
        return {"evaluable": False}
    common = sorted(set(r6["per_item"]) & set(cmp_rates["per_item"]))
    st = sign_test_greater(r6["per_item"][m] - cmp_rates["per_item"][m] for m in common)
    bt = binom_vs_rate(r6["k"], r6["n"], cmp_rates["k"], cmp_rates["n"])
    return {"evaluable": len(common) >= MIN_MELODIES, "n_common": len(common), "sign_test": st, "binomial": bt,
            "both_p_lt_alpha": bool(st["p"] < ALPHA_H4 and bt["p"] < ALPHA_H4)}


def h4_block(rows, include, tau, n_boot=N_BOOT) -> dict:
    rates = {h: item_rates(rows, "seq", "seq", h, include, tau, n_boot=n_boot) for h in CLASS_SETS["seq"] + ["none", "tie"]}
    ctl6 = item_rates(rows, "ctl", "seq", "T+6A", include, tau, n_boot=n_boot)
    null6 = item_rates(rows, "null", "seq", "T+6A", include, tau, n_boot=n_boot)
    return {"rates": rates, "ctl_T+6A": ctl6, "null_T+6A": null6,
            "tests_vs_ctl": h4_tests(rates["T+6A"], ctl6), "tests_vs_null": h4_tests(rates["T+6A"], null6)}


def verdict_h4(all_block: dict, pass_block: dict) -> dict:
    """Registered rule, with X-ctl added as a second comparator: in ALL, T+6A rate >= 0.20 and T+5A, T+7A
    rates <= 0.05 (rates are per-melody shares averaged over melodies). The T+6A share must exceed the NULL
    coincidence rate (the registered comparator) in ALL, and the X-ctl coincidence rate in ALL and in PASS;
    each comparison needs both the sign test and the binomial at p < 0.01."""
    r = all_block["rates"]
    r6 = r["T+6A"]
    if not _ok(r6) or not _ok(r["T+5A"]) or not _ok(r["T+7A"]):
        return {"verdict": "not_evaluable", "reason": "fewer than 12 melodies with scored seq draws"}
    rates_ok = _r(r6["rate"]) >= 0.20 and _r(r["T+5A"]["rate"]) <= 0.05 and _r(r["T+7A"]["rate"]) <= 0.05
    t_null, t_all, t_pass = all_block["tests_vs_null"], all_block["tests_vs_ctl"], pass_block["tests_vs_ctl"]
    out = {"rates_ALL": {h: r[h]["rate"] for h in r if r[h] and "rate" in r[h]}, "rate_clause": bool(rates_ok),
           "tests_ALL_vs_null": t_null, "tests_ALL": t_all, "tests_PASS": t_pass, "comparator": "null and ctl"}
    if not rates_ok or any(t.get("evaluable") and not t["both_p_lt_alpha"] for t in (t_null, t_all)):
        return {"verdict": "not_supported", **out}
    if not (t_null.get("evaluable") and t_all.get("evaluable") and t_pass.get("evaluable")):
        return {"verdict": "not_evaluable", **out}
    return {"verdict": "supported" if t_pass["both_p_lt_alpha"] else "not_supported", **out}


def k3_status(all_h2: dict | None, pass_h2: dict | None) -> str:
    """K3 (CI(X-ctl) >= CI(X-lit) - 0.05, i.e. the H2 difference <= 0.05): triggered when it holds in ALL
    and in PASS, indeterminate when it holds in one of them, not evaluable unless both are evaluable."""
    if not (_ok(all_h2) and _ok(pass_h2)):
        return "not_evaluable"
    a, p = _r(all_h2["mean"]) <= 0.05, _r(pass_h2["mean"]) <= 0.05
    return "triggered" if a and p else ("indeterminate" if a or p else "not_triggered")


def paper_level(decision: str, verdicts: dict[str, str]) -> str:
    if decision == "incomplete":
        return "incomplete_run"
    if decision != "pass":
        return "infill_failure"
    v = [verdicts[h] for h in ("H1", "H2", "H3", "H4", "H5")]
    if any(x == "supported" for x in v):
        return "supported:" + ",".join(h for h in ("H1", "H2", "H3", "H4", "H5") if verdicts[h] == "supported")
    if all(x == "not_supported" for x in v):
        return "negative_benchmark"
    if all(x == "not_evaluable" for x in v):
        return "not_evaluable"
    return "partly_not_evaluable"


INDEX_SPEC = {"CI_lit": ("lit", "B", "B~"), "CI_ctl": ("ctl", "B", "B~"), "ANA": ("ana", "T+5B", "T+5B~"),
              "CI_null": ("null", "B", "B~")}
for _k in KS:
    INDEX_SPEC[f"RI{_k}"] = (f"rel{_k}", f"T{_k}B", f"T{_k}B~")
    INDEX_SPEC[f"LI{_k}"] = (f"rel{_k}", "B", "B~")
    INDEX_SPEC[f"RI{_k}_ctl"] = ("ctl", f"T{_k}B", f"T{_k}B~")      # needs union scoring of ctl draws
COMPARISONS = {"H2": ("CI_lit", "CI_ctl")} | {f"RI{k}-RI{k}_ctl": (f"RI{k}", f"RI{k}_ctl") for k in KS}
PAIR_CONDS = {"H2": ("lit", "ctl"), **{f"H3{k}": (f"rel{k}", "ctl") for k in KS}, "H4": ("seq", "ctl")}


def k2(rows, first_ids: list[str], n_boot: int = N_BOOT) -> dict:
    """K2 after the grid, on the first 6 confirmatory melodies, with ALL: triggered iff CI(X-lit) < 0.10 and
    CI(X-ctl) (the literal index under X-ctl) < 0.10. Uses only the rows of those melodies."""
    sub = [r for r in rows if r["id"] in set(first_ids)]
    a = index(sub, *INDEX_SPEC["CI_lit"], pop_all, n_boot=n_boot)
    b = index(sub, *INDEX_SPEC["CI_ctl"], pop_all, n_boot=n_boot)
    complete = len(first_ids) == 6 and a is not None and b is not None and a["n_melodies"] == 6 == b["n_melodies"]
    trig = bool(_r(a["mean"]) < 0.10 and _r(b["mean"]) < 0.10) if complete else None
    return {"melodies": list(first_ids), "CI_lit": a, "CI_ctl": b, "triggered": trig,
            "status": "not_evaluable" if trig is None else ("triggered" if trig else "not_triggered")}


def summarize(rows, quality: dict, tau: float, first_ids: list[str], n_boot: int = N_BOOT,
              n_boot_nested: int = N_BOOT, diagnostic: bool = False) -> dict:
    """Every registered quantity and verdict, computed automatically from scored rows and the frozen
    quality decision. Verdicts are decided on ALL; comparisons between conditions must also hold in PASS.
    Nothing is computed when the quality decision is not 'pass', unless diagnostic (exploratory)."""
    out = {"quality": quality, "tau": tau, "diagnostic": diagnostic, "primary_population": "ALL"}
    if quality["decision"] != "pass" and not diagnostic:
        out["paper_level"] = paper_level(quality["decision"], {})
        return out
    pops = {"ALL": pop_all, "PASS": pop_pass}
    ix = {p: {n: index(rows, *spec, pred, n_boot=n_boot) for n, spec in INDEX_SPEC.items()} for p, pred in pops.items()}
    ix_oct = {p: {n: index(rows, *spec, pred, key="acc_oct", n_boot=n_boot) for n, spec in INDEX_SPEC.items()}
              for p, pred in pops.items()}
    diffs = {p: {} for p in ("ALL", "PASS", "ALL_paired", "PASS_difference_of_indices")}
    for name, (a, b) in COMPARISONS.items():
        diffs["ALL"][name] = diff_index(rows, INDEX_SPEC[a], INDEX_SPEC[b], pop_all, n_boot=n_boot)
        diffs["PASS"][name] = paired_diff(rows, INDEX_SPEC[a], INDEX_SPEC[b], pop_pass, n_boot=n_boot)
        diffs["ALL_paired"][name] = paired_diff(rows, INDEX_SPEC[a], INDEX_SPEC[b], pop_all, n_boot=n_boot)
        diffs["PASS_difference_of_indices"][name] = diff_index(rows, INDEX_SPEC[a], INDEX_SPEC[b], pop_pass, n_boot=n_boot)
    for k in KS:
        diffs["ALL"][f"RI{k}-LI{k}"] = diff_index(rows, INDEX_SPEC[f"RI{k}"], INDEX_SPEC[f"LI{k}"], pop_all, n_boot=n_boot)
        diffs["PASS"][f"RI{k}-LI{k}"] = paired_diff(rows, INDEX_SPEC[f"RI{k}"], INDEX_SPEC[f"LI{k}"], pop_pass, n_boot=n_boot)
    h4b = {p: h4_block(rows, pred, tau, n_boot) for p, pred in pops.items()}
    A, P, DA, DP = ix["ALL"], ix["PASS"], diffs["ALL"], diffs["PASS"]
    verdicts = {
        "H1": {"verdict": verdict_threshold(A["CI_lit"], 0.25), "ALL": A["CI_lit"], "PASS": P["CI_lit"]},
        "H2": {"verdict": verdict_comparison(DA["H2"], DP["H2"], 0.15), "ALL": DA["H2"], "PASS": DP["H2"]},
        "H3": {**verdict_h3({k: A[f"RI{k}"] for k in KS}, {k: DA[f"RI{k}-LI{k}"] for k in KS},
                            {k: DA[f"RI{k}-RI{k}_ctl"] for k in KS}, {k: DP[f"RI{k}-RI{k}_ctl"] for k in KS},
                            {k: A[f"LI{k}"] for k in KS}),
               "ALL": {k: {"RI": A[f"RI{k}"], "RI-RI_ctl": DA[f"RI{k}-RI{k}_ctl"], "RI-LI": DA[f"RI{k}-LI{k}"]} for k in KS},
               "PASS": {k: {"RI": P[f"RI{k}"], "RI-RI_ctl": DP[f"RI{k}-RI{k}_ctl"], "RI-LI": DP[f"RI{k}-LI{k}"]} for k in KS}},
        "H4": verdict_h4(h4b["ALL"], h4b["PASS"]),
        "H5": {"verdict": verdict_threshold(A["ANA"], 0.15), "ALL": A["ANA"], "PASS": P["ANA"]},
    }
    flat = {h: v["verdict"] for h, v in verdicts.items()}
    rates = {c: pass_rate(rows, c) for c in CLASS_SETS}
    lit_c = {(r["id"], r.get("arm", "base"), r["seed"]): r for r in _select(rows, "lit", pop_all)}
    ctl_c = {(r["id"], r.get("arm", "base"), r["seed"]): r for r in _select(rows, "ctl", pop_all)}
    pairs = [(bool(lit_c[k]["gate"]["pass"]), bool(ctl_c[k]["gate"]["pass"])) for k in sorted(set(lit_c) & set(ctl_c))]
    gating = {h: {"pass_rates": [rates[a], rates[b]],
                  "differential_gating": bool(rates[a] is not None and rates[b] is not None and _r(abs(rates[a] - rates[b])) > 0.10)}
              for h, (a, b) in PAIR_CONDS.items()}
    lee = {"H2": lee_bounds_diff(rows, INDEX_SPEC["CI_lit"], INDEX_SPEC["CI_ctl"], n_boot=n_boot)}
    for k in KS:
        lee[f"H3{k}"] = lee_bounds_diff(rows, INDEX_SPEC[f"RI{k}"], INDEX_SPEC[f"RI{k}_ctl"], n_boot=n_boot)
    labels = _labels(verdicts, flat, gating, lee)
    out.update({"verdicts": verdicts, "labels": labels, "gating": gating,
                "mcnemar_pass_lit_vs_ctl": mcnemar_exact(pairs),
                "K2": k2(rows, first_ids, n_boot), "K3_status": k3_status(DA["H2"], DP["H2"]),
                "indices": ix, "indices_oct": ix_oct, "differences": diffs, "h4": h4b,
                "lee_bounds_exploratory": lee, "paper_level": paper_level(quality["decision"], flat)})
    out["secondary"] = _secondary(rows, tau, n_boot, n_boot_nested)
    return out


def _labels(verdicts, flat, gating, lee) -> dict:
    """Labels printed next to each verdict; a label never changes a verdict."""
    out = {}
    pass_of = {"H1": verdicts["H1"]["PASS"], "H5": verdicts["H5"]["PASS"]}
    thr = {"H1": 0.25, "H5": 0.15}
    for h in ("H1", "H5"):
        p = pass_of[h]
        lab = ["PASS not evaluable"] if not _ok(p) else (["PASS meets the threshold"] if _r(p["mean"]) >= thr[h] and _r(p["lo"]) > 0
                                                          else ["PASS does not meet the threshold"])
        a = verdicts[h]["ALL"]
        if flat[h] == "not_supported" and _ok(a) and _r(a["lo"]) > 0:
            lab.append("below threshold, CI excludes 0")
        out[h] = lab
    h2 = []
    if gating["H2"]["differential_gating"]:
        h2.append("differential gating")
    a = verdicts["H2"]["ALL"]
    if flat["H2"] == "not_supported" and _ok(a) and _r(a["lo"]) > 0:
        h2.append("below threshold, CI excludes 0")
    lb = lee.get("H2")
    if flat["H2"] == "supported" and lb and lb.get("lower") and not lb["lower"]["lo"] > 0:
        h2.append("selection-sensitive (exploratory Lee bound)")
    out["H2"] = h2
    out["H3"] = [f"differential gating rel{k}" for k in KS if gating[f"H3{k}"]["differential_gating"]]
    out["H4"] = ["differential gating"] if gating["H4"]["differential_gating"] else []
    return out


def _secondary(rows, tau, n_boot, n_boot_nested) -> dict:
    """Registered secondary quantities and exploratory read-outs (no thresholds)."""
    pops = {"ALL": pop_all, "PASS": pop_pass}
    absB = {p: {c: pack(per_melody(rows, c, lambda r: r["acc"]["B"], pred), n_boot) for c in ("lit", "null")}
            for p, pred in pops.items()}
    sel_rates, coincidence = {}, {}
    for p, pred in pops.items():
        sel_rates[p], coincidence[p] = {}, {}
        for c, target in RULE_ANSWER.items():
            sel_rates[p][c] = {h: item_rates(rows, c, c, h, pred, tau, n_boot=n_boot) for h in CLASS_SETS[c] + ["none", "tie"]}
            coincidence[p][c] = {}
            for base in ("null", "ctl"):
                cr = item_rates(rows, base, c, target, pred, tau, n_boot=n_boot)
                tr = sel_rates[p][c][target]
                if cr and "error" not in cr and tr and "error" not in tr:
                    common = sorted(set(tr["per_item"]) & set(cr["per_item"]))
                    cr = {**cr, "sign_test_vs_target": sign_test_greater(tr["per_item"][m] - cr["per_item"][m] for m in common)}
                coincidence[p][c][base] = cr
    passk = {c: {p: pass_at_k_summary(rows, c, t, pred, tau, n_boot=n_boot) for p, pred in pops.items()}
             for c, t in RULE_ANSWER.items()}
    repl = {}
    ok_rows = [r for r in rows if pop_all(r)]
    for key in ("nf1", "of1"):
        if not all(key in r for r in ok_rows):
            raise ValueError(f"v2 rows must carry {key} for every scored draw")
        repl[key] = {p: {n: index(rows, *spec, pred, key=key, n_boot=n_boot) for n, spec in INDEX_SPEC.items()}
                     for p, pred in pops.items()}
    repl["onset_B_minus_B2"] = {p: {c: index(rows, c, "B", "B2", pred, key="of1", n_boot=n_boot) for c in CLASS_SETS}
                                for p, pred in pops.items()}
    pass_by_cond = {c: pack(per_melody(rows, c, lambda r: float(r["gate"]["pass"]), pop_all), n_boot) for c in CLASS_SETS}
    alt_by_cond = {g: {c: pack(per_melody(rows, c, lambda r, g=g: float(r["gate"].get(g, False)), pop_all), n_boot)
                       for c in CLASS_SETS} for g in ("pass_octfold", "pass_splice_safe")}
    nested = {n: boot_nested_ci(per_melody_cells(rows, INDEX_SPEC[n][0], contrast(*INDEX_SPEC[n][1:]), pop_all),
                                n_boot=n_boot_nested)
              for n in ("CI_lit", "CI_ctl", "ANA", "RI+2", "RI+5", "RI-3")}
    both_pass = {"H2": diff_cells(rows, INDEX_SPEC["CI_lit"], INDEX_SPEC["CI_ctl"], pop_pass, n_boot=n_boot)}
    null_levels = [r["quality"]["gap_rms_dbfs"] for r in rows if pop_all(r) and r["cond"] == "null"]
    return {"abs_acc_B": absB, "selection_rates": sel_rates, "coincidence": coincidence, "pass_at_2": passk,
            "replications": repl, "pass_rate_by_cond": pass_by_cond, "alternative_gate_pass_rate_by_cond": alt_by_cond,
            "nested_bootstrap_ALL": nested, "H2_both_pass_cells": both_pass,
            "null_gap_rms_dbfs": {"n": len(null_levels), "min": min(null_levels) if null_levels else None,
                                  "median": float(np.median(null_levels)) if null_levels else None,
                                  "max": max(null_levels) if null_levels else None}}
