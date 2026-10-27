"""Unit tests for dacapo.motif_stats on synthetic rows: estimators, tests and every automatic decision rule."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from dacapo import motif_stats as S

r4 = lambda x: round(float(x), 4)
NB = 2000          # bootstrap size for speed in unit tests (the registered value is 10000)


def _rows(spec, arm="base"):
    """spec: list of (id, cond, seed, accs, gate_pass)."""
    return [{"id": i, "cond": c, "seed": s, "arm": arm, "split": "confirm", "status": "ok",
             "acc": a, "gate": {"pass": g}} for i, c, s, a, g in spec]


def test_classify_tie_rule():
    seq = S.CLASS_SETS["seq"]
    a = dict.fromkeys(seq, 0.0) | {"T+6A": 0.8, "T+4A": 0.8}
    assert S.classify(a, seq, 0.7) == "tie" and S.classify(a, seq, 0.7, tie="first") == "T+6A"
    assert S.classify(dict.fromkeys(seq, 0.5), seq, 0.7) == "none"
    assert S.classify(a | {"T+4A": 0.75}, seq, 0.7) == "T+6A"


@pytest.mark.parametrize("n", [5, 12, 13, 24, 36])
def test_boot_stream_matches_v1(n):
    vals = np.random.default_rng(1).normal(size=n)
    r1 = np.random.default_rng(0)
    loop = np.array([r1.choice(vals, n) for _ in range(300)])
    assert np.array_equal(loop, vals[S.boot_index_matrix(n, 300)])


def test_bootstrap_depends_on_melody_order():
    v = [0.1, 0.5, -0.2, 0.3, 0.0, 0.9, -0.4, 0.2]
    assert S.boot_mean_ci(v) != S.boot_mean_ci(v[::-1])          # so melodies are pinned in sorted-id order


def test_t_ci_is_disclosed_alongside():
    rows = _rows([(f"m{m:02d}", "lit", "s0", {"B": 0.1 * (m % 5), "B~": 0.0}, True) for m in range(12)])
    ix = S.index(rows, "lit", "B", "B~", S.pop_pass, n_boot=NB)
    lo, hi = ix["t_ci"]
    assert lo < ix["mean"] < hi


def test_all_vs_pass_population():
    spec = []
    for m in range(12):
        for s in range(4):
            ok = s < 3
            spec.append((f"m{m:02d}", "lit", f"s{s}", {"B": 0.6, "B~": 0.2} if ok else {"B": 0.0, "B~": 0.0}, ok))
    rows = _rows(spec)
    a, p = (S.index(rows, "lit", "B", "B~", pop, n_boot=NB) for pop in (S.pop_all, S.pop_pass))
    assert r4(a["mean"]) == 0.3 and r4(p["mean"]) == 0.4             # ALL = pass rate x PASS


def test_pass_equals_all_when_every_draw_passes():
    rng = np.random.default_rng(4)
    rows = _rows([(f"m{m:02d}", c, f"s{s}", {"B": float(rng.random()), "B~": float(rng.random())}, True)
                  for m in range(12) for c in ("lit", "ctl") for s in range(4)])
    for pop_a, pop_b in ((S.pop_all, S.pop_pass),):
        assert S.index(rows, "lit", "B", "B~", pop_a, n_boot=NB) == S.index(rows, "lit", "B", "B~", pop_b, n_boot=NB)
        d_all = S.diff_index(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), pop_a, n_boot=NB)
        d_pass = S.paired_diff(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), pop_b, n_boot=NB)
        assert d_all["mean"] == pytest.approx(d_pass["mean"]) and d_all["lo"] == pytest.approx(d_pass["lo"])


def test_pass_unbiased_only_when_passing_is_copy_independent():
    """Equal conditional copying (0.3) in lit and ctl, pass rates 0.75 vs 0.5, passing independent of the
    gap's content: ALL shows a spurious H2 of 0.075, PASS shows none. The agreement rule needs both."""
    spec = []
    for m in range(12):
        for s in range(4):
            for c, npass in (("lit", 3), ("ctl", 2)):
                ok = s < npass
                spec.append((f"m{m:02d}", c, f"s{s}", {"B": 0.5, "B~": 0.2} if ok else {"B": 0.0, "B~": 0.0}, ok))
    rows = _rows(spec)
    dall = S.diff_index(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), S.pop_all, n_boot=NB)
    dpass = S.paired_diff(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), S.pop_pass, n_boot=NB)
    assert r4(dall["mean"]) == 0.075 and r4(dpass["mean"]) == 0.0


def _copy_dependent_world(lit_pattern: str):
    """Latent copying is identical in lit and ctl: 1 of 4 draws per melody copies B (contrast 0.8), 3 do not.
    In the 'all' condition every draw passes; in the 'dep' condition the copying draw and 1 of the 3 others
    pass, so passing depends on the content."""
    spec = []
    for m in range(24):
        for c in ("lit", "ctl"):
            dep = (c == "lit") == (lit_pattern == "dep")
            for s in range(4):
                copy = s == 0
                ok = True if not dep else s in (0, 1)
                acc = {"B": 0.9, "B~": 0.1} if copy else {"B": 0.3, "B~": 0.3}
                spec.append((f"m{m:02d}", c, f"s{s}", acc if ok else {"B": 0.0, "B~": 0.0}, ok))
    return _rows(spec)


def test_copy_dependent_passing_biases_pass_and_lee_bounds_cover_truth():
    rows = _copy_dependent_world("dep")                                  # lit passes copy-dependently
    d = S.paired_diff(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), S.pop_pass, n_boot=NB)
    assert r4(d["mean"]) == 0.2                                         # spurious: latent H2 is 0
    lee = S.lee_bounds_diff(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), n_boot=NB)
    assert lee["trimmed"] == "ctl" and lee["q"] == pytest.approx(0.5)
    assert lee["lower"]["mean"] <= 0.05 and lee["lower"]["lo"] <= 0 <= lee["upper"]["mean"] + 1e-9
    rows = _copy_dependent_world("all")                                  # ctl passes copy-dependently
    d = S.paired_diff(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), S.pop_pass, n_boot=NB)
    lee = S.lee_bounds_diff(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), n_boot=NB)
    assert r4(d["mean"]) == -0.2 and lee["lower"]["mean"] <= 0 <= lee["upper"]["mean"] + 0.05


def test_lee_bounds_are_the_estimate_when_pass_rates_match():
    rows = _rows([(f"m{m:02d}", c, f"s{s}", {"B": 0.6 if c == "lit" else 0.3, "B~": 0.2}, s < 3)
                  for m in range(12) for c in ("lit", "ctl") for s in range(4)])
    lee = S.lee_bounds_diff(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), n_boot=NB)
    assert lee["q"] == 0.0 and r4(lee["lower"]["mean"]) == r4(lee["upper"]["mean"]) == 0.3


def test_diff_index_is_difference_of_indices_when_melody_sets_differ():
    rng = np.random.default_rng(3)
    spec = [(f"m{m:02d}", c, f"s{s}", {"B": float(rng.integers(0, 9)) / 8, "B~": float(rng.integers(0, 9)) / 8},
             not (c == "ctl" and m < 3))                                 # 3 melodies have no passing ctl draw
            for m in range(16) for s in range(4) for c in ("lit", "ctl")]
    rows = _rows(spec)
    lit, ctl = (S.index(rows, c, "B", "B~", S.pop_pass, n_boot=NB) for c in ("lit", "ctl"))
    d = S.diff_index(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), S.pop_pass, n_boot=NB)
    assert abs(d["mean"] - (lit["mean"] - ctl["mean"])) < 1e-12
    assert (d["n_melodies_a"], d["n_melodies_b"], d["n_common"]) == (16, 13, 13)
    p = S.paired_diff(rows, ("lit", "B", "B~"), ("ctl", "B", "B~"), S.pop_pass, n_boot=NB)
    assert p["n_melodies"] == 13 and p["mean"] == pytest.approx(d["paired_common_mean"])
    assert d["lo"] < d["mean"] < d["hi"]


def test_min_melodies_makes_not_evaluable():
    rows = _rows([(f"m{m:02d}", "lit", "s0", {"B": 1.0, "B~": 0.0}, True) for m in range(11)])
    ix = S.index(rows, "lit", "B", "B~", S.pop_pass, n_boot=NB)
    assert ix["mean"] == 1.0 and not ix["evaluable"] and S.verdict_threshold(ix, 0.25) == "not_evaluable"


def test_sign_test_values():
    assert S.sign_test_greater([0.25] * 7)["p"] == pytest.approx(0.5 ** 7)
    assert S.sign_test_greater([0.25] * 6 + [0.0] * 10)["p"] == pytest.approx(0.5 ** 6)
    assert S.sign_test_greater([])["p"] == 1.0
    assert S.sign_test_greater([0.5] * 9 + [-0.25])["p"] == pytest.approx(11 / 1024)


def test_binom_vs_rate_floor_and_literal():
    t = S.binom_vs_rate(10, 96, 0, 96)
    assert t["p0"] == pytest.approx(0.5 / 96) and t["p0_literal"] is None and t["p"] < 1e-6
    t = S.binom_vs_rate(10, 96, 3, 96)
    assert t["p0"] == t["p0_literal"] == pytest.approx(3 / 96)
    assert S.binom_vs_rate(0, 0, 0, 96)["p"] == 1.0


def _seq_world(t6_by_melody: dict, n_draws=4, n_mel=24, cmp_hits=0, seq_pass=None, ctl_pass=None, null_hits=0):
    spec = []
    for m in range(n_mel):
        mid = f"m{m:02d}"
        for s in range(n_draws):
            a = dict.fromkeys(S.CLASS_SETS["seq"], 0.0)
            if s < t6_by_melody.get(mid, 0):
                a["T+6A"] = 1.0
            spec.append((mid, "seq", f"s{s}", a, True if seq_pass is None else seq_pass(m, s)))
            c = dict.fromkeys(S.CLASS_SETS["seq"], 0.0) | {"B": 0.0, "B~": 0.0}
            if m < cmp_hits and s == 0:
                c["T+6A"] = 1.0
            spec.append((mid, "ctl", f"s{s}", c, True if ctl_pass is None else ctl_pass(m, s)))
            z = dict.fromkeys(S.CLASS_SETS["seq"], 0.0) | {"B": 0.0, "B~": 0.0}
            if m < null_hits and s < 2:
                z["T+6A"] = 1.0
            spec.append((mid, "null", f"s{s}", z, False))                   # null gaps never pass the gate
    return _rows(spec)


def _h4(rows):
    return S.verdict_h4(S.h4_block(rows, S.pop_all, 0.7, NB), S.h4_block(rows, S.pop_pass, 0.7, NB))


def test_h4_supported_when_spread_over_melodies():
    """Every melody shows T+6A in 1 of 4 draws (rate 0.25): supported in ALL and PASS. An item-majority
    rule would count 0 melodies here."""
    v = _h4(_seq_world({f"m{m:02d}": 1 for m in range(24)}))
    assert v["verdict"] == "supported" and v["tests_ALL"]["sign_test"]["n_pos"] == 24
    assert v["tests_ALL"]["binomial"]["p"] < 1e-6 and v["tests_ALL"]["binomial"]["p0"] == pytest.approx(0.5 / 96)


def test_h4_also_needs_the_registered_null_comparison():
    """The X-ctl comparison holds, but null draws show T+6A at twice the seq rate in every melody: the
    registered comparison against NULL fails, so H4 is not supported."""
    v = _h4(_seq_world({f"m{m:02d}": 1 for m in range(24)}, null_hits=24))
    assert v["tests_ALL"]["both_p_lt_alpha"] and v["tests_PASS"]["both_p_lt_alpha"]
    assert not v["tests_ALL_vs_null"]["both_p_lt_alpha"] and v["verdict"] == "not_supported"


def test_h4_concentrated_effect_fails_the_melody_level_test():
    """6 melodies always show T+6A (rate 0.25): the draw-level binomial is significant, the sign test
    over melodies is not (6 positives: p = 0.0156), so H4 is not supported."""
    v = _h4(_seq_world({f"m{m:02d}": 4 for m in range(6)}))
    assert v["tests_ALL"]["binomial"]["p"] < 1e-6 and v["tests_ALL"]["sign_test"]["p"] == pytest.approx(0.5 ** 6)
    assert v["verdict"] == "not_supported"


def test_h4_ties_count_in_the_denominator():
    rows = _seq_world({})
    for r in rows:
        if r["cond"] == "seq" and r["seed"] == "s0":
            r["acc"]["T+6A"] = r["acc"]["T+4A"] = 0.9                    # a tie: credited to no hypothesis
    rates = S.h4_block(rows, S.pop_all, 0.7, NB)["rates"]
    assert rates["tie"]["rate"] == 0.25 and rates["T+6A"]["rate"] == 0.0 and rates["T+6A"]["n"] == 96


def test_h4_test_clause_must_hold_in_pass_too():
    """ALL supports H4, but in PASS the seq draws that show T+6A fail the gate: not supported."""
    rows = _seq_world({f"m{m:02d}": 1 for m in range(24)}, seq_pass=lambda m, s: s != 0)
    v = _h4(rows)
    assert v["tests_ALL"]["both_p_lt_alpha"] and not v["tests_PASS"]["both_p_lt_alpha"]
    assert v["verdict"] == "not_supported"
    rows = _seq_world({f"m{m:02d}": 1 for m in range(24)}, ctl_pass=lambda m, s: m >= 13)
    v = _h4(rows)                                                        # PASS comparator on 11 melodies
    assert v["verdict"] == "not_evaluable"


def test_h4_rate_clauses_use_all():
    rows = _seq_world({f"m{m:02d}": 1 for m in range(24)})
    for r in rows:
        if r["cond"] == "seq" and r["seed"] == "s1" and r["id"] < "m06":
            r["acc"]["T+5A"] = 1.0                                       # T+5A in 6 of 96 draws: 0.0625 > 0.05
    assert _h4(rows)["verdict"] == "not_supported"


def test_pass_at_k_matches_enumeration():
    for n in range(1, 7):
        for c in range(n + 1):
            for k in range(1, n + 1):
                draws = [1] * c + [0] * (n - c)
                brute = np.mean([any(x) for x in itertools.combinations(draws, k)])
                assert S.pass_at_k(n, c, k) == pytest.approx(brute)
    assert S.pass_at_k(1, 1, 2) is None


def test_mcnemar():
    assert S.mcnemar_exact([(True, False)] * 6 + [(True, True)] * 10)["p"] == pytest.approx(2 * 0.5 ** 6)
    assert S.mcnemar_exact([(True, True)] * 5)["p"] == 1.0


def test_nested_and_crossed_bootstrap_degenerate():
    cells = {f"m{i}": [0.2, 0.2, 0.2] for i in range(10)}
    assert S.boot_nested_ci(cells, n_boot=200) == pytest.approx((0.2, 0.2, 0.2))
    mat = np.full((10, 4), 0.2)
    mat[0, 1] = np.nan
    assert S.boot_crossed_ci(mat, n_boot=200) == pytest.approx((0.2, 0.2, 0.2))


# Quality decisions (Q1, Q2, K1)

def _main_plan(n_mel=36, n_draws=4):
    return [(f"v2c{m:02d}", c, "base", f"d{s}") for m in range(n_mel) for c in S.CLASS_SETS for s in range(n_draws)]


def _qrows(planned, pass_fn, drop=(), lost=()):
    out = []
    for k in planned:
        if k in drop:
            continue
        i, c, a, s = k
        out.append({"id": i, "cond": c, "arm": a, "seed": s, "status": "lost" if k in lost else "ok",
                    "gate": {"pass": pass_fn(i, c, s)}})
    return out


def test_q1_boundary_755_vs_756_of_1008():
    planned = _main_plan()
    nonnull = [k for k in planned if k[1] != "null"]
    assert len(planned) == 1152 and len(nonnull) == 1008
    for n_fail, want in ((252, "pass"), (253, "fail")):
        failing = set(sorted(nonnull, key=lambda k: (k[3], k[0], k[1]))[:n_fail])     # d0 of every cell first
        q = S.quality_decision(_qrows(planned, lambda i, c, s, f=failing: (i, c, "base", s) not in f), planned)
        assert (q["n_pass"], q["need_pass"], q["decision"]) == (1008 - n_fail, 756, want), n_fail


def test_q2_boundary_188_vs_189_of_252_cells():
    planned = _main_plan()
    cells = sorted({(k[0], k[1]) for k in planned if k[1] != "null"})
    assert len(cells) == 252
    for n_bad, want in ((63, "pass"), (64, "fail")):
        bad = set(cells[:n_bad])                                         # these cells pass 1 of 4 draws
        q = S.quality_decision(_qrows(planned, lambda i, c, s, b=bad: not ((i, c) in b and s != "d0")), planned)
        assert q["n_cells_ok"] == 252 - n_bad and q["need_cells"] == 189 and q["decision"] == want, n_bad
        assert q["n_pass"] >= 756                                        # Q1 holds: the cell rule binds alone


def test_null_rows_are_ignored_and_pending_blocks_the_decision():
    planned = _main_plan(n_mel=12)
    q = S.quality_decision(_qrows(planned, lambda i, c, s: c != "null"), planned)
    assert q["decision"] == "pass" and q["n_nonnull"] == 12 * 7 * 4
    one = planned[5]
    q = S.quality_decision(_qrows(planned, lambda i, c, s: True, drop={one}), planned)
    assert q["decision"] == "incomplete" and q["n_missing"] == 1               # no missing-fraction threshold
    q = S.quality_decision(_qrows(planned, lambda i, c, s: True, drop={one}), planned, missing_as_failure=True)
    assert q["decision"] == "pass" and q["n_pass"] == 12 * 7 * 4 - (one[1] != "null")


def test_lost_audio_is_pending_until_the_resume_cap_then_a_failure():
    planned = _main_plan(n_mel=12)
    lost = {k for k in planned if k[3] == "d0" and k[1] != "null"}       # 84 completed draws whose audio vanished
    q = S.quality_decision(_qrows(planned, lambda i, c, s: True, lost=lost), planned)
    assert q["decision"] == "incomplete" and q["n_lost"] == 84
    q = S.quality_decision(_qrows(planned, lambda i, c, s: True, lost=lost), planned, missing_as_failure=True)
    assert q["n_pass"] == 252 and q["draw_pass_rate"] == 0.75 and q["decision"] == "pass"


def test_quality_decision_empty_plan_is_incomplete_not_an_error():
    assert S.quality_decision([], [])["decision"] == "incomplete"
    assert S.pilot_decision([], [], ["a"])["decision"] == "incomplete"


def test_pilot_arms_do_not_collide():
    rows = [{"id": "mel00", "cond": "lit", "arm": "cap1", "seed": "s1", "status": "ok", "gate": {"pass": True}},
            {"id": "mel00", "cond": "lit", "arm": "cap2", "seed": "s1", "status": "ok", "gate": {"pass": False}}]
    planned = [S.draw_key(r) for r in rows]
    d = S.pilot_decision(rows, planned, ["cap1", "cap2"])
    assert d["decision"] == "selected" and d["arm"] == "cap1"
    assert d["per_arm"]["cap1"]["n_pass"] == 1 and d["per_arm"]["cap2"]["n_pass"] == 0


def _p0b(rate_by_arm: dict, arms: list[str]):
    planned = [(f"mel{m:02d}", c, a, f"d{s}") for m in range(14) for c in ("lit", "ctl") for a in arms for s in range(4)]
    rows = []
    for a in arms:
        ks = [k for k in planned if k[2] == a]
        for n, (i, c, _, s) in enumerate(ks):
            rows.append({"id": i, "cond": c, "arm": a, "seed": s, "status": "ok", "gate": {"pass": n < rate_by_arm[a]}})
    return rows, planned


def test_k1_boundary_83_vs_84_of_112_and_tie_order():
    arms = ["rep-rend-flute", "rep-rend-synth_voice", "rep-t18-flute", "t2m-rend-flute"]
    rows, planned = _p0b({arms[0]: 83, arms[1]: 83, arms[2]: 83, arms[3]: 83}, arms)
    assert S.pilot_decision(rows, planned, arms)["decision"] == "k1_triggered"
    rows, planned = _p0b({arms[0]: 84, arms[1]: 90, arms[2]: 90, arms[3]: 84}, arms)
    d = S.pilot_decision(rows, planned, arms)
    assert d["per_arm"][arms[0]]["need"] == 84 and d["per_arm"][arms[0]]["k1_pass"]
    assert d["decision"] == "selected" and d["arm"] == arms[1]           # most passes; tie goes to the earlier arm
    assert S.pilot_decision(rows, planned, arms, excluded=[arms[1]])["arm"] == arms[2]
    rows = [r for r in rows if not (r["arm"] == arms[3] and r["id"] == "mel00")]
    assert S.pilot_decision(rows, planned, arms)["decision"] == "incomplete"
    assert S.pilot_decision(rows, planned, arms, missing_as_failure=True)["decision"] == "selected"


# Verdicts: ALL decides, comparisons between conditions must also hold in PASS

def _ix(mean, lo, n=36):
    return {"mean": mean, "lo": lo, "hi": mean + 0.1, "n_melodies": n, "evaluable": n >= 12}


def test_h1_boundaries_use_unrounded_values():
    assert S.verdict_threshold(_ix(0.2499, 0.1), 0.25) == "not_supported"
    assert S.verdict_threshold(_ix(0.25, 0.0), 0.25) == "not_supported"
    assert S.verdict_threshold(_ix(0.25, 0.001), 0.25) == "supported"


def test_h2_agreement_truth_table():
    yes, no = _ix(0.2, 0.05), _ix(0.1, 0.05)
    assert S.verdict_comparison(yes, yes, 0.15) == "supported"
    assert S.verdict_comparison(yes, no, 0.15) == "not_supported"
    assert S.verdict_comparison(no, yes, 0.15) == "not_supported"
    assert S.verdict_comparison(yes, _ix(0.3, 0.1, n=11), 0.15) == "not_evaluable"   # PASS on 11 common melodies
    assert S.verdict_comparison(_ix(0.3, 0.1, n=11), yes, 0.15) == "not_evaluable"


def test_k3_truth_table():
    lo, hi = _ix(0.04, -0.1), _ix(0.2, 0.1)
    assert S.k3_status(lo, lo) == "triggered"
    assert S.k3_status(lo, hi) == S.k3_status(hi, lo) == "indeterminate"
    assert S.k3_status(hi, hi) == "not_triggered"
    assert S.k3_status(lo, _ix(0.04, -0.1, n=11)) == "not_evaluable"


def test_h3_counting_and_literal_lag_per_k():
    ok = {"mean": 0.3, "lo": 0.1, "hi": 0.5, "evaluable": True}
    low = {"mean": 0.05, "lo": -0.1, "hi": 0.2, "evaluable": True}
    ne = {"mean": 0.3, "lo": 0.1, "hi": 0.5, "evaluable": False}
    dpos = {"mean": 0.12, "lo": 0.0, "hi": 0.2, "evaluable": True}
    lit_hi = {"mean": 0.2, "lo": 0.05, "hi": 0.3, "evaluable": True}
    K = S.KS
    v = S.verdict_h3({"+2": ok, "+5": ok, "-3": low}, {k: dpos for k in K}, {k: dpos for k in K}, {k: dpos for k in K},
                     {k: low for k in K})
    assert v["verdict"] == "supported" and v["per_k"]["-3"] == "not_met"
    v = S.verdict_h3({"+2": ok, "+5": ne, "-3": low}, {k: dpos for k in K}, {k: dpos for k in K}, {k: dpos for k in K},
                     {k: low for k in K})
    assert v["verdict"] == "not_evaluable"
    v = S.verdict_h3({"+2": ok, "+5": ok, "-3": low}, {k: dpos for k in K}, {k: dpos for k in K},
                     {"+2": dpos, "+5": low, "-3": dpos}, {k: low for k in K})        # PASS clause fails for +5
    assert v["verdict"] == "not_supported" and v["per_k"]["+5"] == "not_met"
    v = S.verdict_h3({"+2": ok, "+5": low, "-3": low}, {k: dpos for k in K}, {k: dpos for k in K}, {k: dpos for k in K},
                     {"+2": low, "+5": lit_hi, "-3": low})
    assert v["verdict"] == "not_supported"
    assert v["literal_lag_copier_per_k"] == {"+2": False, "+5": True, "-3": False}


UNION = sorted(set(sum(S.CLASS_SETS.values(), [])) | {"B2"})


def _synth(effect_lit=0.0, effect_seq=0.0, p_pass=0.9, seed=0, n_mel=24, lit_pass=None):
    rng = np.random.default_rng(seed)
    rows, planned = [], []
    for m in range(n_mel):
        mid = f"v2c{m:02d}"
        for c in S.CLASS_SETS:
            for s in range(4):
                planned.append((mid, c, "base", f"d{s}"))
                p = lit_pass if (c == "lit" and lit_pass is not None) else (p_pass if c != "null" else 0.1)
                ok = bool(rng.random() < p)
                acc = {h: (float(rng.integers(0, 4)) / 8 if ok else 0.0) for h in UNION}
                if ok and c == "lit" and rng.random() < effect_lit:
                    acc["B"] = 1.0
                if ok and c == "seq" and rng.random() < effect_seq:
                    acc["T+6A"] = 1.0
                rows.append({"id": mid, "split": "confirm", "cond": c, "arm": "base", "seed": f"d{s}",
                             "status": "ok", "acc": acc, "acc_oct": acc, "nf1": acc, "of1": acc,
                             "quality": {"gap_rms_dbfs": -40.0 if c == "null" else -12.0},
                             "gate": {"pass": ok, "pass_octfold": ok, "pass_splice_safe": ok}})
    return rows, planned


def _summ(rows, planned, **kw):
    q = S.quality_decision(rows, planned)
    first = sorted({k[0] for k in planned})[:6]
    return S.summarize(rows, q, 0.7, first, n_boot=NB, n_boot_nested=200, **kw)


def test_summary_null_world_supports_nothing():
    rows, planned = _synth()
    out = _summ(rows, planned)
    assert out["quality"]["decision"] == "pass" and out["paper_level"] == "negative_benchmark", out["verdicts"]
    assert out["primary_population"] == "ALL" and out["K2"]["status"] == "triggered"
    assert out["verdicts"]["H4"]["comparator"] == "null and ctl"
    json.dumps(out)                                                      # JSON-safe end to end


def test_summary_detects_injected_copying_and_sequence():
    rows, planned = _synth(effect_lit=0.8, effect_seq=0.5)
    out = _summ(rows, planned)
    v = {h: x["verdict"] for h, x in out["verdicts"].items()}
    assert (v["H1"], v["H2"], v["H4"], v["H3"], v["H5"]) == ("supported", "supported", "supported", "not_supported", "not_supported")
    assert out["paper_level"] == "supported:H1,H2,H4" and out["K3_status"] == "not_triggered"
    assert out["K2"]["status"] == "not_triggered"
    assert out["labels"]["H1"] == ["PASS meets the threshold"] and out["labels"]["H2"] == []


def test_differential_gating_flag_and_label():
    rows, planned = _synth(effect_lit=0.8, p_pass=0.9, lit_pass=0.75)
    out = _summ(rows, planned)
    g = out["gating"]["H2"]
    assert g["differential_gating"] == (abs(g["pass_rates"][0] - g["pass_rates"][1]) > 0.10)
    if g["differential_gating"]:
        assert "differential gating" in out["labels"]["H2"]


def test_differential_gating_threshold_is_strictly_above_010():
    rows = _rows([(f"m{m:02d}", c, f"s{s}", {"B": 0.0, "B~": 0.0}, (s + 4 * m) % 100 < (100 if c == "lit" else 90))
                  for m in range(25) for c in ("lit", "ctl") for s in range(4)])
    rates = (S.pass_rate(rows, "lit"), S.pass_rate(rows, "ctl"))
    assert rates == (1.0, 0.9)                                           # exactly 0.10 apart: not flagged
    assert not S._r(abs(rates[0] - rates[1])) > 0.10


def test_k2_reads_only_the_first_six_melodies():
    rows, planned = _synth(effect_lit=0.0)
    first = [f"v2c{m:02d}" for m in range(6)]
    a = S.k2(rows, first, NB)
    for r in rows:
        if r["id"] == "v2c06" and r["cond"] == "lit":
            r["acc"]["B"] = 1.0
    b = S.k2(rows, first, NB)
    assert a == b and a["melodies"] == first
    assert S.k2(rows, first[:5], NB)["status"] == "not_evaluable"


def test_k2_threshold_is_strict():
    spec = [(f"v2c{m:02d}", c, f"d{s}", {"B": 0.10 if c == "lit" else 0.0, "B~": 0.0}, True)
            for m in range(6) for c in ("lit", "ctl") for s in range(4)]
    k = S.k2(_rows(spec), [f"v2c{m:02d}" for m in range(6)], NB)
    assert k["CI_lit"]["mean"] == pytest.approx(0.10) and k["triggered"] is False
    spec = [(i, c, s, {"B": 0.0999 if c == "lit" else 0.0, "B~": 0.0}, g) for i, c, s, _, g in spec]
    assert S.k2(_rows(spec), [f"v2c{m:02d}" for m in range(6)], NB)["triggered"] is True


def test_summary_quality_failure_blocks_verdicts():
    rows, planned = _synth(effect_lit=0.8, p_pass=0.6)
    out = _summ(rows, planned)
    assert out["paper_level"] == "infill_failure" and "verdicts" not in out
    d = _summ(rows, planned, diagnostic=True)
    assert "verdicts" in d and d["diagnostic"]


def test_summary_requires_replication_scores():
    rows, planned = _synth()
    del rows[5]["nf1"]
    with pytest.raises(ValueError):
        _summ(rows, planned)


def test_k3_not_evaluable_when_h2_is_not():
    rows, planned = _synth(n_mel=10)
    out = _summ(rows, planned)
    assert out["K3_status"] == "not_evaluable" and out["verdicts"]["H2"]["verdict"] == "not_evaluable"


def test_paper_level_labels():
    v = lambda *xs: dict(zip(("H1", "H2", "H3", "H4", "H5"), xs))
    assert S.paper_level("pass", v(*["not_evaluable"] * 5)) == "not_evaluable"
    assert S.paper_level("pass", v("not_supported", *["not_evaluable"] * 4)) == "partly_not_evaluable"
    assert S.paper_level("pass", v(*["not_supported"] * 5)) == "negative_benchmark"
    assert S.paper_level("pass", v("supported", *["not_evaluable"] * 4)) == "supported:H1"
    assert S.paper_level("fail", {}) == "infill_failure" and S.paper_level("incomplete", {}) == "incomplete_run"
