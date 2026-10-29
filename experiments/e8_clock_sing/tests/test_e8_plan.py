"""B2 plan: seeds are collision-free and dev and test disjoint, Stage D has 30 configs and 1,080 runs, the Stage T
builder merges coinciding arms and adds S*_rand only for set A, the windows, the fallback, A0 and the trim."""
import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import plan_b2 as P  # noqa: E402

# E6's random sets 3-6 (spec section 1, fact 8), used by S*_rand for draws 0-3
RAND_3_6 = [{2: [4], 3: [8, 9], 4: [5], 5: [10, 11], 6: [2]}, {2: [11], 3: [6, 7], 4: [10], 5: [2, 3], 6: [1]},
            {2: [2], 3: [12, 13], 4: [13], 5: [0, 1], 6: [14]}, {2: [5], 3: [12, 13], 4: [8], 5: [0, 1], 6: [10]}]


def test_seeds_collision_free_and_disjoint():
    tab = P.seed_table()
    assert P.seed_problems(tab) == []
    assert P.read_json(P.SEEDS_JSON) == json.loads(json.dumps(tab))
    for scale in ("full", "tiny"):
        dev = {s for v in tab[scale]["dev"].values() for p in v for s in p}
        test = {s for v in tab[scale]["test"].values() for p in v for s in p}
        assert dev and test and not dev & test
    assert tab["full"]["a0"]["item00"] == [list(P.run_seeds("runB2-a0", "item00", r)) for r in (0, 1)]
    assert set(P.TINY["test_items"]) <= set(tab["tiny"]["test"]) and not set(P.TINY["test_items"]) & set(tab["full"]["test"])


def test_a_collision_is_reported(monkeypatch):
    tab = P.seed_table()
    first = tab["full"]["dev"]["item00"][0][0]
    monkeypatch.setattr(P, "earlier_seeds", lambda: {"b1": {first}})
    assert any("b1" in p for p in P.seed_problems(tab))
    tab2 = copy.deepcopy(tab)
    tab2["full"]["test"]["b2t00"][0][0] = first
    monkeypatch.setattr(P, "earlier_seeds", lambda: {})
    assert any("repeat" in p for p in P.seed_problems(tab2)) and any("share" in p for p in P.seed_problems(tab2))


def test_earlier_seeds_include_the_registered_sets():
    e = P.earlier_seeds()
    assert {1, 2} <= e["e4"] and {63000, 963000, 63112, 963112} <= e["b1"] and {6002, 906002} == e["runb_cpu_test"]
    assert {61000, 961000, 62312, 962314} <= e["e6"] and {20261029, 42} <= e["posthoc_and_heads"]
    assert e["b1_tiny"] == set(P.run_seeds("runB1-tiny", "item00", 0)) and not e["b1_tiny"] & e["b1"]


def test_stage_d_grid():
    cfgs = P.dev_configs()
    assert len(cfgs) == 30 and len({c["config"] for c in cfgs}) == 30
    assert {c["clock"] for c in cfgs} == {"none", "A2", "A4", "A8", "W2"} and {c["k"] for c in cfgs} == {3, 4, 5}
    runs = P.dev_runs()
    assert len(runs) == 1080 and len({r["run_id"] for r in runs}) == 1080
    assert [r["item"] for r in runs[:30]] == ["item00"] * 30 and runs[30]["draw"] == 1 and runs[90]["item"] == "item01"
    by = {}
    for r in runs:
        by.setdefault((r["item"], r["draw"]), set()).add((r["dit_seed"], r["rng_seed"]))
        assert (r["dit_seed"], r["rng_seed"]) == P.run_seeds("runB2-dev", r["item"], r["draw"])
        assert r["task"] == ("cover-nofsq" if r["guide"] else "text2music")
        assert r["cover_strength"] == r["k"] / 8 and r["caption"] == P.CAPTION
    assert all(len(v) == 1 for v in by.values()) and len(by) == 36
    w2 = next(r for r in runs if r["config"] == "oohs_k3_W2")
    assert w2["heads"] == {str(L): list(range(16)) for L in (2, 3, 4, 5, 6)} and P.n_steered(P.parse("oohs_k3_W2")) == 80
    a8 = next(r for r in runs if r["config"] == "clarinet_k5_A8")
    assert a8["beta"] == 8.0 and a8["heads"] == {str(k): v for k, v in P.R.ALIGN.items()}
    none = next(r for r in runs if r["config"] == "oohs_k4_none")
    assert not none["hooked"] and none["heads"] is None


def sec(guided, t2m=None):
    return {"guided": guided, "text2music": guided if t2m is None else t2m}


def test_trim_steps():
    trim = {"drop_clarinet_k3": True, "drop_noguide_test": False}
    assert len(P.dev_configs(trim)) == 25 and not any(c["config"].startswith("clarinet_k3") for c in P.dev_configs(trim))
    assert len(P.dev_runs("full", trim)) == 900
    assert P.timing_trim(sec(10.0))["drop_clarinet_k3"] is False
    t = P.timing_trim(sec(12.0))
    assert t["drop_clarinet_k3"] and not t["drop_noguide_test"] and t["projected_hours"][0] > 7.0 >= t["projected_hours"][1]
    t = P.timing_trim(sec(13.0))
    assert t["drop_clarinet_k3"] and t["drop_noguide_test"] and len(t["projected_hours"]) == 3
    assert P.projected_hours(sec(10.0), P.NO_TRIM) == pytest.approx(35 / 60 + (180 + 1080 + 768) * 10 / 3600)
    # by kind: B1 and Stage T's noguide arm are text2music, Stage D and seven Stage T arms guided
    want = 35 / 60 + ((180 + 96) * 4.0 + (1080 + 672) * 20.0) / 3600
    assert P.projected_hours(sec(20.0, 4.0), P.NO_TRIM) == pytest.approx(want)
    assert P.projected_hours(sec(20.0, 4.0), {"drop_noguide_test": True}) == pytest.approx(want - 96 * 4.0 / 3600)


def test_loop_allowance():
    assert P.seconds_per_run(3.0, 7.0, "guided") == pytest.approx(10.0 + 0.25 * 46.0)
    assert P.seconds_per_run(3.0, 7.0, "text2music") == pytest.approx(10.0 + 0.08 * 46.0)
    # Run A's L4 timings without loops stay under 7 h; with the registered allowance both trim steps apply
    assert P.timing_trim(sec(10.0, 9.0))["projected_hours"] == [pytest.approx(6.14, abs=0.01)]
    t = P.timing_trim(sec(P.seconds_per_run(3.0, 7.0, "guided"), P.seconds_per_run(2.5, 6.5, "text2music")))
    assert t["drop_clarinet_k3"] and t["drop_noguide_test"] and t["loop_allowance"]["rate"]["guided"] == 0.25
    inf = P.timing_trim(sec(float("inf")))
    assert inf["drop_clarinet_k3"] and inf["drop_noguide_test"]


def labels(arms):
    return [a["labels"] for a in arms]


def test_stage_t_builder_merges_and_adds_random_only_for_set_a():
    s, b = P.parse("oohs_k3_A2"), P.parse("oohs_k4_none")
    arms = P.test_arms(s, b)
    assert labels(arms) == [["S*"], ["S*_b0", "ctx_oohs_k3"], ["S*_rand"], ["S*_shuf"], ["ctx_oohs_k4", "B*"], ["noguide"]]
    assert len(P.test_runs(s, b)) == 6 * 96
    w = P.parse("oohs_k5_W2")
    arms = P.test_arms(w, P.parse("clarinet_k5_none"))
    assert "S*_rand" not in [x for a in arms for x in a["labels"]]
    assert labels(arms) == [["S*"], ["S*_b0"], ["S*_shuf"], ["ctx_oohs_k3"], ["ctx_oohs_k4"], ["noguide"], ["B*"]]
    most = P.test_arms(P.parse("clarinet_k5_A8"), P.parse("clarinet_k4_none"))
    assert len(most) == 8 and len(P.test_runs(P.parse("clarinet_k5_A8"), P.parse("clarinet_k4_none"))) == 768
    least = P.test_arms(P.parse("oohs_k3_W2"), P.parse("oohs_k3_none"))
    assert len(least) == 5 and least[1]["labels"] == ["S*_b0", "ctx_oohs_k3", "B*"]
    trimmed = P.test_arms(s, b, {"drop_clarinet_k3": True, "drop_noguide_test": True})
    assert "noguide" not in [x for a in trimmed for x in a["labels"]]


def test_stage_t_runs_order_seeds_and_random_sets():
    s, b = P.parse("oohs_k3_A2"), P.parse("oohs_k4_none")
    runs = P.test_runs(s, b)
    assert [r["config"] for r in runs[:6]] == ["oohs_k3_A2", "oohs_k3_none", "oohs_k3_R2", "oohs_k3_A2_shuf", "oohs_k4_none",
                                                "noguide"]
    assert runs[6]["draw"] == 1 and runs[24]["item"] == "b2t01"
    for r in runs:
        assert (r["dit_seed"], r["rng_seed"]) == P.run_seeds("runB2-test", r["item"], r["draw"])
        if r["config"] == "oohs_k3_R2":
            want = RAND_3_6[r["draw"]]
            assert r["heads"] == {str(k): v for k, v in want.items()} and r["beta"] == 2.0
            assert {int(k): v for k, v in r["heads"].items()} == P.R.random_head_sets()[3 + r["draw"]]
        if r["config"] == "oohs_k3_A2_shuf":
            assert r["shuffle"] and r["heads"] == {str(k): v for k, v in P.R.ALIGN.items()}
    tiny = P.test_runs(s, b, "tiny")   # a dev item: the tiny scale touches no test item
    assert [r["run_id"] for r in tiny] == ["item01_r0_oohs_k3_A2", "item01_r0_oohs_k3_none"]
    assert tiny[0]["dit_seed"] == P.run_seeds("runB2-test-tiny", "item01", 0)[0] and tiny[0]["stage"] == "test"
    assert not {r["item"] for r in tiny} & {it["id"] for it in P.test_items()}


def test_tiny_dev_runs_compare_guide_latents_across_k():
    tiny = P.dev_runs("tiny")
    assert [r["config"] for r in tiny] == ["oohs_k3_none", "oohs_k4_A2"]
    assert len({(r["item"], r["draw"], r["guide"]) for r in tiny}) == 1 and len({r["k"] for r in tiny}) == 2


def test_windows_and_rotation():
    it = P.all_items()["b2t00"]
    keys = P.read_json(P.TOK_JSON)["items"]["b2t00"]["keys"]
    sp = P.spans(it, keys)
    on = P.written_onsets(it)
    assert [a for _, a, _ in sp] == on and [b for _, _, b in sp][:-1] == on[1:]
    last = it["melody"]["notes"][-1]
    assert sp[-1][2] == pytest.approx((last["start"] + last["dur"]) * 0.6)
    rot = P.spans(it, keys, shuffle=True)
    n = len(sp)
    assert [k for k, _, _ in rot] == [k for k, _, _ in sp]
    assert [(a, b) for _, a, b in rot] == [(sp[(j + n // 2) % n][1], sp[(j + n // 2) % n][2]) for j in range(n)]


def test_fallback_and_a0():
    run = P.dev_runs()[0]
    assert run["config"] == "oohs_k3_none" and not run["hooked"]
    fb = P.with_fallback(run)
    assert fb["hooked"] and fb["fallback"] and fb["beta"] == 0.0 and fb["head_set"] == "A"
    clocked = P.dev_runs()[1]
    assert P.with_fallback(clocked) is clocked
    a0 = P.a0_runs()
    assert [r["name"] for r in a0] == ["t2m_nohook", "t2m_nohook_repeat", "t2m_A0", "cover_nohook", "cover_A0", "cover_A2",
                                       "cover_k3", "cover_k5"]
    assert all((r["dit_seed"], r["rng_seed"]) == P.run_seeds("runB2-a0", "item00", r["draw"]) for r in a0)
    assert [r["hooked"] for r in a0] == [False, False, True, False, True, True, False, False] and a0[5]["beta"] == 2.0
    covers = [r for r in a0 if r["guide"]]   # one draw and guide, three k: A0 compares their guide latents
    assert {(r["draw"], r["guide"]) for r in covers} == {(1, "oohs")} and {r["k"] for r in covers} == {3, 4, 5}
    assert [r["name"] for r in P.a0_runs("tiny")] == ["cover_nohook", "cover_A0"]


def test_config_ids_round_trip():
    for c in P.dev_configs() + [P.config("oohs", 3, "A2", random=True), P.config("clarinet", 5, "W2", shuffle=True),
                                P.config(None, None)]:
        assert P.parse(c["config"]) == c
    with pytest.raises(ValueError):
        P.config("oohs", 3, "none", shuffle=True)
