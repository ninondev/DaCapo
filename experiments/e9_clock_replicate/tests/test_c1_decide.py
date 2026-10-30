"""The C1 decision on synthetic values: the selector, R1 at the bootstrap's boundary, R2, every verdict, missing runs
counted as q 0, the 5% limit, the labels, the pooled estimate with B2, and every P0 failure path."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import decide_c1 as D  # noqa: E402
import plan_c1 as P  # noqa: E402
import score_c1 as S  # noqa: E402

RUNS = P.test_runs("full")
ITEMS = sorted({r["item"] for r in RUNS})
CFG = dict(P.ARMS)


def clip(q=0.0, wr=None, npa=None, gate=True, **kw):
    wr = q if wr is None else wr
    npa = q if npa is None else npa
    c = {"scored": True, "gate": gate, "q": q, "wr": wr, "npa_v": npa, "joint": q >= 0.5, "strict": q >= 0.75,
         "npa_mix": npa, "wr_ctc": wr, "q_ctc": q, "wr_ctc_stem": wr, "q_ctc_stem": q, "vowel_only": False,
         "wordless": False, "bleed": False, "recall_first_half": wr, "recall_second_half": wr, "recall_last_word": wr,
         "guide_copy": 0.1}
    c.update(kw)
    return c


def values(per_arm, missing=()):
    """per_arm: {label: f(item index, draw) -> clip}; other arms get q 0; run ids in missing get no value."""
    out = {}
    for r in RUNS:
        if r["run_id"] in missing:
            continue
        f = per_arm.get(r["labels"][0])
        out[r["run_id"]] = f(ITEMS.index(r["item"]), r["draw"]) if f else clip(0.0)
    return out


def pattern(wins, losses, ties_joint=0):
    """S* joint on draw 0 of the first wins items and of the ties; S*_b0 joint on the losses and the ties."""
    def s(i, r):
        return clip(0.6 if r == 0 and (i < wins or wins + losses <= i < wins + losses + ties_joint) else 0.0)

    def b(i, r):
        return clip(0.6 if r == 0 and wins <= i < wins + losses + ties_joint else 0.0)
    return {"S*": s, "S*_b0": b}


def analyse(per_arm, missing=(), b2=None):
    return D.analyse(RUNS, values(per_arm, missing), b2)


def test_selector_keeps_the_largest_q_then_wr_npa_and_lower_draw():
    draws = [{**clip(0.5, wr=0.6, npa=0.5), "draw": 0, "run_id": "a"}, {**clip(0.5, wr=0.7, npa=0.5), "draw": 1, "run_id": "b"},
             {**clip(0.5, wr=0.7, npa=0.9), "draw": 2, "run_id": "c"}, {**clip(0.5, wr=0.7, npa=0.9), "draw": 3, "run_id": "d"}]
    assert S.kept(draws)["run_id"] == "c" and S.kept(draws[:2])["run_id"] == "b" and S.kept(draws[:1])["run_id"] == "a"
    draws.append({**clip(0.51), "draw": 3, "run_id": "e"})
    assert S.kept(draws)["run_id"] == "e"


def test_bootstrap_index_is_the_registered_one():
    a = analyse(pattern(10, 2))
    idx = np.random.default_rng(0).integers(0, 96, (10000, 96))
    import hashlib
    assert a["bootstrap"]["index_sha256"] == hashlib.sha256(idx.tobytes()).hexdigest()
    assert a["bootstrap"] == {**a["bootstrap"], "n_boot": 10000, "seed": 0, "n_items": 96}


@pytest.mark.parametrize("wins,losses,r1", [(10, 3, False), (11, 3, True), (3, 0, False), (4, 0, True)])
def test_r1_at_the_boundary(wins, losses, r1):
    """10 wins and 3 losses give a bootstrap 2.5th percentile of exactly 0 under the registered index: R1 fails, as
    B2's P1 did; one more win lifts it to 1/96 and R1 holds."""
    a = analyse(pattern(wins, losses))
    assert a["rule"]["R1"]["delta_j"] == round((wins - losses) / 96, 6)
    assert (a["rule"]["R1"]["p2_5"] > 0) == r1 and a["rule"]["R1"]["held"] == r1
    if not r1:
        assert a["rule"]["R1"]["p2_5"] <= 0


def test_r1_rounds_before_comparing():
    sums = {"S*": {"J": 0.5, "n_items_joint": 48, "n_items": 96}, "S*_b0": {"J": 0.4}}
    for p25, held in ((0.0, False), (4e-7, False), (6e-7, True), (0.010417, True), (-0.01, False)):
        con = {"delta_j": {"value": 0.1, "p2_5": p25, "ci95": [p25, 0.2]}}
        assert D.rule(sums, con)["R1"]["held"] == held
    sums["S*"]["J"] = 0.4999996
    assert D.rule(sums, {"delta_j": {"value": 0.1, "p2_5": 0.1, "ci95": [0, 0]}})["R2"]["held"]
    sums["S*"]["J"] = 0.499999
    assert not D.rule(sums, {"delta_j": {"value": 0.1, "p2_5": 0.1, "ci95": [0, 0]}})["R2"]["held"]


def test_r2_needs_48_of_96():
    a = analyse(pattern(20, 0, ties_joint=28))           # S* joint on 48 items
    assert a["arms"]["S*"]["n_items_joint"] == 48 and a["rule"]["R2"]["held"]
    a = analyse(pattern(20, 0, ties_joint=27))           # 47
    assert a["arms"]["S*"]["n_items_joint"] == 47 and not a["rule"]["R2"]["held"]


def test_every_verdict():
    rep = analyse(pattern(30, 0, ties_joint=20))["rule"]
    assert rep["R1"]["held"] and rep["R2"]["held"]
    assert D.verdict(True, rep, []) == ("Replicated: the clock helps and the capability holds", [])
    clock = analyse(pattern(30, 0))["rule"]
    assert D.verdict(True, clock, []) == ("Clock helps, capability not reached", ["R2 failed"])
    cap = analyse(pattern(0, 0, ties_joint=60))["rule"]
    assert D.verdict(True, cap, []) == ("Capability holds, clock effect not shown", ["R1 failed"])
    none = analyse(pattern(10, 3))["rule"]
    assert D.verdict(True, none, []) == ("Not replicated", ["R1 failed", "R2 failed"])
    assert D.verdict(False, rep, []) == ("Technical", ["P0 failed"])
    assert D.verdict(True, None, []) == ("Technical", ["P0 failed"])
    assert D.verdict(True, rep, ["scale tiny"])[0] == "test only"


def test_missing_runs_count_as_q_0():
    full = values({"S*": lambda i, r: clip(0.8), "S*_b0": lambda i, r: clip(0.8)})
    gone = {r["run_id"] for r in RUNS if r["labels"] == ["S*"] and r["item"] in ITEMS[:50]}
    a = D.analyse(RUNS, {k: v for k, v in full.items() if k not in gone})
    s = a["arms"]["S*"]
    assert s["n_items_joint"] == 46 and s["n_scored"] == 46 * 4 and s["gate_rate"] == round(46 / 96, 6)
    assert s["wr_ctc_stem"] == round(46 * 4 * 0.8 / 384, 6) and not a["rule"]["R2"]["held"]
    assert D.arm_clips(RUNS, {})["S*"][ITEMS[0]][0]["q"] == 0.0 and S.MISSING["wr_ctc_stem"] == 0.0


def facts(**over):
    """P0 facts of a complete, clean run."""
    plan = RUNS
    code = {"f": "x"}
    gen, scores = {}, {}
    for r in plan:
        g = {k: r[k] for k in D.PLAN_FIELDS}
        g.update(run_id=r["run_id"], status="ok", code_sha256=code, guide_sha256="g" + r["item"], a0_sha256="A",
                 hook_calls=40 if r["hooked"] else 0, max_key=20 if r["hooked"] else None, n_lyr=27, peak_before_norm=0.5,
                 guide_latent_sha256=f"lat{r['item']}{r['draw']}")
        gen[r["run_id"]] = g
        scores[r["run_id"]] = {"whisper": S.CS.WHISPER, "whisper_revision": S.CS.WHISPER_REVISION, "test_flags": []}
    st = {"registration": [], "registered_code": code, "registered_guides": {i: {"oohs": "g" + i} for i in ITEMS},
          "setup_hashes_ok": True, "a0": {"recorded": True, "pass": True, "code_sha256": code}, "a0_sha256": "A",
          "plan": plan, "gen": gen, "scores": scores}
    st.update(over)
    return st


def held(st):
    return {k: v["held"] for k, v in D.integrity(st).items()}


def test_p0_holds_on_a_clean_run():
    assert all(held(facts()).values()) and set(held(facts())) == {"hashes", "a0", "plan", "hooks", "guide_latents",
                                                                  "reference_and_peak", "complete"}


def test_p0_five_percent_limit():
    st = facts()
    ids = [r["run_id"] for r in RUNS]
    for rid in ids[:76]:
        st["scores"].pop(rid)
    p0 = D.integrity(st)
    assert p0["complete"]["held"] and p0["complete"]["detail"] == {"planned": 1536, "missing_failed_or_unscored": 76,
                                                                   "limit": 76}
    st["gen"][ids[100]] = {"run_id": ids[100], "status": "failed"}
    assert not held(st)["complete"]
    st = facts()
    st["scores"][ids[0]] = {"error": "boom"}
    assert held(st)["complete"]


def test_p0_failure_paths():
    rid = RUNS[0]["run_id"]

    def broken(path, value):
        st = copy.deepcopy(facts())
        node = st
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        return held(st)
    assert not broken(("registration",), ["dacapo/attn_bias.py differs from addendum 4"])["hashes"]
    assert not broken(("gen", rid, "code_sha256"), {"f": "y"})["hashes"]
    assert not broken(("gen", rid, "guide_sha256"), "other")["hashes"]
    assert not broken(("setup_hashes_ok",), False)["hashes"]
    assert not broken(("checks_failed",), ["decision: addendum_2 --check failed"])["hashes"]
    assert broken(("checks_failed",), [])["hashes"]
    assert not broken(("a0",), None)["a0"]
    assert not broken(("a0", "pass"), False)["a0"]
    assert not broken(("a0", "code_sha256"), {"f": "old"})["a0"]
    assert not broken(("gen", rid, "a0_sha256"), "B")["a0"]
    assert not broken(("gen", rid, "beta"), 4.0)["plan"]
    assert not broken(("gen", rid, "dit_seed"), 1)["plan"]
    assert not broken(("gen", RUNS[1]["run_id"], "hooked"), True)["plan"]
    assert not broken(("gen", rid, "hook_calls"), 39)["hooks"]
    assert not broken(("gen", rid, "max_key"), 27)["hooks"]
    assert not broken(("gen", RUNS[1]["run_id"], "hook_calls"), 40)["hooks"]
    assert not broken(("gen", rid, "guide_latent_sha256"), "other")["guide_latents"]
    assert not broken(("gen", rid, "guide_latent_sha256"), None)["guide_latents"]
    assert not broken(("gen", rid, "peak_before_norm"), None)["reference_and_peak"]
    assert not broken(("scores", rid, "test_flags"), ["test whisper"])["reference_and_peak"]
    assert not broken(("scores", rid, "whisper_revision"), "main")["reference_and_peak"]


def test_labels():
    def s_star(i, r):
        return clip(0.8, recall_first_half=1.0, recall_second_half=0.6, vowel_only=r == 0) if i < 60 else clip(0.0)
    per = {"S*": s_star, "S*_b0": lambda i, r: clip(0.6 if i < 20 else 0.0),
           "S*_rand": lambda i, r: clip(0.7 if i < 50 else 0.0), "S*_shuf": lambda i, r: clip(0.2)}
    a = analyse(per)
    lab = a["labels"]
    assert a["rule"]["R1"]["held"] and a["rule"]["R2"]["held"]
    assert lab["b2_size_bar"] == {"held": True, "delta_j": round(40 / 96, 6), "needed": 0.2, "with_r1": True}
    assert lab["strict_capability"] and lab["strict_capability_detail"] == {"n_items_strict": 60, "needed": 48}
    assert lab["head_specific"] == "nonspecific bias"     # random heads keep most of the gain
    assert lab["timing"] == "timing-specific" and lab["timing_lower_bound"] > 0
    assert not lab["timbre_copy"]                          # 60 of 384 clips vowel-only: 0.156
    assert lab["words_cut"]                                # first half 1.0 x 60/96, second 0.6 x 60/96: 0.25 apart
    assert lab["frontier_moved"] and "dev_futility" not in lab
    assert lab["independent_asr"] == {"J_ctc_mix": 0.625, "R2_with_ctc_mix": True, "J_ctc_stem": 0.625,
                                      "R2_with_ctc_stem": True}
    assert lab["listeners"] is None and "blind pack" in lab["listeners_note"]
    per["S*_rand"] = lambda i, r: clip(0.0)
    per["S*_shuf"] = lambda i, r: clip(0.8 if i < 60 else 0.0)
    lab = analyse(per)["labels"]
    assert lab["head_specific"] == "head-specific clock" and lab["timing"] == "emphasis, not timing"
    per["S*"] = lambda i, r: clip(0.6 if i < 20 else 0.0, q_ctc=0.0, q_ctc_stem=0.6 if i < 20 else 0.0)
    lab = analyse(per)["labels"]
    assert lab["head_specific"] == "no clock effect to attribute" and not lab["b2_size_bar"]["held"]
    assert lab["independent_asr"]["J_ctc_mix"] == 0.0 and lab["independent_asr"]["J_ctc_stem"] == round(20 / 96, 6)
    assert not lab["strict_capability"]


def test_the_independent_asr_uses_whisper_kept_draws():
    def s(i, r):     # draw 0 has the best Whisper q but no CTC words; draw 1 has CTC words
        return clip(0.8, q_ctc=0.0, q_ctc_stem=0.0) if r == 0 else clip(0.6, q_ctc=0.6, q_ctc_stem=0.6)
    lab = analyse({"S*": s, "S*_b0": lambda i, r: clip(0.0)})["labels"]
    assert lab["independent_asr"]["J_ctc_mix"] == 0.0 and lab["independent_asr"]["J_ctc_stem"] == 0.0


def test_pooled_with_b2():
    b2 = D.b2_item_outcomes(json.loads(P.B2_RESULT.read_text()))
    assert sum(v["j"] for v in b2["S*"].values()) == 14 and sum(v["j"] for v in b2["S*_b0"].values()) == 8
    a = analyse(pattern(30, 0), b2=b2)
    pool = a["pooled_with_b2"]
    assert pool["label"] == "exploratory" and pool["n_items"] == 120 and (pool["n_b2"], pool["n_c1"]) == (24, 96)
    assert pool["delta_j"]["value"] == round((6 + 30) / 120, 6)
    assert pool["delta_j"]["ci95"][0] <= pool["delta_j"]["value"] <= pool["delta_j"]["ci95"][1]
    assert "integers(0, 120, (10000, 120))" in pool["bootstrap"]


def test_demo_pairs_s_star_with_s_b0():
    a = analyse(pattern(10, 2))
    assert len(a["demo"]) == 96 and all(d["comparator"] == "S*_b0" for d in a["demo"])
    d0 = a["demo"][0]
    assert d0["s_star_run"] == f"{ITEMS[0]}_r0_oohs_k3_A8" and d0["comparator_run"].endswith("oohs_k3_none")


def test_summary_reports_every_arm_and_best_of_n():
    a = analyse({lab: (lambda i, r, q=q: clip(q if r == 1 else 0.0)) for lab, q in zip(P.ARM_ORDER, (0.9, 0.6, 0.5, 0.3))})
    assert list(a["arms"]) == list(P.ARM_ORDER)
    assert [a["arms"][x]["J"] for x in P.ARM_ORDER] == [1.0, 1.0, 1.0, 0.0]
    s = a["arms"]["S*"]
    assert s["best_of_n"]["1"] == {"mean_best_q": round(0.9 / 4, 6), "j": 0.25} and s["best_of_n"]["4"]["j"] == 1.0
    assert set(a["contrasts"]["paired"]) == {"S*_b0", "S*_rand", "S*_shuf"}
    assert all(s["items"][i]["kept_draw"] == 1 for i in ITEMS)
