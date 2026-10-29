"""B2 decision on synthetic Stage T values: the selector, P1 with the max retaken, P2, every verdict, the labels,
and every P0 failure path."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import decide_b2 as D  # noqa: E402
import plan_b2 as P  # noqa: E402

S_STAR, B_STAR = P.parse("oohs_k3_A2"), P.parse("oohs_k4_none")
RUNS = P.test_runs(S_STAR, B_STAR)
ITEMS = sorted({r["item"] for r in RUNS})


def clip(q=0.0, wr=None, npa=None, gate=True, **kw):
    wr = q if wr is None else wr
    npa = q if npa is None else npa
    c = {"scored": True, "gate": gate, "q": q, "wr": wr, "npa_v": npa, "joint": q >= 0.5, "strict": q >= 0.75,
         "npa_mix": npa, "wr_ctc": wr, "q_ctc": q, "vowel_only": False, "wordless": False, "bleed": False,
         "recall_first_half": wr, "recall_second_half": wr, "recall_last_word": wr, "guide_copy": 0.1}
    c.update(kw)
    return c


def values(per_config):
    """per_config: {config: f(item index, draw) -> clip}; other configs get q 0."""
    out = {}
    for r in RUNS:
        f = per_config.get(r["config"])
        out[r["run_id"]] = f(ITEMS.index(r["item"]), r["draw"]) if f else clip(0.0)
    return out


def first(n, q=0.6, other=0.0):
    """Joint on draw 0 of the first n items."""
    return lambda i, r: clip(q if i < n and r == 0 else other)


def analyse(per_config, futility=False):
    return D.analyse(RUNS, values(per_config), S_STAR, futility)


def test_selector_and_bo4():
    draws = [{**clip(0.5, wr=0.6, npa=0.5), "draw": 0, "run_id": "a"}, {**clip(0.5, wr=0.7, npa=0.5), "draw": 1, "run_id": "b"},
             {**clip(0.5, wr=0.7, npa=0.9), "draw": 2, "run_id": "c"}, {**clip(0.5, wr=0.7, npa=0.9), "draw": 3, "run_id": "d"}]
    assert D.S.kept(draws)["run_id"] == "c"  # WR, then NPA_v, then the lower draw
    assert D.S.kept(draws[:2])["run_id"] == "b"
    b = D.bo({"x": draws})
    assert b["x"]["bo4"] == 0.5 and b["x"]["j"] and not b["x"]["strict"] and b["x"]["kept_draw"] == 2


def test_best_of_n_averages_every_subset():
    arm = {"x": [{**clip(q), "draw": d, "run_id": str(d)} for d, q in enumerate((0.0, 0.2, 0.6, 0.8))]}
    bon = D.best_of_n(arm)
    assert bon["1"] == {"mean_best_q": 0.4, "j": 0.5}
    assert bon["4"] == {"mean_best_q": 0.8, "j": 1.0}
    assert bon["2"]["mean_best_q"] == pytest.approx(np.mean([0.2, 0.6, 0.8, 0.6, 0.8, 0.8]))


def test_supported():
    a = analyse({"oohs_k3_A2": lambda i, r: clip(0.8)})
    assert a["rule"]["P1"]["held"] and a["rule"]["P2"]["held"] and a["arms"]["S*"]["J"] == 1.0
    assert D.verdict(True, a["rule"], [])[0] == "B2 supported: the lyric clock gives the capability"
    assert a["labels"]["strict_capability"] and a["primary_comparator"] == "S*_b0"
    assert len(a["demo"]) == 24 and all(d["s_star_q"] == 0.8 for d in a["demo"])
    assert a["reel"] == {"label": "top 6 of 24 by rule", "items": ITEMS[:6]}   # all tied: item order


def test_reel_takes_the_six_highest_s_star_items():
    a = analyse({"oohs_k3_A2": lambda i, r: clip(0.5 + 0.01 * i if i % 2 else 0.2)})
    assert a["reel"]["items"] == [ITEMS[i] for i in (23, 21, 19, 17, 15, 13)]


def test_clock_helps_capability_not_reached():
    a = analyse({"oohs_k3_A2": first(10)})
    assert a["rule"]["P1"]["held"] and not a["rule"]["P2"]["held"]
    assert D.verdict(True, a["rule"], []) == ("Clock helps, capability not reached", ["P2 failed"])


def test_capability_without_clock_credit():
    a = analyse({"oohs_k3_A2": first(15), "oohs_k4_none": first(12)})
    assert a["rule"]["P1"]["delta_j"] == 0.125 and not a["rule"]["P1"]["held"] and a["rule"]["P2"]["held"]
    assert D.verdict(True, a["rule"], []) == ("Capability reached, clock not credited", ["P1 failed"])
    assert a["rule"]["P1"]["best_comparator"] in ("ctx_oohs_k4", "B*")


def test_not_supported_names_both_conditions():
    a = analyse({"oohs_k3_A2": first(4), "noguide": first(20)})
    assert D.verdict(True, a["rule"], []) == ("Not supported", ["P1 failed", "P2 failed"])
    assert a["arms"]["noguide"]["J"] > a["arms"]["S*"]["J"]  # noguide is no comparator


def test_p1_retakes_the_max_in_every_resample():
    per = {"oohs_k3_A2": first(11), "oohs_k3_none": lambda i, r: clip(0.6 if 11 <= i < 14 else 0.0),
           "oohs_k4_none": lambda i, r: clip(0.6 if 14 <= i < 17 else 0.0)}
    a = analyse(per)
    vals = values(per)
    ac = D.arm_clips(RUNS, vals)
    j = {lab: np.array([float(D.bo(ac[lab])[i]["j"]) for i in ITEMS]) for lab in ("S*", "S*_b0", "ctx_oohs_k4")}
    idx = D.boot_index(24)
    best = np.maximum(j["S*_b0"][idx].mean(1), j["ctx_oohs_k4"][idx].mean(1))
    lb = round(float(np.percentile(j["S*"][idx].mean(1) - best, 2.5)), 6)
    single = min(round(float(np.percentile(j["S*"][idx].mean(1) - j[b][idx].mean(1), 2.5)), 6) for b in ("S*_b0", "ctx_oohs_k4"))
    assert a["contrasts"]["delta_j"]["p2_5"] == lb < single
    assert a["rule"]["P1"]["delta_j"] == round(11 / 24 - 3 / 24, 6)


def test_p1_needs_a_positive_lower_bound():
    per = {"oohs_k3_A2": first(6), "oohs_k3_none": lambda i, r: clip(0.0)}
    a = analyse(per)
    assert a["rule"]["P1"]["delta_j"] == 0.25 and a["contrasts"]["delta_j"]["p2_5"] > 0 and a["rule"]["P1"]["held"]
    st = copy.deepcopy(a["arms"])
    con = {"delta_j": {**a["contrasts"]["delta_j"], "p2_5": 0.0}}
    assert not D.rule(st, con, ["S*_b0"])["P1"]["held"]


def test_labels():
    lab = analyse({"oohs_k3_A2": lambda i, r: clip(0.8 if r == 0 else 0.1), "oohs_k3_R2": lambda i, r: clip(0.1),
                   "oohs_k3_A2_shuf": lambda i, r: clip(0.0)})["labels"]
    assert lab["strict_capability"] and lab["head_specific"] == "head-specific clock" and lab["timing"] == "timing-specific"
    lab = analyse({"oohs_k3_A2": lambda i, r: clip(0.8), "oohs_k3_R2": lambda i, r: clip(0.75),
                   "oohs_k3_A2_shuf": lambda i, r: clip(0.8)})["labels"]
    assert lab["head_specific"] == "nonspecific bias" and lab["timing"] == "emphasis, not timing"
    lab = analyse({"oohs_k3_A2": lambda i, r: clip(0.1), "oohs_k3_none": lambda i, r: clip(0.3),
                   "oohs_k3_A2_shuf": lambda i, r: clip(0.1)})["labels"]
    assert lab["head_specific"] == "no clock effect to attribute" and lab["timing"] == "timing not shown"
    lab = analyse({"oohs_k3_A2": lambda i, r: clip(0.6, wr=0.6, npa=0.6, vowel_only=True, recall_first_half=0.9,
                                                   recall_second_half=0.6),
                   "oohs_k3_none": lambda i, r: clip(0.0, wr=0.3, npa=0.65)})["labels"]
    assert lab["timbre_copy"] and lab["words_cut"] and lab["frontier_moved"]
    lab = analyse({"oohs_k3_A2": lambda i, r: clip(0.6, wr=0.6, npa=0.6, q_ctc=0.2)}, futility=True)["labels"]
    assert lab["independent_asr"] == {"J_ctc": 0.0, "P2_with_ctc": False} and lab["dev_futility"] is True
    w = D.analyse(P.test_runs(P.parse("oohs_k4_W2"), B_STAR), {}, P.parse("oohs_k4_W2"), False)
    assert w["labels"]["head_specific"] is None and "S*_rand" not in w["arms"]


def test_missing_clips_count_as_zero():
    a = D.analyse(RUNS, {}, S_STAR, False)
    assert a["arms"]["S*"]["J"] == 0.0 and a["arms"]["S*"]["n_scored"] == 0 and a["arms"]["S*"]["gate_rate"] == 0.0
    assert a["arms"]["S*"]["vowel_only_rate"] is None and a["arms"]["S*"]["vowel_only_of_planned"] == 0.0
    assert a["labels"]["timbre_copy"] is False


def test_verdicts_for_flags_p0_and_no_stage_t():
    r = analyse({"oohs_k3_A2": lambda i, r: clip(0.8)})["rule"]
    assert D.verdict(True, r, ["scale tiny"])[0] == "test only"
    assert D.verdict(False, r, [])[0] == "Technical"
    assert D.verdict(True, None, [], stage_t=False)[0] == "Not supported"


# P0

REG = {"a.py": "1", "b.json": "2"}


def gen_row(run, **kw):
    row = {"run_id": run["run_id"], "stage": run["stage"], "item": run["item"], "draw": run["draw"], "guide": run["guide"],
           "clock": run["clock"], "hooked": run["hooked"], "fallback": run["fallback"], "beta": run["beta"],
           "head_set": run["head_set"], "hook_calls": 40 if run["hooked"] else 0, "max_key": 24 if run["hooked"] else None,
           "n_lyr": 27, "code_sha256": dict(REG), "status": "ok", "peak_before_norm": 0.5,
           "guide_latent_sha256": f"lat-{run['item']}-{run['draw']}-{run['guide']}" if run["guide"] else None,
           "selection_sha256": "sel" if run["stage"] == "test" else None}
    row.update(kw)
    return row


def ref_row():
    return {"whisper": D.S.CS.WHISPER, "whisper_revision": D.S.CS.WHISPER_REVISION, "test_flags": []}


def good_state():
    dev = P.dev_runs("tiny")
    test = P.test_runs(S_STAR, B_STAR, "tiny")
    gens = {r["run_id"]: gen_row(r) for r in test}
    dev_gens = {r["run_id"]: gen_row(r) for r in dev}
    rows = list(dev_gens.values()) + list(gens.values())
    guides = {r["item"]: {r["guide"]: f"g-{r['item']}"} for r in rows}
    for r in rows:
        r["guide_sha256"] = f"g-{r['item']}"
    return {"registration": [], "registered_code": dict(REG), "gen_rows": rows,
            "registered_guides": guides, "setup_hashes_ok": True,
            "selection": {"recomputed_equal": True, "file_sha256": "sel", "ledger_head": "sel", "row_hashes": ["sel", "sel"]},
            "cpu_test": {"passed": True, "sha256_ok": True}, "a0": {"recorded": True, "fallback": False},
            "dev_plan": dev, "dev_gen": dev_gens, "dev_scores": {r["run_id"]: ref_row() for r in dev},
            "test_plan": test, "test_gen": gens, "test_scores": {r["run_id"]: ref_row() for r in test}}


def failed(st):
    return sorted(k for k, v in D.integrity(st).items() if not v["held"])


def test_p0_holds_on_a_clean_state():
    assert failed(good_state()) == []


@pytest.mark.parametrize("breaks,check", [
    (lambda st: st["registration"].append("dacapo/attn_bias.py differs from the B1 code addendum"), "hashes"),
    (lambda st: st["gen_rows"][0]["code_sha256"].update({"a.py": "x"}), "hashes"),
    (lambda st: st["gen_rows"][2].update(guide_sha256="other"), "hashes"),
    (lambda st: st.update(setup_hashes_ok=False), "hashes"),
    (lambda st: st["selection"].update(recomputed_equal=False), "selection"),
    (lambda st: st["selection"].update(ledger_head=None), "selection"),
    (lambda st: st["selection"]["row_hashes"].append(None), "selection"),
    (lambda st: st["cpu_test"].update(passed=False), "cpu_test"),
    (lambda st: st.update(a0=None), "a0"),
    (lambda st: st["a0"].update(fallback=True), "a0"),
    (lambda st: st["gen_rows"][0].update(hooked=True, hook_calls=40, max_key=20), "a0"),
    (lambda st: st["gen_rows"][1].update(hook_calls=39), "hooks"),
    (lambda st: st["gen_rows"][1].update(max_key=27), "hooks"),
    (lambda st: st["gen_rows"][0].update(hook_calls=40), "hooks"),
    (lambda st: st["gen_rows"][1].update(guide_latent_sha256="other"), "guide_latents"),
    (lambda st: next(iter(st["test_scores"].values())).update(whisper_revision="abc"), "reference_and_peak"),
    (lambda st: next(iter(st["test_gen"].values())).update(peak_before_norm=None), "reference_and_peak"),
    (lambda st: next(iter(st["dev_scores"].values())).update(whisper="openai/whisper-small.en"), "reference_and_peak"),
    (lambda st: next(iter(st["dev_gen"].values())).update(peak_before_norm=None), "reference_and_peak"),
    (lambda st: st["test_scores"].clear(), "complete"),
])
def test_each_p0_failure(breaks, check):
    st = good_state()
    breaks(st)
    assert check in failed(st)


def test_fallback_runs_pass_p0_when_a0_found_a_difference():
    st = good_state()
    st["a0"]["fallback"] = True
    for r in st["gen_rows"]:
        if r["clock"] == "none":
            r.update(hooked=True, fallback=True, beta=0.0, head_set="A", hook_calls=40, max_key=24)
    assert failed(st) == []


def test_five_percent_of_planned_runs_may_be_missing():
    test = P.test_runs(S_STAR, B_STAR)
    st = good_state()
    gens = {r["run_id"]: gen_row(r) for r in test}
    ok = {"whisper": D.S.CS.WHISPER, "whisper_revision": D.S.CS.WHISPER_REVISION, "test_flags": []}
    st.update(test_plan=test, test_gen=gens, test_scores={r["run_id"]: dict(ok) for r in test[28:]})
    assert "complete" not in failed(st)  # 28 of 576 unscored, limit floor(28.8)
    st["test_scores"].pop(test[28]["run_id"])
    assert "complete" in failed(st)
