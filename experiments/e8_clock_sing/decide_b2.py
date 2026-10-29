"""The B2 decision (addendum 3), applied to Stage T only: integrity (P0), clock credit (P1), capability (P2),
the verdict, the labels and the tables. Pure functions of plain rows; run_b2.py decide gathers them.

  q per clip (score_b2.clip); per item and arm, bo4 = the q of the draw the selector keeps (largest q, then
  WR, then NPA_v, then the lower draw); J_i = 1 if bo4 >= 0.5; J(arm) = mean of J_i over the 24 items.
  P1  Delta_J = J(S*) - max_b J(b) >= 0.20 over the unclocked guided arms b in Stage T (S*_b0, ctx_oohs_k3,
      ctx_oohs_k4, B*), and the 2.5th percentile of the bootstrap Delta_J (max retaken) > 0
  P2  J(S*) >= 0.50
Intervals: cluster bootstrap over items, 10,000 resamples, numpy default_rng(0), all arms resampled together.
Values are rounded to 1e-6 before every comparison.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan_b2 as P  # noqa: E402
import score_b2 as S  # noqa: E402

DELTA_J_MIN, J_MIN = 0.20, 0.50
STRICT_ITEMS = 12
N_BOOT, BOOT_SEED = 10000, 0
MAX_MISSING_SHARE = 0.05
TIMBRE_COPY, WORDS_CUT, FRONTIER_NPA, FRONTIER_WR = 0.30, 0.20, 0.10, 0.20
REEL_ITEMS = 6
VERDICTS = {"supported": "B2 supported: the lyric clock gives the capability",
            "clock_only": "Clock helps, capability not reached",
            "capability_only": "Capability reached, clock not credited",
            "not_supported": "Not supported", "technical": "Technical", "test": "test only"}


def rnd(x) -> float:
    return round(float(x), S.DIGITS)


# per arm and item

def arm_clips(runs: list[dict], clips: dict[str, dict]) -> dict[str, dict[str, list[dict]]]:
    """{arm label: {item: [clip values with draw, run_id]}}; a merged run belongs to all its labels."""
    out: dict = {}
    for run in runs:
        c = {**clips.get(run["run_id"], S.MISSING), "draw": run["draw"], "run_id": run["run_id"]}
        for lab in run["labels"]:
            out.setdefault(lab, {}).setdefault(run["item"], []).append(c)
    return out


def items_of(arm: dict[str, list[dict]]) -> list[str]:
    return sorted(arm)


def bo(arm: dict[str, list[dict]]) -> dict[str, dict]:
    """Per item: the kept draw and its q."""
    out = {}
    for i, draws in arm.items():
        k = S.kept(draws)
        out[i] = {"bo4": k["q"], "kept_draw": k["draw"], "kept_run": k["run_id"], "j": k["q"] >= S.JOINT,
                  "strict": k["q"] >= S.STRICT}
    return out


def best_of_n(arm: dict[str, list[dict]]) -> dict[str, dict]:
    """Best-of-n for n = 1 to the number of draws: the mean over items, and over every subset of n draws,
    of the largest q, and the share of items whose best q >= 0.5."""
    out = {}
    n_max = max(len(v) for v in arm.values())
    for n in range(1, n_max + 1):
        qs, js = [], []
        for draws in arm.values():
            sub = [max(c["q"] for c in s) for s in itertools.combinations(draws, n)]
            qs.append(np.mean(sub))
            js.append(np.mean([q >= S.JOINT for q in sub]))
        out[str(n)] = {"mean_best_q": rnd(np.mean(qs)), "j": rnd(np.mean(js))}
    return out


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return rnd(np.mean(xs)) if xs else None


def _median(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return rnd(np.median(xs)) if xs else None


def summary(arm: dict[str, list[dict]]) -> dict:
    """Section 3.13, one arm. q, NPA, WR and gate count a missing clip as 0; the rates of vowel-only,
    wordless, bleed and the guide-copy index are over scored clips."""
    b = bo(arm)
    clips = [c for v in arm.values() for c in v]
    sc = [c for c in clips if c["scored"]]
    return {"n_items": len(arm), "n_clips": len(clips), "n_scored": len(sc),
            "J": rnd(np.mean([x["j"] for x in b.values()])),
            "n_items_joint": sum(x["j"] for x in b.values()), "n_items_strict": sum(x["strict"] for x in b.values()),
            "joint_clips": sum(c["joint"] for c in clips), "strict_clips": sum(c["strict"] for c in clips),
            "mean_bo4_q": rnd(np.mean([x["bo4"] for x in b.values()])), "mean_q": rnd(np.mean([c["q"] for c in clips])),
            "npa_v": rnd(np.mean([c["npa_v"] for c in clips])), "npa_mix": rnd(np.mean([c["npa_mix"] for c in clips])),
            "wr": rnd(np.mean([c["wr"] for c in clips])), "gate_rate": rnd(np.mean([c["gate"] for c in clips])),
            "wr_ctc": rnd(np.mean([c["wr_ctc"] for c in clips])),
            "npa_v_oct": _mean(c.get("npa_v_oct") for c in sc), "npa_v_chance": _mean(c.get("npa_v_chance") for c in sc),
            "npa_mix_oct": _mean(c.get("npa_mix_oct") for c in sc),
            "npa_mix_chance": _mean(c.get("npa_mix_chance") for c in sc),
            "wr_multiset": _mean(c.get("wr_multiset") for c in sc), "wer_capped": _mean(c.get("wer_capped") for c in sc),
            "vowel_only_rate": _mean(c.get("vowel_only") for c in sc), "wordless_rate": _mean(c.get("wordless") for c in sc),
            "vowel_only_of_planned": rnd(np.mean([bool(c.get("vowel_only")) for c in clips])),
            "bleed_rate": _mean(c.get("bleed") for c in sc), "guide_copy": _mean(c.get("guide_copy") for c in sc),
            "recall_first_half": rnd(np.mean([c.get("recall_first_half", 0.0) for c in clips])),
            "recall_second_half": rnd(np.mean([c.get("recall_second_half", 0.0) for c in clips])),
            "recall_last_word": rnd(np.mean([c.get("recall_last_word", 0.0) for c in clips])),
            "onset_err_abs_median": _median(c.get("onset_err_abs") for c in sc),
            "onset_err_signed_median": _median(c.get("onset_err_signed") for c in sc),
            "n_onset_clips": sum(c.get("onset_err_abs") is not None for c in sc),
            "best_of_n": best_of_n(arm), "items": b}


# bootstrap

def boot_index(n_items: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_items, size=(n_boot, n_items))


def ci(x: np.ndarray) -> list[float]:
    return [rnd(np.percentile(x, 2.5)), rnd(np.percentile(x, 97.5))]


def contrasts(bos: dict[str, dict], items: list[str], comparators: list[str], idx: np.ndarray) -> dict:
    """Delta_J with the max over the comparators retaken in every resample, and the paired contrasts of S*
    against every other arm (mean bo4 q and J)."""
    q = {a: np.array([bos[a][i]["bo4"] for i in items]) for a in bos}
    j = {a: np.array([float(bos[a][i]["j"]) for i in items]) for a in bos}
    out: dict = {}
    if comparators:
        best = np.max(np.stack([j[b][idx].mean(1) for b in comparators]), axis=0)
        d = j["S*"][idx].mean(1) - best
        out["delta_j"] = {"value": rnd(j["S*"].mean() - max(j[b].mean() for b in comparators)),
                          "ci95": ci(d), "p2_5": rnd(np.percentile(d, 2.5)), "comparators": comparators}
    out["paired"] = {}
    for a in bos:
        if a == "S*":
            continue
        dq, dj = q["S*"] - q[a], j["S*"] - j[a]
        out["paired"][a] = {"mean_bo4_q": {"value": rnd(dq.mean()), "ci95": ci(dq[idx].mean(1))},
                            "J": {"value": rnd(dj.mean()), "ci95": ci(dj[idx].mean(1))}}
    return out


# integrity (P0)

def integrity(st: dict) -> dict[str, dict]:
    """P0 as named checks, each {"held": bool, "detail": ...}. st holds plain facts gathered by run_b2:
      registration       problems with the files now (addendum 3, the B1 code addendum, addendum 2)
      registered_code    {path: sha256} that every generation row must have recorded
      registered_guides  {item: {guide: sha256}} of addendum 3, which every guided row's guide must match
      setup_hashes_ok    setup rendered the 72 guides to their sha256 with the registered soundfont
      gen_rows           every ok B2 generation row (dev and test)
      selection          {"recomputed_equal", "file_sha256", "ledger_head", "row_hashes"}
      cpu_test           {"passed", "sha256_ok"}
      a0                 the A0 report or None; fallback is a0["fallback"]
      test_plan, test_gen (run_id -> last ledger row), test_scores (run_id -> score row), and the same three for
      Stage D (dev_plan, dev_gen, dev_scores): every scored clip of either stage must use the registered
      Whisper and every generated clip must have its peak; the 5% limit counts Stage T only"""
    out = {}
    reg = list(st["registration"])
    bad = [r["run_id"] for r in st["gen_rows"] if r.get("code_sha256") != st["registered_code"]]
    guides = st["registered_guides"]
    gbad = [r["run_id"] for r in st["gen_rows"] if r.get("guide")
            and r.get("guide_sha256") != guides.get(r["item"], {}).get(r["guide"])]
    detail = reg + ([f"{len(bad)} runs generated with other code: {bad[:5]}"] if bad else [])
    detail += [f"{len(gbad)} runs used a guide other than the registered one: {gbad[:5]}"] if gbad else []
    detail += [] if st["setup_hashes_ok"] else ["setup did not reproduce the registered soundfont and guides"]
    out["hashes"] = {"held": not detail, "detail": detail}
    sel = st["selection"]
    rows_ok = all(h == sel["file_sha256"] for h in sel["row_hashes"])
    out["selection"] = {"held": bool(sel["recomputed_equal"] and sel["file_sha256"] and sel["ledger_head"] == sel["file_sha256"]
                                     and rows_ok),
                        "detail": {k: sel[k] for k in ("recomputed_equal", "file_sha256", "ledger_head")} |
                        {"rows_carry_hash": rows_ok}}
    cpu = st["cpu_test"]
    out["cpu_test"] = {"held": bool(cpu.get("passed") and cpu.get("sha256_ok")), "detail": cpu}
    a0 = st["a0"]
    fallback = bool(a0 and a0.get("fallback"))
    unclocked = [r for r in st["gen_rows"] if r.get("clock") == "none"]
    fb_bad = [r["run_id"] for r in unclocked if fallback and not (r.get("hooked") and r.get("fallback")
                                                                  and r.get("beta") == 0.0 and r.get("head_set") == "A")]
    nf_bad = [r["run_id"] for r in unclocked if not fallback and r.get("hooked")]
    out["a0"] = {"held": bool(a0 and a0.get("recorded")) and not fb_bad and not nf_bad,
                 "detail": {"recorded": bool(a0 and a0.get("recorded")), "fallback": fallback,
                            "runs_not_following_fallback": fb_bad[:5] + nf_bad[:5]}}
    def hook_ok(r: dict) -> bool:
        if not r.get("hooked"):
            return r.get("hook_calls") == 0
        return r.get("hook_calls") == P.HOOK_CALLS and r.get("max_key") is not None and r["max_key"] < r["n_lyr"]
    hook_bad = [r["run_id"] for r in st["gen_rows"] if not hook_ok(r)]
    out["hooks"] = {"held": not hook_bad, "detail": hook_bad[:10]}
    lat: dict = {}
    for r in st["gen_rows"]:
        if r.get("guide"):
            lat.setdefault((r["stage"], r["item"], r["draw"], r["guide"]), set()).add(r.get("guide_latent_sha256"))
    lat_bad = [list(k) for k, v in lat.items() if len(v) != 1 or None in v]
    out["guide_latents"] = {"held": not lat_bad, "detail": lat_bad[:10]}
    def stage_rows(stage: str) -> tuple[list[dict], list[dict], dict]:
        """(generated runs, their usable score rows, gen rows) of a stage's plan."""
        plan, gens, scores = st.get(f"{stage}_plan", []), st.get(f"{stage}_gen", {}), st.get(f"{stage}_scores", {})
        ok_runs = [r for r in plan if gens.get(r["run_id"], {}).get("status") == "ok"]
        scored = [scores[r["run_id"]] for r in ok_runs if r["run_id"] in scores and "error" not in scores[r["run_id"]]]
        return ok_runs, scored, gens
    wrong_ref, no_peak = {}, []
    for stage in ("dev", "test"):
        ok_runs, scored, gens = stage_rows(stage)
        wrong_ref[stage] = sum(not S.CS.registered_reference(s) for s in scored)
        no_peak += [r["run_id"] for r in ok_runs if gens[r["run_id"]].get("peak_before_norm") is None]
    out["reference_and_peak"] = {"held": not any(wrong_ref.values()) and not no_peak,
                                 "detail": {"other_reference": wrong_ref, "no_peak": no_peak[:5]}}
    plan = st["test_plan"]
    incomplete = len(plan) - len(stage_rows("test")[1])
    limit = math.floor(MAX_MISSING_SHARE * len(plan))
    out["complete"] = {"held": incomplete <= limit, "detail": {"planned": len(plan), "missing_failed_or_unscored": incomplete,
                                                               "limit": limit}}
    return out


# the rule

def rule(sums: dict, con: dict, comparators: list[str]) -> dict:
    jd = con["delta_j"]
    p1 = rnd(jd["value"]) >= DELTA_J_MIN and rnd(jd["p2_5"]) > 0
    p2 = rnd(sums["S*"]["J"]) >= J_MIN
    return {"P1": {"held": p1, "delta_j": jd["value"], "needed": DELTA_J_MIN, "p2_5": jd["p2_5"],
                   "best_comparator": max(comparators, key=lambda b: (sums[b]["J"], sums[b]["mean_bo4_q"],
                                                                       -P.ARM_ORDER.index(b))),
                   "comparators": {b: sums[b]["J"] for b in comparators}},
            "P2": {"held": p2, "J": sums["S*"]["J"], "needed": J_MIN}}


def verdict(p0_ok: bool, r: dict | None, flags: list[str], stage_t: bool = True) -> tuple[str, list[str]]:
    """(verdict, failed conditions)."""
    if flags:
        return VERDICTS["test"], [f"test run ({', '.join(flags)})"]
    if not p0_ok:
        return VERDICTS["technical"], ["P0 failed"]
    if not stage_t:
        return VERDICTS["not_supported"], ["no clocked config passed the gate on dev: Stage T not run"]
    p1, p2 = r["P1"]["held"], r["P2"]["held"]
    if p1 and p2:
        return VERDICTS["supported"], []
    if p1:
        return VERDICTS["clock_only"], ["P2 failed"]
    if p2:
        return VERDICTS["capability_only"], ["P1 failed"]
    return VERDICTS["not_supported"], ["P1 failed", "P2 failed"]


def labels(sums: dict, con: dict, s_star: dict, comparators: list[str], idx: np.ndarray, items: list[str],
           bos: dict, ctc_j: float | None, dev_futility: bool | None) -> dict:
    """Section 3.12; a label never changes the verdict. A label whose arm is absent is None."""
    s = sums["S*"]
    out: dict = {"strict_capability": s["n_items_strict"] >= STRICT_ITEMS}
    gain = rnd(s["mean_bo4_q"] - sums["S*_b0"]["mean_bo4_q"]) if "S*_b0" in sums else None
    if s_star["head_set"] == "A" and "S*_rand" in sums and gain is not None:
        if gain <= 0:
            out["head_specific"] = "no clock effect to attribute"
        else:
            spec = rnd(s["mean_bo4_q"] - sums["S*_rand"]["mean_bo4_q"]) >= rnd(0.5 * gain)
            out["head_specific"] = "head-specific clock" if spec else "nonspecific bias"
    else:
        out["head_specific"] = None
    if "S*_shuf" in sums:
        d = np.array([bos["S*"][i]["bo4"] - bos["S*_shuf"][i]["bo4"] for i in items])
        lb = rnd(np.percentile(d[idx].mean(1), 2.5))
        shuf_gain = rnd(sums["S*_shuf"]["J"] - max(sums[b]["J"] for b in comparators)) if comparators else None
        if lb > 0:
            out["timing"] = "timing-specific"
        elif shuf_gain is not None and shuf_gain >= DELTA_J_MIN:
            out["timing"] = "emphasis, not timing"
        else:
            out["timing"] = "timing not shown"
        out["timing_lower_bound"] = lb
    else:
        out["timing"] = None
    out["timbre_copy"] = s["vowel_only_of_planned"] >= TIMBRE_COPY   # over all 96 planned clips
    out["words_cut"] = rnd(s["recall_first_half"] - s["recall_second_half"]) >= WORDS_CUT
    out["dev_futility"] = dev_futility
    if "S*_b0" in sums:
        b0 = sums["S*_b0"]
        out["frontier_moved"] = (rnd(s["npa_v"]) >= rnd(b0["npa_v"] - FRONTIER_NPA)
                                 and rnd(s["wr"]) >= rnd(b0["wr"] + FRONTIER_WR))
    else:
        out["frontier_moved"] = None
    out["independent_asr"] = None if ctc_j is None else {"J_ctc": ctc_j, "P2_with_ctc": rnd(ctc_j) >= J_MIN}
    return out


def ctc_j(arm: dict[str, list[dict]]) -> float:
    """P2 recomputed with the CTC model's WR on the draws the selector kept (by Whisper's q)."""
    return rnd(np.mean([S.kept(d)["q_ctc"] >= S.JOINT for d in arm.values()]))


def primary_comparator(sums: dict, comparators: list[str]) -> str | None:
    """The unclocked guided arm with the highest J on test; ties go to mean bo4 q, then to the arm order."""
    if not comparators:
        return None
    return max(comparators, key=lambda b: (sums[b]["J"], sums[b]["mean_bo4_q"], -P.ARM_ORDER.index(b)))


def demo(bos: dict, items: list[str], comp: str | None) -> list[dict]:
    """For each test item, the S* draw the selector keeps next to the primary comparator's kept draw."""
    return [{"item": i, "s_star_run": bos["S*"][i]["kept_run"], "s_star_q": bos["S*"][i]["bo4"],
             "comparator": comp, "comparator_run": bos[comp][i]["kept_run"] if comp else None,
             "comparator_q": bos[comp][i]["bo4"] if comp else None} for i in items]


def reel(bos: dict, items: list[str], n: int = REEL_ITEMS) -> dict:
    """The short reel: the n items with the highest S* q of the kept draw (ties to item order)."""
    top = sorted(items, key=lambda i: (-bos["S*"][i]["bo4"], items.index(i)))[:n]
    return {"label": f"top {min(n, len(items))} of {len(items)} by rule", "items": top}


def analyse(runs: list[dict], clips: dict[str, dict], s_star: dict, dev_futility: bool | None) -> dict:
    """Everything decide reports about Stage T, from the planned runs and their per-clip values."""
    ac = arm_clips(runs, clips)
    items = sorted({r["item"] for r in runs})
    sums = {a: summary(ac[a]) for a in P.ARM_ORDER if a in ac}
    bos = {a: sums[a]["items"] for a in sums}
    comparators = [b for b in P.COMPARATORS if b in sums]
    idx = boot_index(len(items))
    con = contrasts(bos, items, comparators, idx)
    r = rule(sums, con, comparators) if comparators else None
    lab = labels(sums, con, s_star, comparators, idx, items, bos, ctc_j(ac["S*"]), dev_futility)
    comp = primary_comparator(sums, comparators)
    return {"items": items, "arms": sums, "contrasts": con, "rule": r, "labels": lab, "primary_comparator": comp,
            "demo": demo(bos, items, comp), "reel": reel(bos, items)}
