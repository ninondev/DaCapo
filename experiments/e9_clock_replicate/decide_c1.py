"""The C1 decision (addendum 4): integrity (P0), the clock effect (R1), the capability (R2), the verdict, the labels
and the tables. Pure functions of plain rows; run_c1.py decide gathers them. B2's decision code
(experiments/e8_clock_sing/decide_b2.py) is imported and never edited: the per-arm summary, the selector's kept
draws, best-of-n, the bootstrap index, the paired contrasts and B2's labels come from it.

  per item and arm, bo4 = the q of the draw the selector keeps (largest q, then WR, then NPA_v, then the lower
  draw); J_i = 1 if bo4 >= 0.5; J(arm) = mean of J_i over the 96 items
  R1  the 2.5th percentile of the bootstrap Delta_J = J(S*) - J(S*_b0) is above 0
  R2  J(S*) >= 0.50 (at least 48 of the 96 melodies)
Bootstrap: clusters are items, 10,000 resamples, numpy default_rng(0).integers(0, 96, (10000, 96)), the draws of
an item kept together, all arms resampled together, numpy's linear percentiles. Values are rounded to 1e-6
before every comparison. A planned run that is missing, failed after 3 attempts or unscored counts as q 0, gate
failed, WR 0, NPA 0 and CTC WR 0.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "experiments" / "e8_clock_sing"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import decide_b2 as D  # noqa: E402
import plan_b2 as P2  # noqa: E402
import plan_c1 as P  # noqa: E402
import score_c1 as S  # noqa: E402

J_MIN = 0.50
SIZE_BAR = D.DELTA_J_MIN           # B2's 0.20, a label in Run C
N_BOOT, BOOT_SEED = D.N_BOOT, D.BOOT_SEED
MAX_MISSING_SHARE = D.MAX_MISSING_SHARE
STRICT_SHARE = 0.5                 # strict capability: bo4 >= 0.75 on at least half of the items (48 of 96)
VERDICTS = {"replicated": "Replicated: the clock helps and the capability holds",
            "clock_only": "Clock helps, capability not reached",
            "capability_only": "Capability holds, clock effect not shown",
            "not_replicated": "Not replicated", "technical": "Technical", "test": "test only"}
LISTENERS = "filled later from the Run C blind pack (exploratory)"

rnd = D.rnd


# per arm

def arm_clips(runs: list[dict], clips: dict[str, dict]) -> dict[str, dict[str, list[dict]]]:
    """{arm label: {item: [clip values with draw and run_id]}}; a planned run without a value counts as missing."""
    return D.arm_clips(runs, {r["run_id"]: clips.get(r["run_id"], dict(S.MISSING)) for r in runs})


def ctc_j(arm: dict[str, list[dict]], field: str) -> float:
    """J recomputed with a CTC WR on the draws the selector kept (by Whisper's q)."""
    return rnd(np.mean([S.kept(d)[field] >= S.JOINT for d in arm.values()]))


def summary(arm: dict[str, list[dict]]) -> dict:
    """decide_b2.summary (section analysis of addendum 3) and the CTC values on the vocal stem."""
    out = D.summary(arm)
    clips = [c for v in arm.values() for c in v]
    out.update(wr_ctc_stem=rnd(np.mean([c.get("wr_ctc_stem", 0.0) for c in clips])),
               q_ctc=rnd(np.mean([c.get("q_ctc", 0.0) for c in clips])),
               q_ctc_stem=rnd(np.mean([c.get("q_ctc_stem", 0.0) for c in clips])),
               J_ctc_mix=ctc_j(arm, "q_ctc"), J_ctc_stem=ctc_j(arm, "q_ctc_stem"))
    return out


# the rule

def rule(sums: dict, con: dict) -> dict:
    jd = con["delta_j"]
    r1 = rnd(jd["p2_5"]) > 0
    r2 = rnd(sums["S*"]["J"]) >= J_MIN
    return {"R1": {"held": r1, "delta_j": jd["value"], "p2_5": jd["p2_5"], "ci95": jd["ci95"], "needed": "p2_5 > 0",
                   "J_s_star": sums["S*"]["J"], "J_s_b0": sums[P.COMPARATOR]["J"]},
            "R2": {"held": r2, "J": sums["S*"]["J"], "n_items_joint": sums["S*"]["n_items_joint"],
                   "n_items": sums["S*"]["n_items"], "needed": J_MIN}}


def verdict(p0_ok: bool, r: dict | None, flags: list[str]) -> tuple[str, list[str]]:
    """(verdict, failed conditions)."""
    if flags:
        return VERDICTS["test"], [f"test run ({', '.join(flags)})"]
    if not p0_ok or r is None:
        return VERDICTS["technical"], ["P0 failed"]
    r1, r2 = r["R1"]["held"], r["R2"]["held"]
    if r1 and r2:
        return VERDICTS["replicated"], []
    if r1:
        return VERDICTS["clock_only"], ["R2 failed"]
    if r2:
        return VERDICTS["capability_only"], ["R1 failed"]
    return VERDICTS["not_replicated"], ["R1 failed", "R2 failed"]


def labels(sums: dict, con: dict, idx: np.ndarray, items: list[str], bos: dict) -> dict:
    """B2's labels (decide_b2.labels with S*_b0 as the only comparator) with Run C's strict threshold, B2's size
    bar, the independent ASR on the mix and on the vocal stem, and the listeners' label left for later. A label
    never changes the verdict; a label whose arm is absent is None."""
    comps = [P.COMPARATOR] if P.COMPARATOR in sums else []
    out = D.labels(sums, con, P2.parse(P.ARMS[0][1]), comps, idx, items, bos, None, None)
    out.pop("dev_futility", None)
    s = sums["S*"]
    need = math.ceil(STRICT_SHARE * s["n_items"])
    out["strict_capability"] = s["n_items_strict"] >= need
    out["strict_capability_detail"] = {"n_items_strict": s["n_items_strict"], "needed": need}
    jd = con.get("delta_j")
    out["b2_size_bar"] = None if jd is None else {"held": rnd(jd["value"]) >= SIZE_BAR, "delta_j": jd["value"],
                                                   "needed": SIZE_BAR, "with_r1": None}
    out["independent_asr"] = {"J_ctc_mix": s["J_ctc_mix"], "R2_with_ctc_mix": rnd(s["J_ctc_mix"]) >= J_MIN,
                              "J_ctc_stem": s["J_ctc_stem"], "R2_with_ctc_stem": rnd(s["J_ctc_stem"]) >= J_MIN}
    out["listeners"] = None
    out["listeners_note"] = LISTENERS
    return out


# pooled with B2 (exploratory)

def b2_item_outcomes(b2_result: dict) -> dict[str, dict[str, dict]]:
    """B2's per-item kept outcomes of S* and S*_b0 from its b2_result.json: {arm: {item: {"bo4", "j"}}}."""
    arms = b2_result["results"]["arms"]
    return {a: {i: {"bo4": v["bo4"], "j": bool(v["j"])} for i, v in arms[a]["items"].items()} for a in ("S*", "S*_b0")}


def pooled(c1_bos: dict, b2: dict, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Delta_J and the mean bo4 q difference of S* against S*_b0 over B2's test items and Run C's, with a cluster
    bootstrap over all of them (default_rng(0), labelled exploratory)."""
    rows = [(b2["S*"][i], b2["S*_b0"][i]) for i in sorted(b2["S*"])]
    rows += [(c1_bos["S*"][i], c1_bos["S*_b0"][i]) for i in sorted(c1_bos["S*"])]
    dj = np.array([float(a["j"]) - float(b["j"]) for a, b in rows])
    dq = np.array([a["bo4"] - b["bo4"] for a, b in rows])
    idx = D.boot_index(len(rows), n_boot, seed)
    bj, bq = dj[idx].mean(1), dq[idx].mean(1)
    return {"label": "exploratory", "n_items": len(rows), "n_b2": len(b2["S*"]), "n_c1": len(c1_bos["S*"]),
            "delta_j": {"value": rnd(dj.mean()), "ci95": D.ci(bj), "p2_5": rnd(np.percentile(bj, 2.5))},
            "mean_bo4_q": {"value": rnd(dq.mean()), "ci95": D.ci(bq)},
            "bootstrap": f"clusters are items, {n_boot} resamples, numpy default_rng({seed}).integers(0, {len(rows)}, "
                         f"({n_boot}, {len(rows)}))"}


# integrity (P0)

PLAN_FIELDS = ("item", "draw", "config", "guide", "k", "clock", "head_set", "heads", "beta", "shuffle", "hooked",
               "task", "cover_strength", "dit_seed", "rng_seed")


def integrity(st: dict) -> dict[str, dict]:
    """P0 as named checks, each {"held": bool, "detail": ...}. st holds plain facts gathered by run_c1:
      registration       problems with the files now (addendum 4, and addendum 2, the B1 code addendum and addendum 3
                         unchanged)
      checks_failed      registration checks (each addendum's own --check) that failed at the last setup or at decision
      registered_code    {path: sha256} that every generation row must have recorded
      registered_guides  {item: {"oohs": sha256}} of addendum 4, which every row's guide must match
      setup_hashes_ok    setup rendered the guides to their sha256 with the registered soundfont, found the seed
                         table as registered and the installed packages at the notebook's pins
      a0, a0_sha256      the A0 report (or None) and the sha256 of the report file now
      plan, gen (run_id -> last ledger row), scores (run_id -> usable score row)"""
    plan, gens, scores = st["plan"], st["gen"], st["scores"]
    ok_rows = [gens[r["run_id"]] for r in plan if gens.get(r["run_id"], {}).get("status") == "ok"]
    out = {}
    detail = list(st["registration"]) + list(st.get("checks_failed", []))
    bad = [g["run_id"] for g in ok_rows if g.get("code_sha256") != st["registered_code"]]
    detail += [f"{len(bad)} runs generated with other code: {bad[:5]}"] if bad else []
    guides = st["registered_guides"]
    gbad = [g["run_id"] for g in ok_rows if g.get("guide_sha256") != guides.get(g["item"], {}).get(g.get("guide"))]
    detail += [f"{len(gbad)} runs used a guide other than the registered one: {gbad[:5]}"] if gbad else []
    detail += [] if st["setup_hashes_ok"] else ["setup did not reproduce the registered soundfont, guides, seed table and "
                                                "package pins"]
    out["hashes"] = {"held": not detail, "detail": detail}
    a0 = st["a0"]
    a0_bad = [g["run_id"] for g in ok_rows if g.get("a0_sha256") != st["a0_sha256"] or st["a0_sha256"] is None]
    a0_code = bool(a0) and a0.get("code_sha256") == st["registered_code"]
    out["a0"] = {"held": bool(a0 and a0.get("recorded") and a0.get("pass")) and a0_code and not a0_bad,
                 "detail": {"recorded": bool(a0 and a0.get("recorded")), "passed": bool(a0 and a0.get("pass")),
                            "ran_on_the_registered_code": a0_code, "rows_following_another_a0": a0_bad[:5]}}
    by_id = {r["run_id"]: r for r in plan}
    off_plan = [g["run_id"] for g in ok_rows if any(g.get(k) != by_id[g["run_id"]][k] for k in PLAN_FIELDS)]
    out["plan"] = {"held": not off_plan, "detail": off_plan[:10]}

    def hook_ok(g: dict) -> bool:
        if not g.get("hooked"):
            return g.get("hook_calls") == 0
        return g.get("hook_calls") == P.HOOK_CALLS and g.get("max_key") is not None and g["max_key"] < g["n_lyr"]
    hook_bad = [g["run_id"] for g in ok_rows if not hook_ok(g)]
    out["hooks"] = {"held": not hook_bad, "detail": hook_bad[:10]}
    lat: dict = {}
    for g in ok_rows:
        lat.setdefault((g["item"], g["draw"]), set()).add(g.get("guide_latent_sha256"))
    lat_bad = [list(k) for k, v in lat.items() if len(v) != 1 or None in v]
    out["guide_latents"] = {"held": not lat_bad, "detail": lat_bad[:10]}
    ok_ids = {g["run_id"] for g in ok_rows}
    scored = [scores[i] for i in ok_ids if i in scores and "error" not in scores[i]]
    wrong_ref = sum(not S.CS.registered_reference(s) for s in scored)
    no_peak = [g["run_id"] for g in ok_rows if g.get("peak_before_norm") is None]
    out["reference_and_peak"] = {"held": not wrong_ref and not no_peak,
                                 "detail": {"other_reference": wrong_ref, "no_peak": no_peak[:5]}}
    incomplete = len(plan) - len(scored)
    limit = math.floor(MAX_MISSING_SHARE * len(plan))
    out["complete"] = {"held": incomplete <= limit,
                       "detail": {"planned": len(plan), "missing_failed_or_unscored": incomplete, "limit": limit}}
    return out


# everything decide reports

def analyse(runs: list[dict], clips: dict[str, dict], b2: dict | None = None) -> dict:
    """The rule, the labels and the tables, from the planned runs and their per-clip values."""
    ac = arm_clips(runs, clips)
    items = sorted({r["item"] for r in runs})
    sums = {a: summary(ac[a]) for a in P.ARM_ORDER if a in ac}
    bos = {a: sums[a]["items"] for a in sums}
    idx = D.boot_index(len(items))
    comps = [P.COMPARATOR] if P.COMPARATOR in sums else []
    con = D.contrasts(bos, items, comps, idx)
    r = rule(sums, con) if comps and "S*" in sums else None
    lab = labels(sums, con, idx, items, bos) if "S*" in sums else {}
    if r is not None and lab.get("b2_size_bar"):
        lab["b2_size_bar"]["with_r1"] = bool(lab["b2_size_bar"]["held"] and r["R1"]["held"])
    pool = pooled(bos, b2) if b2 is not None and comps and "S*" in sums else None
    demo = D.demo(bos, items, P.COMPARATOR if comps else None) if "S*" in sums else []
    return {"items": items, "arms": sums, "contrasts": con, "rule": r, "labels": lab, "pooled_with_b2": pool,
            "demo": demo, "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "n_items": len(items),
                                        "index_sha256": hashlib.sha256(idx.tobytes()).hexdigest()}}
