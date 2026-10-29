"""Power of the B2 rule (addendum 3, section power), simulated from E4's clips (results/gpu_run1/e4_sweep.json,
scored on the mix, so q = min(NPA, WR) without a vocal gate).

  python power_b2.py [--sims 2000] [--out power_b2.json]

Unclocked arms resample E4's ctx_oohs_k3 and ctx_oohs_k4 clips (24 each). Under the alternative each S* clip
starts as a resampled ctx_oohs_k4 clip and becomes worded with an item-varying probability
p_i = expit(logit(p) + N(0, 1)); a worded clip takes its WR from a resampled noguide clip and loses 0.10 of
NPA. 24 items, 4 draws, comparators k3 and k4. P1 and P2 as registered: Delta_J >= 0.20 with the 2.5th
percentile of a 10,000-resample item bootstrap (max retaken) above 0, and J(S*) >= 0.50. For comparison,
the mean-q rule of the first design: mean bo4 q(S*) - max mean bo4 q(b) >= 0.10 with a bootstrap lower bound > 0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "power_b2.json"
PS = (0.0, 0.1, 0.2, 0.3, 0.4)
N_ITEMS, N_DRAWS, NPA_DROP, LOGIT_SD = 24, 4, 0.10, 1.0
N_BOOT, SEED = 10000, 0
DELTA_J, J_MIN, DELTA_Q = 0.20, 0.50, 0.10


def word_recall(words: list[str], hyp: str) -> float:
    """E4's WR (figures/make_figures.py word_recall)."""
    have, want = Counter(re.findall(r"[a-z']+", hyp.lower())), Counter(w.lower() for w in words)
    return sum(min(c, have[w]) for w, c in want.items()) / sum(want.values())


def e4_clips() -> dict[str, np.ndarray]:
    rows = json.loads((ROOT / "results" / "gpu_run1" / "e4_sweep.json").read_text())["results"]["rows"]
    words = {it["id"]: it["words"] for it in json.loads((ROOT / "experiments" / "e4_sing" / "manifest.json").read_text())["items"]}

    def arm(a):
        return np.array([(r["npa"], word_recall(words[r["id"]], r["hyp"])) for r in rows if r["arm"] == a])
    return {"k3": arm("ctx_oohs_k3"), "k4": arm("ctx_oohs_k4"), "noguide": arm("noguide")}


def q(x: np.ndarray) -> np.ndarray:
    return np.minimum(x[..., 0], x[..., 1])


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def one(rng: np.random.Generator, clips: dict, p: float) -> dict:
    k3, k4, ng = clips["k3"], clips["k4"], clips["noguide"]
    pi = expit(np.log(p / (1 - p)) + LOGIT_SD * rng.standard_normal(N_ITEMS)) if p > 0 else np.zeros(N_ITEMS)
    b3 = q(k3[rng.integers(0, len(k3), (N_ITEMS, N_DRAWS))]).max(1)
    b4 = q(k4[rng.integers(0, len(k4), (N_ITEMS, N_DRAWS))]).max(1)
    alt = k4[rng.integers(0, len(k4), (N_ITEMS, N_DRAWS))].copy()
    worded = rng.random((N_ITEMS, N_DRAWS)) < pi[:, None]
    alt[..., 1] = np.where(worded, ng[rng.integers(0, len(ng), (N_ITEMS, N_DRAWS)), 1], alt[..., 1])
    alt[..., 0] = np.where(worded, np.clip(alt[..., 0] - NPA_DROP, 0, 1), alt[..., 0])
    s = q(alt).max(1)
    js, j3, j4 = (s >= 0.5).astype(float), (b3 >= 0.5).astype(float), (b4 >= 0.5).astype(float)
    idx = rng.integers(0, N_ITEMS, (N_BOOT, N_ITEMS))
    dj = round(js.mean() - max(j3.mean(), j4.mean()), 6)
    p1 = False
    if dj >= DELTA_J:
        bd = js[idx].mean(1) - np.maximum(j3[idx].mean(1), j4[idx].mean(1))
        p1 = round(float(np.percentile(bd, 2.5)), 6) > 0
    dq = round(s.mean() - max(b3.mean(), b4.mean()), 6)
    pq = False
    if dq >= DELTA_Q:
        bq = s[idx].mean(1) - np.maximum(b3[idx].mean(1), b4[idx].mean(1))
        pq = round(float(np.percentile(bq, 2.5)), 6) > 0
    joint = float(((alt[..., 0] >= 0.5) & (alt[..., 1] >= 0.5)).mean())
    return {"p1": p1, "p2": round(js.mean(), 6) >= J_MIN, "mean_q_rule": pq, "joint_per_clip": joint,
            "j_s": js.mean(), "j_best_b": max(j3.mean(), j4.mean())}


def simulate(sims: int, ps=PS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    clips = e4_clips()
    out = {}
    for p in ps:
        res = [one(rng, clips, p) for _ in range(sims)]
        m = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        both = float(np.mean([r["p1"] and r["p2"] for r in res]))
        out[f"{p:.1f}"] = {"p_worded": p, "joint_per_clip": round(m["joint_per_clip"], 3), "P1": round(m["p1"], 3),
                           "P2": round(m["p2"], 3), "P1_and_P2": round(both, 3), "mean_q_rule": round(m["mean_q_rule"], 3),
                           "mean_J_s_star": round(m["j_s"], 3), "mean_J_best_comparator": round(m["j_best_b"], 3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    cells = simulate(a.sims)
    print(f"{a.sims} simulations per cell in {time.time() - t0:.0f} s", flush=True)
    doc = {"experiment": "b2_power", "sims_per_cell": a.sims, "seed": SEED, "n_items": N_ITEMS, "n_draws": N_DRAWS,
           "npa_drop": NPA_DROP, "logit_sd": LOGIT_SD, "n_boot": N_BOOT, "delta_j": DELTA_J, "j_min": J_MIN,
           "delta_q_first_design": DELTA_Q, "source": "results/gpu_run1/e4_sweep.json (ctx_oohs_k3, ctx_oohs_k4, noguide)",
           "cells": cells}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    for c in cells.values():
        print(f"p {c['p_worded']:.1f}: joint/clip {c['joint_per_clip']:.2f}  P1 {c['P1']:.2f}  P2 {c['P2']:.2f}  "
              f"both {c['P1_and_P2']:.2f}  mean-q rule {c['mean_q_rule']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
