"""Power of the C1 rule (addendum 4, section power), simulated from B2's per-item outcomes of S* and S*_b0
(results/gpu_run3/b2_result.json: each test item's kept-draw joint J_i for both arms).

  python power_c1.py [--sims 2000] [--out power_c1.json]

Each simulated Run C draws 96 items with replacement from B2's 24 (item, S*, S*_b0) outcomes and applies the
registered rule: R1 holds if the 2.5th percentile of Delta_J = J(S*) - J(S*_b0) over the registered bootstrap
(numpy default_rng(0).integers(0, 96, (10000, 96)), the same index the decision uses) is above 0; R2 holds if
J(S*) >= 0.50. Values are rounded to 1e-6 before comparison. Cells:
  observed       B2's outcomes as they are (8 items won by S*, 2 lost, 14 tied; Delta_J 0.25)
  halved         the effect halved with J(S*) as observed: a drawn win becomes a tie with S*_b0 joint too, with
                 probability 3/8, so the expected Delta_J is 0.125
  halved_s_down  the effect halved by lowering S*: a drawn win becomes a tie with S* not joint, with probability
                 3/8 (expected J(S*) 0.458)
  null           no effect: the two arms of every drawn item swap with probability 1/2 (R1's false-positive rate)
It reports P(R1), P(R2), P(R1 and R2) and P(Delta_J >= 0.20 and R1), B2's size bar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "power_c1.json"
B2_RESULT = ROOT / "results" / "gpu_run3" / "b2_result.json"
N_ITEMS, N_BOOT, BOOT_SEED, SIM_SEED = 96, 10000, 0, 0
J_MIN, SIZE_BAR = 0.50, 0.20
HALF = 3 / 8          # (8 wins x (1 - 3/8) - 2 losses) / 24 = 0.125, half of B2's 0.25
CELLS = ("observed", "halved", "halved_s_down", "null")


def rnd(x) -> float:
    return round(float(x), 6)


def b2_pairs(path: Path = B2_RESULT) -> np.ndarray:
    """(24, 2) array of B2's per-item kept-draw J for S* and S*_b0, in item order."""
    arms = json.loads(Path(path).read_text())["results"]["arms"]
    s, b = arms["S*"]["items"], arms["S*_b0"]["items"]
    return np.array([[float(s[i]["j"]), float(b[i]["j"])] for i in sorted(s)])


def draw(rng: np.random.Generator, pairs: np.ndarray, cell: str, n: int = N_ITEMS) -> np.ndarray:
    """One simulated set of n items: (n, 2) J for S* and S*_b0."""
    x = pairs[rng.integers(0, len(pairs), n)].copy()
    win = (x[:, 0] == 1) & (x[:, 1] == 0)
    if cell == "halved":
        x[win & (rng.random(n) < HALF), 1] = 1.0
    elif cell == "halved_s_down":
        x[win & (rng.random(n) < HALF), 0] = 0.0
    elif cell == "null":
        swap = rng.random(n) < 0.5
        x[swap] = x[swap][:, ::-1]
    elif cell != "observed":
        raise ValueError(cell)
    return x


def decide(x: np.ndarray, idx: np.ndarray) -> dict:
    """The registered rule on one simulated set (no P0 problem, no missing run)."""
    js, jb = x[:, 0], x[:, 1]
    d = js[idx].mean(1) - jb[idx].mean(1)
    p25 = rnd(np.percentile(d, 2.5))
    dj = rnd(js.mean() - jb.mean())
    r1, r2 = p25 > 0, rnd(js.mean()) >= J_MIN
    return {"r1": r1, "r2": r2, "both": r1 and r2, "size_and_r1": r1 and dj >= SIZE_BAR, "dj": dj, "js": js.mean(),
            "p25": p25}


def simulate(sims: int, cells=CELLS, seed: int = SIM_SEED, pairs: np.ndarray | None = None) -> dict:
    pairs = b2_pairs() if pairs is None else pairs
    idx = np.random.default_rng(BOOT_SEED).integers(0, N_ITEMS, (N_BOOT, N_ITEMS))
    out = {}
    for cell in cells:
        rng = np.random.default_rng([seed, CELLS.index(cell)])
        res = [decide(draw(rng, pairs, cell), idx) for _ in range(sims)]
        m = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        out[cell] = {"P_R1": round(m["r1"], 3), "P_R2": round(m["r2"], 3), "P_R1_and_R2": round(m["both"], 3),
                     "P_size_bar_and_R1": round(m["size_and_r1"], 3), "mean_delta_j": round(m["dj"], 4),
                     "mean_J_s_star": round(m["js"], 4), "median_p2_5": round(float(np.median([r["p25"] for r in res])), 4),
                     "mc_se_max": round(float(0.5 / np.sqrt(sims)), 3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    pairs = b2_pairs()
    cells = simulate(a.sims, pairs=pairs)
    wins = int(((pairs[:, 0] == 1) & (pairs[:, 1] == 0)).sum())
    losses = int(((pairs[:, 0] == 0) & (pairs[:, 1] == 1)).sum())
    doc = {"experiment": "c1_power", "sims_per_cell": a.sims, "sim_seed": SIM_SEED, "n_items": N_ITEMS, "n_boot": N_BOOT,
           "boot_seed": BOOT_SEED, "j_min": J_MIN, "size_bar": SIZE_BAR, "half_probability": HALF,
           "source": "results/gpu_run3/b2_result.json (results.arms S* and S*_b0, items: j)",
           "source_sha256": hashlib.sha256(B2_RESULT.read_bytes()).hexdigest(),
           "b2": {"n_items": len(pairs), "J_s_star": rnd(pairs[:, 0].mean()), "J_s_b0": rnd(pairs[:, 1].mean()),
                  "wins": wins, "losses": losses, "ties": len(pairs) - wins - losses},
           "cells": cells}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{a.sims} simulations per cell in {time.time() - t0:.0f} s", flush=True)
    for name, c in cells.items():
        print(f"{name:>14s}: P(R1) {c['P_R1']:.3f}  P(R2) {c['P_R2']:.3f}  P(R1 and R2) {c['P_R1_and_R2']:.3f}  "
              f"P(Delta_J >= 0.20 and R1) {c['P_size_bar_and_R1']:.3f}  mean Delta_J {c['mean_delta_j']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
