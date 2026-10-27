"""Smoke test for the ace_lab trajectory hooks. Run once in each new environment (about 5 min on CPU).

Checks, on short clips:
  1. record=True gives one x0_hat per denoiser call (8 for turbo)
  2. an identity edit (return a copy of x0) gives the same output as no hook (<1% max deviation)
  3. capture + swap_conditioning (lyrics swapped mid-trajectory) runs and changes the output
  4. a combine() that runs the same conditioning again gives the base output
Writes results/smoke_test_<device>.json and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from dacapo import ace as ace_lab

HERE = Path(__file__).parent
CAP = "a cappella, solo female vocal, no instruments, dry studio recording"
L1 = "[Verse]\nSun on the hill\nWind in the trees"
L2 = "[Verse]\nRain on the road\nStars in the sea"


def wav(p: Path) -> np.ndarray:
    return sf.read(str(p))[0]


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).max() / (np.abs(a[:n]).max() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()
    out = HERE / "outputs_smoke"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    h = ace_lab.load(args.device)
    res, ok = {"device": args.device, "load_seconds": round(time.time() - t0, 1)}, True
    kw = dict(caption=CAP, duration=args.duration, seed=5, lyrics=L1, bpm=100)

    base = ace_lab.generate(h, out / "base.wav", record=True, **kw)
    res["base_seconds"] = base["seconds"]
    res["n_steps_recorded"] = len(base["trace"])
    res["t_schedule"] = [round(s["t"], 4) for s in base["trace"]]
    ok &= len(base["trace"]) == 8

    # clone() forces the recompute path v' = (x_t - x0')/t, which must not change the numbers
    ace_lab.generate(h, out / "identity.wav", edit=lambda k, t, x0, xt: x0.clone(), **kw)
    res["identity_rel_maxdiff"] = rel_diff(wav(out / "base.wav"), wav(out / "identity.wav"))
    ok &= res["identity_rel_maxdiff"] < 0.01

    cap2 = ace_lab.generate(h, out / "lyrics2.wav", capture=True, **dict(kw, lyrics=L2))
    cond = {k: v for k, v in cap2["captured"].items() if k in ("encoder_hidden_states", "encoder_attention_mask")}
    sw = ace_lab.generate(h, out / "swap_lyrics_at3.wav", pre=ace_lab.swap_conditioning(3, cond), **kw)
    res["swap_vs_base_rel_maxdiff"] = rel_diff(wav(out / "base.wav"), wav(out / "swap_lyrics_at3.wav"))
    res["swap_seconds"] = sw["seconds"]
    ok &= res["swap_vs_base_rel_maxdiff"] > 0.05  # swapping lyrics must change something

    comb = ace_lab.generate(h, out / "combine_same.wav", combine=lambda k, t, kw_, call: call(kw_), **kw)
    res["combine_rel_maxdiff"] = rel_diff(wav(out / "base.wav"), wav(out / "combine_same.wav"))
    res["combine_denoiser_calls"] = comb["n_denoiser_calls"]
    ok &= res["combine_rel_maxdiff"] < 0.05 and comb["n_denoiser_calls"] == 16

    res["passed"] = bool(ok)
    res["total_seconds"] = round(time.time() - t0, 1)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"smoke_test_{args.device}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
