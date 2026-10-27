"""A0 check logic on synthetic runs and on a short excerpt of a real repaint (no model)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "e5_motif"))
import a0_checks as A  # noqa: E402
import v2_config as C  # noqa: E402


def noise(seed, T):  # ACE-Step's prepare_noise on CPU
    return torch.randn(1, T, 64, generator=torch.Generator().manual_seed(seed)).numpy()


def world():
    runs, arrays, wavs = {}, {}, {}
    for name, mid, cond, L, ins in A.RUNS + [A.FRESH]:
        dit, rng = C.run_seeds(C.TAGS["a0"], mid, 0)
        n = 864000 if L == "t18" else {"mel00": 980800, "mel01": 1016768}[mid]
        T = n // 1920
        runs[name] = {"instruction": C.INSTRUCTIONS[ins], "planned_instruction": C.INSTRUCTIONS[ins],
                      "instruction_logged": C.INSTRUCTIONS[ins], "dcw_enabled": False, "dcw_active_lines": 0,
                      "frames": T, "decode_frames": T, "source_samples": n, "length": L,
                      "duration_meta": int(n / 48000), "repaint_cache_used": 0, "decode_chunk": 256,
                      "cpu_decode_lines": 0, "oom_lines": 0, "peak_before_norm": 0.2, "determinism_warnings": []}
        arrays[name] = {"xt0": noise(dit, T), "pred": np.full((1, T, 64), 0.5, np.float32),
                        "target": np.zeros((1, T, 64), np.float32)}
        wavs[name] = np.zeros((100, 2), np.float32)
    vae = {"raw_same_pin_equal": True, "raw_other_pin_equal": False, "mean_repeat_equal": True,
           "sample_same_pin_equal": True, "sample_other_pin_equal": False}
    facts = {"vae_encode_chunk_samples": 1440000, "attn_implementation": "sdpa", "dit_dtype": "torch.bfloat16",
             "vae_dtype": "torch.bfloat16", "cudnn_deterministic": True, "cudnn_benchmark": False,
             "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"}
    return runs, arrays, vae, facts, wavs


def test_all_pass():
    h = A.hard_checks(*world())
    assert all(v is True for v in h.values()), h


def test_each_hard_failure_is_caught():
    cases = {
        "instruction": lambda r, a, v, f, w: r["b"].update(instruction_logged=C.T2M),
        "dcw_off": lambda r, a, v, f, w: r["c"].update(dcw_active_lines=1),
        "frames": lambda r, a, v, f, w: r["d"].update(frames=509),
        "duration_meta": lambda r, a, v, f, w: r["e"].update(duration_meta=18),
        "noise_paired_across_conditions": lambda r, a, v, f, w: a["b"].update(xt0=a["c"]["xt0"]),
        "noise_same_on_repeat": lambda r, a, v, f, w: a["a_fresh"].update(xt0=a["a"]["xt0"] + 1e-3),
        "window_noise_independent_across_items": lambda r, a, v, f, w: a["c"].update(xt0=a["a"]["xt0"].copy()),
        "vae_pin_reaches_posterior": lambda r, a, v, f, w: v.update(sample_same_pin_equal=False),
        "decode_chunk_256": lambda r, a, v, f, w: r["a"].update(decode_chunk=384),
        "no_cpu_or_oom_decode": lambda r, a, v, f, w: r["a"].update(cpu_decode_lines=1),
        "sdpa_bf16": lambda r, a, v, f, w: f.update(attn_implementation="eager"),
        "determinism_flags": lambda r, a, v, f, w: f.update(cublas_workspace_config=None),
        "finite_and_audible": lambda r, a, v, f, w: a["e"]["pred"].__setitem__((0, 0, 0), np.nan),
        "repeats_within_1e-3": lambda r, a, v, f, w: w["a_repeat"].__setitem__((3, 0), 2e-3),
        "encode_single_shot": lambda r, a, v, f, w: f.update(vae_encode_chunk_samples=720000),
    }
    for key, mutate in cases.items():
        w = copy.deepcopy(world())
        mutate(*w)
        h = A.hard_checks(*w)
        assert h[key] is False, key


def test_kernel_noise_below_the_repeat_tolerance_is_soft():
    runs, arrays, vae, facts, wavs = world()
    vae.update(mean_repeat_equal=False, sample_same_pin_equal=False)  # encoder kernels nondeterministic
    arrays["a_repeat"]["pred"] = arrays["a"]["pred"] + 1e-3
    wavs["a_fresh"][3, 0] = 2 / 32768
    h = A.hard_checks(runs, arrays, vae, facts, wavs)
    assert all(v is True for v in h.values()), h
    s = A.soft_checks(arrays, wavs, runs)
    assert not s["a_repeat"]["pred_latents"]["equal"] and abs(s["a_repeat"]["pred_latents"]["max_abs"] - 1e-3) < 1e-6
    assert round(s["a_fresh"]["wav"]["max_abs_lsb16"]) == 2
    assert s["noise_prefix_identical_450_vs_510"]                        # CPU generator: prefix-identical


def test_tiny_runs_leave_checks_not_run():
    runs, arrays, vae, facts, wavs = world()
    keep = {n for n, *_ in A.TINY_RUNS}
    sub = lambda d: {k: v for k, v in d.items() if k in keep}
    h = A.hard_checks(sub(runs), sub(arrays), vae, facts, sub(wavs), cpu=True)
    assert h["window_noise_independent_across_items"] is None and h["repeats_within_1e-3"] is None
    assert h["sdpa_bf16"] is None and h["encode_single_shot"] is None and h["no_cpu_or_oom_decode"] is None
    assert h["noise_paired_across_conditions"] is True and h["instruction"] is True


def test_context_error_on_a_real_repaint_excerpt():
    """Excerpts (11.8-11.9 s and 16.1-16.2 s) of a real CPU repaint of mel00 lit (peak before normalisation
    0.2199) and of its source: outside the window the output is the source times the normalisation gain."""
    ex = np.load(ROOT / "tests" / "data" / "pilot_splice_excerpt.npy").astype(np.float32) / 32768
    sr, n = 48000, 980800
    out, src = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
    for r, (a, b) in enumerate(((int(11.8 * sr), int(11.9 * sr)), (int(16.1 * sr), int(16.2 * sr)))):
        out[a:b], src[a:b] = ex[0, r], ex[1, r]
    assert A.context_error(out, src, 0.2199) < 2e-4
    assert A.context_error(out, src * 1.01, 0.2199) > 1e-3
    fake = (np.random.default_rng(0).standard_normal(src.shape) * 0.1).astype(np.float32)   # a regenerated context
    assert A.context_error(fake, src, 0.2) > 0.1
