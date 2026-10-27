"""A0: generation-correctness checks before any E5 v2 Phase 0b or confirmatory run (ACE-Step environment).
Pilot melodies mel00 and mel01 only, tag e5v2-a0 (e5v2-tiny-a0 at the tiny scale); no analysis reads these outputs.

  python a0_checks.py --part main    --device cuda   7 repaints and VAE-only pin checks in one process
  python a0_checks.py --part fresh   --device cuda   run "a" again in a new process
  python a0_checks.py --part verdict                 exit 0 (go) or 3 (technical stop, not a strike)

Hard checks: instruction and DCW as intended (resolved by ACE-Step and as logged in the DiT prompt),
latent frames and duration metadata for both source lengths, identical initial noise across the
conditions of an item and on every repeat, window noise uncorrelated across items, the global-RNG pin
reaching the VAE posterior sample, the source kept outside 11.9-16.1 s, the repeats within 1e-3 of each
other, decode chunk 256 with no CPU fallback, single-shot encode, sdpa attention with bf16 DiT and VAE,
determinism flags, finite and audible output, the stored format (16-bit FLAC, 48 kHz, stereo), an NVIDIA
L4, and the pinned ACE-Step commit, weights revision and weight files. Recorded only: bitwise equality of the repeats, determinism warnings, and
the CUDA prefix identity of the noise across lengths (the v1 diagnosis).
At the tiny scale (CPU), A0 runs "a" and "b" only; checks that need other runs or a GPU are recorded
as not run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_config as C  # noqa: E402
from dacapo.motif_integrity import AUDIO_EXT  # noqa: E402

RUNS = [  # name, item, condition, length arm, instruction key; "a_repeat" last, "a_fresh" in a new process
    ("a", "mel00", "lit", "t18", "rep"), ("b", "mel00", "ctl", "t18", "rep"), ("c", "mel01", "lit", "t18", "rep"),
    ("d", "mel00", "lit", "rend", "rep"), ("e", "mel01", "lit", "rend", "rep"), ("f", "mel00", "lit", "t18", "t2m"),
    ("a_repeat", "mel00", "lit", "t18", "rep")]
FRESH = ("a_fresh", "mel00", "lit", "t18", "rep")
TINY_RUNS = RUNS[:2]
W0, W1 = C.WINDOW_FRAMES
LSB16 = 1.0 / 32768
REPEAT_TOL = 1e-3


def out_dir() -> Path:
    return C.run_dir() / "outputs_a0"


def _all(vals) -> bool | None:
    """True/False over the runs that exist; None when no run applies."""
    vals = list(vals)
    return None if not vals else all(vals)


def hard_checks(runs: dict, arrays: dict, vae: dict | None, facts: dict, wavs: dict | None = None,
                cpu: bool = False) -> dict:
    """runs[name]: generate() facts + plan; arrays[name]: {"xt0": [1,T,64] step-0 hidden states,
    "pred": pred_latents, "target": source latents}; vae: pin-check results; facts: runtime_facts().
    A value of None means the check could not run (missing runs, or a GPU-only check on CPU)."""
    chk = {}
    chk["instruction"] = all(r["instruction"] == r["planned_instruction"] == r["instruction_logged"] for r in runs.values())
    chk["dcw_off"] = all(r["dcw_enabled"] is False and r["dcw_active_lines"] == 0 for r in runs.values())
    chk["frames"] = all(r["frames"] == C.expected_frames(r["source_samples"]) == r["decode_frames"] for r in runs.values())
    chk["duration_meta"] = all(r["duration_meta"] == C.expected_duration_meta(r["source_samples"]) for r in runs.values())
    chk["t18_is_450_frames"] = _all(r["frames"] == 450 for r in runs.values() if r["length"] == "t18")
    chk["no_repaint_cache"] = all(r["repaint_cache_used"] == 0 for r in runs.values())
    chk["decode_chunk_256"] = all(r["decode_chunk"] == 256 for r in runs.values())
    chk["no_cpu_or_oom_decode"] = None if cpu else all(r["cpu_decode_lines"] == 0 and r["oom_lines"] == 0 for r in runs.values())
    x = {n: a["xt0"] for n, a in arrays.items()}
    chk["noise_paired_across_conditions"] = _all(np.array_equal(x["a"], x[n]) for n in ("b", "f") if n in x)
    chk["noise_same_on_repeat"] = _all(np.array_equal(x["a"], x[n]) for n in ("a_repeat", "a_fresh") if n in x)
    if "c" in x:
        wa, wc = x["a"][0, W0:W1].ravel(), x["c"][0, W0:W1].ravel()
        chk["window_noise_independent_across_items"] = bool(abs(np.corrcoef(wa, wc)[0, 1]) < 0.05)
    else:
        chk["window_noise_independent_across_items"] = None
    chk["vae_pin_reaches_posterior"] = None if vae is None else bool(
        vae["raw_same_pin_equal"] and not vae["raw_other_pin_equal"] and not vae["sample_other_pin_equal"]
        and (vae["sample_same_pin_equal"] or not vae["mean_repeat_equal"]))
    chk["encode_single_shot"] = None if cpu else all(r["source_samples"] <= facts["vae_encode_chunk_samples"] for r in runs.values())
    chk["sdpa_bf16"] = None if cpu else (facts["attn_implementation"] == "sdpa" and facts["dit_dtype"] == "torch.bfloat16"
                                         and facts["vae_dtype"] == "torch.bfloat16")
    chk["finite_and_audible"] = all(np.isfinite(a["pred"]).all() for a in arrays.values()) and \
        all((r["peak_before_norm"] or 0) > 0.01 for r in runs.values())
    chk["determinism_flags"] = bool(facts["cudnn_deterministic"] and not facts["cudnn_benchmark"]
                                    and facts["deterministic_algorithms"]
                                    and facts["cublas_workspace_config"] in (":4096:8", ":16:8"))
    if wavs is not None:
        reps = [diff(wavs["a"], wavs[n]) for n in ("a_repeat", "a_fresh") if n in wavs]
        chk["repeats_within_1e-3"] = _all(d.get("max_abs", np.inf) <= REPEAT_TOL for d in reps)
    return chk


def context_error(out_wav: np.ndarray, src_wav: np.ndarray, peak_before_norm: float, sr: int = 48000) -> float:
    """Max |output - source x normalisation gain| outside 11.9-16.1 s. ACE-Step splices the source waveform
    back outside the repaint window (25 ms crossfade inside 11.975-12.0 and 16.0-16.025 s) and scales the
    whole clip to -1 dBFS peak; on the CPU pilot this was 6e-5 to 1.5e-4, so a regenerated context shows."""
    n = min(len(out_wav), len(src_wav))
    g = 10 ** (-1 / 20) / peak_before_norm
    idx = np.r_[0:int(11.9 * sr), int(16.1 * sr):n]
    return float(np.abs(out_wav[idx] - src_wav[idx] * g).max())


def weights_sha256(ckpt_dir: Path) -> dict:
    out = {}
    for rel in C.WEIGHTS_SHA256:
        h = hashlib.sha256()
        p = ckpt_dir / rel
        if not p.exists():
            out[rel] = None
            continue
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        out[rel] = h.hexdigest()
    return out


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"equal": False, "shape": [list(a.shape), list(b.shape)]}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"equal": bool(np.array_equal(a, b)), "max_abs": float(d.max()), "rms": float(np.sqrt((d ** 2).mean())),
            "frac_differ": float((d > 0).mean())}


def soft_checks(arrays: dict, wavs: dict, runs: dict) -> dict:
    out = {}
    for n in ("a_repeat", "a_fresh"):
        if n in arrays:
            out[n] = {"pred_latents": diff(arrays["a"]["pred"], arrays[n]["pred"]),
                      "source_latents": diff(arrays["a"]["target"], arrays[n]["target"]),
                      "wav": diff(wavs["a"], wavs[n])}
            out[n]["wav"]["max_abs_lsb16"] = out[n]["wav"].get("max_abs", 0.0) / LSB16
    if "d" in arrays:
        a, d = arrays["a"]["xt0"], arrays["d"]["xt0"]
        out["noise_prefix_identical_450_vs_510"] = bool(np.array_equal(a[0], d[0, : a.shape[1]]))
    out["determinism_warnings"] = sorted({w for r in runs.values() for w in r["determinism_warnings"]})
    return out


def _gen(h, ace, man_items, out: Path, name: str, mid: str, cond: str, L: str, ins: str) -> dict:
    it = man_items[mid]
    dit, rng = C.run_seeds(C.SCALES[C.scale()]["tags"]["a0"], mid, 0)
    xt0 = {}

    def pre(step, t, kw):
        if step == 0:
            xt0["x"] = kw["hidden_states"].detach().float().cpu().numpy()
        return kw

    g = dict(C.GEN)
    r = ace.generate(h, out / f"{name}{AUDIO_EXT}", C.CAPTIONS["flute"], g.pop("duration"), dit, steps=g.pop("steps"),
                     shift=g.pop("shift"), src=str(C.e5_dir() / it["sources"][L][cond]),
                     extra={**C.REPAINT, "instruction": C.INSTRUCTIONS[ins]}, rng_seed=rng, pre=pre, keep_extra=True,
                     out_subtype=C.AUDIO_SUBTYPE, **g)
    ex = r.pop("extra")
    np.savez_compressed(out / f"{name}.npz", xt0=xt0["x"], pred=ex["pred_latents"].float().numpy(),
                        target=ex["target_latents"].float().numpy())
    keep = {k: v for k, v in r.items() if k not in ("trace", "captured")}
    keep.update(item=mid, cond=cond, length=L, planned_instruction=C.INSTRUCTIONS[ins],
                source_samples=it["source_samples"][L], dit_seed=dit, rng_seed=rng)
    return keep


def vae_pin_check(h, ace, src: Path, pin: int) -> dict:
    """Pure RNG: a pinned global draw repeats and another pin changes it. Through the VAE: the posterior
    sample under the same pin repeats whenever the posterior mean repeats, and another pin changes it."""
    import torch
    audio = h.process_src_audio(str(src)).unsqueeze(0)
    T = audio.shape[-1] // 1920
    dev, dt = h.vae.device, h.vae.dtype

    def raw(p):
        ace.pin_global_rng(p)
        return torch.randn(1, 64, T, device=dev, dtype=dt).float().cpu()

    def sample(p):
        ace.pin_global_rng(p)
        return h.tiled_encode(audio, offload_latent_to_cpu=True).float().cpu()

    with torch.inference_mode():
        m1 = h.vae.encode(audio.to(dev, dt)).latent_dist.mean.float().cpu()
        m2 = h.vae.encode(audio.to(dev, dt)).latent_dist.mean.float().cpu()
        s1, s2, s3 = sample(pin), sample(pin), sample(pin + 1)
    g1, g2, g3 = raw(pin), raw(pin), raw(pin + 1)
    return {"raw_same_pin_equal": bool(torch.equal(g1, g2)), "raw_other_pin_equal": bool(torch.equal(g1, g3)),
            "mean_repeat_equal": bool(torch.equal(m1, m2)), "sample_same_pin_equal": bool(torch.equal(s1, s2)),
            "sample_other_pin_equal": bool(torch.equal(s1, s3)),
            "sample_other_pin": diff(s1.numpy(), s3.numpy()), "sample_same_pin": diff(s1.numpy(), s2.numpy())}


def environment_checks(device: str) -> dict:
    """The pinned software and hardware: ACE-Step commit, weights revision, GPU type."""
    from run_analogy import ace_head, device_name, hf_revision
    return {"ace_commit": ace_head() == C.ACE_COMMIT, "hf_revision": hf_revision() == C.HF_REVISION,
            "gpu_l4": None if device != "cuda" else C.GPU_REQUIRED in device_name(device)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["main", "fresh", "verdict"], required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    if a.part == "verdict":
        return verdict(out)
    st = C.addendum_status()
    if not st["ok"]:
        print("A0 refused: code differs from addendum 2:", st["changed"], flush=True)
        return 3
    from dacapo import ace
    det = ace.setup_determinism()
    h = ace.load(a.device)
    facts = {**ace.runtime_facts(h), "setup": det}
    man_items = {it["id"]: it for it in json.loads((C.e5_dir() / C.P0B_MANIFEST.name).read_text())["items"]}
    tiny = C.SCALES[C.scale()]["a0_runs"] == "tiny"
    todo = (TINY_RUNS if tiny else RUNS) if a.part == "main" else [FRESH]
    runs = {name: _gen(h, ace, man_items, out, name, mid, cond, L, ins) for name, mid, cond, L, ins in todo}
    res = {"facts": facts, "runs": runs, "device": a.device, "scale": C.scale()}
    if a.part == "main":
        res["environment"] = environment_checks(a.device)
        res["weights_sha256"] = weights_sha256(Path(os.environ["ACESTEP_CHECKPOINTS_DIR"]))
        dit, rng = C.run_seeds(C.SCALES[C.scale()]["tags"]["a0"], "mel00", 0)
        res["vae"] = vae_pin_check(h, ace, C.e5_dir() / man_items["mel00"]["sources"]["t18"]["lit"], rng)
    C.write_json(out / f"a0_{a.part}.json", res)
    return 0


def verdict(out: Path) -> int:
    import soundfile as sf
    main_ = json.loads((out / "a0_main.json").read_text())
    fresh = C.read_json(out / "a0_fresh.json")
    tiny = main_["scale"] != "full"
    cpu = main_["device"] != "cuda"
    runs = {**main_["runs"], **(fresh["runs"] if fresh else {})}
    arrays = {n: dict(np.load(out / f"{n}.npz")) for n in runs}
    wavs = {n: sf.read(str(out / f"{n}{AUDIO_EXT}"), dtype="float32", always_2d=True)[0] for n in runs}
    hard = hard_checks(runs, arrays, main_["vae"], main_["facts"], wavs, cpu=cpu)
    infos = {n: sf.info(str(out / f"{n}{AUDIO_EXT}")) for n in runs}
    hard["audio_format"] = all((i.format, i.subtype, i.samplerate, i.channels) == ("FLAC", C.AUDIO_SUBTYPE, 48000, 2)
                               for i in infos.values())
    hard["same_config_in_fresh_process"] = None if fresh is None else \
        {k: v for k, v in fresh["facts"].items() if k != "setup"} == {k: v for k, v in main_["facts"].items() if k != "setup"}
    items = {it["id"]: it for it in json.loads((C.e5_dir() / C.P0B_MANIFEST.name).read_text())["items"]}
    ctx = {n: context_error(wavs[n], sf.read(str(C.e5_dir() / items[r["item"]]["sources"][r["length"]][r["cond"]]),
                                             dtype="float32", always_2d=True)[0], r["peak_before_norm"])
           for n, r in runs.items()}
    hard["context_preserved"] = all(v <= 1e-3 for v in ctx.values())
    hard["weights_pinned"] = main_["weights_sha256"] == C.WEIGHTS_SHA256
    hard.update({f"env_{k}": v for k, v in main_["environment"].items()})
    soft = soft_checks(arrays, wavs, runs)
    not_run = sorted(k for k, v in hard.items() if v is None)
    # At full scale every check must have run and passed; a test scale accepts checks it could not run.
    go = all(v is True for v in hard.values()) if not tiny else all(v is not False for v in hard.values())
    C.write_json(C.a0_path(), {"go": go, "scale": main_["scale"], "device": main_["device"], "hard": hard,
                               "not_run": not_run, "soft": soft, "vae": main_["vae"], "context_error": ctx,
                               "facts": main_["facts"], "addendum_sha256": C.addendum_status()["addendum_sha256"],
                               "utc": C.utc()})
    print(json.dumps({"go": go, "hard": hard, "not_run": not_run}, indent=1))
    return 0 if go else 3


if __name__ == "__main__":
    sys.exit(main())
