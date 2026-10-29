"""Run B's CPU test of the lyric clock on the real ACE-Step 1.5 turbo model, before registration.

item00 of the E4 manifest, DiT seed 6002, global pin 906002, 8 steps, bf16 on the CPU (outside every plan):
  a  text2music: no hook against the hook at beta 0 on the alignment heads; pred_latents torch.equal
     (addendum 2 requires this before Run B)
  b  beta 2 on the alignment heads changes pred_latents
  c  cover-nofsq with item00's Voice Oohs guide at k 4: no hook against the hook at beta 0, torch.equal
  d  in c, the encoder states at step 0 (cover) and step 4 (non-cover): the lyric rows [0, n_lyr) must be
     equal; whether the whole tensors are equal is recorded
  e  40 hook calls per hooked run, every biased key < n_lyr, and (Lq, Lk) per phase
The clips are neither scored nor listened to, and their WAVs are deleted. The guide is rendered here, as
E4 renders it, by the measurement interpreter (DACAPO_PY_MAIN, as in experiments/run_all.py), and its
sha256 recorded. The result records the sha256 of the code it ran with; experiments/e6_clock/
register_b1code.py checks them against the files it registers.

  flock <model lock> env ACESTEP_REPO=... ACESTEP_CHECKPOINTS_DIR=... PYTHONPATH=. DACAPO_PY_MAIN=... \\
      <ACE python> experiments/e6_clock/runb_cpu_test.py      writes results/cpu/runb_cpu_test.json
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

ITEM = "item00"
SEED, RNG_SEED = 6002, 906002
K = 4                  # cover steps out of 8 (audio_cover_strength 0.5)
OOHS = 53              # GM Voice Oohs
BETAS = (0.0, 2.0)
OUT = ROOT / "results" / "cpu" / "runb_cpu_test.json"
CODE_FILES = ("dacapo/attn_bias.py", "experiments/e6_clock/runb_cpu_test.py", "dacapo/ace.py", "dacapo/words.py",
              "dacapo/seeds.py", "dacapo/render.py", "dacapo/symbolic.py", "dacapo/provenance.py",
              "experiments/e6_clock/readout.py", "experiments/e4_sing/manifest.json")
WEIGHTS = ("acestep-v15-turbo/model.safetensors", "acestep-v15-turbo/silence_latent.pt",
           "acestep-v15-turbo/modeling_acestep_v15_turbo.py", "vae/diffusion_pytorch_model.safetensors",
           "Qwen3-Embedding-0.6B/model.safetensors")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(x) -> str:
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


RENDER = ("import json, sys\n"
          "from dacapo.render import default_sf2, render\n"
          "from dacapo.symbolic import Melody, Note\n"
          "a = json.loads(sys.argv[1])\n"
          "m = Melody(tuple(Note(**n) for n in a['notes']), a['bpm'])\n"
          "render([m.to_instrument(a['program'])], a['bpm'], a['out'], sr=48000)\n"
          "print(json.dumps({'sf2': str(default_sf2())}))\n")


def render_guide(notes: list[dict], bpm: float, out: Path) -> dict:
    """The guide as experiments/e4_sing/make_items.py renders it (dacapo.render at 48 kHz)."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    arg = json.dumps({"notes": notes, "bpm": bpm, "program": OOHS, "out": str(out)})
    r = subprocess.run([os.environ.get("DACAPO_PY_MAIN", sys.executable), "-c", RENDER, arg], capture_output=True,
                       text=True, check=True, env=env)
    sf2 = Path(json.loads(r.stdout.strip().splitlines()[-1])["sf2"])
    return {"guide_sha256": sha256(out), "soundfont": str(sf2), "soundfont_sha256": sha256(sf2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t_start = time.time()

    import torch

    import readout
    from dacapo import ace
    from dacapo.attn_bias import CrossAttnBias, lyric_keys, word_spans
    from dacapo.provenance import write_result
    from acestep.core.scoring.dit_alignment import MusicStampsAligner

    code = {f: sha256(ROOT / f) for f in CODE_FILES}
    reg = json.loads((ROOT / "experiments" / "e5_motif" / "registration" / "addendum_2.json").read_text())
    ckpt = Path(os.environ.get("ACESTEP_CHECKPOINTS_DIR") or Path(os.environ["ACESTEP_REPO"]) / "checkpoints")
    weights = {w: sha256(ckpt / w) if (ckpt / w).exists() else None for w in WEIGHTS}
    try:
        ace_commit = subprocess.run(["git", "-C", os.environ["ACESTEP_REPO"], "rev-parse", "HEAD"], capture_output=True,
                                    text=True, check=True).stdout.strip()
    except Exception:
        ace_commit = None

    det = ace.setup_determinism()
    h = ace.load("cpu")
    facts = ace.runtime_facts(h)
    it = next(x for x in json.loads((ROOT / "experiments" / "e4_sing" / "manifest.json").read_text())["items"]
              if x["id"] == ITEM)
    bpm, notes = float(it["melody"]["bpm"]), it["melody"]["notes"]
    lay = lyric_keys(h, MusicStampsAligner(h.text_tokenizer), it["lyrics"])
    spans = word_spans(notes, bpm, lay["keys"])
    n_lyr = lay["n_lyr"]

    runs: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        guide = Path(td) / f"{ITEM}_oohs.wav"
        rendered = render_guide(notes, bpm, guide)

        def one(name: str, cover: bool, beta: float | None) -> None:
            steps: list[dict] = []
            enc: dict[int, torch.Tensor] = {}

            def pre(step, t, kw):  # reads the decoder's inputs, changes nothing
                ehs = kw["encoder_hidden_states"]
                steps.append({"step": step, "t": round(t, 6), "lk": int(ehs.shape[1]),
                              "frames": int(kw["hidden_states"].shape[1])})
                if step in (0, K):
                    enc[step] = ehs.detach().clone()
                return kw

            bias = None if beta is None else CrossAttnBias(h.model.decoder, readout.ALIGN, spans, beta, n_lyr)
            t0 = time.time()
            with bias if bias is not None else contextlib.nullcontext():
                r = ace.generate(h, Path(td) / f"{name}.wav", readout.CAPTION_F, float(it["duration"]), SEED,
                                 lyrics=it["lyrics"], steps=readout.STEPS, shift=readout.SHIFT, bpm=int(bpm),
                                 keyscale=it["key"], vocal_language="en", task="cover-nofsq" if cover else "text2music",
                                 src=str(guide) if cover else None, cover_strength=K / readout.STEPS if cover else 1.0,
                                 extra={"dcw_enabled": False}, pre=pre, keep_extra=True, rng_seed=RNG_SEED,
                                 out_subtype=readout.WAV_SUBTYPE)
            ex = r["extra"]
            rec = bias.record() if bias is not None else None
            phases = None
            if bias is not None:  # 5 hooked layers per decoder call, so call i belongs to step i // 5
                calls = [(i // 5, lq, lk) for i, (_, lq, lk) in enumerate(bias.shapes)]
                parts = (("cover", 0, K), ("non_cover", K, readout.STEPS)) if cover else (("text2music", 0, readout.STEPS),)
                phases = {p: sorted({(lq, lk) for s, lq, lk in calls if a <= s < b}) for p, a, b in parts}
            runs[name] = {
                "cover": cover, "beta": beta, "seconds": round(time.time() - t0, 1),
                "n_denoiser_calls": r["n_denoiser_calls"], "dcw_enabled": r["dcw_enabled"], "frames": r["frames"],
                "peak_before_norm": r["peak_before_norm"], "instruction": r["instruction"],
                "steps": steps, "hook": rec, "hook_lq_lk_by_phase": phases,
                "lyric_ids_match": [int(i) for i in list(ex["lyric_token_idss"][0])] == lay["ids"],
                "encoder_mask_all_ones": bool((ex["encoder_attention_mask"] == 1).all()),
                "pred_latents_sha256": tensor_sha256(ex["pred_latents"]),
                "target_latents_sha256": tensor_sha256(ex["target_latents"]) if ex.get("target_latents") is not None else None,
                "_pred": ex["pred_latents"].detach().clone(), "_enc": enc}
            print(f"{name}: {runs[name]['seconds']}s, calls {rec['hook_calls'] if rec else 0}, frames {r['frames']}",
                  flush=True)

        one("t2m_nohook", False, None)
        one("t2m_beta0", False, 0.0)
        one("t2m_beta2", False, 2.0)
        one("cover_nohook", True, None)
        one("cover_beta0", True, 0.0)

    def pred(n):
        return runs[n]["_pred"]

    base, b2 = pred("t2m_nohook").float(), pred("t2m_beta2").float()
    e0, e4 = runs["cover_nohook"]["_enc"][0], runs["cover_nohook"]["_enc"][K]
    h0, h4 = runs["cover_beta0"]["_enc"][0], runs["cover_beta0"]["_enc"][K]
    same_shape = tuple(e0.shape) == tuple(e4.shape)
    checks = {
        "a_t2m_beta0_bit_identical": bool(torch.equal(pred("t2m_nohook"), pred("t2m_beta0"))),
        "b_t2m_beta2_changes": not torch.equal(pred("t2m_nohook"), pred("t2m_beta2")),
        "c_cover_beta0_bit_identical": bool(torch.equal(pred("cover_nohook"), pred("cover_beta0"))),
        "d_lyric_rows_equal_across_switch": bool(torch.equal(e0[:, :n_lyr], e4[:, :n_lyr])),
        "e_hook_calls_40": all(runs[n]["hook"]["hook_calls"] == 40 for n in ("t2m_beta0", "t2m_beta2", "cover_beta0")),
        "e_max_key_below_n_lyr": all(runs[n]["hook"]["max_key"] < n_lyr for n in ("t2m_beta0", "t2m_beta2", "cover_beta0")),
        "runs_ok": all(r["n_denoiser_calls"] == readout.STEPS and r["dcw_enabled"] is False and r["lyric_ids_match"]
                       and r["encoder_mask_all_ones"] and r["peak_before_norm"] is not None for r in runs.values()),
        "alignment_heads_as_registered": {int(k): list(v) for k, v in h.custom_layers_config.items()} == readout.ALIGN,
        "sdpa": facts.get("attn_implementation") == "sdpa",
    }
    recorded = {
        "d_whole_encoder_states_equal_across_switch": bool(same_shape and torch.equal(e0, e4)),
        "d_lk_cover_non_cover": [int(e0.shape[1]), int(e4.shape[1])],
        "d_hooked_run_same_encoder_states": bool(torch.equal(e0, h0) and torch.equal(e4, h4)),
        "c_guide_latent_equal_across_arms": runs["cover_nohook"]["target_latents_sha256"] == runs["cover_beta0"]["target_latents_sha256"],
        "b_beta2_rel_l2_change": float((b2 - base).norm() / base.norm()),
        "b_beta2_max_abs_change": float((b2 - base).abs().max()),
        "lq_lk_by_phase": {n: runs[n]["hook_lq_lk_by_phase"] for n in ("t2m_beta0", "t2m_beta2", "cover_beta0")},
        "weights_match_addendum_2": weights == reg["ace"]["weights_sha256"],
        "ace_commit_matches_addendum_2": ace_commit == reg["ace"]["commit"],
    }
    passed = all(checks.values())
    for r in runs.values():
        r.pop("_pred"), r.pop("_enc")
    cfg = {"experiment": "runb_cpu_test", "item": ITEM, "dit_seed": SEED, "rng_seed": RNG_SEED, "steps": readout.STEPS,
           "shift": readout.SHIFT, "cover_k": K, "guide_program": OOHS, "caption": readout.CAPTION_F,
           "heads": {str(L): hs for L, hs in readout.ALIGN.items()}, "betas": list(BETAS), "device": "cpu", "dtype": "bf16",
           "code_sha256": code, "ace_commit": ace_commit, "weights_sha256": weights}
    results = {"passed": passed, "checks": checks, "recorded": recorded, "n_lyr": n_lyr, "lyric_start": lay["start"],
               "word_keys": lay["keys"], "spans": [[list(k), a, b] for k, a, b in spans],
               **rendered,
               "determinism": det, "runtime": facts, "runs": runs}
    write_result(args.out, cfg, results, t_start)
    print(json.dumps({"passed": passed, "checks": checks, "recorded": recorded}, indent=1), flush=True)
    print("RUN B CPU TEST:", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
