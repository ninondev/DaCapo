"""E6 read-out (exploratory): do the seven lyric-alignment heads of ACE-Step 1.5 give the onset of each
sung word? Its one decision is whether the lyric-clock stage B1 may run. The plan, the rule and the
SHA-256 of this file are registered in experiments/e5_motif/registration/addendum_2.json ("e6_readout").

  python readout.py a0 --device cuda       acceptance checks on two pilot-seed clips (own folder)
  python readout.py score --a0 --device cuda   the registered scoring path on those two clips
  python readout.py gen --device cuda      88 generations, attention capture, onsets for every head set
  python readout.py score --device cuda    vocal gate (htdemucs), Whisper large-v3 words, MMS (diagnostic)
  python readout.py decide                 the registered rule -> results/e6_gate.json
  python readout.py check-gate             exit 0 only for a full-scale, registered gate that allows B1

The registered rule, applied by decision(); B1 runs only if every condition holds:
  0  integrity: the code files and the E4 manifest hash to the registered values, now and in every
     clip; no test flag; every scored clip used Whisper large-v3 at the registered revision; every
     generated clip has its pre-normalisation peak; at most 4 of the 88 planned clips are missing,
     failed or unscored
  1  at least 44 of the 88 planned clips pass the vocal gate (missing, failed, unscored clips fail it)
  2  at least 20 clips are evaluable (vocal and at least 4 lyric words matched to Whisper words)
  3  the median over evaluable clips of the per-clip median |head onset - Whisper onset| is <= 0.25 s
  4  in at least 75% of evaluable clips the alignment heads' error is below the median error of the
     200 random head sets (ties count against the alignment heads)
  5  in every generated clip, ACE-Step's own get_lyric_timestamp succeeds and agrees within one DiT
     frame with ACE-Step's aligner on our replicated attention and with our batched aligner
Errors are rounded to 1e-6 s before every comparison. Everything else written here is exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("E6_OUT", HERE / "outputs_readout"))   # run_all sets E6_OUT/E6_RES/E6_A0 per scale
RES = Path(os.environ.get("E6_RES", HERE / "results"))
OUT_A0 = Path(os.environ.get("E6_A0", HERE / "outputs_readout_a0"))  # never read by decide
REG_DEFAULT = ROOT / "experiments" / "e5_motif" / "registration" / "addendum_2.json"
CODE_FILES = ("experiments/e6_clock/readout.py", "dacapo/__init__.py", "dacapo/ace.py", "dacapo/attn_tap.py",
              "dacapo/provenance.py", "dacapo/seeds.py", "dacapo/words.py", "experiments/e4_sing/manifest.json")
WAV_SUBTYPE = "PCM_16"       # written by soundfile, not by ACE-Step's torchaudio-dependent saver

# acestep/core/generation/handler/lyric_alignment_common.py:12 (_DEFAULT_LAYERS_CONFIG); the turbo
# checkpoint's config.json has no lyric_alignment_layers_config, so this is what ACE-Step uses.
ALIGN = {2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}
TAP_LAYERS = (2, 3, 4, 5, 6)
N_LAYERS, N_HEADS, N_KV = 24, 16, 8
GROUP = N_HEADS // N_KV      # transformers repeat_kv: query head h reads key/value head h // GROUP
N_RANDOM, N_RANDOM_STEPS, HEAD_SEED = 200, 50, 20261027
POSTHOC_SEED = 42            # ACE-Step's auto-LRC seed (ui/gradio/events/results/generation_progress.py:526)
POSTHOC_SEEDS_EXTRA = (43, 44)
STEPS, SHIFT = 8, 3.0
RNG_OFFSET = 900_000         # global RNG pin = DiT seed + RNG_OFFSET (never equal to the DiT seed)
CAPTION_F = "a cappella, solo female vocal, no instruments, no drums, dry studio recording, clear lyrics"  # E4 CAPTION
CAPTION_M = CAPTION_F.replace("female", "male")
VERSES = {
    "v0": "[Verse]\nMorning light upon the river\nCarry me where the willows grow\nEvery step a quiet whisper\nEvery breath a song I know",
    "v1": "[Verse]\nSilver rain across the window\nGentle voices call my name\nUnderneath the yellow lantern\nNothing ever stays the same",
    "v2": "[Verse]\nFollow me along the harbor\nSailing boats are drifting by\nSeven seagulls in the morning\nPainting circles in the sky",
    "v3": "[Verse]\nRemember all the summer evenings\nDancing slowly by the fire\nEvery story that we started\nCarried higher, ever higher",
}
WHISPER = "openai/whisper-large-v3"
WHISPER_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"  # Hugging Face commit, last modified 2024-08-12
# vocal gate and decision thresholds
# vocal presence, not dominance: run-1 no-vocal clips 0.000-0.002, vocal clips 0.74-0.999, a CPU test clip with
# singing and model-added percussion 0.388 (htdemucs vocal-stem energy share)
VOCAL_SHARE_MIN, VOCAL_ACTIVE_MIN_S, VOCAL_ACTIVE_DBFS, PEAK_MIN = 0.2, 1.0, -40.0, 0.01
MIN_MATCHED, MIN_EVAL_CLIPS, MIN_VOCAL_CLIPS = 4, 20, 44
ERR_MAX_S = 0.25
BEAT_NUM, BEAT_DEN = 3, 4    # condition 4 as integers: 4 * n_beat >= 3 * n_evaluable
MAX_INCOMPLETE = 4           # more missing, failed or unscored clips than this: incomplete
MAX_ATTEMPTS = 3             # a clip that raises is run again, at most twice, with the same seeds
DIGITS = 6                   # seconds are rounded to 1e-6 before any comparison
N_BOOT, BOOT_SEED = 10000, 0
NEEDED_EXTRA = ("pred_latents", "encoder_hidden_states", "encoder_attention_mask", "context_latents", "lyric_token_idss")


class SetupError(RuntimeError):
    """A problem that affects every clip (exit before any generation)."""


# plan

def items() -> list[dict]:
    man = json.loads((ROOT / "experiments" / "e4_sing" / "manifest.json").read_text())
    out = []
    for i, it in enumerate(man["items"]):
        for r in range(4):
            out.append({"id": f"{it['id']}_r{r}", "set": "e4", "lyric_id": it["id"], "lyrics": it["lyrics"],
                        "caption": CAPTION_F, "voice": "female", "duration": float(it["duration"]),
                        "bpm": int(it["melody"]["bpm"]), "keyscale": it["key"], "seed": 61000 + 10 * i + r})
    for v, (vid, lyr) in enumerate(VERSES.items()):
        for g, (voice, cap) in enumerate((("female", CAPTION_F), ("male", CAPTION_M))):
            for r in range(5):
                out.append({"id": f"{vid}_{voice[0]}_r{r}", "set": "extra", "lyric_id": vid, "lyrics": lyr,
                            "caption": cap, "voice": voice, "duration": 16.0, "bpm": 90, "keyscale": "",
                            "seed": 62000 + 100 * v + 10 * g + r})
    for x in out:
        x["rng_seed"] = x["seed"] + RNG_OFFSET
    assert len({x["seed"] for x in out}) == len(out) == 88
    return out


def a0_items() -> list[dict]:
    """Two acceptance clips on seeds outside the plan (6000 is the CPU pilot's seed); the tiny CPU test
    scale runs the first one only."""
    base = {x["id"]: x for x in items()}
    out = []
    for src, cid, seed in (("item00_r0", "a0_item00", 6000), ("v0_m_r0", "a0_v0_m", 6001)):
        out.append({**base[src], "id": cid, "seed": seed, "rng_seed": seed + RNG_OFFSET})
    return out[:1] if os.environ.get("E6_SCALE") == "tiny" else out


def lyric_texts() -> dict[str, str]:
    return {x["lyric_id"]: x["lyrics"] for x in items()}


def random_head_sets(k: int = N_RANDOM, seed: int = HEAD_SEED) -> list[dict[int, list[int]]]:
    """7-head sets matched to ALIGN by layer and by GQA group: in layers 3 and 5 (where ALIGN holds a
    whole key/value group) the draw is another whole group (2j, 2j+1); in layers 2, 4 and 6 it is one
    head from another group, so no random head shares a key projection with an alignment head."""
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(k):
        s = {}
        for L, hs in ALIGN.items():
            groups = {h // GROUP for h in hs}
            if len(hs) == 1:
                s[L] = [int(rng.choice([h for h in range(N_HEADS) if h // GROUP not in groups]))]
            else:
                j = int(rng.choice([j for j in range(N_KV) if j not in groups]))
                s[L] = [GROUP * j + q for q in range(GROUP)]
        sets.append(s)
    return sets


# provenance and registration

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    return {f: sha256(ROOT / f) for f in CODE_FILES}


def registered_hashes(reg_path: Path) -> dict[str, str] | None:
    try:
        return json.loads(reg_path.read_text())["e6_readout"]["code_sha256"]
    except (OSError, KeyError, TypeError, ValueError):
        return None


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    scope = ["--", "dacapo", "experiments/e6_clock", "experiments/e4_sing/manifest.json"]
    tracked = run("status", "--porcelain", "--untracked-files=no", *scope)
    return {"git_commit": run("rev-parse", "HEAD"), "git_tracked_changes": None if tracked is None else bool(tracked)}


# attention -> onsets (ACE env)

def consensus_batch(w, violence: float = 2.0) -> np.ndarray:
    """MusicStampsAligner._apply_bidirectional_consensus (acestep/core/scoring/dit_alignment.py:55-101)
    for S head sets at once, medfilt_width=1 (the median filter is then the identity).
    w: float32 tensor [S, H, N tokens, F frames] -> calc [S, N, F]."""
    import torch
    import torch.nn.functional as F

    p = F.softmax(w, dim=-1) * F.softmax(w, dim=-2)
    p = torch.relu(p - violence * torch.quantile(p, 0.5, dim=-1, keepdim=True))
    p = torch.relu(p - violence * torch.quantile(p, 0.5, dim=-2, keepdim=True))
    p = p ** 2
    flat = p.flatten(1)
    std = flat.std(dim=1, unbiased=False).view(-1, 1, 1, 1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1)
    return ((p - mean) / (std + 1e-9)).mean(dim=1).numpy()


def first_frames(calc: np.ndarray) -> np.ndarray:
    """First frame of every token row on ACE-Step's DTW path (dit_alignment.py:257-277)."""
    from acestep.core.scoring._dtw import dtw_cpu

    ti, fi = dtw_cpu(-calc.astype(np.float64))
    first = np.full(calc.shape[0], -1, dtype=np.int64)
    for t, f in zip(ti, fi):
        if first[t] < 0:
            first[t] = f
    return first


def select(att, cfg: dict):
    import torch

    return torch.stack([att[L, h] for L, hs in cfg.items() for h in hs])


def onsets_for_sets(att, cfgs: list[dict], wf: np.ndarray, spf: float) -> np.ndarray:
    import torch

    calc = consensus_batch(torch.stack([select(att, c) for c in cfgs]))
    return np.stack([first_frames(c) for c in calc])[:, wf] * spf


def analyse_clip(atts: dict, steps: np.ndarray, word_first: list[int], spf: float, sets: list[dict]) -> dict:
    """Word onsets (s). atts: {re-noising seed: float32 tensor [24, 16, N, F]} (post-hoc passes, lyric
    slice); steps: [n_steps, 5, 16, N, F] (tap, layers 2-6). Every seed gets ALIGN and all random sets."""
    import torch

    wf = np.asarray(word_first)
    align, rand = {}, {}
    for sd, att in atts.items():
        o = onsets_for_sets(att, [ALIGN] + sets, wf, spf)
        align[sd], rand[sd] = o[0], o[1:]
    att = atts[POSTHOC_SEED]
    single = att.reshape(N_LAYERS * N_HEADS, 1, att.shape[-2], att.shape[-1])
    o_single = np.stack([first_frames(c) for c in consensus_batch(single)])[:, wf] * spf
    per_step = []
    for j in range(steps.shape[0]):
        full = torch.zeros(N_LAYERS, N_HEADS, att.shape[-2], att.shape[-1])
        full[list(TAP_LAYERS)] = torch.tensor(steps[j].astype(np.float32))
        per_step.append(onsets_for_sets(full, [ALIGN] + sets[:N_RANDOM_STEPS], wf, spf))
    return {"align": align, "random": rand, "single": o_single,   # single: [384, words], row = layer * 16 + head
            "steps": np.stack(per_step)}                          # [n_steps, 1 + N_RANDOM_STEPS, words]


# gen and a0 (ACE env)

def setup(device: str, reg_path: Path, unregistered: bool):
    """Checks that apply to every clip; raises SetupError before any diffusion."""
    from dacapo import ace
    from acestep.core.scoring.dit_alignment import MusicStampsAligner
    from dacapo.words import group_words, lyric_words

    code = code_hashes()
    reg = registered_hashes(reg_path)
    registered = reg is not None and reg == code
    if not registered and not unregistered:
        raise SetupError(f"code differs from the registered hashes in {reg_path} (or none registered); "
                         f"current {code}")
    det = ace.setup_determinism()
    h = ace.load(device)
    if {int(k): list(v) for k, v in h.custom_layers_config.items()} != ALIGN:
        raise SetupError(f"unexpected alignment heads {h.custom_layers_config}")
    aligner = MusicStampsAligner(h.text_tokenizer)
    for lid, lyr in lyric_texts().items():  # ACE-Step's own prompt format, tokenizer and lyric slice
        ids = h.text_tokenizer(h._format_lyrics(lyr, "en"), padding="longest", truncation=True, max_length=2048,
                               return_tensors="pt").input_ids
        _, pure, s, e = h._extract_lyric_segment(ids, "en")
        got = [w.norm for w in group_words(aligner._decode_tokens_incrementally(pure))]
        if got != lyric_words(lyr):
            raise SetupError(f"{lid}: token words {got} != lyric words {lyric_words(lyr)}")
    ctx = {"code_sha256": code, "registered": registered, "determinism": det, **git_state()}
    return h, aligner, ctx


def run_clip(h, aligner, it: dict, sets: list[dict], out: Path, ctx: dict, device: str) -> dict:
    """One planned clip: generation with the tap, ACE-Step's own timestamps, three post-hoc passes,
    onsets for every head set. Writes <id>.wav, <id>.npz, <id>.json. Raises on any technical problem."""
    import torch

    from dacapo import ace
    from dacapo.attn_tap import CrossAttnTap
    from dacapo.words import group_words, lyric_words

    t0 = time.time()
    tap = CrossAttnTap(h.model.decoder, TAP_LAYERS)
    r = ace.generate(h, out / f"{it['id']}.wav", it["caption"], it["duration"], it["seed"], lyrics=it["lyrics"],
                     steps=STEPS, shift=SHIFT, bpm=it["bpm"], keyscale=it["keyscale"], vocal_language="en",
                     extra={"dcw_enabled": False}, taps=(tap,), keep_extra=True, rng_seed=it["rng_seed"],
                     out_subtype=WAV_SUBTYPE)
    t_gen = time.time() - t0
    ex = r["extra"]
    missing = [k for k in NEEDED_EXTRA if ex.get(k) is None]
    if missing:
        raise RuntimeError(f"ACE-Step returned no {missing}: is ACESTEP_SAVE_MEMORY set? (generate_music_payload.py:52-63)")
    if r["n_denoiser_calls"] != STEPS or len(tap.records) != STEPS:
        raise RuntimeError(f"{r['n_denoiser_calls']} denoiser calls and {len(tap.records)} tap records, expected {STEPS}")
    if r.get("dcw_enabled") is not False:
        raise RuntimeError(f"DCW resolved to {r.get('dcw_enabled')}, expected False")
    ids = ex["lyric_token_idss"][:1]
    raw, pure, s, e = h._extract_lyric_segment(ids, "en")
    tokens = aligner._decode_tokens_incrementally(pure)
    words = group_words(tokens)
    if [w.norm for w in words] != lyric_words(it["lyrics"]):
        raise RuntimeError(f"{it['id']}: token words {[w.norm for w in words]} != lyric words")
    word_first = [w.tokens[0] for w in words]
    dur = float(it["duration"])
    off = h.get_lyric_timestamp(pred_latent=ex["pred_latents"], encoder_hidden_states=ex["encoder_hidden_states"],
                                encoder_attention_mask=ex["encoder_attention_mask"], context_latents=ex["context_latents"],
                                lyric_token_ids=ids, total_duration_seconds=dur, vocal_language="en",
                                inference_steps=STEPS, seed=POSTHOC_SEED)
    official = [float(off["token_timestamps"][i].start) for i in word_first] if off["success"] else None
    atts = {}
    for sd in (POSTHOC_SEED,) + POSTHOC_SEEDS_EXTRA:
        a = ace.posthoc_lyric_attention(h, ex, seed=sd, steps=STEPS)
        if tuple(a.shape[:2]) != (N_LAYERS, N_HEADS):
            raise RuntimeError(f"post-hoc attention has shape {tuple(a.shape)}")
        atts[sd] = a[:, :, s:e, :].float().cpu()
    att = atts[POSTHOC_SEED]
    n_frames = att.shape[-1]
    spf = dur / n_frames  # ACE-Step's seconds_per_frame (dit_alignment.py:258); 0.08 s here
    steps = tap.stacked(s, e)  # [8, 5, 16, N, F]
    res = analyse_clip(atts, steps, word_first, spf, sets)
    calc = aligner.stamps_align_info(attention_matrix=att, lyrics_tokens=pure, total_duration_seconds=dur,
                                     custom_config=ALIGN, violence_level=2.0, medfilt_width=1)["calc_matrix"]
    replicated = [float(aligner.token_timestamps(calc, pure, dur)[i].start) for i in word_first]
    batched = res["align"][POSTHOC_SEED]

    def within(a, b) -> bool:
        return b is not None and bool(np.max(np.abs(np.asarray(a) - np.asarray(b))) <= spf + 1e-9)

    np.savez_compressed(out / f"{it['id']}.npz", posthoc=att.half().numpy(), steps=steps.astype(np.float16),
                        pure_ids=np.array(pure), onsets_random=res["random"][POSTHOC_SEED],
                        **{f"onsets_random_{sd}": res["random"][sd] for sd in POSTHOC_SEEDS_EXTRA},
                        onsets_single=res["single"], onsets_steps=res["steps"])
    meta = {**it, "status": "ok", "tokens": tokens, "words": [w.norm for w in words],
            "word_tokens": [list(w.tokens) for w in words], "lyric_slice": [s, e], "n_lyric_ids": len(raw),
            "n_frames": n_frames, "seconds_per_frame": spf, "latent_frames": int(ex["pred_latents"].shape[1]),
            "encoder_tokens": int(ex["encoder_hidden_states"].shape[1]),
            "onsets_align": batched.tolist(),                       # primary: same matrix and code as the control
            "onsets_official": official, "onsets_replicated": replicated,
            "official_ok": bool(off["success"]), "official_error": off["error"],
            "replicates_official": within(replicated, official) and within(batched, official),
            "replicates_official_exact": official is not None and replicated == official
            and bool(np.allclose(batched, official, rtol=0, atol=1e-9)),
            "onsets_align_seeds": {str(sd): res["align"][sd].tolist() for sd in POSTHOC_SEEDS_EXTRA},
            "lrc": off["lrc_text"], "peak_before_norm": r.get("peak_before_norm"), "dcw_enabled": r.get("dcw_enabled"),
            "n_denoiser_calls": r["n_denoiser_calls"], "tap_calls": len(tap.records), "device": device,
            "scale": os.environ.get("E6_SCALE", "unknown"), **ctx,
            "tap_seconds": round(tap.seconds, 3), "gen_seconds": round(t_gen, 2), "total_seconds": round(time.time() - t0, 2)}
    (out / f"{it['id']}.json").write_text(json.dumps(meta, indent=1))
    meta["_pred_latents"] = ex["pred_latents"]  # for a0 only; never written
    del atts, att, tap
    if device == "cuda":
        torch.cuda.empty_cache()
    return meta


def gen(args) -> None:
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    h, aligner, ctx = setup(args.device, Path(args.registration or REG_DEFAULT), args.unregistered)
    sets = random_head_sets()
    log_path = out / "log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"runs": []}
    todo = [it for it in items() if not args.only or it["id"] in args.only.split(",")]
    for it in todo[: args.limit or None]:
        if (out / f"{it['id']}.json").exists():
            continue
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                meta = run_clip(h, aligner, it, sets, out, ctx, args.device)
                (out / f"{it['id']}.failed.json").unlink(missing_ok=True)
                log["runs"].append({k: meta[k] for k in ("id", "seed", "gen_seconds", "total_seconds",
                                                         "replicates_official", "peak_before_norm")} | {"attempt": attempt})
                print(f"{it['id']:>12s} {meta['total_seconds']:5.1f}s replicate={meta['replicates_official']} "
                      f"onsets={np.round(meta['onsets_align'][:4], 2).tolist()}...", flush=True)
                break
            except Exception as exc:  # recorded per clip; the stage goes on so that score and decide still run
                fail = {**it, "status": "failed", "attempt": attempt, "error": repr(exc),
                        "traceback": traceback.format_exc(), **ctx}
                (out / f"{it['id']}.failed.json").write_text(json.dumps(fail, indent=1))
                print(f"{it['id']:>12s} attempt {attempt} FAILED: {exc!r}", flush=True)
                if args.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
        log_path.write_text(json.dumps(log, indent=1))


def a0(args) -> None:
    """Acceptance checks on the GPU before gen (exit 3 on a hard failure, which stops the e6read stage).
    Hard: setup (registration, heads, tokenisation of all 16 lyrics), 8 denoiser calls and 8 tap records,
    the intermediate tensors present, DCW off, the peak captured, ACE-Step's timestamps reproduced within
    one frame by both replications. Recorded only: exact replication, tap bit-identity (same seeds with
    and without the tap), per-clip seconds."""
    from dacapo import ace

    OUT_A0.mkdir(parents=True, exist_ok=True)
    rep = {"hard": {}, "soft": {}, "clips": {}}
    try:
        h, aligner, ctx = setup(args.device, Path(args.registration or REG_DEFAULT), args.unregistered)
        rep["hard"]["setup"] = True
    except Exception as exc:
        rep["hard"]["setup"] = False
        rep["error"] = repr(exc)
        h = None
    if h is not None:
        sets = random_head_sets()
        for k, it in enumerate(a0_items()):
            try:
                m = run_clip(h, aligner, it, sets, OUT_A0, ctx, args.device)
            except Exception as exc:
                rep["hard"][f"{it['id']}_runs"] = False
                rep["clips"][it["id"]] = {"error": repr(exc), "traceback": traceback.format_exc()}
                continue
            rep["hard"][f"{it['id']}_runs"] = True
            rep["hard"][f"{it['id']}_peak"] = m["peak_before_norm"] is not None
            rep["hard"][f"{it['id']}_replicates"] = bool(m["replicates_official"])
            rep["soft"][f"{it['id']}_replicates_exact"] = bool(m["replicates_official_exact"])
            if args.device == "cuda":
                rep["soft"][f"{it['id']}_gen_under_6s"] = m["gen_seconds"] < 6
                rep["soft"][f"{it['id']}_total_under_15s"] = m["total_seconds"] < 15
            rep["clips"][it["id"]] = {x: m[x] for x in ("gen_seconds", "total_seconds", "tap_seconds", "peak_before_norm",
                                                         "onsets_align", "onsets_official", "lrc")}
            if k == 0:  # same seeds without the tap: bit-identical pred_latents expected
                import torch
                plain = ace.generate(h, OUT_A0 / f"{it['id']}_plain.wav", it["caption"], it["duration"], it["seed"],
                                     lyrics=it["lyrics"], steps=STEPS, shift=SHIFT, bpm=it["bpm"], keyscale=it["keyscale"],
                                     vocal_language="en", extra={"dcw_enabled": False}, keep_extra=True,
                                     rng_seed=it["rng_seed"], out_subtype=WAV_SUBTYPE)["extra"]["pred_latents"]
                same = bool(torch.equal(plain, m["_pred_latents"]))
                rep["soft"]["tap_bit_identical"] = same
                if not same:
                    again = ace.generate(h, OUT_A0 / f"{it['id']}_plain2.wav", it["caption"], it["duration"], it["seed"],
                                         lyrics=it["lyrics"], steps=STEPS, shift=SHIFT, bpm=it["bpm"],
                                         keyscale=it["keyscale"], vocal_language="en", extra={"dcw_enabled": False},
                                         keep_extra=True, rng_seed=it["rng_seed"], out_subtype=WAV_SUBTYPE)["extra"]["pred_latents"]
                    rep["soft"]["plain_repeat_bit_identical"] = bool(torch.equal(plain, again))
                    rep["soft"]["tap_max_abs_diff"] = float((plain.float() - m["_pred_latents"].float()).abs().max())
    rep["pass"] = all(rep["hard"].values()) and bool(rep["hard"])
    (OUT_A0 / "a0_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"hard": rep["hard"], "soft": rep["soft"]}, indent=1))
    print("E6 A0:", "PASS" if rep["pass"] else "FAIL", flush=True)
    sys.exit(0 if rep["pass"] else 3)


# score (main env)

def score(args) -> None:
    import librosa
    import soundfile as sf
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from transformers import pipeline

    from dacapo.words import lyric_words

    dev = args.device
    out_dir, plan = (OUT_A0, a0_items()) if args.a0 else (OUT, items())
    path = OUT_A0 / "a0_scores.json" if args.a0 else RES / "e6_scores.json"
    rows = json.loads(path.read_text()) if path.exists() and not args.a0 else {}  # a0 always rescores
    sep = get_model("htdemucs").to(dev).eval()
    asr = pipeline("automatic-speech-recognition", model=args.whisper, device=0 if dev == "cuda" else -1,
                   revision=WHISPER_REVISION if args.whisper == WHISPER else None,
                   torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    asr_rev = getattr(asr.model.config, "_commit_hash", None)
    kw = {} if args.whisper.endswith(".en") else {"generate_kwargs": {"language": "en", "task": "transcribe"}}
    fa = torchaudio.pipelines.MMS_FA
    fa_model = fa.get_model(with_star=False).to(dev).eval()
    fa_tok, fa_align = fa.get_tokenizer(), fa.get_aligner()
    path.parent.mkdir(parents=True, exist_ok=True)
    for it in plan:
        wav = out_dir / f"{it['id']}.wav"
        if not (out_dir / f"{it['id']}.json").exists() or (args.only and it["id"] not in args.only.split(",")):
            continue
        old = rows.get(it["id"])
        if old and old.get("whisper") == args.whisper and old.get("whisper_revision") == asr_rev and "error" not in old:
            continue
        t0 = time.time()
        try:
            x, sr = sf.read(str(wav), always_2d=True, dtype="float32")  # ACE-Step output: 48 kHz stereo, peak -1 dBFS
            # vocal gate: htdemucs stem energy shares and vocal-stem activity (50 ms frames >= -40 dBFS)
            torch.manual_seed(0)
            y = torch.tensor(librosa.resample(x.T, orig_sr=sr, target_sr=sep.samplerate))
            ref = y.mean(0)
            yn = (y - ref.mean()) / (ref.std() + 1e-8)
            with torch.no_grad():
                stems = apply_model(sep, yn[None].to(dev), device=dev, shifts=0, split=True, overlap=0.25, progress=False)[0]
            stems = stems.cpu() * (ref.std() + 1e-8) + ref.mean()
            energy = {s: float((stems[i] ** 2).sum()) for i, s in enumerate(sep.sources)}
            tot = sum(energy.values()) or 1.0
            voc = stems[sep.sources.index("vocals")].mean(0).numpy()
            hop = int(0.05 * sep.samplerate)
            n = len(voc) // hop
            rms = np.sqrt((voc[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
            active = float((20 * np.log10(rms) >= VOCAL_ACTIVE_DBFS).sum() * 0.05)
            # reference: Whisper word timestamps (greedy; transformers forces eager attention for them)
            x16 = librosa.resample(x.mean(1), orig_sr=sr, target_sr=16000)
            out = asr({"raw": x16, "sampling_rate": 16000}, return_timestamps="word", **kw)
            chunks = [[c["text"], c["timestamp"][0], c["timestamp"][1]] for c in out["chunks"]]
            # second reference (diagnostic): MMS forced alignment of the known lyric words, on the CPU
            lw = lyric_words(it["lyrics"])
            mms, mms_error = None, None
            try:
                with torch.inference_mode():
                    em, _ = fa_model(torch.tensor(x16)[None].to(dev))
                spans = fa_align(em[0].float().cpu(), fa_tok(lw))
                ratio = len(x16) / em.shape[1] / 16000
                mms = [float(sp[0].start * ratio) for sp in spans]
            except Exception as exc:  # e.g. clip too short for the token sequence; counted in the summary
                mms_error = repr(exc)
            rows[it["id"]] = {"stem_share": {k: v / tot for k, v in energy.items()}, "vocal_active_s": active,
                              "whisper": args.whisper, "whisper_revision": asr_rev, "whisper_text": out["text"],
                              "whisper_chunks": chunks, "mms_onsets": mms, "mms_error": mms_error,
                              "separator": "htdemucs", "seconds": round(time.time() - t0, 2)}
            print(f"{it['id']:>12s} vocals {rows[it['id']]['stem_share']['vocals']:.3f} active {active:4.1f}s "
                  f"words {len(chunks)} {rows[it['id']]['seconds']}s", flush=True)
        except Exception as exc:  # one bad clip must not stop the stage; decide counts it as unscored
            rows[it["id"]] = {"error": repr(exc), "traceback": traceback.format_exc(), "whisper": args.whisper}
            print(f"{it['id']:>12s} SCORE FAILED: {exc!r}", flush=True)
        path.write_text(json.dumps(rows, indent=1))
    if args.a0:  # the registered scoring path on the two acceptance clips, before any planned clip is generated
        ok = len(rows) == len(plan) and all("error" not in r for r in rows.values())
        print("E6 A0 SCORE:", "PASS" if ok else "FAIL", flush=True)
        sys.exit(0 if ok else 3)
    (RES / "e6_scores_meta.json").write_text(json.dumps({"whisper": args.whisper, "whisper_revision": asr_rev,
                                                         "separator": "htdemucs", "aligner": "torchaudio MMS_FA"}, indent=1))


# decide (numpy only)

def rnd(x: float) -> float:
    return round(float(x), DIGITS)


def clip_stats(meta: dict | None, row: dict | None, z: dict | None, it: dict) -> dict:
    """One planned clip. meta None: missing; meta["status"] == "failed": failed in gen."""
    from dacapo.words import match_words, norm_word

    r = {"id": it["id"], "set": it["set"], "lyric_id": it["lyric_id"], "voice": it["voice"]}
    if meta is None or meta.get("status") != "ok":
        return {**r, "status": "missing" if meta is None else "failed", "scored": False, "vocal": False, "evaluable": False}
    r.update(status="ok", peak_before_norm=meta.get("peak_before_norm"), replicates_official=bool(meta["replicates_official"]),
             official_ok=bool(meta.get("official_ok")), code_sha256=meta.get("code_sha256"), registered=meta.get("registered"))
    if row is None or "error" in row:
        return {**r, "scored": False, "vocal": False, "evaluable": False}
    share = row["stem_share"]["vocals"]
    peak = meta.get("peak_before_norm")
    vocal = share >= VOCAL_SHARE_MIN and row["vocal_active_s"] >= VOCAL_ACTIVE_MIN_S and peak is not None and peak >= PEAK_MIN
    hyp_all = [(norm_word(c[0]), c[1]) for c in row["whisper_chunks"]]
    hyp = [(w, t) for w, t in hyp_all if w and t is not None]
    pairs = match_words(meta["words"], [w for w, _ in hyp])
    r.update(scored=True, whisper=row.get("whisper"), whisper_revision=row.get("whisper_revision"),
             mms_failed=row.get("mms_onsets") is None, vocal_share=share, vocal_active_s=row["vocal_active_s"],
             vocal=bool(vocal), n_words=len(meta["words"]), n_whisper_words=len(hyp), n_matched=len(pairs))
    r["evaluable"] = bool(vocal and len(pairs) >= MIN_MATCHED)
    if not pairs:
        return r
    li = np.array([a for a, _ in pairs])
    ref = np.array([hyp[b][1] for _, b in pairs], dtype=float)
    head = np.array(meta["onsets_align"])[li]
    err = np.round(np.abs(head - ref), DIGITS)
    rand_err = np.round(np.median(np.abs(z["onsets_random"][:, li] - ref[None]), axis=1), DIGITS)  # [K]
    e_med, r_med = rnd(np.median(err)), rnd(np.median(rand_err))
    r.update(err_median=e_med, err_signed_median=rnd(np.median(head - ref)), n_within=int((err <= ERR_MAX_S).sum()),
             within_025=float((err <= ERR_MAX_S).mean()), random_err_median=r_med, beat=bool(e_med < r_med),
             random_share_worse=float((rand_err > e_med).mean()), random_err=rand_err.tolist(),
             err_median_excl_first=rnd(np.median(err[1:])) if len(err) > 1 else None)
    # sensitivity to the re-noising seed (exploratory): conditions 3 and 4 with seeds 43 and 44
    r["seeds"] = {}
    for sd in POSTHOC_SEEDS_EXTRA:
        hs = np.array(meta["onsets_align_seeds"][str(sd)])[li]
        es = rnd(np.median(np.abs(hs - ref)))
        rs = rnd(np.median(np.round(np.median(np.abs(z[f"onsets_random_{sd}"][:, li] - ref[None]), axis=1), DIGITS)))
        r["seeds"][str(sd)] = {"err_median": es, "beat": bool(es < rs)}
    # other exploratory read-outs
    st = z["onsets_steps"]  # [n_steps, 1 + N_RANDOM_STEPS, words]
    r["steps_err_median"] = [rnd(np.median(np.abs(st[j, 0][li] - ref))) for j in range(st.shape[0])]
    r["steps_beat"] = [bool(rnd(np.median(np.abs(st[j, 0][li] - ref))) <
                            rnd(np.median(np.median(np.abs(st[j, 1:][:, li] - ref[None]), axis=1)))) for j in range(st.shape[0])]
    r["single_head_err"] = np.round(np.median(np.abs(z["onsets_single"][:, li] - ref[None]), axis=1), DIGITS).tolist()
    if row.get("mms_onsets"):
        mms = np.array(row["mms_onsets"])
        r["mms_vs_whisper_median"] = rnd(np.median(np.abs(mms[li] - ref)))
        r["head_vs_mms_median"] = rnd(np.median(np.abs(np.array(meta["onsets_align"]) - mms)))
        r["head_vs_mms_signed"] = rnd(np.median(np.array(meta["onsets_align"]) - mms))
    r["sung_span_s"] = float(ref.max() - ref.min()) if len(ref) > 1 else None
    return r


def bootstrap(rows: list[dict]) -> dict:
    """Descriptive 95% intervals: lyric clusters resampled within the two strata (12 E4 lines with 4 clips
    each, 4 verses with 10 each), so every resample has the 88 planned clips."""
    rng = np.random.default_rng(BOOT_SEED)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["lyric_id"], []).append(r)
    strata = [sorted({r["lyric_id"] for r in rows if r["set"] == s}) for s in ("e4", "extra")]
    bv, bm, bb = [], [], []
    for _ in range(N_BOOT):
        pick = [r for ids in strata if ids for i in rng.choice(ids, len(ids)) for r in by[i]]
        bv.append(np.mean([r["vocal"] for r in pick]))
        pe = [r for r in pick if r["evaluable"]]
        if pe:
            bm.append(np.median([r["err_median"] for r in pe]))
            bb.append(np.mean([r["beat"] for r in pe]))

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a else None
    return {"vocal_frac": ci(bv), "err_median_of_clip_medians": ci(bm), "beat_frac": ci(bb)}


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    ev = [r for r in rows if r["evaluable"]]
    n_vocal = sum(r["vocal"] for r in rows)
    s = {"n_planned": len(rows), "n_generated": len(ok), "n_failed": sum(r["status"] == "failed" for r in rows),
         "n_missing": sum(r["status"] == "missing" for r in rows), "n_scored": sum(r["scored"] for r in rows),
         "n_vocal": n_vocal, "vocal_frac": n_vocal / len(rows), "n_evaluable": len(ev),
         "n_beat": sum(r.get("beat", False) for r in ev),
         "n_vocal_unmatched": sum(r["vocal"] and not r["evaluable"] for r in rows),
         "n_peak_missing": sum(r.get("peak_before_norm") is None for r in ok),
         "n_official_failed": sum(not r["official_ok"] for r in ok),
         "n_not_replicated": sum(not r["replicates_official"] for r in ok),
         "n_mms_failed": sum(r.get("mms_failed", False) for r in rows if r["scored"]),
         "whisper_models": sorted({f"{r.get('whisper')}@{r.get('whisper_revision')}" for r in rows if r["scored"]})}
    s["n_incomplete"] = s["n_planned"] - s["n_scored"]
    if ev:
        e = np.array([r["err_median"] for r in ev])
        s.update(err_median_of_clip_medians=rnd(np.median(e)), beat_frac=s["n_beat"] / len(ev),
                 pooled_within_025=float(sum(r["n_within"] for r in ev) / sum(r["n_matched"] for r in ev)),
                 err_signed_median_of_clips=rnd(np.median([r["err_signed_median"] for r in ev])),
                 random_err_median_of_clips=rnd(np.median([r["random_err_median"] for r in ev])))
        # rank of the alignment set among the random sets on the pooled statistic (a description, not a test:
        # the alignment heads lie outside the random draw space)
        m_rand = np.median(np.array([r["random_err"] for r in ev]), axis=0)
        s["align_rank_among_random"] = float((1 + (m_rand <= s["err_median_of_clip_medians"]).sum()) / (1 + len(m_rand)))
        s["seeds"] = {sd: {"err_median_of_clip_medians": rnd(np.median([r["seeds"][sd]["err_median"] for r in ev])),
                           "beat_frac": float(np.mean([r["seeds"][sd]["beat"] for r in ev]))} for sd in ev[0]["seeds"]}
        s["steps_err_median"] = np.median(np.array([r["steps_err_median"] for r in ev]), axis=0).tolist()
        s["steps_beat_frac"] = np.mean(np.array([r["steps_beat"] for r in ev]), axis=0).tolist()
        single = np.median(np.array([r["single_head_err"] for r in ev]), axis=0)
        order = np.argsort(single, kind="stable")
        s["best_single_heads"] = [[f"L{i // N_HEADS}h{i % N_HEADS}", float(single[i])] for i in order[:20]]
        s["align_single_head_err"] = {f"L{L}h{h}": float(single[L * N_HEADS + h]) for L, hs in ALIGN.items() for h in hs}
        mm = [r["mms_vs_whisper_median"] for r in ev if "mms_vs_whisper_median" in r]
        if mm:
            s["reference_floor_mms_vs_whisper"] = rnd(np.median(mm))
            s["head_vs_mms_median"] = rnd(np.median([r["head_vs_mms_median"] for r in ev if "head_vs_mms_median" in r]))
        excl = [r["err_median_excl_first"] for r in ev if r["err_median_excl_first"] is not None]
        s["err_median_excl_first_word"] = rnd(np.median(excl)) if excl else None
        for name in ("e4", "extra"):
            sub = [r for r in ev if r["set"] == name]
            if sub:
                s[f"{name}_err_median"] = rnd(np.median([r["err_median"] for r in sub]))
                s[f"{name}_beat_frac"] = float(np.mean([r["beat"] for r in sub]))
                s[f"{name}_n_evaluable"] = len(sub)
    s["ci95_descriptive"] = bootstrap(rows)
    return s


def integrity(rows: list[dict], reg: dict | None, code_now: dict, git: dict) -> list[str]:
    """Condition 0, except the reference model and the test flags (see decision)."""
    out = []
    if reg is None:
        out.append("no registered hashes (addendum_2.json, section e6_readout)")
    elif code_now != reg:
        out.append(f"code now differs from the registration: {sorted(k for k in reg if code_now.get(k) != reg[k])}")
    bad = [r["id"] for r in rows if r["status"] == "ok" and (r.get("code_sha256") != reg or not r.get("registered"))]
    if bad:
        out.append(f"{len(bad)} clips generated with unregistered code: {bad[:5]}")
    if git.get("git_tracked_changes") is not False:
        out.append(f"read-out files have uncommitted changes or git state is unknown ({git})")
    return out


def decision(s: dict, rows: list[dict], integrity_reasons: list[str], test_flags: list[str]) -> tuple[list[str], list[str]]:
    """(reasons that block B1, interpretation labels). Pre-specified; thresholds are module constants."""
    reasons = ([f"test run ({', '.join(test_flags)})"] if test_flags else []) + list(integrity_reasons)
    wrong = [r["id"] for r in rows if r["scored"] and (r.get("whisper"), r.get("whisper_revision")) != (WHISPER, WHISPER_REVISION)]
    if wrong:
        reasons.append(f"{len(wrong)} clips scored with another reference than {WHISPER}@{WHISPER_REVISION[:7]}")
    if s["n_peak_missing"]:
        reasons.append(f"pre-normalisation peak missing in {s['n_peak_missing']} clips")
    if s["n_incomplete"] > MAX_INCOMPLETE:
        reasons.append(f"incomplete: {s['n_incomplete']} of {s['n_planned']} planned clips missing, failed or unscored")
    technical = list(reasons)
    c1 = s["n_vocal"] >= MIN_VOCAL_CLIPS
    c2 = s["n_evaluable"] >= MIN_EVAL_CLIPS
    c3 = s.get("err_median_of_clip_medians", np.inf) <= ERR_MAX_S
    c4 = s["n_evaluable"] > 0 and BEAT_DEN * s["n_beat"] >= BEAT_NUM * s["n_evaluable"]
    c5 = s["n_official_failed"] == 0 and s["n_not_replicated"] == 0
    if not c1:
        reasons.append(f"vocals in {s['n_vocal']}/{s['n_planned']} clips (< {MIN_VOCAL_CLIPS})")
    if not c2:
        reasons.append(f"{s['n_evaluable']} evaluable clips (< {MIN_EVAL_CLIPS})")
    if not c3:
        reasons.append(f"median onset error {s.get('err_median_of_clip_medians')} s (> {ERR_MAX_S})")
    if not c4:
        reasons.append(f"alignment heads beat the random sets in {s['n_beat']}/{s['n_evaluable']} clips (< 75%)")
    if not c5:
        reasons.append(f"ACE-Step timestamps failed in {s['n_official_failed']} clips and were not replicated "
                       f"within one frame in {s['n_not_replicated']}")
    labels = []
    if technical or not c5:
        labels.append("technical")
    if not c1:
        labels.append("vocal yield insufficient")
    elif not c2:
        labels.append("too few evaluable clips")
    elif not c3 and s.get("head_vs_mms_median", np.inf) <= ERR_MAX_S:
        labels.append("reference-limited")
    elif not (c3 and c4):
        labels.append("heads do not give word onsets")
    if not reasons:
        labels.append("B1 allowed")
    return reasons, labels


def decide(args) -> None:
    from dacapo.provenance import write_result

    t0 = time.time()
    scores = json.loads((RES / "e6_scores.json").read_text()) if (RES / "e6_scores.json").exists() else {}
    rows = []
    for it in items():
        mp, fp = OUT / f"{it['id']}.json", OUT / f"{it['id']}.failed.json"
        meta = json.loads(mp.read_text()) if mp.exists() else (json.loads(fp.read_text()) if fp.exists() else None)
        z = None
        if mp.exists():
            with np.load(OUT / f"{it['id']}.npz") as zf:  # onset arrays only; the attention stays on disk
                z = {k: zf[k] for k in zf.files if k.startswith("onsets")}
        rows.append(clip_stats(meta, scores.get(it["id"]), z, it))
    s = summarise(rows)
    reg_path = Path(args.registration or REG_DEFAULT)
    reg, code_now, git = registered_hashes(reg_path), code_hashes(), git_state()
    test_flags = [f for f, on in (("--allow-other-reference", args.allow_other_reference),
                                  ("--registration", bool(args.registration))) if on]
    reasons, labels = decision(s, rows, integrity(rows, reg, code_now, git), test_flags)
    allowed = not reasons
    gate = {"b1_allowed": allowed, "reasons": reasons, "labels": labels, "scale": os.environ.get("E6_SCALE", "unknown"),
            "test_flags": test_flags, "registration": str(reg_path), "registered_sha256": reg, "code_sha256": code_now, **git,
            **{k: s[k] for k in ("n_planned", "n_generated", "n_failed", "n_missing", "n_scored", "whisper_models")},
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cfg = {"experiment": "e6_readout", "align_heads": {str(k): v for k, v in ALIGN.items()},
           "random_sets": [{str(k): v for k, v in st.items()} for st in random_head_sets()],
           "n_random": N_RANDOM, "n_random_steps": N_RANDOM_STEPS, "head_seed": HEAD_SEED, "posthoc_seed": POSTHOC_SEED,
           "posthoc_seeds_extra": list(POSTHOC_SEEDS_EXTRA), "steps": STEPS, "shift": SHIFT, "dcw_enabled": False,
           "rng_offset": RNG_OFFSET, "captions": [CAPTION_F, CAPTION_M], "verses": VERSES, "whisper": WHISPER,
           "whisper_revision": WHISPER_REVISION, "test_flags": test_flags,
           "thresholds": {"vocal_share": VOCAL_SHARE_MIN, "vocal_active_s": VOCAL_ACTIVE_MIN_S, "vocal_active_dbfs": VOCAL_ACTIVE_DBFS,
                          "peak_before_norm": PEAK_MIN, "min_matched": MIN_MATCHED, "min_eval_clips": MIN_EVAL_CLIPS,
                          "min_vocal_clips": MIN_VOCAL_CLIPS, "err_max_s": ERR_MAX_S, "beat": f"{BEAT_NUM}/{BEAT_DEN}",
                          "max_incomplete": MAX_INCOMPLETE, "digits": DIGITS}}
    RES.mkdir(parents=True, exist_ok=True)
    write_result(RES / "e6_readout.json", cfg, {"gate": gate, "summary": s, "rows": rows}, t0)
    (RES / "e6_gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps({k: v for k, v in s.items() if not isinstance(v, (list, dict))}, indent=1))
    print("E6 READ-OUT:", "B1 ALLOWED" if allowed else "B1 BLOCKED: " + "; ".join(reasons), "|", ", ".join(labels), flush=True)


def gate_ok(gate: dict | None, reg: dict | None) -> tuple[bool, str]:
    """Run B's check: a full-scale gate from the registered code that allows B1."""
    if gate is None:
        return False, "no gate file"
    for ok, why in ((gate.get("b1_allowed") is True, "gate does not allow B1"),
                    (gate.get("scale") == "full", f"scale is {gate.get('scale')}, not full"),
                    (not gate.get("test_flags"), "gate came from a test run"),
                    (reg is not None and gate.get("code_sha256") == reg == gate.get("registered_sha256"),
                     "gate code differs from the registration"),
                    (gate.get("n_planned") == 88, "gate does not cover the 88 planned clips")):
        if not ok:
            return False, why
    return True, "ok"


def check_gate(args) -> None:
    p = RES / "e6_gate.json"
    ok, why = gate_ok(json.loads(p.read_text()) if p.exists() else None, registered_hashes(REG_DEFAULT))
    print("E6 GATE:", "B1 may run" if ok else f"B1 skipped ({why})", flush=True)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["a0", "gen", "score", "decide", "check-gate"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of clip ids")
    ap.add_argument("--whisper", default=WHISPER)
    ap.add_argument("--registration", default="", help="tests only: another registration file (forces a blocked gate)")
    ap.add_argument("--unregistered", action="store_true", help="tests only: gen/a0 with code that is not registered")
    ap.add_argument("--a0", action="store_true", help="score: the two acceptance clips of `a0` (exit 3 on any error)")
    ap.add_argument("--allow-other-reference", action="store_true", help="tests only (forces a blocked gate)")
    args = ap.parse_args()
    try:
        {"a0": a0, "gen": gen, "score": score, "decide": decide, "check-gate": check_gate}[args.cmd](args)
    except SetupError as exc:
        print(f"E6 {args.cmd}: SETUP FAILED: {exc}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
