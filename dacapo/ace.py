"""Run ACE-Step 1.5 and read or edit the flow-matching trajectory at every step.

The turbo sampler is Euler on x_t = t*noise + (1-t)*x0 with velocity v = noise - x0, so at each
denoiser call x0_hat = x_t - t*v. We wrap `model.decoder.forward` to read x0_hat, and can replace
it with an edited x0' and return v' = (x_t - x0')/t. This needs no extra DiT calls and no change
to ACE-Step's code.

Needs the official ACE-Step-1.5 repo importable from $ACESTEP_REPO.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.environ["ACESTEP_REPO"])

from acestep.constants import TASK_INSTRUCTIONS  # noqa: E402
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import (GenerationConfig, GenerationParams, _candidate_repaint_sidecars,  # noqa: E402
                               generate_music)

from dacapo.seeds import pin_global_rng  # noqa: E402,F401  (re-exported: ace.pin_global_rng)

SR = 48000
FPS = 25  # VAE latent frames per second

EditFn = Callable[[int, float, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]
PreFn = Callable[[int, float, dict], dict]
# combine(step, t, kw, call) -> velocity. `call(kw)` runs the denoiser once with those kwargs and
# returns its velocity, so one step can use several contexts (contrastive guidance).
CombineFn = Callable[[int, float, dict, Callable[[dict], torch.Tensor]], torch.Tensor]


def load(device: str = "cpu", dtype: str = "bf16") -> AceStepHandler:
    h = AceStepHandler()
    status, ok = h.initialize_service(project_root=os.environ["ACESTEP_REPO"], config_path="acestep-v15-turbo",
                                      device=device, use_mlx_dit=False)
    if not ok:
        raise RuntimeError(status)
    if device == "cpu" and dtype == "bf16":  # AMX/AVX512-BF16 CPUs: halves RAM, faster matmuls
        h.model = h.model.to(torch.bfloat16)
        h.text_encoder = h.text_encoder.to(torch.bfloat16)
        h.silence_latent = h.silence_latent.to(torch.bfloat16)
        h.dtype = torch.bfloat16
    return h


class StepHook:
    """Context manager wrapping the DiT decoder.

    edit(step, t, x0_hat, x_t) -> new x0 or None   per-step x0-space intervention
    pre(step, t, kwargs) -> kwargs                 swap conditioning before the call (e.g. context
                                                   latents). Swapping encoder states needs a fresh
                                                   cross-attention cache, see `swap_conditioning`.
    record=True   keep every step's x0_hat (float16, batch item 0)
    capture=True  keep the first call's conditioning tensors (to replay them in another run)
    """

    def __init__(self, h: AceStepHandler, edit: Optional[EditFn] = None, pre: Optional[PreFn] = None,
                 combine: Optional[CombineFn] = None, record: bool = False, capture: bool = False):
        self.decoder = h.model.decoder
        self.edit, self.pre, self.combine, self.record, self.capture = edit, pre, combine, record, capture
        self.trace: list[dict] = []
        self.captured: dict = {}
        self.k = 0
        self.extra_calls = 0
        self.frames: Optional[int] = None  # latent frames T of this run (hidden_states [B, T, 64] at the first call)

    def __enter__(self):
        self._orig = self.decoder.forward
        self.decoder.forward = self._wrapped
        return self

    def __exit__(self, *exc):
        del self.decoder.forward  # drop the instance attribute; the class method is visible again
        return False

    def _wrapped(self, *args, **kw):
        if self.k == 0:
            self.frames = int(kw["hidden_states"].shape[1])
        if self.capture and self.k == 0:
            self.captured = {n: kw[n].detach().clone() for n in
                             ("context_latents", "encoder_hidden_states", "encoder_attention_mask") if kw.get(n) is not None}
        if self.pre is not None:
            kw = self.pre(self.k, float(kw["timestep"][0]), kw)
        out = self._orig(*args, **kw)
        xt, t, v = kw["hidden_states"], kw["timestep"], out[0]
        if self.combine is not None:
            def call(kw2: dict) -> torch.Tensor:
                # extra evaluations must not touch the sampler's cross-attention cache
                from transformers.cache_utils import DynamicCache, EncoderDecoderCache
                kw2 = {n: x for n, x in kw2.items() if n != "_v"}  # `_v` carries the base velocity to the combiner only
                kw2.update(past_key_values=EncoderDecoderCache(DynamicCache(), DynamicCache()), use_cache=True)
                self.extra_calls += 1
                return self._orig(*args, **kw2)[0]
            v = self.combine(self.k, float(t[0]), dict(kw, _v=v), call)
        # x0 arithmetic in float32: in bf16, (x_t - x0)/t amplifies rounding error at small t
        tt = t.view(-1, 1, 1).float()
        x0 = xt.float() - tt * v.float()
        if self.record:
            self.trace.append({"step": self.k, "t": float(t[0]), "x0": x0[0].cpu().numpy().astype(np.float16)})
        if self.edit is not None:
            x0_new = self.edit(self.k, float(t[0]), x0, xt)
            if x0_new is not None and x0_new is not x0:
                v = ((xt.float() - x0_new.float()) / tt.clamp_min(1e-4)).to(v.dtype)
        self.k += 1
        return (v,) + tuple(out[1:])


# Tensors on the 25 Hz latent time axis ([B, T, C], T must match this run's hidden_states length).
# encoder_* tensors are only read by cross-attention, so another run's can have any length.
TIME_ALIGNED = ("context_latents",)


def fit_frames(x: torch.Tensor, n: int, silence_latent: Optional[torch.Tensor] = None, max_diff: int = 2) -> torch.Tensor:
    """Crop or right-pad a captured `context_latents` tensor [B, T, C] to n frames.

    Cover/repaint runs get their length from the source audio (floor(samples / 1920) frames), so
    two sources a few ms apart can give 372 vs 373 frames, and their contexts cannot be swapped
    as-is. Cropping drops the last frames.
    Padding fills frames t..n-1 of the source half (first C/2 channels) with `silence_latent`
    frames t..n-1. ACE-Step's own no-source context (text2music, cover->text hand-off) holds the
    same values there, and they match the latent of a digitally silent guide tail. The chunk-mask
    half (last C/2) repeats the last mask frame of `x`. Returns `x` itself if it already has n frames.
    Raises if the lengths differ by more than `max_diff` frames (80 ms at 25 Hz), so the
    swapped-in melody is never shifted silently.
    """
    t = x.shape[1]
    if t == n:
        return x
    if abs(t - n) > max_diff:
        raise ValueError(f"swapped-in context has {t} frames but this run has {n} (more than {max_diff} apart)")
    if t > n:
        return x[:, :n]
    if silence_latent is None:
        raise ValueError(f"swapped-in context has {t} frames but this run has {n}; pass silence_latent to pad it")
    c = x.shape[-1] // 2
    sil = silence_latent.reshape(-1, silence_latent.shape[-1])
    if sil.shape[-1] != c:
        raise ValueError(f"silence_latent has {sil.shape[-1]} channels, context source half has {c}")
    if sil.shape[0] < n:
        sil = sil.repeat((n + sil.shape[0] - 1) // sil.shape[0], 1)
    src = sil[t:n].to(x.device, x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
    mask = x[:, -1:, c:].expand(-1, n - t, -1)
    return torch.cat([x, torch.cat([src, mask], dim=-1)], dim=1)


def swap_conditioning(from_step: int, cond: dict, silence_latent: Optional[torch.Tensor] = None) -> PreFn:
    """PreFn that uses conditioning tensors captured from another run, from `from_step` on.
    The DiT caches only cross-attention K/V, so a new cache starts at the switch.
    `TIME_ALIGNED` tensors with a different length are cropped or padded by `fit_frames`
    (padding needs `silence_latent`, e.g. `h.silence_latent`). Same-length tensors pass unchanged."""
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    fitted: dict = {}

    def pre(step: int, t: float, kw: dict) -> dict:
        if step < from_step:
            return kw
        kw = dict(kw)
        frames = kw["hidden_states"].shape[1]
        for n, v in cond.items():
            if n in TIME_ALIGNED and v.shape[1] != frames:
                if (n, frames) not in fitted:
                    fitted[(n, frames)] = fit_frames(v, frames, silence_latent)
                v = fitted[(n, frames)]
            kw[n] = v
        if "encoder_hidden_states" in cond and step == from_step:
            kw["past_key_values"] = EncoderDecoderCache(DynamicCache(), DynamicCache())
        return kw
    return pre


def determinism_state() -> dict:
    """Flags that decide kernel choice and reproducibility, for provenance."""
    return {"cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "fill_uninitialized_memory": getattr(getattr(torch.utils, "deterministic", None),
                                                 "fill_uninitialized_memory", None),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "vae_decode_chunk_env": os.environ.get("ACESTEP_VAE_DECODE_CHUNK_SIZE"),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def setup_determinism() -> dict:
    """One determinism configuration for every E5 v2 and E6 process. Call it before the model loads.
    cuBLAS reads CUBLAS_WORKSPACE_CONFIG when its first handle is created, so the variable is set here
    only while CUDA is still uninitialised. The VAE decode chunk is fixed at 256 latent frames (every
    v1 L4 decode used 256; without the variable ACE-Step picks the chunk from free VRAM).
    warn_only=True: an op without a deterministic CUDA kernel warns (generate() records the warnings)."""
    want = ":4096:8"
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (want, ":16:8"):
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be set to {want} before CUDA starts")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = want
    os.environ["ACESTEP_VAE_DECODE_CHUNK_SIZE"] = "256"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return determinism_state()


def runtime_facts(h: AceStepHandler) -> dict:
    """What actually runs: attention kernel family, dtypes, encode chunking, versions, device."""
    from importlib import metadata
    from acestep.gpu_config import get_gpu_memory_gb
    vers = {}
    for p in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "numpy"):
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            vers[p] = None
    cuda = torch.cuda.is_available()
    gpu_gb = float(get_gpu_memory_gb())
    return {"device": str(h.device), "attn_implementation": getattr(h.model.config, "_attn_implementation", None),
            "dit_dtype": str(next(h.model.parameters()).dtype), "handler_dtype": str(h.dtype),
            "vae_dtype": str(h.vae.dtype), "text_encoder_dtype": str(next(h.text_encoder.parameters()).dtype),
            "gpu_memory_gb_for_encode_chunk": gpu_gb,
            "vae_encode_chunk_samples": 48000 * (15 if gpu_gb <= 8 else 30),  # handler vae_encode.py:46-52
            "versions": vers, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_sm_count": torch.cuda.get_device_properties(0).multi_processor_count if cuda else None,
            **determinism_state()}


# ACE-Step log lines that show what a run actually did (one generate() call at a time).
_LOG_MARKERS = ("text_prompt:", "[tiled_decode] chunk_size=", "[DCW]", "auto-enabling CPU VAE decode",
                "falling back to CPU", "OOM", "[repaint_cache]", "BEFORE: Peak=", "Length mismatch")


def parse_run_log(messages: Sequence[str]) -> dict:
    """Facts from ACE-Step's own log messages of one run (inference.py, conditioning_text.py:118-124,
    vae_decode.py:56, generate_music_decode.py:153-163, dcw_correction.py:97-101). On a CPU device
    ACE-Step always logs "auto-enabling CPU VAE decode" (it sees 0 GB of VRAM), so cpu_decode_lines
    is only meaningful on CUDA."""
    text = "\n".join(messages)
    instr = re.findall(r"# Instruction\n(.*)\n", text)
    dur = re.findall(r"- duration: (\d+) seconds", text)
    chunk = re.findall(r"\[tiled_decode\] chunk_size=(\d+).*?latents_shape=torch\.Size\(\[\d+, \d+, (\d+)\]\)", text)
    peak = re.findall(r"BEFORE: Peak=([0-9.]+)", text)
    return {"instruction_logged": instr[0] if instr else None,
            "n_prompts_logged": len(instr),
            "duration_meta": int(dur[0]) if dur else None,
            "decode_chunk": int(chunk[0][0]) if chunk else None,
            "decode_frames": int(chunk[0][1]) if chunk else None,
            "dcw_active_lines": text.count("[DCW] Active"),
            "cpu_decode_lines": sum(text.count(s) for s in ("auto-enabling CPU VAE decode", "falling back to CPU")),
            "oom_lines": text.count("OOM"),
            "repaint_cache_used": text.count("[repaint_cache] Loaded") + text.count("[repaint_cache] Replacing"),
            "peak_before_norm": float(peak[0]) if peak else None}


def generate(h: AceStepHandler, out_wav: str | Path, caption: str, duration: float, seed: int,
             lyrics: str = "[Instrumental]", steps: int = 8, shift: float = 3.0, task: str = "text2music",
             src: str | None = None, codes: str = "", cover_strength: float = 1.0, cover_noise: float = 0.0,
             vocal_language: str = "en", bpm: int | None = None, keyscale: str = "", timesignature: str = "",
             edit: Optional[EditFn] = None, pre: Optional[PreFn] = None, combine: Optional[CombineFn] = None,
             record: bool = False, capture: bool = False, extra: Optional[dict] = None,
             taps: Sequence = (), keep_extra: bool = False, rng_seed: Optional[int] = None,
             out_subtype: Optional[str] = None) -> dict:
    """extra: further GenerationParams fields (e.g. repainting_start/end, chunk_mask_mode, repaint_mode,
    dcw_enabled, instruction).
    out_subtype: write ACE-Step's normalised output with soundfile at this subtype (e.g. "PCM_16"; the
    container follows the suffix of out_wav, .wav or .flac). ACE-Step's own saver goes through torchaudio,
    whose output subtype depends on the installed torchaudio and torchcodec. None keeps that saver (v1).
    task="repaint" gets ACE-Step's repaint instruction (TASK_INSTRUCTIONS["repaint"]) unless `extra` sets
    one. GenerationParams defaults to the text2music instruction for every task; ACE-Step's UI, REST API
    and CLI swap in the task's instruction, its Python API does not. Other tasks keep that default.
    rng_seed: if given, pin_global_rng(rng_seed) runs right before generation (None: global RNG untouched).
    taps: context managers entered around the call (e.g. attn_tap.CrossAttnTap); they must not change
    the forward pass. keep_extra: also return ACE-Step's intermediate tensors (EXTRA_KEYS).
    The result also holds what ACE-Step actually used (resolved instruction and DCW flag, the logged
    prompt instruction and duration, latent frames, decode chunk) and `peak_before_norm`."""
    kw = dict(extra or {})
    if task == "repaint":
        kw.setdefault("instruction", TASK_INSTRUCTIONS["repaint"])
        # A json next to the source with the same stem (or ./gradio_outputs/batch_*/<stem>.json) makes
        # ACE-Step load cached latents instead of encoding the source, and may replace the seed
        # (inference.py:452-504). Only the Gradio UI writes such files.
        found = [p for p in _candidate_repaint_sidecars(str(src)) if os.path.exists(p)] if src else []
        if found:
            raise RuntimeError(f"repaint source has ACE-Step sidecar(s) {found}; move them away")
    instrumental = lyrics.strip() == "[Instrumental]"
    params = GenerationParams(task_type=task, caption=caption, lyrics=lyrics, instrumental=instrumental,
                              vocal_language="unknown" if instrumental else vocal_language,
                              duration=duration, inference_steps=steps, seed=seed, shift=shift, thinking=False,
                              use_cot_caption=False, use_cot_language=False, use_cot_metas=False,
                              bpm=bpm, keyscale=keyscale, timesignature=timesignature,
                              src_audio=src, audio_codes=codes, audio_cover_strength=cover_strength,
                              cover_noise_strength=cover_noise, **kw)
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
    out_wav = Path(out_wav)
    t0 = time.time()
    msgs: list[str] = []
    from loguru import logger  # ACE-Step's logger
    sink = logger.add(lambda m: msgs.append(m.record["message"]), level="DEBUG",
                      filter=lambda rec: any(s in rec["message"] for s in _LOG_MARKERS), format="{message}")
    try:
        with warnings.catch_warnings(record=True) as warned, \
                StepHook(h, edit=edit, pre=pre, combine=combine, record=record, capture=capture) as hook, \
                contextlib.ExitStack() as stack:
            warnings.simplefilter("always")
            for tp in taps:
                stack.enter_context(tp)
            if rng_seed is not None:
                pin_global_rng(rng_seed)
            r = generate_music(h, None, params, cfg, save_dir=None if out_subtype else str(out_wav.parent / "_tmp"))
    finally:
        logger.remove(sink)
    if not r.success:
        raise RuntimeError(r.error)
    if out_subtype:
        a = r.audios[0]
        sf.write(str(out_wav), a["tensor"].float().cpu().numpy().T, int(a.get("sample_rate", SR)), subtype=out_subtype)
    else:
        Path(r.audios[0]["path"]).replace(out_wav)
    det = sorted({str(w.message)[:200] for w in warned if "deterministic" in str(w.message)})
    res = {"seconds": round(time.time() - t0, 1), "n_denoiser_calls": hook.k + hook.extra_calls, "trace": hook.trace,
           "captured": hook.captured, "frames": hook.frames, "dit_seed": seed, "rng_seed": rng_seed,
           "instruction_requested": params.instruction,
           "instruction": h._resolve_generate_music_task(params.task_type, params.audio_codes, params.instruction)[1],
           "dcw_requested": params.dcw_enabled, "dcw_enabled": h._resolve_dcw_enabled(params.dcw_enabled),
           "determinism_warnings": det, **parse_run_log(msgs)}
    if keep_extra:
        res["extra"] = {k: r.extra_outputs.get(k) for k in EXTRA_KEYS}
    return res


# ACE-Step intermediates kept by generate(keep_extra=True) (core/generation/handler/generate_music_payload.py:76-99)
# target_latents: the encoded cover/repaint source as the DiT received it (the VAE posterior sample).
EXTRA_KEYS = ("pred_latents", "target_latents", "encoder_hidden_states", "encoder_attention_mask", "context_latents",
              "lyric_token_idss")


@torch.inference_mode()
def posthoc_lyric_attention(h: AceStepHandler, extra: dict, seed: int = 42, steps: int = 8) -> torch.Tensor:
    """Cross-attention of every decoder layer, [layers, heads, encoder tokens, DiT frames], from the same
    decoder call as ACE-Step's LyricTimestampMixin.get_lyric_timestamp (core/generation/handler/
    lyric_timestamp.py:71-104): the final latent re-noised to t = 1/steps with `seed`, one pass with
    output_attentions=True. Call it outside StepHook. The lyric tokens are encoder positions
    [0, len(lyric_token_ids)); h._extract_lyric_segment gives the pure-lyric slice."""
    pred = extra["pred_latents"][:1].to(h.device, h.dtype)
    ehs = extra["encoder_hidden_states"][:1].to(h.device, h.dtype)
    eam = extra["encoder_attention_mask"][:1].to(h.device, h.dtype)
    ctx = extra["context_latents"][:1].to(h.device, h.dtype)
    t_last = 1.0 / steps
    t = torch.tensor([t_last], device=pred.device, dtype=pred.dtype)
    xt = t_last * h._sample_noise_like(pred, seed) + (1.0 - t_last) * pred
    am = torch.ones(1, pred.shape[1], device=pred.device, dtype=pred.dtype)
    with h._load_model_context("model"):
        out = h.model.decoder(hidden_states=xt, timestep=t, timestep_r=t, attention_mask=am, encoder_hidden_states=ehs,
                              use_cache=False, past_key_values=None, encoder_attention_mask=eam, context_latents=ctx,
                              output_attentions=True, custom_layers_config=h.custom_layers_config, enable_early_exit=True)
    return torch.stack([a[0].transpose(-1, -2) for a in out[2] if a is not None])


@torch.inference_mode()
def encode(h: AceStepHandler, wav: str | Path, deterministic: bool = True) -> np.ndarray:
    """Audio file -> [T, 64] VAE latent at 25 Hz (posterior mean if deterministic, else a sample,
    which is what ACE-Step uses for cover/repaint sources)."""
    audio = h.process_src_audio(str(wav))  # [2, N] at 48 kHz
    vae = h.vae
    dist = vae.encode(audio.unsqueeze(0).to(vae.device, vae.dtype)).latent_dist
    lat = dist.mean if deterministic else dist.sample()
    return lat[0].T.float().cpu().numpy()


@torch.inference_mode()
def decode(h: AceStepHandler, latent: np.ndarray, out_wav: str | Path | None = None) -> np.ndarray:
    """[T, 64] latent -> [N, 2] waveform at 48 kHz (optionally written to disk)."""
    vae = h.vae
    x = torch.tensor(latent.T[None], dtype=vae.dtype, device=vae.device)
    wav = vae.decode(x).sample[0].float().cpu().numpy().T
    if out_wav is not None:
        sf.write(str(out_wav), wav, SR)
    return wav
