"""Run ACE-Step 1.5 and read or edit the flow-matching trajectory at every step.

The turbo sampler is Euler on x_t = t*noise + (1-t)*x0 with velocity v = noise - x0, so at each
denoiser call x0_hat = x_t - t*v. We wrap `model.decoder.forward` to read x0_hat, and can replace
it with an edited x0' and return v' = (x_t - x0')/t. This needs no extra DiT calls and no change
to ACE-Step's code.

Needs the official ACE-Step-1.5 repo importable from $ACESTEP_REPO.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.environ["ACESTEP_REPO"])

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

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

    def __enter__(self):
        self._orig = self.decoder.forward
        self.decoder.forward = self._wrapped
        return self

    def __exit__(self, *exc):
        del self.decoder.forward  # drop the instance attribute; the class method is visible again
        return False

    def _wrapped(self, *args, **kw):
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


def generate(h: AceStepHandler, out_wav: str | Path, caption: str, duration: float, seed: int,
             lyrics: str = "[Instrumental]", steps: int = 8, shift: float = 3.0, task: str = "text2music",
             src: str | None = None, codes: str = "", cover_strength: float = 1.0, cover_noise: float = 0.0,
             vocal_language: str = "en", bpm: int | None = None, keyscale: str = "", timesignature: str = "",
             edit: Optional[EditFn] = None, pre: Optional[PreFn] = None, combine: Optional[CombineFn] = None,
             record: bool = False, capture: bool = False, extra: Optional[dict] = None) -> dict:
    """extra: further GenerationParams fields (e.g. repainting_start/end, chunk_mask_mode, repaint_mode)."""
    instrumental = lyrics.strip() == "[Instrumental]"
    params = GenerationParams(task_type=task, caption=caption, lyrics=lyrics, instrumental=instrumental,
                              vocal_language="unknown" if instrumental else vocal_language,
                              duration=duration, inference_steps=steps, seed=seed, shift=shift, thinking=False,
                              use_cot_caption=False, use_cot_language=False, use_cot_metas=False,
                              bpm=bpm, keyscale=keyscale, timesignature=timesignature,
                              src_audio=src, audio_codes=codes, audio_cover_strength=cover_strength,
                              cover_noise_strength=cover_noise, **(extra or {}))
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
    out_wav = Path(out_wav)
    t0 = time.time()
    with StepHook(h, edit=edit, pre=pre, combine=combine, record=record, capture=capture) as hook:
        r = generate_music(h, None, params, cfg, save_dir=str(out_wav.parent / "_tmp"))
    if not r.success:
        raise RuntimeError(r.error)
    Path(r.audios[0]["path"]).replace(out_wav)
    return {"seconds": round(time.time() - t0, 1), "n_denoiser_calls": hook.k + hook.extra_calls, "trace": hook.trace,
            "captured": hook.captured}


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
