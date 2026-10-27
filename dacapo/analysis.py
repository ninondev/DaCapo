"""Audio -> pitch / notes. Every extractor here is noisy. Check its measured error in
experiments/e0_ruler (audio with known notes) before you trust it on generated music."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from .render import load_mono

HOP = 0.01


def f0_crepe(path: str | Path, model: str = "full", periodicity_threshold: float = 0.5,
             fmin: float = 65.0, fmax: float = 1500.0) -> tuple[np.ndarray, np.ndarray]:
    """Frame-level f0 at 10 ms (Hz, 0 = unvoiced). CREPE is monophonic: on a mix it follows
    the strongest pitch, and we need to measure this error."""
    import librosa
    import torch
    import torchcrepe

    x, sr = load_mono(path)
    x = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
    audio = torch.tensor(x, dtype=torch.float32)[None]
    f0, per = torchcrepe.predict(audio, 16000, hop_length=160, fmin=fmin, fmax=fmax, model=model,
                                 return_periodicity=True, batch_size=512, device="cpu")
    per = torchcrepe.filter.median(per, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    f0 = f0[0].numpy()
    f0[per[0].numpy() < periodicity_threshold] = 0.0
    times = np.arange(len(f0)) * HOP
    return times, f0


def _to_grid(times: np.ndarray, f0: np.ndarray, total: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-frame resample of an f0 track onto the shared 10 ms grid."""
    grid = np.arange(0.0, total, HOP)
    idx = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    return grid, f0[idx]


def f0_swift(path: str | Path, fmin: float = 65.0, fmax: float = 1500.0) -> tuple[np.ndarray, np.ndarray]:
    """SwiftF0 (MIT, ONNX): ~100x faster than CREPE on CPU; voiced when confidence >= 0.5."""
    from swift_f0 import SwiftF0

    x, sr = load_mono(path)
    r = SwiftF0().detect(x.astype(np.float32), sr, fmin=fmin, fmax=fmax)
    f0 = np.where(np.asarray(r.confidence) >= 0.5, np.asarray(r.pitch_hz), 0.0)
    return _to_grid(np.asarray(r.timestamps), f0, len(x) / sr)


@lru_cache(maxsize=1)
def _fcpe():
    import torchfcpe
    return torchfcpe.spawn_bundled_infer_model(device="cpu")


def f0_fcpe(path: str | Path, fmin: float = 65.0, fmax: float = 1500.0, threshold: float = 0.006) -> tuple[np.ndarray, np.ndarray]:
    """FCPE (MIT): fast, robust in noise; 10 ms hop at 16 kHz."""
    import librosa
    import torch

    x, sr = load_mono(path)
    x = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
    n = len(x) // 160 + 1
    with torch.no_grad():
        f0 = _fcpe().infer(torch.tensor(x, dtype=torch.float32)[None, :, None], sr=16000, decoder_mode="local_argmax",
                           threshold=threshold, f0_min=fmin, f0_max=fmax, interp_uv=False, output_interp_target_length=n)
    f0 = f0.squeeze().numpy()
    return np.arange(len(f0)) * HOP, f0


F0_METHODS = {"crepe": f0_crepe, "swift": f0_swift, "fcpe": f0_fcpe}


def chroma_argmax(path: str | Path, fmin: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
    """Strongest pitch class per 10 ms frame, from an STFT chromagram with bins below `fmin`
    removed (drops the bass and most low accompaniment). Same melody-accuracy proxy as
    Music ControlNet / MuseControlLite-style evaluations. Chance is not 1/12 when the accompaniment
    shares pitch classes with the melody, so always compare with a mismatched reference."""
    import librosa

    x, sr = load_mono(path)
    x = librosa.resample(x, orig_sr=sr, target_sr=22050) if sr != 22050 else x
    S = np.abs(librosa.stft(x, n_fft=4096, hop_length=220)) ** 2
    freqs = librosa.fft_frequencies(sr=22050, n_fft=4096)
    S[freqs < fmin] = 0.0
    chroma = librosa.feature.chroma_stft(S=S, sr=22050, tuning=0.0)
    pc = chroma.argmax(axis=0).astype(float)
    pc[chroma.max(axis=0) <= 0] = -1  # silence
    return np.arange(len(pc)) * 220 / 22050, pc


def f0(path: str | Path, method: str = "swift") -> tuple[np.ndarray, np.ndarray]:
    return F0_METHODS[method](path)


@lru_cache(maxsize=1)
def _bp_model():
    logging.getLogger().setLevel(logging.ERROR)
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model
    return Model(build_icassp_2022_model_path(FilenameSuffix.onnx))


def notes_basic_pitch(path: str | Path, min_note_ms: float = 80.0) -> list[tuple[float, float, int]]:
    """Polyphonic note transcription -> [(onset_s, offset_s, midi_pitch)]."""
    from basic_pitch.inference import predict
    import contextlib, io

    with contextlib.redirect_stdout(io.StringIO()):
        _, midi, _ = predict(str(path), _bp_model(), minimum_note_length=min_note_ms)
    return sorted((n.start, n.end, n.pitch) for inst in midi.instruments for n in inst.notes)


def skyline(notes: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    """Crude melody guess from polyphonic notes: keep a note only if no higher note is sounding
    at its onset. A baseline heuristic, not a melody extractor."""
    out = []
    for s, e, p in notes:
        if not any(s2 <= s + 0.03 and e2 > s + 0.03 and p2 > p for s2, e2, p2 in notes if (s2, e2, p2) != (s, e, p)):
            out.append((s, e, p))
    return out
