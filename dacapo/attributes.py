"""Attribute-level similarity between two versions of 'the same' music.

Shows which attribute an intervention changed, or at which denoising step it stopped changing:
pitch, harmony (pitch classes), rhythm, loudness contour and timbre. Each score is 1.0 for
identical audio and falls toward a chance level. Measure that level (e.g. against an unrelated
clip), do not assume it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .render import load_mono

SR = 22050
HOP = 220  # ~10 ms


def _load(path: str | Path) -> np.ndarray:
    import librosa

    x, sr = load_mono(path)
    return librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def features(path: str | Path) -> dict:
    import librosa

    from .analysis import f0_swift

    x = _load(path)
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=HOP))
    _, f0 = f0_swift(path)
    return {
        "f0": f0,  # 10 ms grid, Hz, 0 = unvoiced
        "chroma": librosa.feature.chroma_stft(S=S ** 2, sr=SR, tuning=0.0),
        "onset": librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=SR, hop_length=HOP),
        "rms": librosa.feature.rms(S=S)[0],
        "mfcc": librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=SR)), n_mfcc=20)[1:],
    }


def similarity(fa: dict, fb: dict) -> dict:
    n = min(len(fa["f0"]), len(fb["f0"]))
    a, b = fa["f0"][:n], fb["f0"][:n]
    both = (a > 0) & (b > 0)
    pitch = float((np.abs(1200 * np.log2(a[both] / b[both])) < 50).mean()) if both.sum() > 20 else float("nan")
    m = min(fa["chroma"].shape[1], fb["chroma"].shape[1])
    ca, cb = fa["chroma"][:, :m], fb["chroma"][:, :m]
    na, nb = np.linalg.norm(ca, axis=0), np.linalg.norm(cb, axis=0)
    live = (na > 1e-6) & (nb > 1e-6)  # silent frames have no pitch content to compare
    chroma = float(np.mean(np.sum(ca * cb, 0)[live] / (na[live] * nb[live]))) if live.any() else float("nan")
    ma, mb = fa["mfcc"].mean(1), fb["mfcc"].mean(1)
    return {
        "pitch_agree": pitch,  # frames voiced in both whose f0 agrees within 50 cents
        "voicing_agree": float(((a > 0) == (b > 0)).mean()),
        "chroma_cos": chroma,
        "onset_corr": _corr(fa["onset"], fb["onset"]),
        "rms_corr": _corr(fa["rms"], fb["rms"]),
        "timbre_cos": float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-9)),
    }
