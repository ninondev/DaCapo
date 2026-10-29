"""The scorer that Run B's B1 and B2 share.

Scorer.measure runs E6's measurement on one clip: htdemucs stem shares and vocal-stem activity, then
Whisper large-v3 word chunks on the full mix. These statements are copied unchanged from readout.score
(experiments/e6_clock/tests/test_b1_scorer.py compares their syntax trees), and the thresholds come from
readout.py. The functions below it apply E6's vocal gate, the registered in-order word matching
(dacapo.words.match_words over Whisper's normalised word chunks), ordered recall and the onset error
against the written notes. measure also returns the vocal stem, for B2's pitch measures.

The registered scorer runs Whisper large-v3 at revision 06f233fe06e710322aca913c1bc4249a0d71fce1 in fp16
on CUDA, and raises on the CPU, on any other model and on any other revision. Only a test run (the tiny
CPU scale) may name another Whisper model; its rows carry a test flag, and every verdict computed from
them is "test only".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readout import (DIGITS, ERR_MAX_S, MIN_MATCHED, PEAK_MIN, VOCAL_ACTIVE_DBFS, VOCAL_ACTIVE_MIN_S,  # noqa: E402
                     VOCAL_SHARE_MIN, WHISPER, WHISPER_REVISION, rnd)

from dacapo.words import match_words, norm_word  # noqa: E402


class Scorer:
    """htdemucs and Whisper, loaded once. test_whisper: another Whisper model for a test run (any device)."""

    def __init__(self, dev: str = "cuda", test_whisper: str | None = None):
        if test_whisper is None and dev != "cuda":
            raise RuntimeError("the registered scorer runs Whisper large-v3 in fp16 on CUDA; a CPU run is a test run "
                               "and must name its test model")
        import torch
        from demucs.pretrained import get_model
        from transformers import pipeline

        whisper = test_whisper or WHISPER
        sep = get_model("htdemucs").to(dev).eval()
        asr = pipeline("automatic-speech-recognition", model=whisper, device=0 if dev == "cuda" else -1,
                       revision=WHISPER_REVISION if whisper == WHISPER else None,
                       torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
        asr_rev = getattr(asr.model.config, "_commit_hash", None)
        kw = {} if whisper.endswith(".en") else {"generate_kwargs": {"language": "en", "task": "transcribe"}}
        if test_whisper is None and asr_rev != WHISPER_REVISION:
            raise RuntimeError(f"{WHISPER} loaded at revision {asr_rev}, registered {WHISPER_REVISION}")
        self.dev, self.sep, self.asr, self.kw = dev, sep, asr, kw
        self.whisper, self.whisper_revision = whisper, asr_rev
        self.test_flags = [] if test_whisper is None else [f"test whisper {test_whisper} on {dev}"]
        self.stem_sr = sep.samplerate

    def measure(self, wav) -> tuple[dict, np.ndarray]:
        """(row, vocal stem): the row holds the stem energy shares, the vocal-stem activity in seconds, the
        Whisper text and word chunks [text, start s, end s]; the stem is mono float32 at self.stem_sr."""
        import librosa
        import soundfile as sf
        import torch
        from demucs.apply import apply_model

        sep, asr, kw, dev = self.sep, self.asr, self.kw, self.dev
        t0 = time.time()
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
        row = {"stem_share": {k: v / tot for k, v in energy.items()}, "vocal_active_s": active,
               "whisper": self.whisper, "whisper_revision": self.whisper_revision, "whisper_text": out["text"],
               "whisper_chunks": chunks, "separator": "htdemucs", "test_flags": list(self.test_flags),
               "seconds": round(time.time() - t0, 2)}
        return row, voc


def registered_reference(row: dict) -> bool:
    """The clip was scored with Whisper large-v3 at the registered revision and without a test flag."""
    return (row.get("whisper"), row.get("whisper_revision")) == (WHISPER, WHISPER_REVISION) and not row.get("test_flags")


def vocal(row: dict, peak) -> bool:
    """E6's vocal gate (readout.clip_stats). peak: the clip's pre-normalisation peak; None fails."""
    share = row["stem_share"]["vocals"]
    vocal = share >= VOCAL_SHARE_MIN and row["vocal_active_s"] >= VOCAL_ACTIVE_MIN_S and peak is not None and peak >= PEAK_MIN
    return bool(vocal)


def words_of(row: dict) -> list[tuple[str, float]]:
    """Whisper's word chunks as (normalised word, onset s); empty words and missing onsets are dropped,
    as in readout.clip_stats."""
    hyp_all = [(norm_word(c[0]), c[1]) for c in row["whisper_chunks"]]
    hyp = [(w, t) for w, t in hyp_all if w and t is not None]
    return hyp


def matches(lyric: list[str], hyp: list[tuple[str, float]]) -> list[tuple[int, int]]:
    """(lyric index, Whisper index) pairs of the registered in-order exact matching."""
    return match_words(lyric, [w for w, _ in hyp])


def ordered_recall(lyric: list[str], hyp: list[tuple[str, float]], positions=None) -> float:
    """Share of the lyric words (or of the lyric positions given) that the in-order matching finds."""
    pos = list(range(len(lyric)) if positions is None else positions)
    if not pos:
        raise ValueError("no lyric positions")
    hit = {a for a, _ in matches(lyric, hyp)}
    return sum(i in hit for i in pos) / len(pos)


def onset_error(lyric: list[str], hyp: list[tuple[str, float]], written: list[float]) -> dict:
    """Whisper onset minus written onset over the matched words, in seconds rounded to 1e-6 as in E6:
    the median absolute error, the median signed error (positive: sung late), the share within 0.25 s
    and the median without the first matched word. written: the written onset of every lyric word."""
    pairs = matches(lyric, hyp)
    if len(written) != len(lyric):
        raise ValueError(f"{len(written)} written onsets for {len(lyric)} lyric words")
    if not pairs:
        return {"n_matched": 0, "err_median": None, "err_signed_median": None, "within_025": None,
                "err_median_excl_first": None}
    signed = np.array([hyp[b][1] - written[a] for a, b in pairs], dtype=float)
    err = np.round(np.abs(signed), DIGITS)
    return {"n_matched": len(pairs), "err_median": rnd(np.median(err)), "err_signed_median": rnd(np.median(signed)),
            "within_025": float((err <= ERR_MAX_S).mean()),
            "err_median_excl_first": rnd(np.median(err[1:])) if len(err) > 1 else None}


def clip(row: dict | None, peak, lyric: list[str], written: list[float]) -> dict:
    """One clip as B1 counts it. row None or with an error: unscored, so recall 0, gate failed and not
    evaluable. Evaluable: vocal and at least MIN_MATCHED lyric words matched (E6's rule)."""
    if row is None or "error" in row:
        return {"scored": False, "vocal": False, "recall": 0.0, "n_matched": 0, "evaluable": False, "err_median": None}
    hyp = words_of(row)
    v = vocal(row, peak)
    out = {"scored": True, "vocal": v, "recall": ordered_recall(lyric, hyp), **onset_error(lyric, hyp, written),
           "n_whisper_words": len(hyp), "vocal_share": row["stem_share"]["vocals"], "vocal_active_s": row["vocal_active_s"],
           "registered_reference": registered_reference(row)}
    out["evaluable"] = bool(v and out["n_matched"] >= MIN_MATCHED)
    if not out["evaluable"]:
        out["err_median"] = None  # a clip that is not evaluable has no error
    return out
