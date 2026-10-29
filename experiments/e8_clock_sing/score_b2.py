"""Scores of B2 clips (main environment), registered in addendum 3. One scorer computes everything.

measure() runs clock_score.Scorer (E6's htdemucs stems and Whisper large-v3 word chunks on the full mix),
then SwiftF0 on the htdemucs vocal stem and on the mix with E4's note pitch accuracy (score_sing.pitch_scores:
confidence >= 0.5, 65-1500 Hz, one global lag within +-0.5 s from frame_metrics, the median f0 over each
note's central 60%, +-50 cents), pitch accuracy against the other melodies of the set (chance), the pinned
CTC model's transcript of the mix, and the guide-copy index. Its row holds raw measurements only;
clip() turns a row into the registered per-clip values, so select and decide recompute everything from rows.

  q = gate x min(NPA_v, WR)     joint: q >= 0.5     strict: q >= 0.75
  NPA_v   note pitch accuracy on the vocal stem (absolute octave; unvoiced notes are wrong)
  WR      lyric words matched in order (dacapo.words.match_words) to Whisper's word chunks / lyric words
  gate    E6's vocal gate (clock_score.vocal)
"""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "e6_clock"))
import clock_score as CS  # noqa: E402

from dacapo.metrics import levenshtein  # noqa: E402
from dacapo.words import lyric_words, norm_word  # noqa: E402

CTC = "facebook/wav2vec2-large-960h-lv60-self"
CTC_REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
JOINT, STRICT = 0.5, 0.75
DIGITS = CS.DIGITS
# vowel tokens: ooh, oh, ah, la, mm, hmm and their spellings (oo, oooh, ohh, aah, lala, mmm, hmmm)
VOWEL = re.compile(r"(o+h*|a+h+|(la)+|m+|h+m+)")
VOWEL_SHARE = 0.5
# what Whisper large-v3 wrote on E4's wordless clips (results/gpu_run1/e4_sweep.json), longest first
HALLUCINATIONS = ("thank you for watching", "transcript emily beynon", "satsang with mooji", "thank you", "oh my god",
                  "bf watch tv", "bells chiming", "oh god", "oh yeah", "so", "um")
GUIDE_COPY_LAG_S, MEL_HOP_S, MEL_SR, N_MELS = 0.5, 0.01, 16000, 64
BLEED_SHARE = 0.5


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def e4_score():
    """E4's scorer module (experiments/e4_sing/score_sing.py), imported by path, never run."""
    return _load("e4_score_sing", ROOT / "experiments" / "e4_sing" / "score_sing.py")


def rnd(x) -> float:
    return round(float(x), DIGITS)


# text measures

def tokens(text: str | None) -> list[str]:
    return [w for w in (norm_word(x) for x in re.findall(r"[a-z0-9']+", (text or "").lower())) if w]


def vowel_only(text: str | None) -> bool:
    """Transcript empty, or at least half of its words are vowel sounds."""
    t = tokens(text)
    return not t or sum(bool(VOWEL.fullmatch(w)) for w in t) / len(t) >= VOWEL_SHARE


def hallucinated(text: str | None) -> bool:
    """The transcript is one or more registered E4 hallucination strings, with nothing else but vowel sounds."""
    t = " " + " ".join(tokens(text)) + " "
    hit = False
    for h in HALLUCINATIONS:
        while f" {h} " in t:
            t, hit = t.replace(f" {h} ", " ", 1), True
    rest = t.split()
    return hit and all(VOWEL.fullmatch(w) for w in rest)


def wordless(text: str | None) -> bool:
    return vowel_only(text) or hallucinated(text)


def multiset_wr(words: list[str], hyp: str | None) -> float:
    """E4's word recall (figures/make_figures.py word_recall): multiset match, case-insensitive."""
    have, want = Counter(re.findall(r"[a-z']+", (hyp or "").lower())), Counter(w.lower() for w in words)
    return sum(min(c, have[w]) for w, c in want.items()) / sum(want.values())


def wer_capped(words: list[str], hyp: str | None) -> float:
    """E4's WER on normalised words (score_sing.norm_words), capped at 1."""
    h = re.sub(r"[^a-z' ]", " ", (hyp or "").lower()).split()
    return min(levenshtein([w.lower() for w in words], h) / max(1, len(words)), 1.0)


def halves(n: int) -> tuple[list[int], list[int]]:
    """Word positions of the first and second half (the middle word of an odd line goes to the second)."""
    return list(range(n // 2)), list(range(n // 2, n))


def ctc_words(text: str | None) -> list[tuple[str, None]]:
    return [(w, None) for w in tokens(text)]


# audio measures

def guide_copy_index(voc: np.ndarray, sr: int, guide: Path) -> float | None:
    """Best Pearson correlation, over lags within +-0.5 s, between the log-mel spectrograms of the vocal
    stem and the guide, on the frames where the guide sounds (within 40 dB of its loudest frame)."""
    import librosa
    import soundfile as sf

    g, gsr = sf.read(str(guide), always_2d=True, dtype="float32")

    def logmel(x, s):
        y = librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=s, target_sr=MEL_SR)
        m = librosa.feature.melspectrogram(y=y, sr=MEL_SR, n_fft=1024, hop_length=int(MEL_HOP_S * MEL_SR), n_mels=N_MELS)
        return np.log(m + 1e-8)

    a, b = logmel(voc, sr), logmel(g.mean(1), gsr)
    n = min(a.shape[1], b.shape[1])
    e = b[:, :n].max(0)
    on = np.where(e >= e.max() - np.log(1e4))[0]   # 40 dB in power
    best = None
    lag_max = int(round(GUIDE_COPY_LAG_S / MEL_HOP_S))
    for lag in range(-lag_max, lag_max + 1):
        idx = on + lag
        ok = (idx >= 0) & (idx < a.shape[1])
        if ok.sum() < 10:
            continue
        x, y = a[:, idx[ok]].ravel(), b[:, on[ok]].ravel()
        if x.std() == 0 or y.std() == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        best = c if best is None else max(best, c)
    return None if best is None else rnd(best)


class B2Scorer:
    """clock_score.Scorer plus the pinned CTC model. test_whisper: another Whisper for a test run (the tiny
    CPU scale only); clock_score refuses the CPU without it."""

    def __init__(self, dev: str = "cuda", test_whisper: str | None = None):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.base = CS.Scorer(dev, test_whisper=test_whisper)
        self.dev = dev
        self.ctc_proc = Wav2Vec2Processor.from_pretrained(CTC, revision=CTC_REVISION)
        self.ctc = Wav2Vec2ForCTC.from_pretrained(CTC, revision=CTC_REVISION).to(dev).eval()
        self.ctc_revision = getattr(self.ctc.config, "_commit_hash", None)
        if self.ctc_revision != CTC_REVISION:
            raise RuntimeError(f"{CTC} loaded at revision {self.ctc_revision}, registered {CTC_REVISION}")
        self.e4 = e4_score()
        self.test_flags = list(self.base.test_flags) + ([f"CTC on {dev}"] if dev != "cuda" else [])
        self.torch = torch

    def measure(self, wav: Path, item: dict, others: list[dict], guide: Path | None) -> dict:
        """The raw row of one clip. guide: the run's guide WAV (for noguide runs, the item's Voice Oohs
        guide, as a reference level for the guide-copy index)."""
        import librosa
        import soundfile as sf

        from dacapo.analysis import f0_swift

        row, voc = self.base.measure(wav)
        t0 = time.time()
        sr = self.base.stem_sr
        mel = self.e4.melody(item)
        with tempfile.TemporaryDirectory() as td:
            stem = Path(td) / "vocals.wav"
            sf.write(str(stem), voc.astype(np.float32), sr, subtype="FLOAT")
            et, ef = f0_swift(stem)
        own = self.e4.pitch_scores(et, ef, mel)
        chance = [self.e4.pitch_scores(et, ef, self.e4.melody(o))["npa"] for o in others]
        mt, mf = f0_swift(wav)
        mix = self.e4.pitch_scores(mt, mf, mel)
        mix_chance = [self.e4.pitch_scores(mt, mf, self.e4.melody(o))["npa"] for o in others]
        x, xsr = sf.read(str(wav), always_2d=True, dtype="float32")
        x16 = librosa.resample(x.mean(1), orig_sr=xsr, target_sr=16000)
        with self.torch.no_grad():
            iv = self.ctc_proc(x16, sampling_rate=16000, return_tensors="pt").input_values.to(self.dev)
            ids = self.ctc(iv).logits.argmax(-1)
        ctc_text = self.ctc_proc.batch_decode(ids.cpu())[0]
        row.update(npa_v=float(own["npa"]), npa_v_oct=float(own["npa_oct"]), voiced_notes_v=float(own["voiced_notes"]),
                   npa_v_chance=float(np.mean(chance)) if chance else None, npa_mix=float(mix["npa"]),
                   npa_mix_oct=float(mix["npa_oct"]), npa_mix_chance=float(np.mean(mix_chance)) if mix_chance else None,
                   voiced_frac_v=float((ef > 0).mean()), ctc=CTC, ctc_revision=self.ctc_revision, ctc_text=ctc_text,
                   guide_copy=guide_copy_index(voc, sr, guide) if guide is not None else None,
                   guide_copy_ref=None if guide is None else guide.name,
                   test_flags=list(self.test_flags), seconds_b2=round(time.time() - t0, 2))
        return row


# per clip (pure)

MISSING = {"scored": False, "gate": False, "npa_v": 0.0, "wr": 0.0, "q": 0.0, "joint": False, "strict": False,
           "wr_ctc": 0.0, "q_ctc": 0.0, "npa_mix": 0.0}


def clip(row: dict | None, peak, item: dict) -> dict:
    """The registered per-clip values. row None or with an error: unscored, so q 0, gate failed, WR 0 and
    NPA 0; the rates of vowel-only, wordless, bleed and the guide-copy index then leave the clip out."""
    if row is None or "error" in row:
        return dict(MISSING)
    lyric = lyric_words(item["lyrics"])
    hyp = CS.words_of(row)
    gate = CS.vocal(row, peak)
    wr = CS.ordered_recall(lyric, hyp)
    first, second = halves(len(lyric))
    npa_v = float(row["npa_v"])
    q = rnd((1.0 if gate else 0.0) * min(npa_v, wr))
    wr_ctc = CS.ordered_recall(lyric, ctc_words(row.get("ctc_text")))
    q_ctc = rnd((1.0 if gate else 0.0) * min(npa_v, wr_ctc))
    on = CS.onset_error(lyric, hyp, written(item))
    ok_on = on["n_matched"] >= CS.MIN_MATCHED
    words = item.get("words") or lyric
    return {"scored": True, "gate": bool(gate), "npa_v": rnd(npa_v), "wr": rnd(wr), "q": q, "joint": q >= JOINT,
            "strict": q >= STRICT, "npa_v_oct": rnd(row["npa_v_oct"]), "npa_v_chance": row.get("npa_v_chance"),
            "npa_mix": rnd(row["npa_mix"]), "npa_mix_oct": rnd(row["npa_mix_oct"]),
            "npa_mix_chance": row.get("npa_mix_chance"),
            "wr_multiset": rnd(multiset_wr(words, row.get("whisper_text"))),
            "wer_capped": rnd(wer_capped(words, row.get("whisper_text"))),
            "recall_first_half": rnd(CS.ordered_recall(lyric, hyp, first)),
            "recall_second_half": rnd(CS.ordered_recall(lyric, hyp, second)),
            "recall_last_word": rnd(CS.ordered_recall(lyric, hyp, [len(lyric) - 1])),
            "n_matched": on["n_matched"], "onset_err_abs": on["err_median"] if ok_on else None,
            "onset_err_signed": on["err_signed_median"] if ok_on else None,
            "vowel_only": vowel_only(row.get("whisper_text")), "wordless": wordless(row.get("whisper_text")),
            "guide_copy": row.get("guide_copy"), "bleed": (1.0 - row["stem_share"]["vocals"]) > BLEED_SHARE,
            "vocal_share": rnd(row["stem_share"]["vocals"]), "wr_ctc": rnd(wr_ctc), "q_ctc": q_ctc,
            "registered_reference": CS.registered_reference(row),
            "ctc_pinned": (row.get("ctc"), row.get("ctc_revision")) == (CTC, CTC_REVISION)}


def written(item: dict) -> list[float]:
    spb = 60.0 / float(item["melody"]["bpm"])
    return [rnd(n["start"] * spb) for n in item["melody"]["notes"]]


def kept(draws: list[dict]) -> dict:
    """The selector: the draw with the largest q; ties go to higher WR, then higher NPA_v, then the lower
    draw index. draws: per-clip values with a "draw" key."""
    return max(draws, key=lambda c: (c["q"], c["wr"], c["npa_v"], -c["draw"]))
