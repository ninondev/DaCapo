"""Scores of C1 clips (main environment), registered in addendum 4. One scorer computes everything.

C1Scorer is B2's scorer (experiments/e8_clock_sing/score_b2.py, imported and never edited) with one more
measure: the pinned CTC model (facebook/wav2vec2-large-960h-lv60-self at 54074b1c...) on the htdemucs vocal
stem, decoded exactly as B2 decodes the mix (resampled to 16 kHz mono, fp32, greedy argmax). The stem is the
one B2's scorer separates for NPA_v: the wrapper keeps the stem that clock_score.Scorer.measure returns, so
htdemucs runs once per clip.

clip() is score_b2.clip with two more values, reported and never in the rule:
  wr_ctc_stem   lyric words matched in order to the CTC transcript of the vocal stem / lyric words
  q_ctc_stem    gate x min(NPA_v, wr_ctc_stem)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments" / "e8_clock_sing", ROOT / "experiments" / "e6_clock"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import score_b2 as S  # noqa: E402

from dacapo.words import lyric_words  # noqa: E402

CS = S.CS
CTC, CTC_REVISION = S.CTC, S.CTC_REVISION
JOINT, STRICT, DIGITS = S.JOINT, S.STRICT, S.DIGITS
CTC_SR = 16000
MISSING = {**S.MISSING, "wr_ctc_stem": 0.0, "q_ctc_stem": 0.0}
kept = S.kept
rnd = S.rnd


class C1Scorer(S.B2Scorer):
    """B2Scorer plus the CTC transcript of the vocal stem. test_whisper: another Whisper for a test run (the tiny
    CPU scale only); clock_score refuses the CPU without it."""

    def __init__(self, dev: str = "cuda", test_whisper: str | None = None):
        super().__init__(dev, test_whisper=test_whisper)
        self.keep_stem()

    def keep_stem(self) -> None:
        """Wrap the base scorer's measure so the vocal stem it returns stays available after B2's measure."""
        inner = self.base.measure

        def measure(wav):
            row, voc = inner(wav)
            self._stem = voc
            return row, voc
        self._stem = None
        self.base.measure = measure

    def transcribe(self, x: np.ndarray, sr: int) -> str:
        """The pinned CTC model on mono audio, as score_b2.B2Scorer.measure runs it on the mix."""
        import librosa

        x16 = librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=sr, target_sr=CTC_SR)
        with self.torch.no_grad():
            iv = self.ctc_proc(x16, sampling_rate=CTC_SR, return_tensors="pt").input_values.to(self.dev)
            ids = self.ctc(iv).logits.argmax(-1)
        return self.ctc_proc.batch_decode(ids.cpu())[0]

    def measure(self, wav: Path, item: dict, others: list[dict], guide: Path | None) -> dict:
        self._stem = None
        row = super().measure(wav, item, others, guide)
        if self._stem is None:
            raise RuntimeError("the base scorer returned no vocal stem")
        t0 = time.time()
        row.update(ctc_stem_text=self.transcribe(self._stem, self.base.stem_sr), ctc_stem_sr=int(self.base.stem_sr),
                   ctc_stem_seconds=round(time.time() - t0, 2), scorer="score_c1")
        self._stem = None
        return row


def clip(row: dict | None, peak, item: dict) -> dict:
    """The registered per-clip values (score_b2.clip) and the CTC values on the vocal stem. row None or with an
    error: unscored, so q 0, gate failed, WR 0, NPA 0 and both CTC WRs 0."""
    if row is None or "error" in row:
        return dict(MISSING)
    c = S.clip(row, peak, item)
    has = row.get("ctc_stem_text") is not None
    wr = CS.ordered_recall(lyric_words(item["lyrics"]), S.ctc_words(row.get("ctc_stem_text"))) if has else 0.0
    c.update(wr_ctc_stem=rnd(wr), q_ctc_stem=rnd((1.0 if c["gate"] else 0.0) * min(c["npa_v"], wr)), ctc_stem_scored=has)
    return c
