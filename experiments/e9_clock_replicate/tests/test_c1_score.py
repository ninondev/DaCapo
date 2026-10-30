"""The C1 scorer: B2's per-clip values unchanged, the CTC values on the vocal stem, the missing rule, the wrapper's
CTC-on-stem field on a stub (the stem B2's scorer separated, decoded as B2 decodes the mix), and the refusal of the
CPU without a test Whisper."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import plan_c1 as P  # noqa: E402
import score_c1 as S  # noqa: E402

IT = P.all_items()["c1t00"]
LINE = IT["lyrics"].split("\n", 1)[1]


def row(text: str, stem: str | None = "", share=0.9, active=5.0, npa_v=0.8, ctc="", **kw) -> dict:
    words = text.split()
    r = {"stem_share": {"vocals": share, "drums": 0.0, "bass": 0.0, "other": 1 - share}, "vocal_active_s": active,
         "whisper": S.CS.WHISPER, "whisper_revision": S.CS.WHISPER_REVISION, "whisper_text": " " + text,
         "whisper_chunks": [[" " + w, 2.4 + 0.6 * k, 2.7 + 0.6 * k] for k, w in enumerate(words)], "test_flags": [],
         "npa_v": npa_v, "npa_v_oct": npa_v, "npa_v_chance": 0.05, "npa_mix": npa_v, "npa_mix_oct": npa_v,
         "npa_mix_chance": 0.05, "ctc": S.CTC, "ctc_revision": S.CTC_REVISION, "ctc_text": ctc, "guide_copy": 0.3}
    if stem is not None:
        r["ctc_stem_text"] = stem
    r.update(kw)
    return r


def test_b2_values_are_unchanged():
    r = row(LINE, stem=LINE.upper(), ctc=LINE.upper())
    c, b2 = S.clip(r, 0.5, IT), S.S.clip(r, 0.5, IT)
    assert {k: c[k] for k in b2} == b2 and c["q"] == 0.8 and c["joint"] and c["strict"]


def test_ctc_on_the_stem():
    n = len(IT["words"])
    half = " ".join(IT["words"][: n // 2]).upper()
    c = S.clip(row(LINE, stem=half, ctc=""), 0.5, IT)
    assert c["wr_ctc"] == 0.0 and c["q_ctc"] == 0.0
    assert c["wr_ctc_stem"] == round((n // 2) / n, 6) and c["q_ctc_stem"] == round(min(0.8, (n // 2) / n), 6)
    assert c["ctc_stem_scored"]
    gated = S.clip(row(LINE, stem=LINE.upper(), share=0.1), 0.5, IT)
    assert gated["wr_ctc_stem"] == 1.0 and gated["q_ctc_stem"] == 0.0 and gated["q"] == 0.0
    old = S.clip(row(LINE, stem=None), 0.5, IT)            # a row without the stem transcript
    assert old["wr_ctc_stem"] == 0.0 and not old["ctc_stem_scored"] and old["q"] == 0.8


def test_missing_and_failed_clips():
    assert S.clip(None, 0.5, IT) == S.MISSING and S.clip({"error": "x"}, 0.5, IT) == S.MISSING
    assert S.MISSING["q"] == 0.0 and not S.MISSING["gate"] and S.MISSING["wr_ctc_stem"] == S.MISSING["q_ctc_stem"] == 0.0
    assert {k: S.MISSING[k] for k in S.S.MISSING} == S.S.MISSING


class FakeBase:
    stem_sr = 44100
    whisper, whisper_revision, test_flags = "w", "r", []

    def __init__(self):
        self.calls = 0

    def measure(self, wav):
        self.calls += 1
        voc = np.full(2 * self.stem_sr, 0.25, dtype=np.float32)          # 2 s of stem at 44.1 kHz
        return {"stem_share": {"vocals": 1.0}, "whisper_text": "x"}, voc


class FakeProc:
    def __init__(self):
        self.seen = []

    def __call__(self, x, sampling_rate, return_tensors):
        import torch
        self.seen.append((len(x), sampling_rate))
        return type("V", (), {"input_values": torch.tensor(np.asarray(x))[None]})()

    def batch_decode(self, ids):
        return [f"LEN{ids.shape[-1]}"]


class FakeCTC:
    def __call__(self, iv):
        import torch
        return type("O", (), {"logits": torch.zeros(1, iv.shape[-1], 3)})()


def stub(monkeypatch):
    import torch
    sc = object.__new__(S.C1Scorer)
    sc.base, sc.ctc_proc, sc.ctc, sc.torch, sc.dev = FakeBase(), FakeProc(), FakeCTC(), torch, "cpu"
    sc.keep_stem()

    def b2_measure(self, wav, item, others, guide):    # stands for B2Scorer.measure: it calls base.measure once
        r, _ = self.base.measure(wav)
        r["ctc_text"] = "MIX"
        return r
    monkeypatch.setattr(S.S.B2Scorer, "measure", b2_measure)
    return sc


def test_wrapper_adds_the_stem_transcript(monkeypatch):
    sc = stub(monkeypatch)
    r = sc.measure(Path("x.wav"), IT, [], None)
    assert r["ctc_text"] == "MIX" and r["ctc_stem_text"] == "LEN32000" and r["ctc_stem_sr"] == 44100
    assert r["scorer"] == "score_c1" and sc.base.calls == 1          # htdemucs runs once per clip
    assert sc.ctc_proc.seen == [(32000, 16000)] and sc._stem is None
    r2 = sc.measure(Path("y.wav"), IT, [], None)
    assert r2["ctc_stem_text"] == "LEN32000" and sc.base.calls == 2


def test_wrapper_refuses_without_a_stem(monkeypatch):
    sc = stub(monkeypatch)
    monkeypatch.setattr(S.S.B2Scorer, "measure", lambda self, *a: {"ctc_text": "MIX"})
    with pytest.raises(RuntimeError, match="no vocal stem"):
        sc.measure(Path("x.wav"), IT, [], None)


def test_cpu_without_a_test_whisper_is_refused():
    with pytest.raises(RuntimeError, match="CUDA"):
        S.C1Scorer("cpu")
