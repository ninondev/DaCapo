"""B2 scorer on fixtures: q, joint, WR by position, CTC WR, vowel-only and wordless, E4's multiset WR and WER,
the guide-copy index, and the refusal of the CPU."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import plan_b2 as P  # noqa: E402
import score_b2 as S  # noqa: E402

IT = P.all_items()["b2t00"]   # "The dogs bark at the gate when the moon climbs up the sky", 13 words


def row(text: str, onsets=None, share=0.9, active=5.0, npa_v=0.8, ctc="", **kw) -> dict:
    words = text.split()
    on = onsets or [2.4 + 0.6 * k for k in range(len(words))]
    r = {"stem_share": {"vocals": share, "drums": 0.0, "bass": 0.0, "other": 1 - share}, "vocal_active_s": active,
         "whisper": S.CS.WHISPER, "whisper_revision": S.CS.WHISPER_REVISION, "whisper_text": " " + text,
         "whisper_chunks": [[" " + w, t, t + 0.3] for w, t in zip(words, on)], "test_flags": [], "npa_v": npa_v,
         "npa_v_oct": npa_v, "npa_v_chance": 0.05, "npa_mix": npa_v, "npa_mix_oct": npa_v, "npa_mix_chance": 0.05,
         "ctc": S.CTC, "ctc_revision": S.CTC_REVISION, "ctc_text": ctc, "guide_copy": 0.3}
    r.update(kw)
    return r


LINE = IT["lyrics"].split("\n", 1)[1]


def test_q_is_gate_times_min_of_npa_and_wr():
    c = S.clip(row(LINE, npa_v=0.8), 0.5, IT)
    assert c["wr"] == 1.0 and c["q"] == 0.8 and c["joint"] and c["strict"] and c["gate"]
    c = S.clip(row(LINE, npa_v=0.6), 0.5, IT)
    assert c["q"] == 0.6 and c["joint"] and not c["strict"]
    c = S.clip(row("the dogs bark at the gate", npa_v=0.9), 0.5, IT)
    assert c["wr"] == round(6 / 13, 6) and c["q"] == c["wr"] and not c["joint"]
    assert c["recall_first_half"] == 1.0 and c["recall_second_half"] == 0.0 and c["recall_last_word"] == 0.0


def test_the_gate_zeroes_q():
    assert S.clip(row(LINE, share=0.1), 0.5, IT)["q"] == 0.0           # vocal share below 0.2
    assert S.clip(row(LINE, active=0.5), 0.5, IT)["q"] == 0.0          # under 1 s of active vocal frames
    assert S.clip(row(LINE), None, IT)["q"] == 0.0                     # no pre-normalisation peak
    assert S.clip(row(LINE), 0.005, IT)["q"] == 0.0
    assert S.clip(None, 0.5, IT) == S.MISSING and S.clip({"error": "x"}, 0.5, IT)["q"] == 0.0


def test_wr_is_in_order_and_exact():
    c = S.clip(row("the sky up climbs moon the when gate the at bark dogs the"), 0.5, IT)
    assert c["wr"] < 0.3   # reversed: few words match in order
    c = S.clip(row("the dog barks at the gate when the moon climbs up the sky"), 0.5, IT)
    assert c["wr"] == round(11 / 13, 6)  # dog, barks differ from dogs, bark
    assert c["wr_multiset"] == round(11 / 13, 6)


def test_ctc_wr_and_onsets():
    c = S.clip(row(LINE, ctc="THE DOGS BARK AT THE GATE WHEN THE MOON CLIMBS UP THE SKY"), 0.5, IT)
    assert c["wr_ctc"] == 1.0 and c["q_ctc"] == 0.8
    c = S.clip(row(LINE, ctc=""), 0.5, IT)
    assert c["wr_ctc"] == 0.0 and c["q_ctc"] == 0.0
    written = S.written(IT)
    c = S.clip(row(LINE, onsets=[t + 0.25 for t in written]), 0.5, IT)
    assert c["onset_err_abs"] == 0.25 and c["onset_err_signed"] == 0.25 and c["n_matched"] == 13
    c = S.clip(row("the dogs bark"), 0.5, IT)
    assert c["onset_err_abs"] is None  # fewer than 4 matched words


@pytest.mark.parametrize("text,vowel,wordless", [
    ("", True, True), ("oh oh oh oh", True, True), ("Ooh, ooh, aah, mmm", True, True), ("la la la the", True, True),
    ("thank you", False, True), ("Thank you for watching.", False, True), ("transcript emily beynon transcript emily beynon", False, True),
    ("oh my god oh my god oh", False, True), ("so so so so", False, True),
    ("the dogs bark", False, False), ("thank you the dogs bark", False, False), ("oh the dogs", False, False),
    ("oh the dogs bark", False, False), ("hmm hmm we sing", True, True), ("oh oh the dogs", True, True),
])
def test_vowel_only_and_wordless(text, vowel, wordless):
    assert S.vowel_only(text) is vowel and S.wordless(text) is wordless


def test_e4_rates_on_e4_transcripts():
    rows = json.loads((ROOT / "results/gpu_run1/e4_sweep.json").read_text())["results"]["rows"]
    k4 = [r for r in rows if r["arm"] == "ctx_oohs_k4"]
    ng = [r for r in rows if r["arm"] == "noguide"]
    assert np.mean([S.wordless(r["hyp"]) for r in k4]) > 0.7 > 0.1 > np.mean([S.wordless(r["hyp"]) for r in ng])


def test_multiset_wr_and_wer_match_e4():
    sys.path.insert(0, str(ROOT / "figures"))
    mf = pytest.importorskip("make_figures")
    rows = json.loads((ROOT / "results/gpu_run1/e4_sweep.json").read_text())["results"]["rows"]
    words = {it["id"]: it["words"] for it in P.dev_items()}
    for r in rows[::7]:
        assert S.multiset_wr(words[r["id"]], r["hyp"]) == mf.word_recall(words[r["id"]], r["hyp"])
        lower = [w.lower() for w in words[r["id"]]]
        assert S.wer_capped(words[r["id"]], r["hyp"]) == pytest.approx(min(S.levenshtein(lower, r["hyp"].split()) / len(lower), 1))


def test_guide_copy_index(tmp_path):
    import soundfile as sf
    sr = 44100
    t = np.arange(int(3.0 * sr)) / sr
    x = (np.sin(2 * np.pi * 440 * t) * (t > 0.5) * (t < 2.5)).astype(np.float32)
    g = tmp_path / "g.wav"
    sf.write(str(g), np.stack([x, x], 1), sr)
    assert S.guide_copy_index(x, sr, g) > 0.99
    late = np.roll(x, int(0.2 * sr))
    assert S.guide_copy_index(late, sr, g) > 0.99   # found at a lag within 0.5 s
    rng = np.random.default_rng(0)
    assert S.guide_copy_index(rng.standard_normal(len(x)).astype(np.float32) * 0.1, sr, g) < 0.5


def test_halves_and_selector_order():
    assert S.halves(13) == (list(range(6)), list(range(6, 13))) and S.halves(12) == (list(range(6)), list(range(6, 12)))


def test_scorer_refuses_the_cpu():
    with pytest.raises(RuntimeError, match="CUDA"):
        S.B2Scorer("cpu")
