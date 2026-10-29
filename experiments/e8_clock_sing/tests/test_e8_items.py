"""B2 items: the 24 lines meet every line rule, manifest_b2.json rebuilds identically from seed 22, the guide and
tokenizer records cover all 36 items, and the rules catch what they forbid.

  python -m pytest experiments/e8_clock_sing/tests
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import make_items_b2 as M  # noqa: E402
import plan_b2 as P  # noqa: E402

from dacapo.words import lyric_words  # noqa: E402


def test_24_lines_meet_every_rule():
    lines = M.read_lines()
    assert len(lines) == 24 and len(set(lines)) == 24
    old = M.earlier_lines()
    assert {ln: M.line_problems(ln, old) for ln in lines if M.line_problems(ln, old)} == {}
    for ln in lines:
        words = ln.split()
        assert 11 <= len(words) <= 13
        assert not {w.lower() for w in words} & (M.HOMOPHONES | M.HALLUCINATION_WORDS)


def test_rules_catch_each_violation():
    old = M.earlier_lines()
    assert M.line_problems("I walk down to the sea and watch the light go slow", old)  # an E4 line
    assert any("words" in p for p in M.line_problems("The cat sat on the mat", old))
    assert any("syllables" in p for p in M.line_problems("The yellow cat sat on the mat in the warm hall at dusk", old))
    assert any("homophone" in p for p in M.line_problems("The cat sat on the mat in the red hall at dusk now", old))
    assert any("wordless" in p for p in M.line_problems("The cat sat on the mat and you sang in the hall at dusk", old))
    assert any("capitalised" in p for p in M.line_problems("the cat sat on the mat in the warm hall at dusk", old))
    assert any("capitalised" in p for p in M.line_problems("The Cat sat on the mat in the warm hall at dusk", old))
    assert any("letters" in p for p in M.line_problems("The cat's mat lay in the warm hall at the dusk", old))


def test_syllable_counter():
    assert all(M.syllables(w) == 1 for ln in M.e4().LINES for w in ln.split())
    for w in ("little", "yellow", "water", "window", "voices", "ended", "table", "garden"):
        assert M.syllables(w) == 2, w
    for w in ("while", "once", "where", "leaves", "climbed", "fixed", "years", "house", "square", "tune"):
        assert M.syllables(w) == 1, w


def test_manifest_rebuilds_from_seed_22():
    built = M.manifest_text(M.build(M.read_lines()))
    assert P.B2_MANIFEST.read_text() == built
    man = json.loads(built)
    assert man["seed"] == 22 and "written by the agents" in man["author"]
    keys = ["C major", "D major", "F major", "G major"]
    for i, it in enumerate(man["items"]):
        notes = it["melody"]["notes"]
        assert it["id"] == f"b2t{i:02d}" and it["key"] == keys[i % 4] and it["duration"] == 14.0
        assert len(notes) == len(it["words"]) == len(lyric_words(it["lyrics"]))
        assert notes[0]["start"] == 4.0 and all(57 <= n["pitch"] <= 76 for n in notes)
        ends = [n["start"] for n in notes[1:]] + [20.0]
        assert all(abs(n["dur"] - 0.9 * (e - n["start"])) < 1e-9 for n, e in zip(notes, ends))


def test_tok_check_covers_36_items():
    tok = P.read_json(P.TOK_JSON)
    assert tok["passed"] and set(tok["items"]) == set(P.all_items())
    for iid, v in tok["items"].items():
        it = P.all_items()[iid]
        assert v["start"] == 9 and v["keys"][0][0] == 12 and not v["problems"]
        assert v["words"] == lyric_words(it["lyrics"]) and len(v["keys"]) == len(it["melody"]["notes"])
        assert all(1 <= len(k) <= 2 for k in v["keys"])
        flat = [k for ks in v["keys"] for k in ks]
        assert flat == sorted(flat) and max(flat) < v["n_lyr"]


def test_tok_check_recomputes_when_the_tokenizer_is_here():
    ckpt = M.default_ckpt()
    if not (ckpt / "Qwen3-Embedding-0.6B" / "tokenizer.json").exists():
        pytest.skip("no ACE-Step checkpoint (ACESTEP_CHECKPOINTS_DIR)")
    assert M.tok_check(ckpt) == P.read_json(P.TOK_JSON)


def test_guide_record():
    g = P.read_json(P.GUIDES_JSON)
    assert set(g["sha256"]) == set(P.all_items()) and all(set(v) == {"oohs", "clarinet"} for v in g["sha256"].values())
    assert len({h for v in g["sha256"].values() for h in v.values()}) == 72
    b1 = json.loads((ROOT / "experiments/e5_motif/registration/addendum_2_b1code.json").read_text())["cpu_test"]
    assert g["sha256"]["item00"]["oohs"] == b1["guide_sha256"]
    assert g["soundfont_sha256"] == "74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0"


@pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="no FluidSynth")
def test_two_guides_render_to_their_hashes(tmp_path):
    from dacapo.render import default_sf2
    try:
        default_sf2()
    except FileNotFoundError:
        pytest.skip("no General MIDI soundfont")
    g = P.read_json(P.GUIDES_JSON)["sha256"]
    for iid, guide in (("item00", "oohs"), ("b2t07", "clarinet")):
        _, _, h = M.render_one(P.all_items()[iid], guide, tmp_path)
        assert h == g[iid][guide]
    assert os.listdir(tmp_path)
