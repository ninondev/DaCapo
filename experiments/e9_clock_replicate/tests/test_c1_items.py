"""C1 items: the 96 lines meet every line rule (addendum 3's, through make_items_b2, and Run C's three), manifest_c1.json
rebuilds identically from seed 33, no melody repeats an earlier one, the guide and tokenizer records cover the 98
items and agree with B2's on item00 and item01, and the rules catch what they forbid.

  python -m pytest experiments/e9_clock_replicate/tests
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
import make_items_c1 as M  # noqa: E402
import plan_b2 as P2  # noqa: E402
import plan_c1 as P  # noqa: E402

from dacapo.words import lyric_words  # noqa: E402


def test_96_lines_meet_every_rule():
    lines = M.read_lines()
    assert len(lines) == 96 and len(set(lines)) == 96
    assert M.rule_problems(lines) == {}
    old = M.earlier_lines()
    for ln in lines:
        words = ln.split()
        assert 11 <= len(words) <= 13 and ln.lower() not in old
        assert not {w.lower() for w in words} & (M.M.HOMOPHONES | M.M.HALLUCINATION_WORDS)
        assert all(M.M.syllables(w) == 1 for w in words)
    assert max(M.shared_words(a, b) for i, a in enumerate(lines) for b in lines[i + 1:]) <= 6


def test_earlier_lines_cover_e4_e6_b2_and_e7():
    old = M.earlier_lines()
    assert "i walk down to the sea and watch the light go slow" in old                 # E4
    assert "the dogs bark at the gate when the moon climbs up the sky" in old         # B2
    assert "i watch the sun climb past the hill" in old                               # E7 mel00
    assert M.M.earlier_lines() <= old and len(M.e7_lines()) == 48 and len(M.b2_lines()) == 24


def test_rules_catch_each_violation():
    old = M.earlier_lines()
    assert "a line from E4, E6, B2 or E7" in M.line_problems("The dogs bark at the gate when the moon climbs up the sky", old)
    assert any("syllables" in p for p in M.line_problems("The yellow cat sat on the mat in the warm hall at dusk", old))
    assert any("homophone" in p for p in M.line_problems("The cat sat on the mat in the red hall at dusk now", old))
    assert any("capitalised" in p for p in M.line_problems("the cat sat on the mat in the warm hall at dusk", old))
    a = "The cat sat on the mat and the dog lay in the sun"
    b = "The cat sat on the mat and the dog ran to the barn"
    assert M.shared_words(a, b) == 10 and M.overlap_problems([a, b]) == ["lines 0 and 1 share 10 words"]
    assert M.shared_words("the the the a", "the the a a") == 3          # counted with repeats
    lines = M.read_lines()
    bad = M.rule_problems(lines[:95] + [lines[0]])
    assert "(file)" in bad and "(overlap)" in bad
    old_c1t48 = "We climbed through the pines to the top and gazed at the stars"   # 8 words of B2's b2t14
    assert M.earlier_overlap_problems([old_c1t48]) == [
        "line 0 shares 8 words with the earlier line 'we climbed the steep path to the top and saw the coast'"]
    assert "(overlap with earlier)" in M.rule_problems(lines[:48] + [old_c1t48] + lines[49:])


def test_no_line_shares_more_than_6_words_with_an_earlier_line():
    lines = M.read_lines()
    assert M.earlier_overlap_problems(lines) == []
    assert max(M.shared_words(a, b) for a in lines for b in M.earlier_lines()) <= 6
    assert lines[48] == "We walked through the pines at dusk and gazed up at the stars"


def test_manifest_rebuilds_from_seed_33():
    built = M.manifest_text(M.build(M.read_lines()))
    assert P.C1_MANIFEST.read_text() == built
    man = json.loads(built)
    assert man["seed"] == 33 and "written by the agents" in man["author"] and len(man["items"]) == 96
    keys = ["C major", "D major", "F major", "G major"]
    for i, it in enumerate(man["items"]):
        notes = it["melody"]["notes"]
        assert it["id"] == f"c1t{i:02d}" and it["key"] == keys[i % 4] and it["duration"] == 14.0
        assert it["melody"]["bpm"] == 100.0 and it["guides"] == {"oohs": f"guides/c1t{i:02d}_oohs.wav"}
        assert len(notes) == len(it["words"]) == len(lyric_words(it["lyrics"]))
        assert notes[0]["start"] == 4.0 and all(57 <= n["pitch"] <= 76 for n in notes)
        ends = [n["start"] for n in notes[1:]] + [20.0]
        assert all(abs(n["dur"] - 0.9 * (e - n["start"])) < 1e-9 for n, e in zip(notes, ends))
    assert M.melody_problems(man) == []


def test_build_procedure_is_b2s():
    """make_items_c1.build is make_items_b2.build with seed 33 and the c1t ids: on B2's lines and seed 22 it gives
    B2's melodies."""
    b2 = P2.read_json(P2.B2_MANIFEST)
    old = M.SEED
    try:
        M.SEED = 22
        mine = M.build(M.b2_lines())
    finally:
        M.SEED = old
    assert [it["melody"] for it in mine["items"]] == [it["melody"] for it in b2["items"]]
    assert [it["key"] for it in mine["items"]] == [it["key"] for it in b2["items"]]


def test_melody_check_catches_a_repeat():
    man = json.loads(P.C1_MANIFEST.read_text())
    man["items"][5]["melody"] = P2.test_items()[3]["melody"]
    man["items"][7]["melody"] = man["items"][6]["melody"]
    bad = M.melody_problems(man)
    assert "c1t05 repeats b2t03" in bad and "c1t07 repeats c1t06" in bad


def test_tok_check_covers_98_items():
    tok = P.read_json(P.TOK_JSON)
    assert tok["passed"] and set(tok["items"]) == set(P.all_items()) and len(tok["items"]) == 98
    for iid, v in tok["items"].items():
        it = P.all_items()[iid]
        assert v["start"] == 9 and v["keys"][0][0] == 12 and not v["problems"]
        assert v["words"] == lyric_words(it["lyrics"]) and len(v["keys"]) == len(it["melody"]["notes"])
        assert all(1 <= len(k) <= 2 for k in v["keys"])
        flat = [k for ks in v["keys"] for k in ks]
        assert flat == sorted(flat) and max(flat) < v["n_lyr"]
    assert M.b2_tok_problems(tok) == []


def test_tok_check_recomputes_with_the_tokenizer():
    ckpt = M.M.default_ckpt()
    if not (ckpt / "Qwen3-Embedding-0.6B" / "tokenizer.json").exists():
        pytest.skip("no ACE-Step checkpoint (ACESTEP_CHECKPOINTS_DIR)")
    assert M.tok_check(ckpt) == P.read_json(P.TOK_JSON)


def test_guide_record():
    g = P.read_json(P.GUIDES_JSON)
    assert set(g["sha256"]) == set(P.all_items()) and all(set(v) == {"oohs"} for v in g["sha256"].values())
    assert len({v["oohs"] for v in g["sha256"].values()}) == 98 and g["programs"] == {"oohs": 53}
    assert g["soundfont_sha256"] == "74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0"
    assert M.b2_guide_problems(g) == []


@pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="no FluidSynth")
def test_two_guides_render_to_their_hashes(tmp_path):
    from dacapo.render import default_sf2
    try:
        default_sf2()
    except FileNotFoundError:
        pytest.skip("no General MIDI soundfont")
    g = P.read_json(P.GUIDES_JSON)["sha256"]
    for iid in ("item01", "c1t57"):
        _, _, h = M.M.render_one(P.all_items()[iid], "oohs", tmp_path)
        assert h == g[iid]["oohs"]
    assert os.listdir(tmp_path)
