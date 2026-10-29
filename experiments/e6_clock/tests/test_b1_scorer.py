"""experiments/e6_clock/clock_score.py: the measurement statements are readout.score's, unchanged; the gate,
the word matching, recall and onset error on fixtures; the registered scorer refuses the CPU."""
import ast
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import clock_score as CS  # noqa: E402
import readout  # noqa: E402

TREE_R = ast.parse((HERE / "readout.py").read_text())
TREE_C = ast.parse((HERE / "clock_score.py").read_text())


def fn(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def assigned(body, names):
    """Top-level assignments to these names, in order."""
    return [st for st in body if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name)
            and st.targets[0].id in names]


def segment(body):
    """From `x, sr = sf.read(...)` to `chunks = [...]`: the measurement of one clip."""
    def is_read(st):
        return isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Tuple) and \
            [e.id for e in st.targets[0].elts] == ["x", "sr"]
    for node in [body] + [n.body for n in ast.walk(ast.Module(body=body, type_ignores=[])) if isinstance(n, ast.Try)]:
        starts = [i for i, st in enumerate(node) if is_read(st)]
        if starts:
            ends = [i for i, st in enumerate(node) if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name)
                    and st.targets[0].id == "chunks"]
            return node[starts[0]: ends[0] + 1]
    raise AssertionError("measurement statements not found")


class _ArgsWhisper(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "args" and node.attr == "whisper":
            return ast.copy_location(ast.Name(id="whisper", ctx=node.ctx), node)
        return self.generic_visit(node)


def dumps(stmts):
    return [ast.dump(s) for s in stmts]


def test_measurement_statements_are_readouts():
    r, c = segment(fn(TREE_R, "score").body), segment(fn(TREE_C, "measure").body)
    assert len(r) == 17 and dumps(r) == dumps(c)


def test_model_setup_matches_readout_with_the_model_name_as_a_variable():
    names = ("sep", "asr", "asr_rev", "kw")
    r = [_ArgsWhisper().visit(st) for st in assigned(fn(TREE_R, "score").body, names)]
    c = assigned(fn(TREE_C, "__init__").body, names)
    assert [st.targets[0].id for st in c] == list(names) and dumps(r) == dumps(c)


def test_gate_and_word_statements_are_clip_stats():
    r = assigned(fn(TREE_R, "clip_stats").body, ("share", "vocal", "hyp_all", "hyp"))
    c = assigned(fn(TREE_C, "vocal").body, ("share", "vocal")) + assigned(fn(TREE_C, "words_of").body, ("hyp_all", "hyp"))
    assert len(r) == 4 and dumps(r) == dumps(c)


def test_constants_come_from_readout():
    for k in ("VOCAL_SHARE_MIN", "VOCAL_ACTIVE_MIN_S", "VOCAL_ACTIVE_DBFS", "PEAK_MIN", "MIN_MATCHED", "WHISPER",
              "WHISPER_REVISION", "DIGITS"):
        assert getattr(CS, k) is getattr(readout, k)


def _row(words_times, share=0.9, active=5.0, whisper=readout.WHISPER, rev=readout.WHISPER_REVISION, flags=()):
    return {"stem_share": {"vocals": share, "drums": 1 - share, "bass": 0.0, "other": 0.0}, "vocal_active_s": active,
            "whisper": whisper, "whisper_revision": rev, "test_flags": list(flags),
            "whisper_chunks": [[w, t, None if t is None else t + 0.3] for w, t in words_times]}


def test_gate_matches_readout_clip_stats():
    for share, active, peak in ((0.9, 5.0, 0.5), (0.19, 5.0, 0.5), (0.2, 1.0, 0.01), (0.9, 0.95, 0.5), (0.9, 5.0, None),
                                (0.9, 5.0, 0.009)):
        row = _row([(" oh", 2.5)], share=share, active=active)  # no word matched: clip_stats stops after the gate
        meta = {"status": "ok", "words": ["i"], "onsets_align": [2.5], "peak_before_norm": peak, "official_ok": True,
                "replicates_official": True}
        it = {"id": "x", "set": "e4", "lyric_id": "x", "voice": "female"}
        assert CS.vocal(row, peak) == readout.clip_stats(meta, row, None, it)["vocal"]


def test_words_matching_recall_and_onset_error():
    lyric = ["i", "walk", "down", "to"]
    row = _row([(" I", 2.5), (" walked", 3.1), ("", 3.2), (" Down,", 3.7), (" to", None), (" To", 4.1)])
    hyp = CS.words_of(row)
    assert hyp == [("i", 2.5), ("walked", 3.1), ("down", 3.7), ("to", 4.1)]
    assert CS.matches(lyric, hyp) == [(0, 0), (2, 2), (3, 3)]
    assert CS.ordered_recall(lyric, hyp) == 0.75 and CS.ordered_recall(lyric, hyp, positions=[1, 2]) == 0.5
    e = CS.onset_error(lyric, hyp, [2.4, 3.0, 3.6, 4.2])
    assert e["n_matched"] == 3 and e["err_median"] == 0.1 and e["err_signed_median"] == 0.1
    assert e["within_025"] == 1.0 and e["err_median_excl_first"] == 0.1
    assert CS.onset_error(lyric, [], [2.4, 3.0, 3.6, 4.2])["err_median"] is None
    with pytest.raises(ValueError):
        CS.onset_error(lyric, hyp, [2.4])
    # out of order: the in-order matching keeps the longest ordered run
    assert CS.ordered_recall(lyric, [("to", 1.0), ("down", 2.0), ("walk", 3.0), ("i", 4.0)]) == 0.25


def test_clip_counts_unscored_as_recall_zero_and_gate_fail():
    lyric, written = ["a", "b", "c", "d"], [1.0, 2.0, 3.0, 4.0]
    for row in (None, {"error": "boom"}):
        c = CS.clip(row, 0.5, lyric, written)
        assert c == {"scored": False, "vocal": False, "recall": 0.0, "n_matched": 0, "evaluable": False, "err_median": None}
    good = _row([(" a", 1.2), (" b", 2.2), (" c", 3.2), (" d", 4.2)])
    c = CS.clip(good, 0.5, lyric, written)
    assert c["evaluable"] and c["recall"] == 1.0 and c["err_median"] == 0.2 and c["registered_reference"]
    c = CS.clip(good, None, lyric, written)  # a missing peak fails the gate: not evaluable, no error
    assert not c["vocal"] and not c["evaluable"] and c["err_median"] is None and c["recall"] == 1.0
    three = _row([(" a", 1.2), (" b", 2.2), (" c", 3.2)])
    assert not CS.clip(three, 0.5, lyric, written)["evaluable"]  # fewer than MIN_MATCHED words


def test_registered_reference():
    assert CS.registered_reference(_row([]))
    assert not CS.registered_reference(_row([], whisper="openai/whisper-small"))
    assert not CS.registered_reference(_row([], rev="0" * 40))
    assert not CS.registered_reference(_row([], flags=["test whisper openai/whisper-small on cpu"]))


def test_registered_scorer_refuses_the_cpu():
    with pytest.raises(RuntimeError, match="CUDA"):
        CS.Scorer("cpu")
