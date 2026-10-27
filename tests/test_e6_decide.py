"""E6 read-out decision logic on synthetic clips (no ACE-Step, no audio)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "e6_clock"))
import readout  # noqa: E402

K = readout.N_RANDOM
PLAN = readout.items()
HASHES = {f: "0" * 64 for f in readout.CODE_FILES}


def _clip(it, head_err=0.1, rand_err=1.0, share=0.95, n=8, peak=0.6, ok=True, replicated=True, ref=None, head=None,
          whisper=readout.WHISPER, rev=readout.WHISPER_REVISION):
    ref = np.arange(n) * 0.5 + 0.3 if ref is None else np.asarray(ref, float)
    head = ref + head_err if head is None else np.asarray(head, float)
    words = [chr(97 + k) * 2 for k in range(len(ref))]  # normalised lyric words, as gen stores them
    meta = {"status": "ok", "words": words, "onsets_align": head.tolist(), "peak_before_norm": peak,
            "official_ok": ok, "replicates_official": replicated and ok, "code_sha256": HASHES, "registered": True,
            "onsets_align_seeds": {"43": head.tolist(), "44": head.tolist()}}
    row = {"stem_share": {"vocals": share}, "vocal_active_s": 5.0, "whisper": whisper, "whisper_revision": rev,
           "whisper_chunks": [[" " + w, t, t + 0.3] for w, t in zip(words, ref)], "mms_onsets": None}
    rand = np.tile(ref + rand_err, (K, 1))
    z = {"onsets_random": rand, "onsets_random_43": rand, "onsets_random_44": rand,
         "onsets_steps": np.zeros((8, 1 + readout.N_RANDOM_STEPS, len(ref))), "onsets_single": np.zeros((384, len(ref)))}
    return readout.clip_stats(meta, row, z, it)


def _rows(n_vocal=60, missing=(), failed=(), **kw):
    rows = []
    for i, it in enumerate(PLAN):
        if i in missing:
            rows.append(readout.clip_stats(None, None, None, it))
        elif i in failed:
            rows.append(readout.clip_stats({"status": "failed"}, None, None, it))
        else:
            rows.append(_clip(it, share=0.95 if i < n_vocal else 0.0, **kw))
    return rows


def _decide(rows, integrity=(), flags=()):
    s = readout.summarise(rows)
    return s, *readout.decision(s, rows, list(integrity), list(flags))


def test_all_criteria_met():
    s, reasons, labels = _decide(_rows(60))
    assert s["n_vocal"] == 60 and s["n_evaluable"] == 60 and reasons == [] and labels == ["B1 allowed"]


def test_threshold_edges_and_ties():
    s, reasons, _ = _decide(_rows(44, head_err=0.25, rand_err=0.25))  # 44/88 vocal; error at 0.25; tie with random
    assert s["n_vocal"] == 44 and s["err_median_of_clip_medians"] == 0.25
    assert len(reasons) == 1 and "beat" in reasons[0]  # ties count against the alignment heads


def test_grid_rounding_at_the_threshold():
    # head onsets on the 80 ms DiT grid, Whisper on its 20 ms grid: errors 0.24 and 0.26, median 0.25
    head = np.array([5, 10, 15, 20]) * (14.0 / 175)
    ref = head + np.array([0.24, -0.26, 0.26, -0.24])
    ref = np.round(ref / 0.02) * 0.02
    r = _clip(PLAN[0], ref=ref, head=head)
    assert r["err_median"] == 0.25
    rows = [_clip(it, ref=ref, head=head) if i < 50 else _clip(it, share=0.0) for i, it in enumerate(PLAN)]
    s, reasons, _ = _decide(rows)
    assert s["err_median_of_clip_medians"] == 0.25 and not any("median onset error" in x for x in reasons)


def test_missing_clips_count_as_non_vocal_and_incomplete():
    s, reasons, _ = _decide(_rows(88, missing=set(range(19, 88))))
    assert any("vocals in 19/88" in x for x in reasons) and any("19 evaluable" in x for x in reasons)
    assert any(x.startswith("incomplete") for x in reasons)


def test_failed_clips_up_to_four_are_tolerated_five_block():
    s, reasons, _ = _decide(_rows(88, failed={0, 1, 2, 3}))
    assert s["n_failed"] == 4 and reasons == []
    s, reasons, labels = _decide(_rows(88, failed={0, 1, 2, 3, 4}))
    assert any(x.startswith("incomplete: 5 of 88") for x in reasons) and "technical" in labels


def test_missing_peak_fails_the_gate_and_blocks():
    rows = _rows(60)
    rows[0] = _clip(PLAN[0], peak=None)
    s, reasons, _ = _decide(rows)
    assert not rows[0]["vocal"] and s["n_peak_missing"] == 1 and any("peak missing in 1" in x for x in reasons)


def test_timestamp_failure_or_replication_miss_blocks():
    rows = _rows(60)
    rows[5] = _clip(PLAN[5], ok=False)
    rows[6] = _clip(PLAN[6], replicated=False)
    s, reasons, labels = _decide(rows)
    assert s["n_official_failed"] == 1 and s["n_not_replicated"] == 2
    assert any("ACE-Step timestamps failed in 1" in x for x in reasons) and "technical" in labels


def test_wrong_reference_and_test_flags_block():
    rows = _rows(60, whisper="openai/whisper-small.en", rev="e872752")
    _, reasons, _ = _decide(rows)
    assert any("another reference" in x for x in reasons)
    _, reasons, _ = _decide(_rows(60), flags=["--allow-other-reference"])
    assert reasons and reasons[0].startswith("test run")


def test_integrity_reasons():
    rows = _rows(60)
    code = dict(HASHES)
    assert readout.integrity(rows, HASHES, code, {"git_tracked_changes": False}) == []
    assert readout.integrity(rows, None, code, {"git_tracked_changes": False})[0].startswith("no registered")
    other = {**HASHES, "dacapo/ace.py": "1" * 64}
    out = readout.integrity(rows, other, code, {"git_tracked_changes": False})
    assert any("differs" in x for x in out) and any("88 clips generated with unregistered code" in x for x in out)
    assert readout.integrity(rows, HASHES, code, {"git_tracked_changes": None})
    rows[3]["registered"] = False
    assert any("1 clips" in x for x in readout.integrity(rows, HASHES, code, {"git_tracked_changes": False}))


def test_unmatched_vocal_clip_is_not_evaluable():
    r = _clip(PLAN[0], n=3)
    assert r["vocal"] and r["n_matched"] == 3 and not r["evaluable"]


def test_labels():
    _, _, labels = _decide(_rows(30))
    assert "vocal yield insufficient" in labels
    rows = _rows(60, head_err=0.4)
    for r in rows:
        if r["evaluable"]:
            r["head_vs_mms_median"] = 0.1
    s = readout.summarise(rows)
    s["head_vs_mms_median"] = 0.1
    assert "reference-limited" in readout.decision(s, rows, [], [])[1]
    s["head_vs_mms_median"] = 0.6
    assert "heads do not give word onsets" in readout.decision(s, rows, [], [])[1]


def test_bootstrap_is_stratified_over_the_88_planned_clips():
    ci = readout.bootstrap(_rows(88))
    assert ci["vocal_frac"] == [1.0, 1.0]
    rows = _rows(88, missing=set(range(48, 88)))  # every verse clip missing: vocal share 48/88 in every resample
    assert readout.bootstrap(rows)["vocal_frac"] == [48 / 88, 48 / 88]


def test_random_head_sets_share_no_key_value_group_with_the_alignment_heads():
    sets = readout.random_head_sets()
    assert len(sets) == K and sets == readout.random_head_sets()
    for st in sets:
        assert sorted(st) == sorted(readout.ALIGN)
        for L, hs in st.items():
            assert len(hs) == len(readout.ALIGN[L])
            assert not {h // readout.GROUP for h in hs} & {h // readout.GROUP for h in readout.ALIGN[L]}
            if len(hs) == 2:
                assert hs[0] % 2 == 0 and hs[1] == hs[0] + 1


def test_items_plan():
    assert len(PLAN) == 88 and sum(x["set"] == "e4" for x in PLAN) == 48
    assert {x["duration"] for x in PLAN} == {14.0, 16.0}
    seeds = {x["seed"] for x in PLAN}
    assert not seeds & {x["rng_seed"] for x in PLAN}
    assert not seeds & {x["seed"] for x in readout.a0_items()}
    assert len(readout.lyric_texts()) == 16


def test_gate_check():
    reg = dict(HASHES)
    good = {"b1_allowed": True, "scale": "full", "test_flags": [], "code_sha256": reg, "registered_sha256": reg,
            "n_planned": 88}
    assert readout.gate_ok(good, reg) == (True, "ok")
    assert not readout.gate_ok(None, reg)[0]
    for bad in ({"b1_allowed": False}, {"scale": "quick"}, {"test_flags": ["--registration"]},
                {"code_sha256": {**reg, "dacapo/ace.py": "1" * 64}}, {"n_planned": 4}):
        assert not readout.gate_ok({**good, **bad}, reg)[0]
    assert not readout.gate_ok(good, None)[0]
