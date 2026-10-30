"""The sing tool without a model: melody text and MIDI parsing, the word-note check and the warnings, items, windows
and runs against B2's own functions on B2's items, the seeds, the selector, the output JSON, resuming, and the
notebook. ACE-Step is never run here.

  python -m pytest -q -p no:cacheprovider experiments/sing/tests
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments"))
import sing as S  # noqa: E402

import plan_b2 as P  # noqa: E402  (sing put experiments/e8_clock_sing on the path)
import readout as R  # noqa: E402

LINE = "Hold my hand and walk with me now"
TUNE = "C4:1 E4:1 G4:2 A4:1 G4:1 E4:2 D4:2 C4:2"


def args(*extra: str):
    return S.parser().parse_args(["run", "--melody", TUNE, "--lyric", LINE, *extra])


def b2_events(it: dict) -> list[tuple[int, float]]:
    """A B2 item's notes back as written lengths: onset differences, and the rest of the 16 beats for the last."""
    starts = [n["start"] for n in it["melody"]["notes"]]
    lens = [b - a for a, b in zip(starts, starts[1:])] + [S.LEAD_IN + 16 - starts[-1]]
    return [(n["pitch"], d) for n, d in zip(it["melody"]["notes"], lens)]


# melody text

def test_parse_melody_names_lengths_and_rests():
    assert S.parse_melody("C4:1 D4:1 E4:2 G4:1") == [(60, 1.0), (62, 1.0), (64, 2.0), (67, 1.0)]
    assert S.parse_melody("F#4:0.5, Bb3:1/2 | c5 R:3/2 rest:1 -:2 69:1 C-1:1 E♭4:1") == [
        (66, 0.5), (58, 0.5), (72, 1.0), (None, 1.5), (None, 1.0), (None, 2.0), (69, 1.0), (0, 1.0), (63, 1.0)]


@pytest.mark.parametrize("text, why", [("", "empty"), ("H4:1", "not a note"), ("C4:0", "more than 0"),
                                       ("C4:-1", "more than 0"), ("C4:x", "not a number"), ("C4:17", "at most 16"),
                                       ("A9:1", "outside MIDI"), ("G9:1x", "not a number"), ("200:1", "outside MIDI"), ("C4:1:2", "not a note"),
                                       ("C4:1/25", "at least 1/24"), ("C4:1/10000000", "at least 1/24")])
def test_parse_melody_refuses(text, why):
    with pytest.raises(S.InputError, match=why):
        S.parse_melody(text)


def test_shortest_length_is_the_midi_grid():
    assert S.parse_melody("C4:1/24") == [(60, 1 / 24)]


def test_melody_text_round_trips():
    ev = [(60, 1.0), (None, 0.5), (61, 1.5), (70, 11 / 12), (59, 2.0)]
    assert S.melody_text(ev) == "C4:1 R:1/2 C#4:3/2 Bb4:11/12 B3:2"
    assert S.parse_melody(S.melody_text(ev)) == ev
    for it in P.test_items():
        assert S.parse_melody(S.melody_text(b2_events(it))) == b2_events(it)


# MIDI

def write_midi(path, notes, bpm=100.0, extra_track=None, drums=False):
    """notes: (pitch, start beat, end beat)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    spb = 60.0 / bpm
    for track in [notes] + ([extra_track] if extra_track else []):
        inst = pretty_midi.Instrument(program=0)
        inst.notes = [pretty_midi.Note(100, p, a * spb, b * spb) for p, a, b in track]
        pm.instruments.append(inst)
    if drums:
        d = pretty_midi.Instrument(program=0, is_drum=True)
        d.notes = [pretty_midi.Note(100, 36, 0.0, 0.1)]
        pm.instruments.append(d)
    pm.write(str(path))
    return path


@pytest.mark.parametrize("bpm", [100.0, 120.0, 73.0])
def test_midi_gives_the_same_events_as_text_at_any_file_tempo(tmp_path, bpm):
    ev = S.parse_melody("C4:1 D4:1/2 E4:3/2 R:1 G4:2 F4:1/3 E4:2/3 D4:1")
    notes, t = [], 0.0
    for p, d in ev:
        if p is not None:
            notes.append((p, t, t + d))
        t += d
    got, info = S.parse_midi(write_midi(tmp_path / "x.mid", notes, bpm, drums=True))
    assert got == ev
    assert info["notes"] == 7 and abs(info["tempo_bpm"] - bpm) < 1e-3 and len(info["sha256"]) == 64


def test_midi_short_gaps_are_legato_long_gaps_rests_and_leading_bars_dropped(tmp_path):
    # notes played a little short (0.9 of a beat) stay one beat long; a 1.1-beat gap is a rest; the first note
    # sits on beat 2 of the third bar: the two empty bars go, the pickup offset stays
    notes = [(60, 9.0, 9.9), (62, 10.0, 10.9), (64, 11.0, 11.9), (65, 13.0, 14.0)]
    got, info = S.parse_midi(write_midi(tmp_path / "x.mid", notes))
    assert info["dropped_leading_beats"] == 8
    assert got == [(None, 1.0), (60, 1.0), (62, 1.0), (64, 22 / 24), (None, 26 / 24), (65, 1.0)]


def test_midi_refusals(tmp_path):
    with pytest.raises(S.InputError, match="not monophonic"):
        S.parse_midi(write_midi(tmp_path / "a.mid", [(60, 0, 1), (64, 0, 1)]))
    with pytest.raises(S.InputError, match="overlap"):
        S.parse_midi(write_midi(tmp_path / "b.mid", [(60, 0, 2), (64, 1, 3)]))
    with pytest.raises(S.InputError, match="2 tracks"):
        S.parse_midi(write_midi(tmp_path / "c.mid", [(60, 0, 1)], extra_track=[(64, 0, 1)]))
    (tmp_path / "d.mid").write_bytes(b"not a midi file")
    with pytest.raises(S.InputError, match="cannot read"):
        S.parse_midi(tmp_path / "d.mid")


def test_midi_tempo_note_and_text_or_midi(tmp_path):
    mid = write_midi(tmp_path / "x.mid", [(60 + k, k, k + 1) for k in range(4)], bpm=120.0)
    a = S.parser().parse_args(["check", "--midi", str(mid), "--lyric", "one two three four"])
    events, words, midi, notes = S.load_inputs(a)
    assert len(S.pitched(events)) == 4 and midi["file"] == "x.mid"
    assert notes and "120 BPM" in notes[0] and "100 BPM" in notes[0]
    with pytest.raises(S.InputError, match="not both"):
        S.load_inputs(S.parser().parse_args(["check", "--midi", str(mid), "--melody", "C4:1", "--lyric", "a"]))


# the word-note check and the warnings

def check(melody=TUNE, lyric=LINE, bpm=100, voice="female", draws=4, key="auto"):
    return S.check_inputs(S.parse_melody(melody), S.split_lyric(lyric), bpm, voice, draws, key)


def test_one_word_per_note_or_refused():
    assert check() == []
    with pytest.raises(S.InputError, match="7 words and the melody 8 notes"):
        check(lyric="Hold my hand and walk with me")
    with pytest.raises(S.InputError, match="9 words and the melody 8 notes"):
        check(lyric=LINE + " please")
    w = check(melody="C4:1 R:1 E4:1 G4:1 R:2 A4:1 G4:1 E4:2 D4:2 C4:2")   # rests take no word
    assert len(w) == 1 and "through any rest" in w[0] and w[0].endswith(": Hold, hand")
    with pytest.raises(S.InputError, match="no letters"):
        check(lyric="Hold my hand and walk with me --")
    with pytest.raises(S.InputError, match="English"):
        check(melody="C4:1 D4:1 E4:1", lyric="我 爱 你")
    with pytest.raises(S.InputError, match="structure tags"):
        check(lyric="[Verse] my hand and walk with me now")
    with pytest.raises(S.InputError, match="only rests"):
        check(melody="R:4", lyric="a")


@pytest.mark.parametrize("kw, why", [({"bpm": 20}, "30-300"), ({"bpm": 92.5}, "whole number"), ({"draws": 0}, "1-8"),
                                     ({"draws": 9}, "1-8"), ({"voice": "child"}, "female or male"),
                                     ({"key": "H major"}, "key"), ({"melody": "C4:16 " * 7 + "C4:16", "bpm": 100},
                                                                   "stops at 60 s")])
def test_settings_refused(kw, why):
    with pytest.raises(S.InputError, match=why):
        check(**kw)


def test_warnings_name_what_b2_never_tried():
    w = " | ".join(check(lyric="Hold my yellow hand by the sea you"))
    assert "more than one syllable" in w and "yellow" in w
    assert "homophones" in w and "by" in w and "sea" in w
    assert "wordless audio" in w and "you" in w
    w = " | ".join(check(melody="C3:1 E4:1 G4:2 A4:1 G4:1 E4:2 D4:2 C6:2"))
    assert "C3" in w and "C6" in w and "A3-E5" in w
    assert any("A2-E4" in x for x in check(voice="male"))
    assert any("fast passages" in x for x in check(melody="C4:1/4 E4:1 G4:2 A4:1 G4:1 E4:2 D4:2 C4:2"))
    assert any("work best" in x for x in check(melody="C4:4 E4:4 G4:4 A4:4 G4:4 E4:4 D4:4 C4:4"))   # 22 s
    assert any("punctuation" in x and "now!" in x for x in check(lyric="Hold my hand and walk with me now!"))
    assert not any("punctuation" in x for x in check(lyric="Hold my hand and don't walk 7 now"))


@pytest.mark.parametrize("lyric", ["I stay 'cause you are mine", "Hold my 'hand' and walk with me", 'Hold "my" hand and walk with me',
                                   "(oh) my hand and walk with me", "Hold my *hand* and walk with me", "Hold my ¿hand and walk with me",
                                   "Hold my élan and walk with me", "Hold my ’cause and walk with me"])
def test_words_starting_with_a_mark_are_refused(lyric):
    """ACE-Step's tokenizer joins these words to the one before or drops them (checked with the checkpoint's Qwen3
    tokenizer through dacapo.words.group_words); the check refuses them before the install."""
    melody = " ".join(["C4:1"] * len(lyric.split()))
    with pytest.raises(S.InputError, match="start every word with a letter"):
        check(melody=melody, lyric=lyric)


def test_trailing_marks_and_inner_apostrophes_pass():
    for lyric in ["Hold my hand!", "Hold my now...", "don't you go", "it’s my song"]:
        check(melody="C4:1 D4:1 E4:1", lyric=lyric)


def test_clock_windows_through_rests_and_too_short_for_the_clock():
    assert S.words_before_rests(S.parse_melody("R:2 C4:1 R:1 D4:1 E4:1 R:1 R:1 F4:1 R:3"), list("abcd")) == ["a", "c"]
    assert not any("through any rest" in x for x in check(melody="R:2 C4:1 E4:1 G4:2 A4:1 G4:1 E4:2 D4:2 C4:2 R:2"))
    w = check(melody="C4:1/4 E4:1/4 G4:1/4 A4:1 G4:1/4 E4:1/4 D4:1/4 C4:2", bpm=300)
    assert any("between two steps" in x and x.endswith(": my, me") for x in w)   # windows of 0.05 s
    assert not any("between two steps" in x for x in check())


def test_estimate_key():
    assert S.estimate_key(S.parse_melody("C4:1 E4:1 G4:1 C5:2 G4:1 E4:1 C4:2")) == "C major"
    assert S.estimate_key(S.parse_melody("A3:2 C4:1 E4:1 A4:2 E4:1 C4:1 A3:2 G#3:1 A3:2")) == "A minor"
    assert S.estimate_key(S.parse_melody("D4:1 F#4:1 A4:1 D5:2 A4:1 F#4:1 D4:2")) == "D major"
    assert S.estimate_key(S.parse_melody(TUNE)) == "C major"
    assert all(S.valid_key(S.estimate_key(S.parse_melody(f"{p}:1"))) for p in range(48, 72))


def test_auto_key_holds_every_note_of_every_b2_and_e4_melody():
    """B2 told ACE-Step each melody's own major key, which held all its notes; auto must not pick a key that
    leaves notes out (the Krumhansl-Kessler correlation alone did on 24 of these 36)."""
    items = P.test_items() + P.dev_items()
    got = {it["id"]: S.estimate_key(b2_events(it)) for it in items}
    assert all(not S.outside_key(b2_events(it), got[it["id"]]) for it in items)
    assert all(not S.outside_key(b2_events(it), it["key"]) for it in items)
    assert sum(got[it["id"]] == it["key"] for it in items) >= 32


def test_keys_any_case_and_notes_outside_the_key():
    assert [S.norm_key(k) for k in ("d major", "F♯ MINOR", "bb major", "DB major", " C  major ")] == [
        "D major", "F# minor", "Bb major", "Db major", "C major"]
    assert S.norm_key("H major") is None and S.norm_key("C") is None
    assert check(key="d major") == []
    assert S.outside_key(S.parse_melody("C4:1 F#4:1 Bb4:1 C5:1"), "C major") == ["F#4", "Bb4"]
    assert S.outside_key(S.parse_melody("A3:1 G#3:1 G3:1"), "A minor") == []
    assert "F#4" in S.key_warning(S.parse_melody("C4:1 F#4:1"), "C major")[0] and not S.key_warning(S.parse_melody("C4:1"), "C major")


# items, windows and runs against B2

def test_constants_are_b2s():
    m = P.dev_items()[0]
    assert S.LEAD_IN == __import__("make_items_b2").e4().LEAD_IN == m["melody"]["notes"][0]["start"]
    assert S.caption("female") == P.CAPTION == R.CAPTION_F and S.caption("male") == R.CAPTION_M
    assert P.GUIDES["oohs"] == 53 and P.parse(S.S_STAR)["k"] == 3 and P.parse(S.S_STAR)["beta"] == 8.0
    sel = json.loads((ROOT / "results" / "gpu_run3" / "b2_selection.json").read_text())
    assert (sel["s_star"], sel["b_star"]) == (S.S_STAR, S.S_B0)   # the recipes Run B chose


@pytest.mark.parametrize("it", P.test_items() + P.dev_items(), ids=lambda it: it["id"])
def test_items_equal_b2_items(it):
    """Every B2 test item and every E4 dev item, given to the tool as text, gives the same notes, length, lyric."""
    ev = S.parse_melody(S.melody_text(b2_events(it)))
    item = S.build_item(ev, it["words"], 100, it["key"])
    assert item["melody"] == it["melody"]
    assert item["duration"] == it["duration"] and item["lyrics"] == it["lyrics"] and item["key"] == it["key"]
    assert S.duration_s(ev, 100) == 14.0


def test_short_notes_and_rests_follow_b2s_rule():
    item = S.build_item(S.parse_melody("C4:1/2 R:1 D4:2"), ["a", "b"], 120, "C major")
    assert item["melody"]["notes"] == [{"pitch": 60, "start": 4.0, "dur": 0.5 * 0.95, "vel": 100},
                                       {"pitch": 62, "start": 5.5, "dur": 2 * 0.9, "vel": 100}]
    assert item["duration"] == float(np.ceil((4 + 3.5 + 2) * 60 / 120)) == 5.0


def test_spans_equal_b2s_on_b2_items():
    tok = P.read_json(P.TOK_JSON)["items"]
    for it in P.test_items() + P.dev_items():
        mine = S.build_item(S.parse_melody(S.melody_text(b2_events(it))), it["words"], 100, it["key"])
        keys = tok[it["id"]]["keys"]
        assert S.spans(mine, keys) == P.spans(it, keys)
        assert S.word_windows(mine) == [(a, b) for _, a, b in P.spans(it, keys)]


def test_window_rule_and_clock_frames_equal_attn_bias():
    from dacapo import attn_bias as A
    assert (S.HOP_S, S.OFFSET_S, S.DIGITS) == (A.HOP_S, A.OFFSET_S, A.DIGITS)
    for mel, bpm in [("R:2 C#4:1 R:4 Db4:1/2 E4:1/24 F4:1/12 G4:16 R:1 A4:1", 100), ("C4:1/4 " * 8 + "C4:1", 300),
                     ("C4:1/24 D4:1/24 E4:1", 300), (TUNE, 30)]:
        ev = S.parse_melody(mel)
        item = S.build_item(ev, ["w"] * len(S.pitched(ev)), bpm, "C major")
        win = S.word_windows(item)
        sp = P.spans(item, [[k] for k in range(len(win))])
        assert win == [(a, b) for _, a, b in sp]
        lq = int(np.ceil(win[-1][1] / A.HOP_S)) + 2
        t = A.template(sp, lq, len(win))
        assert [S.clock_frames(a, b) for a, b in win] == [int(t[:, k].sum()) for k in range(len(win))]


def test_runs_are_b2s_s_star_and_b0_with_the_tools_seeds_and_voice():
    it = P.test_items()[0]
    item = S.build_item(b2_events(it), it["words"], 100, it["key"])
    runs = S.make_runs(item, 3, 7, "male", compare=True)
    assert [r["run_id"] for r in runs] == ["take1_clock", "take2_clock", "take3_clock", "take1_noclock",
                                           "take2_noclock", "take3_noclock"]
    same = ("config", "guide", "k", "clock", "head_set", "heads", "beta", "shuffle", "hooked", "fallback", "task",
            "cover_strength", "lyrics", "duration", "bpm", "keyscale")
    for r in runs:
        b2 = P.make_run("test", "full", it, r["draw"], P.parse(r["config"]))
        assert {k: r[k] for k in same} == {k: b2[k] for k in same}
        assert r["caption"] == R.CAPTION_M and (r["dit_seed"], r["rng_seed"]) == S.seeds(7, r["draw"])
        assert (r["dit_seed"], r["rng_seed"]) != (b2["dit_seed"], b2["rng_seed"])
    clock = [r for r in runs if r["hooked"]]
    assert all(r["heads"] == {str(L): hs for L, hs in R.ALIGN.items()} and r["beta"] == 8.0 for r in clock)
    assert all(r["cover_strength"] == 3 / 8 and r["task"] == "cover-nofsq" for r in runs)
    assert [(r["dit_seed"], r["rng_seed"]) for r in runs[:3]] == [(r["dit_seed"], r["rng_seed"]) for r in runs[3:]]
    assert len({(r["dit_seed"], r["rng_seed"]) for r in runs}) == 3


def test_seed_tag_is_the_tools_own():
    assert S.TAG not in {t for sc in P.TAGS.values() for t in sc.values()}
    from dacapo.seeds import run_seeds
    assert S.seeds(0, 0) == run_seeds(S.TAG, "0", 0) and S.seeds(0, 0) != S.seeds(1, 0) != S.seeds(0, 1)


# planning, resuming, scoring and the output JSON

@pytest.fixture
def planned(tmp_path):
    rundir, plan, _ = S.make_plan(args("--draws", "3", "--compare"), tmp_path)
    return rundir, plan


def test_plan_renders_the_guide_like_b2_and_resumes(planned, tmp_path):
    rundir, plan = planned
    guide = rundir / "sing_oohs.wav"
    assert guide.exists() and plan["guide"]["sha256"] == S.sha256(guide) and plan["guide"]["program"] == 53
    b2 = P.read_json(P.GUIDES_JSON)
    assert (plan["guide"]["soundfont"], plan["guide"]["soundfont_sha256"]) == (b2["soundfont"], b2["soundfont_sha256"])
    assert plan["guide"]["soundfont_is_b2s"] is True and not any("soundfont" in w.lower() for w in plan["warnings"])
    assert abs(plan["guide"]["seconds"] - S.guide_seconds(plan["item"])) < 0.05
    assert plan["recipe"]["config"] == S.S_STAR and plan["recipe"]["comparison"] == S.S_B0
    assert [d["take"] for d in plan["seeds"]["draws"]] == [1, 2, 3] and plan["seeds"]["tag"] == S.TAG
    assert plan["code"]["sha256"]["experiments/sing/sing.py"] == S.sha256(HERE / "sing.py")
    again, plan2, _ = S.make_plan(args("--draws", "3", "--compare"), tmp_path)
    assert again == rundir and plan2 == plan
    other, _, _ = S.make_plan(args("--draws", "3", "--compare", "--seed", "1"), tmp_path)
    assert other != rundir
    with pytest.raises(S.InputError, match="other settings"):
        S.make_plan(args("--seed", "2", "--name", rundir.name), tmp_path)
    lower, plan3, _ = S.make_plan(args("--draws", "3", "--compare", "--key", "c major"), tmp_path)
    assert plan3["settings"]["key"] == plan3["item"]["key"] == "C major"
    assert S.make_plan(args("--draws", "3", "--compare", "--key", "AUTO"), tmp_path)[0] == rundir


def test_another_soundfont_is_flagged(tmp_path, monkeypatch):
    import make_items_b2
    real = make_items_b2.soundfont
    monkeypatch.setattr(make_items_b2, "soundfont", lambda: {**real(), "soundfont_sha256": "0" * 64})
    _, plan, _ = S.make_plan(args("--draws", "1"), tmp_path)
    assert plan["guide"]["soundfont_is_b2s"] is False and any("not B2's" in w for w in plan["warnings"])


def test_gen_stops_on_a_gpu_older_than_compute_capability_8(planned, monkeypatch):
    import torch
    rundir, _ = planned
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "Tesla T4")
    with pytest.raises(S.SetupError, match="Tesla T4 is older than compute capability 8.0.*L4 or A100"):
        S.gen(rundir, "cuda")


def test_the_tools_own_lines_always_reach_the_user():
    for line in ["refused: x\n", "setup problem: y\n", "warning: z\n", "    take1_clock generated in 2.0 s\n",
                 "  take2_noclock FAILED: RuntimeError()\n"]:
        assert S.OWN_LINE.match(line)
    assert not S.OWN_LINE.match("2026-10-30 | INFO | something\n")


def fake_gen(rundir, run, peak=0.5):
    import soundfile as sf
    wav = rundir / f"{run['run_id']}.wav"
    sf.write(str(wav), np.zeros((4800, 2), np.float32), 48000, subtype="PCM_16")
    row = {"run_id": run["run_id"], "status": "ok", "peak_before_norm": peak, "wav_sha256": S.sha256(wav),
           "hook_calls": 40 if run["hooked"] else 0, "gen_seconds": 1.0}
    S.append_jsonl(rundir / "ledger.jsonl", row)
    return row


def fake_measure(text, npa):
    words = text.split()
    return {"stem_share": {"vocals": 0.9, "other": 0.1}, "vocal_active_s": 4.0, "whisper": S.CPU_WHISPER,
            "whisper_revision": "x", "whisper_text": text,
            "whisper_chunks": [[" " + w, 2.4 + 0.6 * k, 2.6 + 0.6 * k] for k, w in enumerate(words)], "npa_v": npa,
            "npa_v_oct": npa, "npa_mix": npa, "npa_mix_oct": npa, "ctc_text": text.upper(), "ctc": "c",
            "ctc_revision": "r", "guide_copy": 0.1, "test_flags": [f"test whisper {S.CPU_WHISPER} on cpu"]}


def test_selector_and_output_json(planned):
    rundir, plan = planned
    runs = {r["run_id"]: r for r in plan["runs"]}
    gens = {rid: fake_gen(rundir, r) for rid, r in runs.items() if rid != "take3_noclock"}
    gens["take3_noclock"] = {"run_id": "take3_noclock", "status": "failed", "error": "RuntimeError('x')"}
    half = "Hold my hand and"
    scores = {"take1_clock": fake_measure(half, 0.5), "take2_clock": fake_measure(LINE, 0.5),
              "take3_clock": fake_measure(LINE, 0.25), "take1_noclock": fake_measure("oh oh", 0.75),
              "take2_noclock": {"error": "ValueError('scorer')"}}
    for rid, s in scores.items():
        s["wav_sha256"] = gens[rid]["wav_sha256"]
    doc = S.result_doc(plan, gens, scores, {"whisper": S.CPU_WHISPER, "test_flags": ["t"]}, {})
    t = {x["run_id"]: x for x in doc["takes"]}
    # take 1 and take 2 both reach q 0.5; take 2 has the higher WR, so B2's selector keeps it
    assert t["take1_clock"]["q"] == t["take2_clock"]["q"] == 0.5 and doc["kept"]["clock"] == "take2_clock"
    assert t["take2_clock"]["notes_right"] == "4/8" and t["take2_clock"]["words_recognised"] == "8/8"
    assert t["take1_clock"]["words_recognised"] == "4/8" and t["take3_clock"]["q"] == 0.25
    assert t["take1_noclock"]["q"] == 0.0 and t["take1_noclock"]["wordless"] is True
    assert t["take2_noclock"]["status"] == "generated, not scored" and t["take2_noclock"]["q"] == 0.0
    assert t["take3_noclock"]["status"] == "not generated" and "RuntimeError" in t["take3_noclock"]["error"]
    assert doc["kept"]["no clock"] == "take1_noclock"
    assert [x["run_id"] for x in S.ordered(doc)] == ["take2_clock", "take1_clock", "take3_clock", "take1_noclock",
                                                    "take2_noclock", "take3_noclock"]
    assert doc["cpu_test"] and doc["settings"]["seed"] == 0 and doc["seeds"]["tag"] == S.TAG
    assert doc["code"]["commit"] == plan["code"]["commit"] and doc["recipe"]["config"] == S.S_STAR
    assert json.loads(json.dumps(doc)) == doc
    text = S.table(doc)
    assert text.splitlines()[1].startswith("  2*") and "Hold my hand and walk with me now" in text
    # a score row of another file is ignored
    scores["take2_clock"] = {**scores["take2_clock"], "wav_sha256": "other"}
    assert S.result_doc(plan, gens, scores, {}, {})["kept"]["clock"] == "take1_clock"


def test_score_writes_mp3s_the_kept_take_and_the_json(planned, monkeypatch):
    import score_b2
    rundir, plan = planned
    for r in plan["runs"]:
        fake_gen(rundir, r)
    made = []

    class FakeScorer:
        def __init__(self, dev, test_whisper=None):
            made.append((dev, test_whisper))

        def measure(self, wav, item, others, guide):
            assert others == [] and guide.name == "sing_oohs.wav" and item == plan["item"]
            return fake_measure(LINE if "take2" in wav.name else "Hold my", 0.75)
    monkeypatch.setattr(score_b2, "B2Scorer", FakeScorer)
    assert S.score(rundir, "cpu", None) == 0
    assert made == [("cpu", S.CPU_WHISPER)]
    doc = json.loads((rundir / "sing_result.json").read_text())
    assert doc["kept"] == {"clock": "take2_clock", "no clock": "take2_noclock"}
    assert (rundir / "kept.wav").read_bytes() == (rundir / "take2_clock.wav").read_bytes()
    if shutil.which("ffmpeg"):
        assert (rundir / "take1_clock.mp3").exists() and (rundir / "kept.mp3").exists()
        assert doc["takes"][0]["mp3"] == "take1_clock.mp3"
    assert S.score(rundir, "cpu", None) == 0 and len(made) == 1   # scored takes are not scored again
    with pytest.raises(S.SetupError, match="large-v3"):
        S.score(rundir, "cuda", "openai/whisper-tiny")


def test_cpu_whisper_is_run_bs_tiny_one():
    import run_b
    assert S.CPU_WHISPER == run_b.TINY_WHISPER


def test_cli_exit_codes(capsys, tmp_path):
    assert S.main(["check", "--melody", TUNE, "--lyric", LINE]) == 0
    out = capsys.readouterr().out
    assert "8 words on 8 notes" in out and "the takes last about 12 s" in out
    assert S.main(["check", "--melody", TUNE, "--lyric", "too few words"]) == 1
    assert "refused" in capsys.readouterr().out
    assert S.main(["run", "--melody", TUNE, "--lyric", LINE, "--device", "cuda", "--test-whisper", "x",
                   "--out", str(tmp_path)]) == 1
    assert S.main(["gen"]) == 2


# the notebook

def test_notebook_is_built_from_the_builder_and_says_what_it_must():
    import build_notebook as A
    import build_notebook_sing as B
    nb = json.loads((ROOT / "notebook_sing.ipynb").read_text())
    assert nb == B.notebook(None)
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for pin in A.ACE_PINS + A.PINS + [A.BASIC_PITCH, A.ACE_COMMIT, A.HF_REVISION]:
        assert pin in src
    for field in ("MELODY", "MIDI_FILE", "LYRIC", "TEMPO", "VOICE", "DRAWS", "SEED", "COMPARE_NO_CLOCK"):
        assert f"{field} = " in src and "#@param" in src
    assert "torch.cuda.is_available()" in src and "needs a GPU" in src
    assert "get_device_capability(0)[0] < 8" in src and "is_bf16_supported" not in src and "free T4 does not work" in src
    assert "14 of 24" in src and "listening test" in src and "Not a product" in src
    assert "Audio(" in src and "sing.py" in src
    for field in ("TEMPO", "VOICE", "KEY", "DRAWS", "SEED", "COMPARE_NO_CLOCK", "USE_DRIVE"):
        assert f"#@markdown {field}: " in src
    assert "quarter note" in src and "How to read the scores" in src and "(best without clock)" in src
    assert "the input was refused" in src and "no take was made" in src
    order = ["get_device_capability", "# The Da Capo code", "# Check the melody and the words now", "# Google Drive keeps",
             "!apt-get"]
    pos = [src.find(x) for x in order]
    assert -1 not in pos and pos == sorted(pos), pos   # GPU, code, input check, then Drive, then the long install
