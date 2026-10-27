"""One test per integrity check (addendum 2, S10) and the ledger rules (D4)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dacapo import motif_integrity as I

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5_motif"


def _design(mels=("v2c00", "v2c01"), conds=("lit", "null"), draws=2, base=1000):
    jobs = []
    for i, m in enumerate(mels):
        for c in conds:
            for d in range(draws):
                jobs.append({"id": m, "split": "confirm", "cond": c, "arm": "base", "seed": f"d{d}",
                             "dit_seed": base + 10 * i + d, "rng_seed": base + 500 + 10 * i + d, "src": f"sources/{m}_{c}.wav"})
    return {"phase": "main", "attempt": 1, "ci_method": "melody", "jobs": jobs, "manifest_sha256": "m" * 64,
            "ruler_program": 73, "sources_sha256": {}, "addendum_sha256": "a" * 64}


def _events(design, utc="2026-11-01T10:00:00Z"):
    ev = []
    for j in design["jobs"]:
        k = {f: j[f] for f in I.JOB_FIELDS}
        ev += [{"event": "start", **k, "pass": 1, "utc": utc},
               {"event": "done", **k, "utc": utc, "wav_sha256": "w" + j["id"] + j["cond"] + j["seed"]}]
    return ev


def test_check1_lossless():
    good = {"format": "WAV", "subtype": "PCM_16", "samplerate": 48000, "channels": 2}
    assert I.check_lossless({("a", "lit", "base", "d0"): good})["ok"]
    assert I.check_lossless({("a", "lit", "base", "d0"): good | {"subtype": "FLOAT"}})["ok"]
    assert not I.check_lossless({("a", "lit", "base", "d0"): good | {"format": "MP3", "subtype": "MPEG_LAYER_III"}})["ok"]
    assert not I.check_lossless({("a", "lit", "base", "d0"): good | {"samplerate": 44100}})["ok"]


def test_check2_peaks():
    row = lambda p: {"id": "a", "cond": "lit", "arm": "base", "seed": "d0", "status": "ok", "quality": {"peak_dbfs": p}}
    assert I.check_peaks([row(-1.0), row(-0.6), row(-200.0)])["ok"]
    assert not I.check_peaks([row(-3.0)])["ok"]


@pytest.fixture(scope="module")
def mel04_sources(tmp_path_factory):
    if shutil.which("fluidsynth") is None:
        pytest.skip("fluidsynth not installed")
    sys.path.insert(0, str(E5))
    import make_items
    it = next(x for x in json.loads((E5 / "manifest_p0b.json").read_text())["items"] if x["id"] == "mel04")
    root = tmp_path_factory.mktemp("src")
    make_items.render_sources(it, root, arms=("rend",))
    return root / "sources"


def test_check3_source_levels_on_rendered_sources(mel04_sources):
    lit, null = I.source_levels(mel04_sources / "mel04_lit.wav"), I.source_levels(mel04_sources / "mel04_null.wav")
    assert lit["ctx_dbfs"] > -40 and null["ctx_dbfs"] < -90 and null["d_dbfs"] > -40
    d = _design(mels=("mel04",))
    levels = {"sources/mel04_lit.wav": lit, "sources/mel04_null.wav": null}
    assert I.check_sources(d, levels)["ok"]
    assert not I.check_sources(d, {"sources/mel04_lit.wav": null, "sources/mel04_null.wav": null})["ok"]
    assert not I.check_sources(d, {"sources/mel04_lit.wav": lit, "sources/mel04_null.wav": {**null, "d_dbfs": -200.0}})["ok"]


def test_audio_sha256_ignores_the_float_peak_chunk(tmp_path):
    x = (np.random.default_rng(0).standard_normal((4800, 2)) * 0.1).astype(np.float32)
    sf.write(str(tmp_path / "a.wav"), x, 48000, subtype="FLOAT")
    import time
    time.sleep(1.1)                                                      # the PEAK chunk stores a timestamp
    sf.write(str(tmp_path / "b.wav"), x, 48000, subtype="FLOAT")
    assert I.audio_sha256(tmp_path / "a.wav") == I.audio_sha256(tmp_path / "b.wav")
    y = x.copy()
    y[100, 0] += 1e-6
    sf.write(str(tmp_path / "c.wav"), y, 48000, subtype="FLOAT")
    assert I.audio_sha256(tmp_path / "a.wav") != I.audio_sha256(tmp_path / "c.wav")


def test_check4_seeds():
    d = _design()
    assert I.check_seeds(d, _events(d))["ok"]
    shared = json.loads(json.dumps(d))
    shared["jobs"][-1]["dit_seed"], shared["jobs"][-1]["rng_seed"] = d["jobs"][0]["dit_seed"], d["jobs"][0]["rng_seed"]
    assert not I.check_seeds(shared, _events(shared))["ok"]              # a seed pair in two melodies
    ev = _events(d)
    ev[1]["dit_seed"] += 1
    assert not I.check_seeds(d, ev)["ok"]                                # generation used other seeds
    within = json.loads(json.dumps(d))
    within["jobs"][1]["dit_seed"], within["jobs"][1]["rng_seed"] = d["jobs"][0]["dit_seed"], d["jobs"][0]["rng_seed"]
    assert not I.check_seeds(within, _events(within))["ok"]              # two draws of a melody share noise


def test_check4_against_the_committed_table():
    d = _design(mels=("v2c00",), conds=("lit",), draws=2)
    table = {"v2c00": [[j["dit_seed"], j["rng_seed"]] for j in d["jobs"]]}
    assert I.check_seeds(d, _events(d), table)["ok"]
    table["v2c00"][1][0] += 1
    assert not I.check_seeds(d, _events(d), table)["ok"]


def test_check5_disjoint_from_pilot():
    main, pilot = _design(), _design(mels=("mel00",), base=90000)
    assert I.check_disjoint(main, pilot)["ok"]
    assert not I.check_disjoint(main, _design(mels=("v2c00",), base=90000))["ok"]
    assert not I.check_disjoint(main, _design(mels=("mel00",), base=1000))["ok"]
    assert not I.check_disjoint(main, None)["ok"]


def test_check6_files_vs_design(tmp_path):
    d = _design()
    d["sources_sha256"] = {"sources/v2c00_lit.wav": "x"}
    for j in d["jobs"]:
        (tmp_path / I.audio_name(j)).write_bytes(b"")
    assert I.check_files_vs_design(d, tmp_path, {"sources/v2c00_lit.wav": "x"})["ok"]
    assert not I.check_files_vs_design(d, tmp_path, {"sources/v2c00_lit.wav": "y"})["ok"]
    (tmp_path / "_orphans").mkdir()
    (tmp_path / "_orphans" / f"v2c09__lit__base__d0.123{I.AUDIO_EXT}").write_bytes(b"")
    assert I.check_files_vs_design(d, tmp_path, {"sources/v2c00_lit.wav": "x"})["ok"]   # moved-aside files are not scored
    (tmp_path / f"v2c09__lit__base__d0{I.AUDIO_EXT}").write_bytes(b"")
    assert not I.check_files_vs_design(d, tmp_path, {"sources/v2c00_lit.wav": "x"})["ok"]
    (tmp_path / f"v2c09__lit__base__d0{I.AUDIO_EXT}").unlink()
    (tmp_path / "v2c00__lit__base__d0.wav").write_bytes(b"")                 # any stray audio file counts
    assert not I.check_files_vs_design(d, tmp_path, {"sources/v2c00_lit.wav": "x"})["ok"]


def test_check7_ruler_gate():
    d = _design()
    g = {"results": {"summary": {"passed": True, "manifest_sha256": "m" * 64, "program": 73, "tau": 0.7}}}
    assert I.check_ruler(g, d)["ok"]
    for bad in ({"passed": False}, {"manifest_sha256": "n" * 64}, {"program": 71}):
        gg = json.loads(json.dumps(g))
        gg["results"]["summary"].update(bad)
        assert not I.check_ruler(gg, d)["ok"]
    assert not I.check_ruler(None, d)["ok"]


def test_check8_wav_hashes():
    rows = [{"id": "a", "cond": "lit", "arm": "base", "seed": "d0", "status": "ok", "wav_sha256": "h1"}]
    assert I.check_wav_hashes(rows, {("a", "lit", "base", "d0"): "h1"})["ok"]
    assert not I.check_wav_hashes(rows, {("a", "lit", "base", "d0"): "h2"})["ok"]
    assert not I.check_wav_hashes(rows, {})["ok"]                        # file gone since the quality stage


def test_check9_ledger():
    d = _design()
    ev = _events(d)
    sha = {I.job_key(e): e["wav_sha256"] for e in ev if e["event"] == "done"}
    assert I.check_ledger(d, ev, sha)["ok"]
    assert not I.check_ledger(d, ev + [ev[1]], sha)["ok"]               # a completed draw generated twice
    k = I.job_key(d["jobs"][0])
    assert not I.check_ledger(d, ev, {**sha, k: "other"})["ok"]
    key = {f: d["jobs"][0][f] for f in I.JOB_FIELDS}
    regen = ev + [{"event": "lost", **key, "reason": "audio missing"}, {"event": "start", **key, "pass": 1},
                  {"event": "done", **key, "wav_sha256": "new"}]
    assert I.check_ledger(d, regen, {**sha, k: "new"})["ok"]            # lost before the freeze, generated again
    assert not I.check_ledger(d, regen, sha)["ok"]                      # the wav must be the last one recorded
    bad = ev + [{"event": "check_failed", **key, "failures": ["dcw"]}]
    assert not I.check_ledger(d, bad, sha)["ok"]


def test_draw_status_missing_vs_lost(tmp_path):
    d = _design()
    ev = [e for e in _events(d) if not (e["event"] == "done" and e["id"] == "v2c01" and e["cond"] == "null")]
    for j in d["jobs"]:
        if not (j["id"] == "v2c00" and j["cond"] == "lit" and j["seed"] == "d0"):
            (tmp_path / I.audio_name(j)).write_bytes(b"")
    st = I.draw_status(d, ev, tmp_path)
    assert st[("v2c00", "lit", "base", "d0")] == "lost"                  # completed, wav gone
    assert st[("v2c01", "null", "base", "d0")] == "missing"              # never completed
    assert st[("v2c00", "null", "base", "d1")] == "ok"
    key = {f: d["jobs"][0][f] for f in I.JOB_FIELDS}
    st = I.draw_status(d, ev + [{"event": "lost", **key}], tmp_path)
    assert st[("v2c00", "lit", "base", "d0")] == "missing"               # after a lost event it may run again


def test_check10_registration():
    add = {"addendum_utc": "2026-10-31T12:00:00Z", "code_sha256": {"score_analogy.py": "s1", "motif_stats.py": "s2"}}
    d = _design()
    d["addendum_sha256"] = "A"
    ev = _events(d)
    assert I.check_registration(add, {"score_analogy.py": "s1", "motif_stats.py": "s2"}, d, "A", ev)["ok"]
    assert not I.check_registration(add, {"score_analogy.py": "s1", "motif_stats.py": "edited"}, d, "A", ev)["ok"]
    assert not I.check_registration(add, {"score_analogy.py": "s1", "motif_stats.py": "s2"}, d, "B", ev)["ok"]
    late = add | {"addendum_utc": "2026-11-02T00:00:00Z"}
    assert not I.check_registration(late, {"score_analogy.py": "s1", "motif_stats.py": "s2"}, d, "A", ev)["ok"]
    assert not I.check_registration(None, {}, d, None, ev)["ok"]
    assert not I.check_registration({**add, "code_sha256": {}}, {}, d, "A", ev)["ok"]


def test_ledger_reader_counts_resumes_and_skips_a_torn_line(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ev = _events(_design())[:2] + [{"event": "pass", "pass": 1}, {"event": "resume", "pass": 2}, {"event": "pass", "pass": 2}]
    p.write_text("\n".join(json.dumps(e) for e in ev) + "\n" + '{"event": "start", "id": "v2')
    events = I.read_ledger(p)
    st, n = I.ledger_state(events)
    assert n == 1 and len(st) == 1 and len(events) == 5 and I.current_pass(events) == 2


def test_ledger_state_counts_starts_per_pass_and_resets_on_lost():
    k = {"id": "v2c00", "cond": "lit", "arm": "base", "seed": "d0", "dit_seed": 1, "rng_seed": 2}
    ev = [{"event": "start", **k, "pass": 1}, {"event": "error", **k}, {"event": "start", **k, "pass": 2}]
    st, _ = I.ledger_state(ev)
    s = st[I.job_key(k)]
    assert s["starts"] == 2 and s["starts_by_pass"] == {1: 1, 2: 1} and s["errors"] == 1
    st, _ = I.ledger_state(ev + [{"event": "done", **k, "wav_sha256": "x"}, {"event": "lost", **k}])
    assert st[I.job_key(k)]["starts"] == 0 and st[I.job_key(k)]["done"] == []
