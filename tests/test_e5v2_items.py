"""E5 items: the v1 generator unchanged, the fresh v2 items, the 18 s length arm and rendering from the
committed manifests (main environment; rendering tests need FluidSynth)."""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5_motif"
sys.path.insert(0, str(E5))

import make_items as M  # noqa: E402
from dacapo.motif_integrity import audio_sha256  # noqa: E402

V1 = json.loads((E5 / "manifest.json").read_text())
V2 = json.loads((E5 / "manifest_v2.json").read_text())
P0B = json.loads((E5 / "manifest_p0b.json").read_text())
needs_fluidsynth = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not installed")


def test_v1_manifest_regenerates_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "HERE", tmp_path)
    monkeypatch.setattr(M, "render", lambda *a, **k: None)
    M.main()
    assert (tmp_path / "manifest.json").read_text() == (E5 / "manifest.json").read_text()


def test_v1_main_refuses_to_replace_a_different_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "HERE", tmp_path)
    monkeypatch.setattr(M, "render", lambda *a, **k: None)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(RuntimeError):
        M.main()


def test_contexts_rebuilt_from_the_manifest_equal_build():
    rng = np.random.default_rng(101)
    for i, it in enumerate(V1["items"]):
        b = M.build(rng, i)
        ctx, D = M.contexts_from_hyps(it["hyps"])
        for c in M.CONDITIONS:
            assert M.source(ctx[c], D) == M.source(b["contexts"][c], b["D"]), (it["id"], c)


def test_v2_items_are_fresh_and_match_the_committed_manifest():
    items = M.v2_items()
    assert [it["id"] for it in items] == [f"v2c{i:02d}" for i in range(36)]
    assert all(it["split"] == "confirm" for it in items) and V2["seed"] == 303
    M.disjoint_from_v1(items, V1)
    for it, cm in zip(items, V2["items"]):
        assert (it["id"], it["tonic"], it["hyps"], it["B2"], it["sources"]) == (cm["id"], cm["tonic"], cm["hyps"], cm["B2"], cm["sources"])
        for p in (73, 54, 71):
            assert it[M.hyp_key(p)] == cm[M.hyp_key(p)]
    with pytest.raises(RuntimeError):
        M.disjoint_from_v1(items[:1], {"items": [{"hyps": items[0]["hyps"], "B2": items[0]["B2"]}]})


def test_d_is_the_notes_starting_before_beat_4_at_full_length():
    rng = np.random.default_rng(303)
    for i, it in enumerate(V2["items"][:6]):
        b = M.build(rng, i, unit_fn=M.unit_in_window)
        assert [asdict(n) for n in b["D"].notes] == it["hyps"]["D"]
        assert all(n["start"] < 4.0 for n in it["hyps"]["D"])
    ends = {it["t18_cut"]["D_end_beat"] for it in V2["items"]}
    assert ends <= {4.0, 4.5, 5.0, 5.5}


def test_v2_units_lie_inside_the_scored_window():
    from score_analogy import notes_in_gap
    for it in V2["items"]:
        for h, notes in list(it["hyps"].items()) + [("B2", it["B2"])]:
            if h != "D":
                assert len(notes_in_gap(notes)) == len(notes), (it["id"], h)
    first = json.loads((E5 / "k0_first_run" / "manifest_v2.json").read_text())
    assert first["seed"] == 202 and all(a["hyps"]["B"] != b["hyps"]["B"] for a, b in zip(first["items"], V2["items"]))


def test_manifests_record_lengths_and_hashes():
    for man in (V2, P0B):
        for it in man["items"]:
            assert it["source_samples"]["t18"] == 864000 and it["source_samples"]["rend"] > 864000
            assert set(it["source_sha256"]) == {"rend", "t18"} and len(it["source_sha256"]["t18"]) == 8
    assert all(len(it["hyp_render_sha256"][str(p)]) == 35 for it in V2["items"] for p in (73, 54, 71))
    assert [it["hyps"] for it in P0B["items"]] == [it["hyps"] for it in V1["items"]]


def test_trim_t18(tmp_path):
    x = (np.random.default_rng(0).standard_normal((900000, 2)) * 0.1).astype(np.float32)
    sf.write(str(tmp_path / "src.wav"), x, 48000, subtype="FLOAT")
    info = M.trim_t18(tmp_path / "src.wav", tmp_path / "t18.wav")
    y, sr = sf.read(str(tmp_path / "t18.wav"), dtype="float32", always_2d=True)
    assert sr == 48000 and len(y) == 864000 and sf.info(str(tmp_path / "t18.wav")).subtype == "FLOAT"
    assert np.array_equal(y[:861600], x[:861600])
    ramp = (0.5 * (1 + np.cos(np.linspace(0, np.pi, 2400))))[:, None]
    assert np.allclose(y[861600:], x[861600:864000] * ramp, atol=1e-7) and y[-1].max() == 0.0
    assert info["samples_in"] == 900000
    with pytest.raises(ValueError):
        M.trim_t18(tmp_path / "t18.wav", tmp_path / "again.wav")       # a source must be longer than 18 s


@needs_fluidsynth
def test_render_from_the_committed_manifests_reproduces_every_hash(tmp_path):
    res = M.render_from_manifest(E5 / "manifest_p0b.json", tmp_path, ["mel00"])
    assert res["ok"] and res["n_files"] == 16
    assert len({sf.info(str(p)).frames for p in (tmp_path / "sources").glob("mel00_*.wav")}) == 1
    res = M.render_from_manifest(E5 / "manifest_v2.json", tmp_path, ["v2c00"], hyp_programs=(54,), keep_hyps=False)
    assert res["ok"] and res["n_files"] == 16 + 35
    assert not list((tmp_path / "hyp_renders_v2_p54").glob("*.wav"))


@needs_fluidsynth
def test_a_changed_render_is_reported(tmp_path):
    M.render_from_manifest(E5 / "manifest_p0b.json", tmp_path, ["mel01"])
    p = tmp_path / "sources_t18" / "mel01_lit.wav"
    x, sr = sf.read(str(p), dtype="float32")
    x[1000] += 0.01
    sf.write(str(p), x, sr, subtype="FLOAT")
    assert audio_sha256(p) != P0B["items"][1]["source_sha256"]["t18"]["lit"]
    res = M.render_from_manifest(E5 / "manifest_p0b.json", tmp_path, ["mel01"])
    assert res["ok"]                                                     # a wrong file is rendered again
    with pytest.raises(RuntimeError):
        M.render_from_manifest(E5 / "manifest_p0b.json", tmp_path, ["mel01"], expect_sha256="0" * 64)


@needs_fluidsynth
def test_render_command_writes_where_e5_dir_points(tmp_path, monkeypatch):
    shutil.copy(E5 / "manifest_p0b.json", tmp_path / "manifest_p0b.json")
    monkeypatch.setenv("E5_DIR", str(tmp_path))
    assert M.cli(["render", "--manifest", "manifest_p0b.json", "--ids", "mel00"]) == 0
    assert len(list((tmp_path / "sources").glob("mel00_*.wav"))) == 8 and len(list((tmp_path / "sources_t18").glob("*.wav"))) == 8
