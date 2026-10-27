"""Check resumable measurement, input faults, E4 controls and export privacy."""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import numpy as np
import pytest
import soundfile as sf

from experiments.e7_melody_adapter import pack, registration, suno


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=suno.TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(item, route, take, npa, wr, *, source="suno"):
    row = {"source": source, "item_id": item, "route": route, "take": take,
           "pitch": {"own": {"strict": {"npa": npa, "start_s": 5., "speed_ratio": 1.03,
                                         "transpose_semitones": 2}}}, "words": {"wr": wr}}
    if source == "ruler":
        row["recovery"] = {"start_error_s": 0., "speed_error": 0., "transpose_error": 0}
    return row


@pytest.fixture
def clean_provenance(monkeypatch):
    from dacapo import provenance
    monkeypatch.setattr(provenance, "environment", lambda: {"git_commit": "1" * 40, "git_dirty": False})


def test_inventory_lists_missing_and_unreadable_and_accepts_extension_case(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    good = audio / "S_mel00_R1_a.MP3"
    # A real short WAV payload is decodable independent of its supplied extension.
    sf.write(good, np.zeros((160, 2)), 16000, format="WAV")
    (audio / "S_mel00_R1_b.mp3").write_bytes(b"")
    (audio / "S_mel01_R1_a.mp3").write_bytes(b"not audio")
    (audio / "extra.mp3").write_bytes(b"unregistered")
    records = tmp_path / "table.csv"
    table(records, [{"item_id": "mel00", "route": "R1", "take": "a", "filename": good.name,
                     "route_available": "yes", "bpm": 100, "key": "C major", "style": pack.STYLE}])
    inv = suno.inventory(audio, records, {"items": [{"id": "mel00"}, {"id": "mel01"}]})
    assert len(inv["clips"]) == 1
    assert inv["clips"][0]["prefix_samples"] == 160
    assert inv["clips"][0]["filename"] == "S_mel00_R1_a.mp3"
    assert len(inv["unreadable_audio"]) == 2
    assert len(inv["missing_audio"]) == 9
    assert inv["invalid_names"] == ["extra.mp3"]
    assert all("same returned song" in row["action"] for row in inv["unreadable_audio"])
    (audio / "S_mel00_R1_a.mp3").write_bytes(good.read_bytes())
    with pytest.raises(ValueError, match="Duplicate audio"):
        suno.inventory(audio, records, {"items": [{"id": "mel00"}, {"id": "mel01"}]})


def test_inventory_rejects_invalid_route_values_and_duplicate_table_rows(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    records = tmp_path / "table.csv"
    row = {"item_id": "mel00", "route": "R1", "take": "a", "filename": "S_mel00_R1_a.mp3",
           "route_available": "sometimes"}
    table(records, [row])
    with pytest.raises(ValueError, match="route_available"):
        suno.inventory(audio, records, {"items": [{"id": "mel00"}]})
    row["route_available"] = "unknown"
    table(records, [row, row])
    with pytest.raises(ValueError, match="Duplicate recording"):
        suno.inventory(audio, records, {"items": [{"id": "mel00"}]})


def test_pair_average_then_median_and_failed_takes_do_not_count():
    rows = []
    for item, values in enumerate(((0., 1.), (.6, .8), (.8, 1.))):
        for take, value in zip(suno.TAKES, values):
            rows.append(metric_row(str(item), "R1", take, value, .7))
    rows.append(metric_row("incomplete", "R1", "a", 1., 1.))
    rows.append({"route": "R1", "item_id": "incomplete", "take": "b", "status": "failed"})
    result = suno.paired_summary(rows, ["R1", "R2"], minimum=3)
    assert result["R1"]["complete_items"] == 3
    assert result["R1"]["metrics"]["pitch.own.strict.npa"]["median"] == pytest.approx(.7)
    assert result["R1"]["metrics"]["pitch.own.strict.npa"]["item_pair_means"]["0"] == .5
    assert result["R2"]["measured"] is False
    with pytest.raises(ValueError, match="Duplicate scored take"):
        suno.paired_summary(rows + [rows[0]], ["R1"])


def test_decision_never_pools_thresholds_between_routes():
    rows = [metric_row(str(item), route, take, npa, wr)
            for item in range(10) for take in suno.TAKES
            for route, npa, wr in (("R1", .95, .4), ("R2", .5, .8))]
    routes = suno.paired_summary(rows, pack.ROUTES)
    ruler = {"complete_items": 12, "median_npa_ceiling": .99}
    assert suno.decide(routes, ruler)["status"] == "provisional_negative"
    routes["R1"]["metrics"]["words.wr"]["median"] = .6
    assert suno.decide(routes, ruler)["status"] == "already_accurate"
    assert suno.decide(routes, ruler, complete=False)["status"] == "incomplete_run"
    assert suno.decide(routes, {**ruler, "median_npa_ceiling": .949})["status"] == "ruler_insufficient_pending"
    assert suno.decide(suno.paired_summary([], pack.ROUTES), ruler)["status"] == "not_measured"


def test_ruler_auc_differs_from_matched_pair_win_rate():
    rows = [metric_row(str(item), route, "a", corrupt if route == "two_wrong_notes" else clean, 0., source="ruler")
            for item, (clean, corrupt) in enumerate(((.6, .5), (.9, .8)))
            for route in ("clean", "speed_1.03", "transpose_2", "two_wrong_notes")]
    result = suno.ruler_summary(rows)
    assert result["paired_auc"] == .75 and result["matched_pair_win_rate"] == 1.
    assert result["paired_auc_ci95"][0] <= .75 <= result["paired_auc_ci95"][1]


def test_import_does_not_load_models_or_start_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import started a subprocess")
    monkeypatch.setattr(suno.subprocess, "run", forbidden)
    before = "dacapo.ace" in sys.modules
    importlib.reload(suno)
    assert ("dacapo.ace" in sys.modules) is before


def test_budget_counts_only_wrapped_generation_calls_and_resumes(monkeypatch):
    ticks = iter((100., 102., 1000., 1002.))
    monkeypatch.setattr(suno.time, "monotonic", lambda: next(ticks))
    # Avoid real signals in this deterministic clock test.
    monkeypatch.delattr(suno.signal, "SIGALRM")
    updates = []
    budget = suno.Budget(5., used=1., on_update=updates.append)
    assert budget.call(lambda: "a") == "a"
    budget.check()
    assert budget.call(lambda: "b") == "b"
    assert updates == [3., 5.]
    with pytest.raises(suno.BudgetExceeded):
        budget.call(lambda: "must not run")


def test_resume_store_reuses_hash_bound_scores_and_rejects_changed_assets(tmp_path, clean_provenance):
    cfg = {"prereg_sha256": "a" * 64, "ruler": "swift"}
    store = suno.ResumeStore(tmp_path / "drive", tmp_path / "scratch", cfg)
    path = store.root / "assets" / "a.wav"
    path.parent.mkdir()
    path.write_bytes(b"original")
    clip = {"source": "ruler", "item_id": "mel00", "route": "clean", "take": "a",
            "path": path, "filename": path.name}
    store.put_asset(clip)
    row = metric_row("mel00", "clean", "a", 1., 0., source="ruler")
    store.put_score(clip, row)
    resumed = suno.ResumeStore(tmp_path / "drive", tmp_path / "new-runtime", cfg)
    assert resumed.score(clip)["pitch"] == row["pitch"]
    assert resumed.asset(suno.clip_key(clip))["path"] == path
    with pytest.raises(ValueError, match="Checkpoint registration"):
        suno.ResumeStore(tmp_path / "drive", tmp_path / "scratch2", {**cfg, "ruler": "different"})
    path.write_bytes(b"changed")
    assert resumed.score(clip) is None
    with pytest.raises(RuntimeError, match="restore the same"):
        resumed.asset(suno.clip_key(clip))


def fake_run_environment(tmp_path, monkeypatch):
    """Keep real checkpoint, sequencing, summary and archive code in run()."""
    cfg = registration.build_config()
    prereg = {"config": cfg, "code_sha256": {}, "manifest_sha256": "a" * 64}
    here = tmp_path / "e7"
    here.mkdir()
    (here / "prereg_suno.json").write_text(json.dumps(prereg))
    monkeypatch.setattr(suno, "HERE", here)
    monkeypatch.setattr(suno, "ROOT", tmp_path / "clone")
    events, scored, generated = [], [], []
    def guard():
        events.append("gpu_guard")
        assert events[0] == "inventory"
        return prereg
    monkeypatch.setattr(suno, "require_gpu_run", guard)
    audio = tmp_path / "audio"
    audio.mkdir()
    clips = []
    for item in pack.load_manifest()["items"]:
        for take in suno.TAKES:
            path = audio / f"S_{item['id']}_R1_{take}.mp3"
            path.write_bytes(path.name.encode())
            clips.append({"source": "suno", "item_id": item["id"], "route": "R1", "take": take,
                          "path": path, "filename": path.name, "owner_record": {}})
    def inventory(*args):
        events.append("inventory")
        return {"clips": clips, "missing_audio": [], "unreadable_audio": [], "invalid_names": [], "warnings": []}
    monkeypatch.setattr(suno, "inventory", inventory)
    def render(manifest, store, ruler_cfg):
        events.append("ruler_render")
        result = []
        for item in manifest["items"]:
            for variant in ruler_cfg["variants"]:
                meta = {"source": "ruler", "item_id": item["id"], "route": variant, "take": "a"}
                saved = store.asset(suno.clip_key(meta))
                if saved:
                    result.append(saved)
                    continue
                path = store.root / "assets" / (item["id"] + variant + ".wav")
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(path.name.encode())
                generated.append(suno.clip_key(meta))
                result.append(store.put_asset({**meta, "path": path, "filename": path.name}))
        return result
    monkeypatch.setattr(suno, "render_ruler", render)
    return cfg, clips, events, scored, generated, audio, tmp_path / "drive"


def read_private(archives):
    with zipfile.ZipFile(archives["suno_private.zip"]) as archive:
        return json.loads(archive.read("records/e7_stage_s_private.json"))["results"]


def test_run_resumes_after_interruption_without_rescoring_or_regenerating(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, scored, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    fail_once = {"active": True}
    class Scorer:
        def __init__(self, *args): pass
        def close(self): pass
        def score(self, clip, items, work):
            if clip["source"] == "suno" and clip["item_id"] == "mel01" and fail_once["active"]:
                fail_once["active"] = False
                raise KeyboardInterrupt("runtime stopped")
            scored.append(suno.clip_key(clip))
            events.append(clip["source"])
            return metric_row(clip["item_id"], clip["route"], clip["take"], 1., .8, source=clip["source"])
    monkeypatch.setattr(suno, "AudioScorer", Scorer)
    def controls(manifest, store, control_cfg):
        events.append("ace_generate")
        assert sum(1 for key in scored if key.startswith("suno/")) == 24
        store.state["controls_generation_status"] = "budget_exceeded"
        return []
    monkeypatch.setattr(suno, "generate_controls", controls)
    with pytest.raises(KeyboardInterrupt):
        suno.run(audio, tmp_path / "table.csv", output)
    assert len(generated) == 48
    first_scored = list(scored)
    assert "ace_generate" not in events
    archives = suno.run(audio, tmp_path / "table.csv", output)
    assert len(generated) == 48
    assert len(scored) == len(set(scored)) == 72
    assert scored[:len(first_scored)] == first_scored
    assert events.index("suno") > max(i for i, value in enumerate(events[:events.index("suno")]) if value == "ruler")
    private = read_private(archives)
    assert private["complete"] is True
    assert private["decision_for_reviewer"]["status"] == "already_accurate"
    with zipfile.ZipFile(archives["e7_suno_public.zip"]) as archive:
        public = json.loads(archive.read("records/e7_stage_s_public.json"))["results"]
        assert public["controls_status"] == "budget_exceeded"
        assert public["stage_s_statement"].startswith("Stage S was evaluated")
    # A third runtime only reads saved audio and scores; no scorer work remains.
    suno.run(audio, tmp_path / "table.csv", output)
    assert len(scored) == 72 and len(generated) == 48


def test_suno_clip_failure_does_not_stop_batch_and_is_retryable(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, scored, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    fail_once = {"active": True}
    class Scorer:
        def __init__(self, *args): pass
        def close(self): pass
        def score(self, clip, items, work):
            key = suno.clip_key(clip)
            if key == "suno/mel00/R1/a" and fail_once["active"]:
                fail_once["active"] = False
                raise ValueError("decode failed later in the file")
            scored.append(key)
            return metric_row(clip["item_id"], clip["route"], clip["take"], 1., .8, source=clip["source"])
    monkeypatch.setattr(suno, "AudioScorer", Scorer)
    def controls(manifest, store, cfg):
        store.state["controls_generation_status"] = "complete"
        return []
    monkeypatch.setattr(suno, "generate_controls", controls)
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and len(result["rows"]) == 23
    assert result["routes"]["R1"]["complete_items"] == 11
    assert result["decision_for_reviewer"]["status"] == "already_accurate"
    assert result["failed_clips"][0]["reason"].startswith("ValueError:")
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert len(result["rows"]) == 24 and result["failed_clips"] == []
    assert len(scored) == len(set(scored)) == 72


def test_public_archive_incomplete_wording_and_private_data_separation(tmp_path, monkeypatch, clean_provenance):
    original = pack.sha256_file
    monkeypatch.setattr(pack, "sha256_file", lambda path: "f" * 64 if Path(path).name == "prereg_suno.json" else original(path))
    work = tmp_path / "work"
    work.mkdir()
    prereg = {"config": registration.build_config(), "code_sha256": {}, "manifest_sha256": "a" * 64}
    archives = suno.publish_archives(work, tmp_path / "out", [], [metric_row("mel00", "R1", "a", .812345, .723456)],
                                    {"clips": []}, prereg, 0., False)
    with zipfile.ZipFile(archives["e7_suno_public.zip"]) as archive:
        text = "\n".join(archive.read(name).decode() for name in archive.namelist())
        assert ".812345" not in text and ".723456" not in text
        public = json.loads(archive.read("records/e7_stage_s_public.json"))["results"]
        assert "did not complete" in public["stage_s_statement"]
        assert "was evaluated" not in public["stage_s_statement"]
        assert all(Path(name).suffix in {".json", ".csv"} for name in archive.namelist())
    assert read_private(archives)["decision_for_reviewer"]["status"] == "incomplete_run"


def test_dirty_checkpoint_provenance_is_rejected(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"environment": {"git_commit": "1" * 40, "git_dirty": True}}))
    with pytest.raises(RuntimeError, match="clean known commit"):
        suno.assert_result_provenance([path])


def test_control_generation_uses_e4_arguments_and_resumes_without_model_load(tmp_path, monkeypatch, clean_provenance):
    import dacapo
    from dacapo import render
    cfg = registration.build_config()["suno"]
    fake_ace = types.ModuleType("dacapo.ace")
    calls, loads = [], []
    fake_ace.load = lambda device: loads.append(device) or object()
    def generate(handler, path, caption, duration, seed, **kwargs):
        calls.append({"caption": caption, "duration": duration, "seed": seed, **kwargs})
        sf.write(path, np.zeros((160, 2)), 16000)
        return {"seconds": .001, "n_denoiser_calls": 8}
    fake_ace.generate = generate
    monkeypatch.setitem(sys.modules, "dacapo.ace", fake_ace)
    monkeypatch.setattr(dacapo, "ace", fake_ace, raising=False)
    seed_module = types.ModuleType("dacapo.seeds")
    seed_module.run_seeds = pack.run_seeds
    monkeypatch.setitem(sys.modules, "dacapo.seeds", seed_module)
    e4 = types.ModuleType("experiments.e4_sing.run_sing")
    e4.arms = lambda guides: cfg["arm_definitions"]
    monkeypatch.setitem(sys.modules, "experiments.e4_sing.run_sing", e4)
    def render_guide(melody, path, **kwargs):
        sf.write(path, np.zeros((480, 2)), 48000)
    monkeypatch.setattr(render, "render_melody", render_guide)
    manifest = {"items": pack.load_manifest()["items"][:1]}
    store = suno.ResumeStore(tmp_path / "drive", tmp_path / "work", {"prereg_sha256": "a" * 64})
    clips = suno.generate_controls(manifest, store, cfg)
    assert len(clips) == len(calls) == 8 and loads == ["cuda"]
    for draw in range(2):
        batch = calls[draw * 4:draw * 4 + 4]
        assert len({call["seed"] for call in batch}) == 1
        assert len({call["rng_seed"] for call in batch}) == 1
        for call, arm in zip(batch, cfg["arm_definitions"]):
            assert call["task"] == arm["task"]
            assert call["cover_strength"] == arm.get("cover_strength", 1.)
            assert call["cover_noise"] == arm.get("cover_noise", 0.)
            assert (call["src"] is not None) == ("guide" in arm)
            assert call["caption"] == pack.STYLE
            assert "pre" not in call and call["codes"] == ""
    resumed = suno.ResumeStore(tmp_path / "drive", tmp_path / "new-runtime", store.config)
    again = suno.generate_controls(manifest, resumed, cfg)
    assert len(again) == 8 and len(calls) == 8 and loads == ["cuda"]
    assert "generation_in_flight" not in resumed.state
    assert resumed.state["generation_elapsed_seconds"] > 0

    # A hard reset after the native output move still recovers the final audio,
    # even though its unresolved in-flight call reserves all remaining capacity.
    reset = suno.ResumeStore(tmp_path / "reset-drive", tmp_path / "reset-work", store.config)
    reset_directory = reset.root / "assets" / "controls"
    reset_directory.mkdir(parents=True)
    first = clips[0]
    (reset_directory / first["filename"]).write_bytes(first["path"].read_bytes())
    reset.begin_generation(cfg["ace_generation_budget_seconds"])
    after_reset = suno.ResumeStore(tmp_path / "reset-drive", tmp_path / "after-reset", store.config)
    recovered = suno.generate_controls(manifest, after_reset, cfg)
    assert len(recovered) == 1 and recovered[0]["recovered_completed_output"]
    assert after_reset.state["controls_generation_status"] == "budget_exceeded"
    assert after_reset.state["generation_elapsed_seconds"] == 0
    assert after_reset.generation_budget_used == cfg["ace_generation_budget_seconds"]
    assert len(calls) == 8 and loads == ["cuda"]

    # A remaining generation must verify its saved guide before it can use it.
    missing = resumed.state["assets"].pop("ace/mel00/ctx_oohs_k3/b")
    (resumed.root / missing["asset_path"]).unlink()
    guide = resumed.state["assets"]["guide/mel00/oohs/a"]
    (resumed.root / guide["asset_path"]).write_bytes(b"changed guide")
    resumed.save()
    with pytest.raises(RuntimeError, match="restore the same saved file"):
        suno.generate_controls(manifest, resumed, cfg)
    assert len(calls) == 8
