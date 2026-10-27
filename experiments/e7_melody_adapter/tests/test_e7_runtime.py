"""Exercise private-file handling, aggregation and dense-context boundaries."""
import csv
import importlib
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import pytest
import torch

from experiments.e7_melody_adapter import pack, suno


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=suno.TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(item, route, take, npa, wr, *, source="suno"):
    return {"source": source, "item_id": item, "route": route, "take": take,
            "pitch": {"own": {"strict": {"npa": npa, "start_s": 5., "speed_ratio": 1.03,
                                           "transpose_semitones": 2}}},
            "words": {"wr": wr}}


def test_inventory_lists_missing_takes_and_does_not_select_duplicates(tmp_path):
    directory = tmp_path / "audio"
    directory.mkdir()
    manifest = {"items": [{"id": "mel00"}, {"id": "mel01"}]}
    name = "S_mel00_R1_a.mp3"
    (directory / name).write_bytes(b"fixture")
    (directory / "rerolled.mp3").write_bytes(b"fixture")
    records = tmp_path / "table.csv"
    row = {"item_id": "mel00", "route": "R1", "take": "a", "filename": name,
           "route_available": "yes", "actual_option": "MIDI prompt", "model_version": "shown version",
           "date_utc": "2026-10-27", "credits_used": "10"}
    table(records, [row])
    result = suno.inventory(directory, records, manifest)
    assert len(result["missing_audio"]) == 11
    assert result["files_by_route"] == {"R1": 1, "R2": 0, "R3": 0}
    assert result["invalid_names"] == ["rerolled.mp3"]
    assert result["clips"][0]["owner_record"]["model_version"] == "shown version"
    duplicate = directory / "second"
    duplicate.mkdir()
    (duplicate / name).write_bytes(b"different output")
    with pytest.raises(ValueError, match="Duplicate audio"):
        suno.inventory(directory, records, manifest)


def test_record_table_duplicate_or_unavailable_with_audio_is_rejected(tmp_path):
    manifest = {"items": [{"id": "mel00"}]}
    inputs = tmp_path / "audio"
    inputs.mkdir()
    name = "S_mel00_R1_a.mp3"
    (inputs / name).write_bytes(b"fixture")
    records = tmp_path / "table.csv"
    row = {"item_id": "mel00", "route": "R1", "take": "a", "filename": name}
    table(records, [row, row])
    with pytest.raises(ValueError, match="Duplicate recording"):
        suno.inventory(inputs, records, manifest)
    table(records, [{**row, "route_available": "no"}])
    with pytest.raises(ValueError, match="unavailable"):
        suno.inventory(inputs, records, manifest)


def test_pair_average_precedes_item_median_and_single_takes_do_not_count():
    rows = []
    for item, values in enumerate(((0., 1.), (.6, .8), (.8, 1.))):
        for take, value in zip(suno.TAKES, values):
            rows.append(metric_row(str(item), "R1", take, value, .7))
    rows.append(metric_row("incomplete", "R1", "a", 1., 1.))
    result = suno.paired_summary(rows, ["R1", "R2"], minimum=3)
    assert result["R1"]["complete_items"] == 3
    assert result["R1"]["incomplete_item_ids"] == ["incomplete"]
    assert result["R1"]["metrics"]["pitch.own.strict.npa"]["median"] == pytest.approx(.7)
    assert result["R1"]["metrics"]["pitch.own.strict.npa"]["item_pair_means"]["0"] == .5
    assert result["R1"]["measured"] is True
    assert result["R2"]["measured"] is False
    with pytest.raises(ValueError, match="Duplicate scored take"):
        suno.paired_summary(rows + [rows[0]], ["R1"])


def test_decision_never_pools_thresholds_between_routes():
    rows = []
    for item in range(10):
        for take in suno.TAKES:
            rows.extend((metric_row(str(item), "R1", take, .95, .4),
                         metric_row(str(item), "R2", take, .5, .8)))
    routes = suno.paired_summary(rows, pack.ROUTES)
    ruler = {"complete_items": 12, "median_npa_ceiling": .99}
    assert suno.decide(routes, ruler)["status"] == "provisional_negative"
    routes["R1"]["metrics"]["words.wr"]["median"] = .6
    result = suno.decide(routes, ruler)
    assert result["status"] == "already_accurate" and result["passing_routes"] == ["R1"]
    assert suno.decide(routes, ruler, complete=False)["status"] == "incomplete_run"
    assert suno.decide(routes, {**ruler, "median_npa_ceiling": .949})["status"] == "ruler_insufficient_pending"
    assert suno.decide(suno.paired_summary([], pack.ROUTES), ruler)["status"] == "not_measured"


def test_dense_hook_keeps_actual_mask_and_silent_tail_then_stops():
    original = torch.zeros((2, 7, 128), dtype=torch.bfloat16)
    original[..., :64] = -1
    original[..., 64:] = 2
    guide = torch.full((5, 64), 3.)
    hook = suno.context_pre(guide, 3)
    kwargs = {"context_latents": original, "is_covers": torch.tensor([False, False])}
    updated = hook(0, 1., kwargs)
    assert torch.equal(updated["context_latents"][..., 64:], original[..., 64:])
    assert torch.all(updated["context_latents"][:, :5, :64] == 3)
    assert torch.all(updated["context_latents"][:, 5:, :64] == -1)
    assert torch.all(original[..., :64] == -1)
    assert hook(3, .5, kwargs) is kwargs
    with pytest.raises(RuntimeError, match="is_covers=False"):
        hook(0, 1., {**kwargs, "is_covers": True})


def test_ruler_auc_is_roc_auc_with_paired_item_resampling():
    rows = []
    # Every matched pair improves, yet the first clean is below the second corrupt.
    for item, (clean, corrupt) in enumerate(((.6, .5), (.9, .8))):
        for route in ("clean", "speed_1.03", "transpose_2", "two_wrong_notes"):
            row = metric_row(str(item), route, "a", corrupt if route == "two_wrong_notes" else clean,
                             0., source="ruler")
            row["recovery"] = {"start_error_s": .01, "speed_error": 0., "transpose_error": 0}
            rows.append(row)
    result = suno.ruler_summary(rows)
    assert result["paired_auc"] == .75
    assert result["matched_pair_win_rate"] == 1.
    assert result["paired_auc_ci95"][0] <= .75 <= result["paired_auc_ci95"][1]
    assert result["median_npa_ceiling"] == .75


def test_runtime_import_never_imports_the_song_model_or_starts_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import started a subprocess")
    monkeypatch.setattr(suno.subprocess, "run", forbidden)
    before = "dacapo.ace" in sys.modules
    importlib.reload(suno)
    assert ("dacapo.ace" in sys.modules) is before


def test_archives_separate_private_scores_and_exclude_audio(tmp_path, monkeypatch):
    from dacapo import provenance
    from experiments.e7_melody_adapter import registration
    monkeypatch.setattr(provenance, "environment", lambda: {"git_commit": "1" * 40, "git_dirty": False})
    # Prereg hashing uses the path only; no immutable registration is changed.
    original_digest = pack.sha256_file
    monkeypatch.setattr(pack, "sha256_file", lambda path: "f" * 64 if Path(path).name == "prereg_suno.json" else original_digest(path))
    work, output = tmp_path / "work", tmp_path / "out"
    work.mkdir()
    (work / "private-song.mp3").write_bytes(b"private audio")
    private_row = metric_row("mel00", "R1", "a", .812345, .723456)
    public_row = metric_row("mel00", "noguide", "a", .1, .2, source="ace")
    prereg = {"config": registration.build_config(), "code_sha256": {}, "manifest_sha256": "a" * 64}
    archives = suno.publish_archives(work, output, [public_row], [private_row], {"clips": []},
                                    prereg, 0., complete=False, failure="BudgetExceeded")
    with zipfile.ZipFile(archives["e7_suno_public.zip"]) as archive:
        assert all(Path(name).suffix in {".csv", ".json"} for name in archive.namelist())
        text = "\n".join(archive.read(name).decode() for name in archive.namelist())
        assert ".812345" not in text and ".723456" not in text
        data = json.loads(archive.read("records/e7_stage_s_public.json"))
        assert "routes" not in data["results"] and "decision_for_reviewer" not in data["results"]
        assert data["results"]["complete"] is False
    with zipfile.ZipFile(archives["suno_private.zip"]) as archive:
        assert all(Path(name).suffix in {".csv", ".json"} for name in archive.namelist())
        data = json.loads(archive.read("records/e7_stage_s_private.json"))
        assert data["results"]["decision_for_reviewer"]["status"] == "incomplete_run"
        assert data["results"]["rows"][0]["words"]["wr"] == .723456


def test_dirty_result_provenance_is_rejected(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"environment": {"git_commit": "1" * 40, "git_dirty": True}}))
    with pytest.raises(RuntimeError, match="clean known commit"):
        suno.assert_result_provenance([path])


def test_registered_budget_cannot_be_extended_between_clips(monkeypatch):
    clock = iter((10., 20.))
    monkeypatch.setattr(suno.time, "monotonic", lambda: next(clock))
    budget = suno.Budget(5)
    with pytest.raises(suno.BudgetExceeded):
        budget.check()


def test_generation_shares_seeds_uses_text2music_and_observes_masks(tmp_path, monkeypatch):
    import soundfile as sf
    import dacapo
    from dacapo import render
    from experiments.e7_melody_adapter.registration import build_config

    calls, checkpoints = [], []
    ace = types.ModuleType("dacapo.ace")
    seeds = types.ModuleType("dacapo.seeds")
    seeds.run_seeds = pack.run_seeds
    ace.load = lambda device: object()
    ace.encode = lambda handler, path: np.ones((40, 64), dtype=np.float32)

    def generate(handler, path, caption, duration, seed, **kwargs):
        calls.append({"seed": seed, "caption": caption, "duration": duration, **kwargs})
        original = torch.zeros((1, 60, 128))
        original[..., 64:] = 2
        if kwargs["pre"]:
            changed = kwargs["pre"](0, 1., {"context_latents": original, "is_covers": False})
            assert torch.equal(changed["context_latents"][..., 64:], original[..., 64:])
        return {"seconds": .2, "n_denoiser_calls": 8, "captured": {"context_latents": original}}

    def render_stub(melody, path, program, sr):
        sf.write(path, np.zeros((sr, 2), np.float32), sr)

    ace.generate = generate
    monkeypatch.setitem(sys.modules, "dacapo.ace", ace)
    monkeypatch.setitem(sys.modules, "dacapo.seeds", seeds)
    monkeypatch.setattr(dacapo, "ace", ace, raising=False)
    monkeypatch.setattr(render, "render_melody", render_stub)
    monkeypatch.setattr(suno, "write_checkpoint", lambda path, cfg, results, started: checkpoints.append(results))
    manifest = pack.build_manifest()
    manifest["items"] = manifest["items"][:1]
    cfg = build_config()["suno"]
    clips = suno.generate_controls(manifest, tmp_path, suno.Budget(60), cfg, {})
    assert len(clips) == len(calls) == len(checkpoints) == 8
    assert {call["task"] for call in calls} == {"text2music"}
    assert {call["caption"] for call in calls} == {"pop, full band, female vocal"}
    assert all(call["codes"] == "" and call["src"] is None and call["capture"] for call in calls)
    assert all(clip["context_mask_values"] == [2.] for clip in clips)
    assert len({(call["seed"], call["rng_seed"]) for call in calls[:4]}) == 1
    assert len({(call["seed"], call["rng_seed"]) for call in calls[4:]}) == 1
    assert calls[0]["seed"] != calls[4]["seed"]


def test_ruler_renders_registered_variants_and_exactly_repeats_block(tmp_path, monkeypatch):
    import soundfile as sf
    from dacapo import render
    from experiments.e7_melody_adapter.registration import build_config

    rendered = []

    def voice_stub(melody, path, program, sr):
        rendered.append(melody)
        assert program == 53 and sr == 48000
        # A finite nonzero render tests sample layout and peak scaling cheaply.
        sf.write(path, np.full((sr, 2), .7, dtype=np.float32), sr, subtype="FLOAT")

    def band_stub(instruments, bpm, path, sr):
        sf.write(path, np.full((sr, 2), .2, dtype=np.float32), sr, subtype="FLOAT")

    monkeypatch.setattr(render, "render_melody", voice_stub)
    monkeypatch.setattr(render, "render", band_stub)
    manifest = pack.build_manifest()
    manifest["items"] = manifest["items"][:1]
    cfg = build_config()["ruler_self_check"]
    clips = suno.render_ruler(manifest, tmp_path, suno.Budget(60), cfg)
    assert len(clips) == 4
    assert rendered[1].bpm == pytest.approx(rendered[0].bpm * 1.03)
    assert rendered[2].pitches() == [pitch + 2 for pitch in rendered[0].pitches()]
    differences = np.asarray(rendered[3].pitches()) - rendered[0].pitches()
    assert np.flatnonzero(differences).tolist() == [5, 21]
    assert differences[[5, 21]].tolist() == [1, -1]
    for clip, melody in zip(clips, rendered):
        audio, sr = sf.read(clip["path"], always_2d=True)
        assert sr == 48000 and len(audio) % 2 == 0
        assert np.array_equal(audio[:len(audio) // 2], audio[len(audio) // 2:])
        assert melody.notes[0].start * melody.spb == pytest.approx(5.)
        assert clip["expected_start_s"][1] - clip["expected_start_s"][0] == pytest.approx((len(audio) // 2) / sr)


@pytest.mark.parametrize("complete,mismatch,error", [
    (True, False, None),
    (False, False, "snapshot is incomplete"),
    (True, True, "source and checkpoint code differ"),
])
def test_gpu_guard_rejects_loader_downloads_and_checkpoint_edits(tmp_path, monkeypatch,
                                                               complete, mismatch, error):
    import dacapo
    from experiments.e7_melody_adapter.registration import build_config

    config = build_config()
    models = config["models"]
    root, source = tmp_path / "repo", tmp_path / "ace-source"
    root.mkdir()
    source.mkdir()
    weights = tmp_path / "snapshots" / models["ace_weights_revision"]
    weights.mkdir(parents=True)
    (source / "checkpoints").symlink_to(weights, target_is_directory=True)
    monkeypatch.chdir(root)
    monkeypatch.setattr(suno, "ROOT", root)
    monkeypatch.setenv("ACESTEP_REPO", str(source))
    monkeypatch.setenv("E7_ACE_WEIGHTS_DIR", str(weights))
    monkeypatch.setattr(suno, "checkout_state", lambda: None)
    monkeypatch.setattr(suno, "verify_registration", lambda: {"config": config})
    monkeypatch.setattr(suno, "require_e5_source", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command == [sys.executable, "experiments/e5_motif/register.py", "--check"]

    def check_output(command, **kwargs):
        assert command[:3] == ["git", "-C", str(source)]
        return models["ace_source_commit"] + "\n" if "rev-parse" in command else ""

    monkeypatch.setattr(suno.subprocess, "run", run)
    monkeypatch.setattr(suno.subprocess, "check_output", check_output)
    ace = types.ModuleType("dacapo.ace")
    ace.generate = lambda rng_seed=None: None

    def forbidden_load(*args, **kwargs):
        raise AssertionError("The guard must not load, download, or edit model files.")

    ace.load = forbidden_load
    monkeypatch.setitem(sys.modules, "dacapo.ace", ace)
    monkeypatch.setattr(dacapo, "ace", ace, raising=False)
    package = types.ModuleType("acestep")
    package.__path__ = []
    downloader = types.ModuleType("acestep.model_downloader")
    checks = []

    def check_exists(path):
        checks.append("exists")
        assert path == weights
        return complete

    def check_mismatch(model, path):
        checks.append("mismatch")
        assert model == "acestep-v15-turbo" and path == weights
        return mismatch

    downloader.check_main_model_exists = check_exists
    downloader._check_code_mismatch = check_mismatch
    monkeypatch.setitem(sys.modules, "acestep", package)
    monkeypatch.setitem(sys.modules, "acestep.model_downloader", downloader)
    if error is None:
        assert suno.require_gpu_run() == {"config": config}
    else:
        with pytest.raises(RuntimeError, match=error):
            suno.require_gpu_run()
    assert checks == (["exists", "mismatch"] if complete else ["exists"])
    assert len(commands) == 1
    assert list(weights.iterdir()) == []
