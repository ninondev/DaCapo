"""Check resumable measurement, input faults, E4 controls and export privacy."""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
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
    assert set(inv["unreadable_paths"]) == {"S_mel00_R1_b.mp3", "S_mel01_R1_a.mp3"}
    assert [(row["item_id"], row["route"], row["take"]) for row in inv["unreadable_audio"]] == [
        ("mel00", "R1", "b"), ("mel01", "R1", "a")]
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


def test_null_metric_removes_only_that_item_from_that_metric():
    rows = []
    for item in range(12):
        for take in suno.TAKES:
            row = metric_row(f"mel{item:02d}", "R1", take, .9, .7)
            short = item == 3 and take == "b"
            row["pitch"]["own"]["relaxed"] = {"voiced_deviation_semitones": None if short else .2 + item / 100,
                                              "dtw_cost_semitones": None if short else 1.}
            rows.append(row)
    route = suno.paired_summary(rows, ["R1"])["R1"]
    assert route["complete_items"] == 12
    voiced = route["metrics"]["pitch.own.relaxed.voiced_deviation_semitones"]
    assert voiced["items"] == 11 and "mel03" not in voiced["item_pair_means"]
    assert route["metrics"]["pitch.own.relaxed.dtw_cost_semitones"]["items"] == 11
    assert route["metrics"]["pitch.own.strict.npa"]["items"] == 12
    assert route["metrics"]["words.wr"]["items"] == 12


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
    # Each update carries only that call's measured time.
    assert updates == [2., 2.] and budget.used == 5.
    with pytest.raises(suno.BudgetExceeded):
        budget.call(lambda: "must not run")


def test_budget_alarm_is_enforced_when_the_call_swallows_it():
    def swallowed_then_failed():
        try:
            time.sleep(2)
        except Exception as exc:
            # ACE returns success=False and the wrapper raises a plain error.
            raise RuntimeError(str(exc))
    def swallowed_then_returned():
        try:
            time.sleep(2)
        except Exception:
            return {"success": False}
    for function in (swallowed_then_failed, swallowed_then_returned):
        budget = suno.Budget(10., call_limit=.05)
        with pytest.raises(suno.BudgetExceeded):
            budget.call(function)
        assert budget.alarm_fired and budget.used < 1.


def test_budget_reserves_a_bounded_call_allowance(monkeypatch):
    monkeypatch.delattr(suno.signal, "SIGALRM")
    reserved = []
    budget = suno.Budget(925., call_limit=120., on_start=reserved.append)
    budget.call(lambda: None)
    budget.used = 900.
    budget.call(lambda: None)
    assert reserved[0] == 120. and reserved[1] == pytest.approx(25.)


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


def public_of(archives):
    with zipfile.ZipFile(archives["e7_suno_public.zip"]) as archive:
        return json.loads(archive.read("records/e7_stage_s_public.json"))["results"]


def fake_scorer(monkeypatch, fault=None, *, start_fault=None, npa=lambda clip: 1.):
    """A fast scorer whose faults are chosen per clip by the test."""
    scored, starts = [], []
    class Scorer:
        def __init__(self, *args):
            starts.append(True)
            if start_fault is not None and start_fault():
                raise OSError("Model download failed")
        def close(self): pass
        def score(self, clip, items, work):
            if fault is not None:
                fault(clip)
            scored.append(suno.clip_key(clip))
            return metric_row(clip["item_id"], clip["route"], clip["take"], npa(clip), .8, source=clip["source"])
    monkeypatch.setattr(suno, "AudioScorer", Scorer)
    return scored, starts


def no_controls(monkeypatch, status="complete"):
    def controls(manifest, store, cfg):
        store.state["controls_generation_status"] = status
    monkeypatch.setattr(suno, "generate_controls", controls)


def test_single_failed_clip_continues_but_leaves_the_run_pending(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    fail_once = {"active": True}
    def fault(clip):
        if suno.clip_key(clip) == "suno/mel00/R1/a" and fail_once["active"]:
            fail_once["active"] = False
            raise ValueError("decode failed later in the file")
    scored, _ = fake_scorer(monkeypatch, fault)
    no_controls(monkeypatch)
    archives = suno.run(audio, tmp_path / "table.csv", output)
    result = read_private(archives)
    assert archives["complete"] is False and result["complete"] is False
    assert len(result["rows"]) == 23 and result["pending_clips"] == ["suno/mel00/R1/a"]
    assert result["decision_for_reviewer"]["status"] == "scoring_failures_pending"
    assert result["failed_clips"][0]["reason"].startswith("ValueError:")
    assert "did not complete" in public_of(archives)["stage_s_statement"]
    archives = suno.run(audio, tmp_path / "table.csv", output)
    result = read_private(archives)
    assert result["complete"] is True and len(result["rows"]) == 24 and result["failed_clips"] == []
    assert result["decision_for_reviewer"]["status"] == "already_accurate"
    assert public_of(archives)["stage_s_statement"].startswith("Stage S was evaluated")
    assert len(scored) == len(set(scored)) == 72


def test_scorer_start_failure_on_resume_ends_the_run_incomplete(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    state = {"interrupt": True, "start_fails": False}
    def fault(clip):
        if clip["source"] == "suno" and state["interrupt"]:
            state["interrupt"] = False
            raise KeyboardInterrupt("runtime lost after the ruler")
    scored, starts = fake_scorer(monkeypatch, fault, start_fault=lambda: state["start_fails"])
    no_controls(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        suno.run(audio, tmp_path / "table.csv", output)
    state["start_fails"] = True
    with pytest.raises(suno.RunWideFault, match="could not start"):
        suno.run(audio, tmp_path / "table.csv", output)
    archives = {name: output / name for name in ("e7_suno_public.zip", "suno_private.zip")}
    result = read_private(archives)
    assert result["complete"] is False and result["failed_clips"] == []
    assert result["decision_for_reviewer"]["status"] == "incomplete_run"
    assert result["failure"] == "RunWideFault"
    assert "did not complete" in public_of(archives)["stage_s_statement"]
    state["start_fails"] = False
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and len(result["rows"]) == 24


def test_sticky_cuda_fault_ends_the_run_without_clip_failures(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    state = {"suno": 0, "poisoned": True}
    def fault(clip):
        # After the fault every call fails, so the runtime probe fails as well.
        if clip["source"] == "suno" and state["poisoned"]:
            state["suno"] += 1
        if state["poisoned"] and state["suno"] > 6:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
    scored, _ = fake_scorer(monkeypatch, fault)
    no_controls(monkeypatch)
    with pytest.raises(RuntimeError, match="CUDA error"):
        suno.run(audio, tmp_path / "table.csv", output)
    archives = {name: output / name for name in ("e7_suno_public.zip", "suno_private.zip")}
    result = read_private(archives)
    assert result["complete"] is False and len(result["rows"]) == 6
    assert result["failed_clips"] == []
    assert result["decision_for_reviewer"]["status"] == "incomplete_run"
    assert "did not complete" in public_of(archives)["stage_s_statement"]
    state["poisoned"] = False
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and len(result["rows"]) == 24


@pytest.mark.parametrize("make", [
    lambda: __import__("concurrent.futures.process", fromlist=["BrokenProcessPool"]).BrokenProcessPool("pool"),
    lambda: MemoryError(),
    lambda: RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    lambda: ValueError("wrapped"),
])
def test_run_wide_fault_classification(make):
    exc = make()
    if isinstance(exc, ValueError):
        assert not suno.run_wide_fault(exc)
        exc.__cause__ = RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")
    assert suno.run_wide_fault(exc)


def test_consecutive_failures_with_a_failed_probe_abort_and_do_not_count(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    broken = {"active": False}
    def fault(clip):
        # A decoder library breaks at mel02; the ruler probe then fails too.
        broken["active"] |= clip["source"] == "suno" and clip["item_id"] == "mel02"
        if broken["active"]:
            raise ValueError("unexpected decoder state")
    fake_scorer(monkeypatch, fault)
    no_controls(monkeypatch)
    limit = cfg["suno"]["consecutive_failure_abort"]
    with pytest.raises(suno.RunWideFault, match=f"{limit} consecutive"):
        suno.run(audio, tmp_path / "table.csv", output)
    state = json.loads((output / "checkpoints/state.json").read_text())["results"]
    assert len(state["scoring_failures"]) == limit
    assert all(record["counted_runtimes"] == [] for record in state["scoring_failures"].values())


def test_identical_failure_in_three_finished_passes_becomes_a_listed_exclusion(
        tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    def fault(clip):
        if suno.clip_key(clip) == "suno/mel05/R1/b":
            raise ValueError("the decoder stops in the middle of this file")
    scored, starts = fake_scorer(monkeypatch, fault)
    no_controls(monkeypatch)
    assert cfg["suno"]["failure_exclusion_runtimes"] == 3
    for _ in range(2):
        result = read_private(suno.run(audio, tmp_path / "table.csv", output))
        assert result["complete"] is False
        assert result["decision_for_reviewer"]["status"] == "scoring_failures_pending"
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and result["pending_clips"] == []
    assert result["excluded_clips"] == ["suno/mel05/R1/b"]
    assert result["routes"]["R1"]["complete_items"] == 11
    assert result["decision_for_reviewer"]["status"] == "already_accurate"
    assert [row["status"] for row in result["failed_clips"]] == ["excluded"]
    # An excluded take is not retried; the scorer is not even started.
    before = (len(scored), len(starts))
    assert read_private(suno.run(audio, tmp_path / "table.csv", output))["complete"] is True
    assert (len(scored), len(starts)) == before
    # A new download with different bytes starts a new attempt.
    (audio / "S_mel05_R1_b.mp3").write_bytes(b"downloaded again")
    fake_scorer(monkeypatch)
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and result["excluded_clips"] == []
    assert result["routes"]["R1"]["complete_items"] == 12 and result["failed_clips"] == []
    history = result["resolved_failures"]
    assert len(history) == 1 and history[0]["status"] == "excluded"
    assert history[0]["scored_audio_sha256"] != history[0]["audio_sha256"]


def test_empty_inputs_are_not_reported_as_evaluated(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    clips.clear()
    fake_scorer(monkeypatch)
    no_controls(monkeypatch)
    archives = suno.run(audio, tmp_path / "table.csv", output)
    assert read_private(archives)["decision_for_reviewer"]["status"] == "incomplete_run"
    assert "did not complete" in public_of(archives)["stage_s_statement"]


def test_main_exits_nonzero_for_an_unfinished_run(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["suno", "--inputs", "a", "--records-csv", "b", "--output-dir", "c"])
    monkeypatch.setattr(suno, "run", lambda *args: {"complete": False})
    with pytest.raises(SystemExit) as stopped:
        suno.main()
    assert stopped.value.code == 3
    monkeypatch.setattr(suno, "run", lambda *args: {"complete": True})
    suno.main()


def test_decision_is_published_before_controls_and_saved_controls_are_scored(
        tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    scored, _ = fake_scorer(monkeypatch)
    def controls(manifest, store, control_cfg):
        # A hard kill here must leave the finished Suno and ruler result behind.
        early = read_private({"suno_private.zip": output / "suno_private.zip"})
        assert early["complete"] is True
        assert early["decision_for_reviewer"]["status"] == "already_accurate"
        directory = store.root / "assets" / "controls"
        directory.mkdir(parents=True, exist_ok=True)
        for arm in suno.ARMS[:2]:
            path = directory / f"A_mel00_{arm}_a.wav"
            path.write_bytes(arm.encode())
            store.put_asset({"source": "ace", "item_id": "mel00", "route": arm, "take": "a",
                             "path": path, "filename": path.name})
        store.state["controls_generation_status"] = "failed"
        raise RuntimeError("ACE generation exhausted its registered active-call budget.")
    monkeypatch.setattr(suno, "generate_controls", controls)
    archives = suno.run(audio, tmp_path / "table.csv", output)
    public = public_of(archives)
    assert archives["complete"] is True
    assert public["controls_status"] == "generation_incomplete"
    assert [row["route"] for row in public["rows"] if row["source"] == "ace"] == list(suno.ARMS[:2])
    assert sum(key.startswith("ace/") for key in scored) == 2


def test_replaced_suno_file_leaves_a_private_trace(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    values = {"npa": .5}
    fake_scorer(monkeypatch, npa=lambda clip: values["npa"] if clip["source"] == "suno" else 1.)
    no_controls(monkeypatch)
    suno.run(audio, tmp_path / "table.csv", output)
    original = suno.pack.sha256_file(audio / "S_mel00_R1_a.mp3")
    (audio / "S_mel00_R1_a.mp3").write_bytes(b"a different take")
    values["npa"] = .99
    archives = suno.run(audio, tmp_path / "table.csv", output)
    trace = read_private(archives)["replaced_after_scoring"]
    assert len(trace) == 1 and trace[0]["clip"] == "suno/mel00/R1/a"
    assert trace[0]["previous_audio_sha256"] == original
    assert trace[0]["previous_scores"] == {"npa": .5, "wr": .8}
    assert trace[0]["new_scores"] == {"npa": .99, "wr": .8}
    assert public_of(archives)["replaced_after_scoring"] == []


def test_unreadable_score_checkpoint_is_rescored_and_logged(tmp_path, clean_provenance):
    store = suno.ResumeStore(tmp_path / "drive", tmp_path / "scratch", {"prereg_sha256": "a" * 64})
    path = tmp_path / "S_mel00_R1_a.mp3"
    path.write_bytes(b"audio")
    clip = {"source": "suno", "item_id": "mel00", "route": "R1", "take": "a",
            "path": path, "filename": path.name, "owner_record": {}}
    store.put_failure(clip, ValueError("first attempt"))
    store.put_score(clip, metric_row("mel00", "R1", "a", .7, .6))
    checkpoint = store.score_path(clip)
    checkpoint.write_text(checkpoint.read_text()[:40])
    resumed = suno.ResumeStore(tmp_path / "drive", tmp_path / "scratch2", store.config)
    assert resumed.score(clip) is None and not checkpoint.exists()
    logged = resumed.state["unreadable_score_checkpoints"]
    assert len(logged) == 1 and logged[0]["clip"] == "suno/mel00/R1/a"
    assert (resumed.root / logged[0]["checkpoint"]).exists()
    resumed.put_score(clip, metric_row("mel00", "R1", "a", .7, .6))
    assert resumed.score(clip)["pitch"]["own"]["strict"]["npa"] == .7


def test_stale_failure_is_cleared_when_a_score_exists(tmp_path, clean_provenance):
    store = suno.ResumeStore(tmp_path / "drive", tmp_path / "scratch", {"prereg_sha256": "a" * 64})
    path = tmp_path / "S_mel00_R1_a.mp3"
    path.write_bytes(b"audio")
    clip = {"source": "suno", "item_id": "mel00", "route": "R1", "take": "a",
            "path": path, "filename": path.name, "owner_record": {}}
    store.put_failure(clip, ValueError("first attempt"))
    row = metric_row("mel00", "R1", "a", .7, .6)
    row["audio_sha256"] = suno.pack.sha256_file(path)
    # A kill between the score write and the state write leaves both entries.
    store._write(store.score_path(clip), {"row": row})
    resumed = suno.ResumeStore(tmp_path / "drive", tmp_path / "scratch2", store.config)
    assert "suno/mel00/R1/a" in resumed.state["scoring_failures"]
    assert resumed.score(clip) is not None
    assert resumed.state["scoring_failures"] == {}
    again = suno.ResumeStore(tmp_path / "drive", tmp_path / "scratch3", store.config)
    assert again.state["scoring_failures"] == {}


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
            assert "timesignature" not in call
    resumed = suno.ResumeStore(tmp_path / "drive", tmp_path / "new-runtime", store.config)
    again = suno.generate_controls(manifest, resumed, cfg)
    assert len(again) == 8 and len(calls) == 8 and loads == ["cuda"]
    assert "generation_in_flight" not in resumed.state
    assert resumed.state["generation_elapsed_seconds"] > 0

    # A hard reset after the native output move still recovers the final audio.
    # The lost call is charged its bounded allowance, so generation continues.
    first = clips[0]
    limit = cfg["ace_generation_call_limit_seconds"]
    reset = suno.ResumeStore(tmp_path / "reset-drive", tmp_path / "reset-work", store.config)
    (reset.root / "assets" / "controls").mkdir(parents=True)
    (reset.root / "assets" / "controls" / first["filename"]).write_bytes(first["path"].read_bytes())
    reset.begin_generation(limit)
    after_reset = suno.ResumeStore(tmp_path / "reset-drive", tmp_path / "after-reset", store.config)
    assert after_reset.generation_budget_used == limit < cfg["ace_generation_budget_seconds"]
    recovered = suno.generate_controls(manifest, after_reset, cfg)
    assert len(recovered) == 8 and recovered[0]["recovered_completed_output"]
    assert after_reset.state["controls_generation_status"] == "complete"
    assert after_reset.state["generation_unresolved_reserved_seconds"] == limit
    # The reservation is charged once; measured time holds only the new calls.
    assert after_reset.state["generation_elapsed_seconds"] < 1.
    assert after_reset.generation_budget_used < limit + 1.
    assert len(calls) == 15 and loads == ["cuda", "cuda"]

    # A reservation that uses the whole allowance still stops before any new call.
    spent = suno.ResumeStore(tmp_path / "spent-drive", tmp_path / "spent-work", store.config)
    (spent.root / "assets" / "controls").mkdir(parents=True)
    (spent.root / "assets" / "controls" / first["filename"]).write_bytes(first["path"].read_bytes())
    spent.begin_generation(cfg["ace_generation_budget_seconds"])
    after_spent = suno.ResumeStore(tmp_path / "spent-drive", tmp_path / "after-spent", store.config)
    recovered = suno.generate_controls(manifest, after_spent, cfg)
    assert len(recovered) == 1 and recovered[0]["recovered_completed_output"]
    assert after_spent.state["controls_generation_status"] == "budget_exceeded"
    assert after_spent.state["generation_elapsed_seconds"] == 0
    assert len(calls) == 15 and loads == ["cuda", "cuda"]

    # A remaining generation must verify its saved guide before it can use it.
    missing = resumed.state["assets"].pop("ace/mel00/ctx_oohs_k3/b")
    (resumed.root / missing["asset_path"]).unlink()
    guide = resumed.state["assets"]["guide/mel00/oohs/a"]
    (resumed.root / guide["asset_path"]).write_bytes(b"changed guide")
    resumed.save()
    with pytest.raises(RuntimeError, match="restore the same saved file"):
        suno.generate_controls(manifest, resumed, cfg)
    assert resumed.state["controls_generation_status"] == "failed"
    assert len(calls) == 15


def unreadable_inventory(monkeypatch, events, clips, bad):
    """Inventory that lists the named files as unreadable, as the real one does."""
    def inventory(*args):
        events.append("inventory")
        return {"clips": [clip for clip in clips if clip["filename"] not in bad], "missing_audio": [],
                "unreadable_audio": [{"filename": name, "reason": "empty file"} for name in sorted(bad)],
                "unreadable_paths": {clip["filename"]: clip["path"] for clip in clips if clip["filename"] in bad},
                "invalid_names": [], "warnings": []}
    monkeypatch.setattr(suno, "inventory", inventory)


def test_unreadable_inventory_files_keep_the_run_pending(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    bad = {"S_mel00_R1_a.mp3", "S_mel01_R1_a.mp3", "S_mel02_R1_b.mp3"}
    for name in bad:
        (audio / name).write_bytes(b"")
    unreadable_inventory(monkeypatch, events, clips, bad)
    fake_scorer(monkeypatch)
    no_controls(monkeypatch)
    archives = suno.run(audio, tmp_path / "table.csv", output)
    result = read_private(archives)
    assert archives["complete"] is False and result["complete"] is False
    assert result["decision_for_reviewer"]["status"] == "scoring_failures_pending"
    assert result["pending_clips"] == ["suno/mel00/R1/a", "suno/mel01/R1/a", "suno/mel02/R1/b"]
    assert {row["exception_type"] for row in result["failed_clips"]} == {"UnreadableAudio"}
    assert "unreadable_paths" not in result["inventory"]
    assert "did not complete" in public_of(archives)["stage_s_statement"]
    # The same songs are downloaded again and now decode.
    original = {name: (audio / name) for name in bad}
    for name, path in original.items():
        path.write_bytes(name.encode())
    bad.clear()
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and result["routes"]["R1"]["complete_items"] == 12
    # A scored take that is unreadable in a later inventory blocks completion again.
    bad.add("S_mel05_R1_b.mp3")
    scored_bytes = (audio / "S_mel05_R1_b.mp3").read_bytes()
    (audio / "S_mel05_R1_b.mp3").write_bytes(b"")
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is False and result["pending_clips"] == ["suno/mel05/R1/b"]
    # Bytes identical to the scored file keep that score even if inventory flags them.
    (audio / "S_mel05_R1_b.mp3").write_bytes(scored_bytes)
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and result["routes"]["R1"]["complete_items"] == 12


def test_unreadable_file_in_three_finished_passes_becomes_an_exclusion(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    bad = {"S_mel03_R1_a.mp3"}
    (audio / "S_mel03_R1_a.mp3").write_bytes(b"")
    unreadable_inventory(monkeypatch, events, clips, bad)
    fake_scorer(monkeypatch)
    no_controls(monkeypatch)
    for _ in range(2):
        result = read_private(suno.run(audio, tmp_path / "table.csv", output))
        assert result["decision_for_reviewer"]["status"] == "scoring_failures_pending"
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and result["excluded_clips"] == ["suno/mel03/R1/a"]
    assert result["routes"]["R1"]["complete_items"] == 11


def all_routes(clips, audio):
    clips.clear()
    for item in pack.load_manifest()["items"]:
        for route in pack.ROUTES:
            for take in suno.TAKES:
                path = audio / f"S_{item['id']}_{route}_{take}.mp3"
                path.write_bytes(path.name.encode())
                clips.append({"source": "suno", "item_id": item["id"], "route": route, "take": take,
                              "path": path, "filename": path.name, "owner_record": {}})
    clips.sort(key=lambda clip: clip["filename"])


def test_adjacent_file_failures_pass_the_probe_and_become_exclusions(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    all_routes(clips, audio)
    bad = {"suno/mel04/R2/a", "suno/mel04/R2/b", "suno/mel04/R3/a"}
    def fault(clip):
        if suno.clip_key(clip) in bad:
            raise ValueError("decoder stops in the middle of this file")
    scored, starts = fake_scorer(monkeypatch, fault)
    no_controls(monkeypatch)
    assert cfg["suno"]["consecutive_failure_abort"] == len(bad)
    for attempt in range(cfg["suno"]["failure_exclusion_runtimes"] - 1):
        result = read_private(suno.run(audio, tmp_path / "table.csv", output))
        assert result["decision_for_reviewer"]["status"] == "scoring_failures_pending"
        assert result["pending_clips"] == sorted(bad)
    # Each pass rebuilt the scorer once for the probe and finished every other clip.
    assert len(starts) == 4 and sum(key.startswith("suno/") for key in scored) == 69
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and result["excluded_clips"] == sorted(bad)
    assert result["routes"]["R2"]["complete_items"] == 11


def test_out_of_memory_on_one_file_is_a_clip_failure_when_the_probe_passes(
        tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    def fault(clip):
        if suno.clip_key(clip) == "suno/mel07/R1/a":
            raise RuntimeError("CUDA out of memory. Tried to allocate 30.00 GiB")
    fake_scorer(monkeypatch, fault)
    no_controls(monkeypatch)
    result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["pending_clips"] == ["suno/mel07/R1/a"]
    assert result["failed_clips"][0]["reason"].startswith("RuntimeError: CUDA out of memory")
    for _ in range(2):
        result = read_private(suno.run(audio, tmp_path / "table.csv", output))
    assert result["complete"] is True and result["excluded_clips"] == ["suno/mel07/R1/a"]


def test_probe_npa_outside_tolerance_is_run_wide(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    state = {"oom": False}
    def fault(clip):
        if suno.clip_key(clip) == "suno/mel07/R1/a":
            state["oom"] = True
            raise MemoryError()
    tolerance = cfg["suno"]["runtime_probe_npa_tolerance"]
    fake_scorer(monkeypatch, fault, npa=lambda clip: 1. - (2 * tolerance if state["oom"] else 0.))
    no_controls(monkeypatch)
    with pytest.raises(suno.RunWideFault, match="probe also failed"):
        suno.run(audio, tmp_path / "table.csv", output)
    result = read_private({"suno_private.zip": output / "suno_private.zip"})
    assert result["decision_for_reviewer"]["status"] == "incomplete_run"
    assert result["failed_clips"] == []


def null_relaxed_scorer(monkeypatch):
    class Scorer:
        def __init__(self, *args): pass
        def close(self): pass
        def score(self, clip, items, work):
            row = metric_row(clip["item_id"], clip["route"], clip["take"], .5, .2, source=clip["source"])
            if clip["source"] in {"suno", "ace"}:
                # Take b had no voiced frames, so its relaxed deviation is null.
                row["pitch"]["own"]["relaxed"] = {
                    "voiced_deviation_semitones": None if clip["take"] == "b" else .3}
            return row
    monkeypatch.setattr(suno, "AudioScorer", Scorer)


def test_metric_without_a_usable_pair_is_null_for_routes_and_controls(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    del clips[2:]
    null_relaxed_scorer(monkeypatch)
    def controls(manifest, store, control_cfg):
        directory = store.root / "assets" / "controls"
        directory.mkdir(parents=True, exist_ok=True)
        for take in suno.TAKES:
            path = directory / f"A_mel00_noguide_{take}.wav"
            path.write_bytes(take.encode())
            store.put_asset({"source": "ace", "item_id": "mel00", "route": "noguide", "take": take,
                             "path": path, "filename": path.name})
        store.state["controls_generation_status"] = "complete"
    monkeypatch.setattr(suno, "generate_controls", controls)
    archives = suno.run(audio, tmp_path / "table.csv", output)
    empty = {"median": None, "items": 0, "ci95": None, "item_pair_means": {}}
    name = "pitch.own.relaxed.voiced_deviation_semitones"
    route = read_private(archives)["routes"]["R1"]
    assert route["complete_items"] == 1 and route["metrics"][name] == empty
    assert route["metrics"]["pitch.own.strict.npa"]["items"] == 1
    control = public_of(archives)["controls"]["noguide"]
    assert control["complete_items"] == 1 and control["metrics"][name] == empty
    rows = [metric_row("mel00", arm, take, .5, .2, source="ace") for arm in suno.ARMS for take in suno.TAKES]
    for row in rows:
        row["pitch"]["own"]["relaxed"] = {"voiced_deviation_semitones": None if row["take"] == "a" else .1}
    summary = suno.paired_summary(rows, list(suno.ARMS))
    assert all(summary[arm]["metrics"][name] == empty for arm in suno.ARMS)


def test_scorer_release_fault_still_writes_the_record(tmp_path, monkeypatch, clean_provenance):
    cfg, clips, events, _, generated, audio, output = fake_run_environment(tmp_path, monkeypatch)
    state = {"n": 0}
    class Scorer:
        def __init__(self, *args): pass
        def close(self):
            if state["n"] > 6:
                raise RuntimeError("CUDA error: an illegal memory access was encountered")
        def score(self, clip, items, work):
            state["n"] += clip["source"] == "suno"
            if state["n"] > 6:
                raise RuntimeError("CUDA error: an illegal memory access was encountered")
            return metric_row(clip["item_id"], clip["route"], clip["take"], 1., .8, source=clip["source"])
    monkeypatch.setattr(suno, "AudioScorer", Scorer)
    no_controls(monkeypatch)
    with pytest.raises(suno.RunWideFault, match="CUDA error"):
        suno.run(audio, tmp_path / "table.csv", output)
    result = read_private({"suno_private.zip": output / "suno_private.zip"})
    assert result["decision_for_reviewer"]["status"] == "incomplete_run" and len(result["rows"]) == 6


def test_lost_generation_reservation_is_charged_once_across_resumes(tmp_path, monkeypatch, clean_provenance):
    monkeypatch.delattr(suno.signal, "SIGALRM")
    cfg = {"prereg_sha256": "a" * 64}
    suno.ResumeStore(tmp_path / "drive", tmp_path / "w1", cfg).begin_generation(120.)
    for run in range(2, 6):
        store = suno.ResumeStore(tmp_path / "drive", tmp_path / f"w{run}", cfg)
        budget = suno.Budget(925., used=store.generation_budget_used, call_limit=120.,
                             on_start=store.begin_generation, on_update=store.update_generation_time)
        budget.call(lambda: None)
        assert store.state["generation_unresolved_reserved_seconds"] == 120.
        assert store.state["generation_elapsed_seconds"] < 1.
        assert 120. <= store.generation_budget_used < 121.
