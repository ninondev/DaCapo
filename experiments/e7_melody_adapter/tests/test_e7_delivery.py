"""Check registration integrity and owner notebook guards without model downloads."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import wave

import pytest

from experiments.e7_melody_adapter import build_notebooks, stage_s_setup, registration


def fixture_registration(tmp_path):
    root = tmp_path / "repo"
    here = root / "experiments/e7_melody_adapter"
    (here / "records").mkdir(parents=True)
    for name in registration.CODE_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("runtime fixture\n")
    (here / "melodies_suno.json").write_bytes((registration.HERE / "melodies_suno.json").read_bytes())
    (here / "records/suno_pack.json").write_bytes((registration.HERE / "records/suno_pack.json").read_bytes())
    data = registration.build_registration("2026-10-27T00:00:00Z", root)
    path = here / "prereg_suno.json"
    path.write_text(json.dumps(data))
    return root, path, data


def test_registration_detects_code_and_definition_tampering(tmp_path):
    root, path, data = fixture_registration(tmp_path)
    assert registration.verify_registration(path, root) == data
    changed = copy.deepcopy(data)
    changed["config"]["suno"]["npa_threshold"] = .99
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="definitions"):
        registration.verify_registration(path, root)
    path.write_text(json.dumps(data))
    (root / registration.CODE_FILES[0]).write_text("changed scoring\n")
    with pytest.raises(ValueError, match="Registered code changed"):
        registration.verify_registration(path, root)


def test_registration_detects_melody_and_lyrics_changes(tmp_path):
    root, path, _ = fixture_registration(tmp_path)
    manifest = root / registration.PREFIX / "melodies_suno.json"
    data = json.loads(manifest.read_text())
    data["items"][0]["lyrics"] = "Changed words"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="melody"):
        registration.verify_registration(path, root)


def test_early_notebook_guard_rejects_missing_coverage_and_changed_inputs(tmp_path):
    root, path, data = fixture_registration(tmp_path)
    assert stage_s_setup.verify_registered_files(root) == data
    changed = copy.deepcopy(data)
    del changed["code_sha256"][registration.CODE_FILES[0]]
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="every runtime"):
        stage_s_setup.verify_registered_files(root)
    path.write_text(json.dumps(data))
    (root / registration.PREFIX / "melodies_suno.json").write_text("{}")
    with pytest.raises(RuntimeError, match="upload source"):
        stage_s_setup.verify_registered_files(root)


def test_notebooks_are_reproducible_plain_python_and_unexecuted():
    for name, expected in build_notebooks.build().items():
        for i, cell in enumerate(expected["cells"]):
            cell["id"] = f"e7-{i:02d}"
        actual = json.loads((registration.HERE / name).read_text())
        assert actual == expected
        for cell in actual["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                assert cell["execution_count"] is None and cell["outputs"] == []
        combined = "\n".join("".join(c["source"]) for c in actual["cells"])
        assert "os.chdir(REPO)" in combined
        assert "verify_registered_files()" in combined
        assert "stage_s_run.sh" in combined
        assert "default Colab" in combined
        assert "pip install -e" not in combined
        if name == "colab_e7_suno.ipynb":
            assert "inspect.signature(generate)" in combined
            assert 'arms(["clarinet", "oohs"])' in combined
            assert '"--no-deps", "basic-pitch==0.4.0"' in combined
            assert 'result["environment"]["git_commit"] != "unknown"' in combined
            assert 'result["environment"]["git_dirty"] is False' in combined
            assert combined.index("require_e5_source()") < combined.index('prepare("gpu", WORK)')
            assert combined.index('"experiments/e5_motif/register.py", "--check"') > combined.index('prepare("gpu", WORK)')
            assert combined.index('drive.mount("/content/drive")') < combined.index('prepare("gpu", WORK)')
            assert combined.index("preflight_audio_inputs(INPUTS)") < combined.index('prepare("gpu", WORK)')
            assert 'Path("/content/drive/MyDrive/e7-suno/scored")' in combined
            assert "Scoring has no time stop" in combined


def test_notebook_preflight_lists_bad_audio_and_keeps_short_uppercase_file(tmp_path, capsys):
    manifest = tmp_path / "melodies.json"
    manifest.write_text(json.dumps({"items": [{"id": "mel00"}]}))
    inputs = tmp_path / "audio"
    inputs.mkdir()
    (inputs / "S_mel00_R1_a.mp3").write_bytes(b"")
    (inputs / "S_mel00_R1_b.mp3").write_bytes(b"corrupt audio")
    short = inputs / "S_mel00_R2_a.MP3"
    # Header-based decoding accepts the short valid waveform regardless of suffix.
    with wave.open(str(short), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 160)
    result = stage_s_setup.preflight_audio_inputs(inputs, manifest)
    assert result["readable_audio"] == [short.name]
    assert {item["filename"] for item in result["unreadable_audio"]} == {
        "S_mel00_R1_a.mp3", "S_mel00_R1_b.mp3"}
    assert len(result["missing_audio"]) == 3
    output = capsys.readouterr().out
    assert "Download the same song again; do not regenerate" in output


@pytest.mark.parametrize("version", [(3, 12, 14), (3, 13, 16)])
def test_default_colab_and_development_python_are_supported(version):
    stage_s_setup.require_python_version(version)


@pytest.mark.parametrize("version", [(3, 11, 0), (3, 14, 0)])
def test_unregistered_python_versions_are_rejected(version):
    with pytest.raises(RuntimeError, match="3.12 and 3.13"):
        stage_s_setup.require_python_version(version)


def alignment_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    weights = tmp_path / "weights"
    source_dir = source / "acestep/models/turbo"
    model_dir = weights / "acestep-v15-turbo"
    source_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    originals = {}
    for index, name in enumerate(stage_s_setup.ORIGINAL_MODEL_CODE_SHA256):
        original = f"original code {index}\n".encode()
        (model_dir / name).write_bytes(original)
        (source_dir / name).write_text(f"registered source code {index}\n")
        originals[name] = hashlib.sha256(original).hexdigest()
    monkeypatch.setattr(stage_s_setup, "ORIGINAL_MODEL_CODE_SHA256", originals)
    monkeypatch.setattr(stage_s_setup, "_model_code_mismatches", lambda *_: [])
    return source, weights, originals


def test_snapshot_download_uses_pinned_real_local_directory(tmp_path, monkeypatch):
    weights = tmp_path / "local-weights"
    captured = {}
    def snapshot_download(repo_id, **kwargs):
        captured.update(repo_id=repo_id, **kwargs)
        Path(kwargs["local_dir"]).mkdir()
        (weights / "snapshot-file.txt").write_text("local materialized content")
        return kwargs["local_dir"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    assert stage_s_setup.download_model_snapshot(weights) == weights
    assert captured["repo_id"] == "ACE-Step/Ace-Step1.5"
    assert captured["revision"] == stage_s_setup.ACE_REVISION
    assert captured["local_dir"] == str(weights)
    assert "acestep-v15-turbo/*" in captured["allow_patterns"]
    assert not any(path.is_symlink() for path in weights.rglob("*"))


def test_model_copy_checks_originals_and_is_idempotent(tmp_path, monkeypatch):
    source, weights, originals = alignment_fixture(tmp_path, monkeypatch)
    record = stage_s_setup.prepare_model_code_alignment(source, weights, originals)
    assert record["complete"] is True
    assert record["original_sha256"] == originals
    assert record["source_commit"] == stage_s_setup.ACE_COMMIT
    assert record["weights_revision"] == stage_s_setup.ACE_REVISION
    for name in originals:
        target = weights / "acestep-v15-turbo" / name
        assert not target.is_symlink()
        assert target.read_bytes() == (source / "acestep/models/turbo" / name).read_bytes()
        assert record["installed_sha256"][name] == hashlib.sha256(target.read_bytes()).hexdigest()
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in weights.rglob("*") if p.is_file()}
    assert stage_s_setup.prepare_model_code_alignment(source, weights, originals) == record
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in weights.rglob("*") if p.is_file()}


def test_model_copy_refuses_unknown_original_before_any_replacement(tmp_path, monkeypatch):
    source, weights, originals = alignment_fixture(tmp_path, monkeypatch)
    names = list(originals)
    first = weights / "acestep-v15-turbo" / names[0]
    before = first.read_bytes()
    (weights / "acestep-v15-turbo" / names[1]).write_text("unexpected upstream bytes")
    with pytest.raises(RuntimeError, match="Original checkpoint code hash mismatch"):
        stage_s_setup.prepare_model_code_alignment(source, weights, originals)
    assert first.read_bytes() == before
    assert not (weights / stage_s_setup.MODEL_CODE_RECORD).exists()


def test_model_copy_resumes_interrupted_approved_replacement(tmp_path, monkeypatch):
    source, weights, originals = alignment_fixture(tmp_path, monkeypatch)
    copyfile = stage_s_setup.shutil.copyfile
    calls = []
    def interrupted(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise InterruptedError("runtime disconnected")
        return copyfile(src, dst)
    monkeypatch.setattr(stage_s_setup.shutil, "copyfile", interrupted)
    with pytest.raises(InterruptedError):
        stage_s_setup.prepare_model_code_alignment(source, weights, originals)
    record = json.loads((weights / stage_s_setup.MODEL_CODE_RECORD).read_text())
    assert record["complete"] is False
    monkeypatch.setattr(stage_s_setup.shutil, "copyfile", copyfile)
    assert stage_s_setup.prepare_model_code_alignment(source, weights, originals)["complete"] is True


@pytest.mark.parametrize("tampering", ["code", "record", "symlink", "source"])
def test_prepared_model_copy_rejects_tampering(tmp_path, monkeypatch, tampering):
    source, weights, originals = alignment_fixture(tmp_path, monkeypatch)
    stage_s_setup.prepare_model_code_alignment(source, weights, originals)
    name = next(iter(originals))
    target = weights / "acestep-v15-turbo" / name
    if tampering == "code":
        target.write_text("modified prepared file")
    elif tampering == "record":
        record_path = weights / stage_s_setup.MODEL_CODE_RECORD
        record = json.loads(record_path.read_text())
        record["weights_revision"] = "unregistered"
        record_path.write_text(json.dumps(record))
    elif tampering == "source":
        (source / "acestep/models/turbo" / name).write_text("modified source")
    else:
        target.unlink()
        target.symlink_to(source / "acestep/models/turbo" / name)
    with pytest.raises(RuntimeError):
        stage_s_setup.prepare_model_code_alignment(source, weights, originals)


def test_stage_s_lock_does_not_include_future_h1_files(tmp_path):
    root, path, data = fixture_registration(tmp_path)
    for name in ("h1_lib.py", "run_h1.sh", "requirements-h1.txt"):
        (root / registration.PREFIX / name).write_text("Future H1 implementation.\n")
    assert registration.verify_registration(path, root) == data
    assert not any(name.endswith(("/lib.py", "/run_stage.sh", "/colab_setup.py", "/requirements-e7.txt"))
                   for name in registration.CODE_FILES)


def test_stage_s_lock_covers_the_control_generation_path():
    # generate_controls imports these modules; their code is part of the registered controls.
    for name in ("dacapo/__init__.py", "dacapo/ace.py", "dacapo/seeds.py",
                 "experiments/e4_sing/run_sing.py", "experiments/e4_sing/make_items.py"):
        assert name in registration.CODE_FILES
        assert (registration.ROOT / name).is_file()


def test_generation_budget_matches_its_stated_basis():
    from experiments.e7_melody_adapter import pack
    cfg = registration.build_config()["suno"]
    longest = max((32 + cfg["duration_extra_beats"]) * 60 / item["melody"]["bpm"]
                  for item in pack.load_manifest()["items"])
    assert round(longest, 2) == 26.09
    warm = 96 * 3 * longest / 14 * 1.5
    assert round(warm) == 805
    assert cfg["ace_generation_budget_seconds"] == round(warm) + cfg["ace_generation_call_limit_seconds"]
    assert "805 seconds" in cfg["ace_generation_budget_basis"]


def test_gpu_source_gate_requires_files_and_rng_seed(tmp_path):
    with pytest.raises(RuntimeError, match="E5 v2"):
        stage_s_setup.require_e5_source(tmp_path)
    files = ["dacapo/seeds.py", "dacapo/words.py", "experiments/e6_clock/readout.py",
             "experiments/e5_motif/register.py", "experiments/e5_motif/registration/addendum_2.json"]
    for name in files:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    ace = tmp_path / "dacapo/ace.py"
    ace.write_text("def generate(seed): pass\n")
    with pytest.raises(RuntimeError, match="rng_seed"):
        stage_s_setup.require_e5_source(tmp_path)
    ace.write_text("def generate(seed, rng_seed=None): pass\n")
    stage_s_setup.require_e5_source(tmp_path)


def test_runtime_import_has_no_install_side_effects(monkeypatch):
    import importlib
    def reject(*args, **kwargs):
        raise AssertionError("A subprocess ran at import time")
    monkeypatch.setattr(subprocess, "run", reject)
    importlib.reload(stage_s_setup)


def test_stage_shell_rejects_training():
    result = subprocess.run(["bash", str(registration.HERE / "stage_s_run.sh"), "train"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "not part of this delivery" in result.stderr
