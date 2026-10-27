"""Check registration integrity and owner notebook guards without model downloads."""
import ast
import copy
import json
from pathlib import Path
import subprocess

import pytest

from experiments.e7_melody_adapter import build_notebooks, colab_setup, registration


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
    assert colab_setup.verify_registered_files(root) == data
    changed = copy.deepcopy(data)
    del changed["code_sha256"][registration.CODE_FILES[0]]
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="every runtime"):
        colab_setup.verify_registered_files(root)
    path.write_text(json.dumps(data))
    (root / registration.PREFIX / "melodies_suno.json").write_text("{}")
    with pytest.raises(RuntimeError, match="upload source"):
        colab_setup.verify_registered_files(root)


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
        assert "run_stage.sh" in combined
        assert "pip install -e" not in combined
        if name == "colab_e7_suno.ipynb":
            assert "inspect.signature(generate)" in combined
            assert 'arms(["clarinet", "oohs"])' in combined
            assert '"--no-deps", "basic-pitch==0.4.0"' in combined
            assert 'result["environment"]["git_commit"] != "unknown"' in combined
            assert 'result["environment"]["git_dirty"] is False' in combined
            assert combined.index("require_e5_source()") < combined.index('prepare("gpu", WORK)')
            assert combined.index('"experiments/e5_motif/register.py", "--check"') < combined.index('prepare("gpu", WORK)')


def test_gpu_source_gate_requires_files_and_rng_seed(tmp_path):
    with pytest.raises(RuntimeError, match="E5 v2"):
        colab_setup.require_e5_source(tmp_path)
    files = ["dacapo/seeds.py", "dacapo/words.py", "experiments/e6_clock/readout.py",
             "experiments/e5_motif/register.py", "experiments/e5_motif/registration/addendum_2.json"]
    for name in files:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    ace = tmp_path / "dacapo/ace.py"
    ace.write_text("def generate(seed): pass\n")
    with pytest.raises(RuntimeError, match="rng_seed"):
        colab_setup.require_e5_source(tmp_path)
    ace.write_text("def generate(seed, rng_seed=None): pass\n")
    colab_setup.require_e5_source(tmp_path)


def test_runtime_import_has_no_install_side_effects(monkeypatch):
    import importlib
    def reject(*args, **kwargs):
        raise AssertionError("A subprocess ran at import time")
    monkeypatch.setattr(subprocess, "run", reject)
    importlib.reload(colab_setup)


def test_stage_shell_rejects_training():
    result = subprocess.run(["bash", str(registration.HERE / "run_stage.sh"), "train"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "not part of this delivery" in result.stderr
