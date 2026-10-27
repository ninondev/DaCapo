"""Run the E7 guard in isolated Git fixtures under ignored E7 outputs."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import pytest


E7 = Path(__file__).resolve().parents[1]
GUARD = E7 / "precommit_check.sh"
PREFIX = Path("experiments/e7_melody_adapter")


def command(repo, *args):
    return subprocess.run(args, cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo():
    fixture_root = E7 / "outputs" / "guard-tests"
    fixture_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="repo-", dir=fixture_root) as directory:
        root = Path(directory)
        command(root, "git", "init", "-q")
        command(root, "git", "config", "user.name", "Guard Test")
        command(root, "git", "config", "user.email", "guard@example.invalid")
        command(root, "git", "config", "commit.gpgsign", "false")
        (root / "README.md").write_text("Protected baseline.\n")
        (root / PREFIX).mkdir(parents=True)
        (root / PREFIX / ".gitignore").write_text("outputs/\n*.wav\n")
        command(root, "git", "add", ".")
        command(root, "git", "commit", "-qm", "Fixture baseline")
        baseline = command(root, "git", "rev-parse", "HEAD")
        (root / ".git" / "e7_base").write_text(baseline + "\n")
        yield root


def guard(repo, **overrides):
    environment = os.environ.copy()
    environment.pop("E7_BASE", None)
    environment.update(overrides)
    return subprocess.run(["bash", str(GUARD)], cwd=repo, env=environment, capture_output=True, text=True)


def test_allows_e7_text_and_ignores_generated_audio(repo):
    (repo / PREFIX / "new file.md").write_text("Allowed new text.\n")
    (repo / PREFIX / "empty.txt").write_text("")
    output = repo / PREFIX / "outputs"
    output.mkdir()
    (output / "large.wav").write_bytes(b"\0" * 1_000_001)
    result = guard(repo)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("relative", ["README.md", "tests/new.py", "dacapo/new.py"])
def test_rejects_protected_changes_and_untracked_paths(repo, relative):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Forbidden change.\n")
    result = guard(repo)
    assert result.returncode == 1
    assert "Outside E7" in result.stderr


@pytest.mark.parametrize("name,data,message", [
    ("large.txt", b"x" * 1_000_001, "exceeds 1 MB"),
    ("binary.data", b"\x00\x01binary", "Binary or non-text"),
], ids=["oversized-text", "binary-content"])
def test_rejects_large_and_binary_files(repo, name, data, message):
    (repo / PREFIX / name).write_bytes(data)
    result = guard(repo)
    assert result.returncode == 1
    assert message in result.stderr


def test_rejects_staged_binary_with_text_worktree_copy(repo):
    path = repo / PREFIX / "pending.data"
    path.write_bytes(b"\0binary")
    command(repo, "git", "add", str(PREFIX / path.name))
    path.write_text("Text worktree copy.\n")
    result = guard(repo)
    assert result.returncode == 1
    assert "Staged binary" in result.stderr


def test_missing_baseline_has_actionable_exit_two(repo):
    (repo / ".git" / "e7_base").unlink()
    result = guard(repo)
    assert result.returncode == 2
    assert "git rev-parse HEAD > .git/e7_base" in result.stderr


def test_origin_main_fallback_and_environment_priority(repo):
    baseline = command(repo, "git", "rev-parse", "HEAD")
    command(repo, "git", "update-ref", "refs/remotes/origin/main", baseline)
    (repo / ".git" / "e7_base").unlink()
    assert guard(repo).returncode == 0
    (repo / ".git" / "e7_base").write_text("invalid-baseline\n")
    result = guard(repo, E7_BASE=baseline)
    assert result.returncode == 0, result.stderr
    assert "from E7_BASE" in result.stdout


def test_baseline_file_wins_over_origin_main(repo):
    command(repo, "git", "update-ref", "refs/remotes/origin/main", command(repo, "git", "rev-parse", "HEAD"))
    (repo / ".git" / "e7_base").write_text("invalid-baseline\n")
    result = guard(repo)
    assert result.returncode == 2
    assert "from .git/e7_base" in result.stderr


def test_calls_existing_register_only_with_check(repo):
    register = repo / "experiments/e5_motif/register.py"
    register.parent.mkdir(parents=True)
    register.write_text("import sys\nassert sys.argv[1:] == ['--check']\nraise SystemExit(7)\n")
    command(repo, "git", "add", str(register.relative_to(repo)))
    command(repo, "git", "commit", "-qm", "Fixture registration")
    (repo / ".git" / "e7_base").write_text(command(repo, "git", "rev-parse", "HEAD") + "\n")
    result = guard(repo)
    assert result.returncode == 7
    assert "E5 registration check failed" in result.stderr
