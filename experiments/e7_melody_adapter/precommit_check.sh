#!/usr/bin/env bash
# Check the whole E7 change, including untracked files. Do not install anything.
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
python - "$repo_root" <<'PY'
from pathlib import Path
import os
import subprocess
import sys

root = Path(sys.argv[1])
prefix = "experiments/e7_melody_adapter/"
limit = 1_000_000


def git(*args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


baseline_file = root / os.fsdecode(git("rev-parse", "--git-path", "e7_base").strip())
baseline = os.environ.get("E7_BASE", "").strip()
source = "E7_BASE"
if not baseline and baseline_file.is_file():
    baseline = baseline_file.read_text().strip()
    source = ".git/e7_base"
if not baseline:
    check = subprocess.run(
        ["git", "rev-parse", "--verify", "origin/main^{commit}"],
        cwd=root, capture_output=True,
    )
    if check.returncode == 0:
        baseline = "origin/main"
        source = "origin/main"
if not baseline:
    print("No E7 baseline. First run: git rev-parse HEAD > .git/e7_base", file=sys.stderr)
    sys.exit(2)
try:
    # Resolve to a commit before passing the value to git diff.
    commit = git("rev-parse", "--verify", "--end-of-options", baseline + "^{commit}").strip().decode("ascii")
except subprocess.CalledProcessError:
    print(f"Invalid E7 baseline from {source}; use a valid commit.", file=sys.stderr)
    sys.exit(2)

changed = git("diff", "--name-only", "--no-renames", "-z", commit, "--").split(b"\0")
untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
paths = sorted({os.fsdecode(path) for path in changed + untracked if path})
errors = []
for name in paths:
    if not name.startswith(prefix):
        errors.append(f"Outside E7: {name!r}")
        continue
    path = root / name
    if path.is_symlink():
        errors.append(f"Symlink is not an E7 text file: {name!r}")
        continue
    if not path.exists():
        # Deletions in E7 are allowed; protected-path deletions fail above.
        continue
    if not path.is_file():
        errors.append(f"Not a regular E7 file: {name!r}")
        continue
    size = path.stat().st_size
    if size > limit:
        errors.append(f"File exceeds 1 MB ({size} bytes): {name!r}")
    if size and subprocess.run(["grep", "-Iq", ".", "--", str(path)], capture_output=True).returncode != 0:
        errors.append(f"Binary or non-text file: {name!r}")

# A staged binary must not pass merely because its worktree copy was replaced.
index_entries = git("ls-files", "--stage", "-z").split(b"\0")
changed_set = set(paths)
for entry in index_entries:
    if not entry:
        continue
    metadata, raw_name = entry.split(b"\t", 1)
    name = os.fsdecode(raw_name)
    if name not in changed_set or not name.startswith(prefix):
        continue
    mode, oid, stage = metadata.split()
    if stage != b"0":
        errors.append(f"Unresolved index entry: {name!r}")
        continue
    if mode != b"100644" and mode != b"100755":
        errors.append(f"Staged entry is not a regular text file: {name!r}")
        continue
    blob = git("cat-file", "blob", oid.decode("ascii"))
    if len(blob) > limit:
        errors.append(f"Staged file exceeds 1 MB ({len(blob)} bytes): {name!r}")
    if blob and subprocess.run(["grep", "-Iq", "."], input=blob, capture_output=True).returncode != 0:
        errors.append(f"Staged binary or non-text file: {name!r}")
if errors:
    print("E7 precommit check failed:", file=sys.stderr)
    for error in errors:
        print("- " + error, file=sys.stderr)
    sys.exit(1)

register = root / "experiments/e5_motif/register.py"
if register.is_file():
    result = subprocess.run([sys.executable, str(register), "--check"], cwd=root)
    if result.returncode:
        print("E5 registration check failed. Stop and report; do not rewrite it.", file=sys.stderr)
        sys.exit(result.returncode if result.returncode > 0 else 1)
    e5_status = "passed"
else:
    e5_status = "absent (E5 v2 GPU work remains blocked)"
print(f"E7 precommit check passed: {len(paths)} paths; baseline {commit} from {source}; E5 check {e5_status}.")
PY

