"""Minimal experiment provenance: every result file records what produced it."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def environment(packages: tuple[str, ...] = ("numpy", "torch", "torchcrepe", "basic-pitch", "mir_eval", "pretty_midi", "librosa")) -> dict:
    versions = {}
    for p in packages:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    try:
        fs = subprocess.run(["fluidsynth", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        fs = None
    status = _git("status", "--porcelain")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "fluidsynth": fs,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status == "unknown" else bool(status),
    }


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def write_result(path: str | Path, cfg: dict, results: dict, started: float) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "config_hash": config_hash(cfg),
        "environment": environment(),
        "wall_seconds": round(time.time() - started, 2),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path
