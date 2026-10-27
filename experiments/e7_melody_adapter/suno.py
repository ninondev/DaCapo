"""Run the registered Stage S controls and private external-audio measurement.

Model imports are local to GPU functions. Importing this module does not load
ACE-Step, download weights, read private audio, or start an experiment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from contextlib import contextmanager
import shutil
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np

from . import stage_s_lib as lib, pack
from .stage_s_setup import checkout_state, inspect_audio, require_e5_source, verify_model_code_alignment
from .registration import verify_registration


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("noguide", "ctx_oohs_k3", "ctx_oohs_k6", "ctx_clarinet_k4")
TAKES = ("a", "b")
TABLE_FIELDS = pack.TABLE_FIELDS


class BudgetExceeded(RuntimeError):
    """Only active ACE generation calls consume this budget."""


class RunWideFault(RuntimeError):
    """A runtime fault, not a fault of one input file. The run stops incomplete."""


class UnreadableAudio(ValueError):
    """An inventory file that could not be decoded. It blocks completion like a failure."""


RUN_WIDE_TYPES = ("BrokenProcessPool", "MemoryError", "OutOfMemoryError", "AcceleratorError")
RUN_WIDE_MARKERS = ("CUDA", "CUBLAS", "cuDNN", "out of memory")


def run_wide_fault(exc: BaseException) -> bool:
    """Check an exception and its causes for the registered runtime faults."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, RunWideFault) or type(exc).__name__ in RUN_WIDE_TYPES:
            return True
        if any(marker in str(exc) for marker in RUN_WIDE_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class Budget:
    """A cumulative generation-call budget, retained across runtime restarts."""

    def __init__(self, seconds: float, *, used: float = 0.0, call_limit: float | None = None,
                 on_start=None, on_update=None):
        self.seconds, self.used = float(seconds), float(used)
        self.call_limit = None if call_limit is None else float(call_limit)
        self.on_start, self.on_update = on_start, on_update
        self.alarm_fired = False

    def check(self):
        if self.used >= self.seconds:
            raise BudgetExceeded("ACE generation exhausted its registered active-call budget.")

    def call(self, function, *args, **kwargs):
        self.check()
        allowance = self.seconds - self.used
        if self.call_limit is not None:
            allowance = min(allowance, self.call_limit)
        if self.on_start is not None:
            # A lost runtime is charged this bounded call allowance, not the remainder.
            self.on_start(allowance)
        started = time.monotonic()
        previous = None
        self.alarm_fired = False
        message = "ACE generation exhausted its registered active-call budget or per-call limit."
        def expired(*_):
            # ACE catches broad exceptions inside generation, so the flag is checked as well.
            self.alarm_fired = True
            raise BudgetExceeded(message)
        if hasattr(signal, "SIGALRM"):
            previous = signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, allowance)
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            if self.alarm_fired and not isinstance(exc, BudgetExceeded):
                raise BudgetExceeded(message) from exc
            raise
        finally:
            if previous is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            measured = max(0.0, time.monotonic() - started)
            self.used += measured
            if self.on_update is not None:
                # Report this call only; the starting total may hold unresolved reservations.
                self.on_update(measured)
        if self.alarm_fired:
            raise BudgetExceeded(message)
        return result


def canonical_name(name: str) -> str:
    path = Path(name)
    return path.stem + ".mp3" if path.suffix.lower() == ".mp3" else name


def probe_audio(path: Path) -> dict:
    """Decode a short prefix before GPU setup. No duration cutoff is applied."""
    return {**inspect_audio(path), "audio_sha256": pack.sha256_file(path)}


def inventory(inputs: Path, table_path: Path, manifest: dict) -> dict:
    """List missing and unreadable takes; never select an alternative output."""
    inputs, table_path = Path(inputs), Path(table_path)
    if not inputs.is_dir() or not table_path.is_file():
        raise FileNotFoundError("Provide the audio directory and completed recording CSV.")
    item_ids = {item["id"] for item in manifest["items"]}
    expected = {f"S_{item_id}_{route}_{take}.mp3": (item_id, route, take)
                for item_id in item_ids for route in pack.ROUTES for take in TAKES}
    with table_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing_columns = sorted(set(TABLE_FIELDS) - set(reader.fieldnames or []))
        rows = list(reader)
    if missing_columns:
        raise ValueError("Recording CSV is missing columns: " + ", ".join(missing_columns))
    table, warnings, invalid = {}, [], []
    for index, row in enumerate(rows, 2):
        name = canonical_name((row.get("filename") or "").strip())
        key = tuple((row.get(field) or "").strip() for field in ("item_id", "route", "take"))
        if name not in expected or expected[name] != key:
            invalid.append(f"CSV row {index}: invalid item/route/take/filename")
            continue
        if name in table:
            raise ValueError(f"Duplicate recording CSV row for {name}; do not choose a take.")
        cleaned = {field: (row.get(field) or "").strip() for field in TABLE_FIELDS}
        available = cleaned["route_available"].lower() or "unknown"
        if available not in pack.ROUTE_AVAILABLE_VALUES:
            raise ValueError(f"Invalid route_available for {name}; use yes, no, or unknown.")
        cleaned["route_available"] = available
        table[name] = cleaned
    paths = {}
    for path in sorted(inputs.rglob("*")):
        if not path.is_file():
            continue
        name = canonical_name(path.name)
        if name not in expected:
            invalid.append(path.relative_to(inputs).as_posix())
            continue
        if name in paths:
            raise ValueError(f"Duplicate audio filename {name}; do not choose a take.")
        paths[name] = path
    clips, unreadable, unreadable_paths = [], [], {}
    for name, path in sorted(paths.items()):
        item_id, route, take = expected[name]
        row = table.get(name, {})
        if not row:
            warnings.append(f"Missing CSV row: {name}")
        elif row["route_available"] == "no":
            raise ValueError(f"Audio exists for a route marked unavailable: {name}")
        missing = [field for field in ("actual_option", "model_version", "date_utc", "credits_used",
                                      "actual_bpm", "actual_key", "actual_style") if not row.get(field)]
        if missing:
            warnings.append(f"Missing metadata for {name}: {', '.join(missing)}")
        try:
            checked = probe_audio(path)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            unreadable.append({"filename": name, "item_id": item_id, "route": route, "take": take,
                               "reason": str(exc),
                               "action": "download the same returned song again; do not generate a replacement"})
            unreadable_paths[name] = path
            continue
        clips.append({"source": "suno", "item_id": item_id, "route": route, "take": take,
                      "path": path, "filename": name, "owner_record": row, **checked})
    return {"clips": clips, "missing_audio": sorted(set(expected) - set(paths)),
            "unreadable_audio": unreadable, "unreadable_paths": unreadable_paths,
            "invalid_names": invalid, "warnings": warnings,
            "files_by_route": {route: sum(c["route"] == route for c in clips) for route in pack.ROUTES},
            "table_sha256": pack.sha256_file(table_path)}


def require_gpu_run() -> dict:
    """Check immutable inputs and E5 v2 before importing or loading any model."""
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run Stage S from the DaCapo clone root.")
    checkout_state()
    registration = verify_registration()
    require_e5_source()
    subprocess.run([sys.executable, "experiments/e5_motif/register.py", "--check"],
                   cwd=ROOT, check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Stage S scoring requires CUDA and Whisper large-v3. No CPU fallback is allowed.")
    if not os.environ.get("ACESTEP_REPO"):
        raise RuntimeError("Prepare the pinned owner GPU runtime first.")
    models = registration["config"]["models"]
    source = Path(os.environ["ACESTEP_REPO"]).resolve()
    actual_commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != models["ace_source_commit"]:
        raise RuntimeError("ACE-Step source does not match the registered commit.")
    source_changes = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all",
         "--", ".", ":(exclude)checkpoints"], text=True)
    if source_changes:
        raise RuntimeError("The pinned ACE-Step source has local changes.")
    weight_dir = Path(os.environ.get("E7_ACE_WEIGHTS_DIR", "")).resolve()
    if (not weight_dir.is_dir() or (source / "checkpoints").resolve() != weight_dir):
        raise RuntimeError("ACE-Step must use the registered weight snapshot from the prepared runtime.")
    from dacapo import ace
    if "rng_seed" not in inspect.signature(ace.generate).parameters:
        raise RuntimeError("Wait for the reviewed E5 v2 generate(rng_seed=...) change.")
    # Setup validates the original snapshot code before a registered source-code sync.
    registration["runtime_model_alignment"] = verify_model_code_alignment(
        source, weight_dir, models["original_model_code_sha256"])
    from acestep.model_downloader import check_main_model_exists
    if not check_main_model_exists(weight_dir):
        raise RuntimeError("The pinned ACE snapshot is incomplete. Restore the exact registered revision.")
    return registration


def write_checkpoint(path: Path, cfg: dict, results: dict, started: float) -> Path:
    """All generation and scoring result JSON files carry clean provenance."""
    from dacapo.provenance import write_result
    result = write_result(path, cfg, results, started)
    assert_result_provenance([result])
    return result


def clip_key(clip: dict) -> str:
    return "/".join(str(clip[name]) for name in ("source", "item_id", "route", "take"))


def headline_scores(row: dict) -> dict:
    """The two decision quantities, kept for a replaced-after-scoring trace."""
    strict = row.get("pitch", {}).get("own", {}).get("strict", {})
    return {"npa": strict.get("npa"), "wr": row.get("words", {}).get("wr")}


class ResumeStore:
    """Durable, private Drive checkpoints bound to one immutable registration."""

    def __init__(self, output: Path, work: Path, config: dict):
        self.root, self.work, self.config = output / "checkpoints", work, config
        self.root.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text())
            if payload["config"] != config:
                raise ValueError("Checkpoint registration or scoring config changed. Use the original registration or a new output folder.")
            assert_result_provenance([self.state_path])
            self.state = payload["results"]
            for key, default in (("scored_audio", {}), ("replaced_after_scoring", []),
                                 ("unreadable_score_checkpoints", []), ("resolved_failures", [])):
                self.state.setdefault(key, default)
        else:
            self.state = {"assets": {}, "stage_seconds": {}, "generation_elapsed_seconds": 0.0,
                          "generation_unresolved_reserved_seconds": 0.0,
                          "scoring_failures": {}, "controls_generation_status": "pending",
                          "scored_audio": {}, "replaced_after_scoring": [],
                          "unreadable_score_checkpoints": [], "resolved_failures": []}
            self.save()
        self._hashes = {}
        pending = self.state.pop("generation_in_flight", None)
        if pending is not None:
            # A vanished runtime cannot report the end of its active call. Keep
            # its reserved capacity separate from measured active seconds.
            self.state["generation_unresolved_reserved_seconds"] = (
                self.state.get("generation_unresolved_reserved_seconds", 0.0) + pending["reserved_seconds"])
            self.state.setdefault("interrupted_generation_calls", []).append(pending)
            self.save()

    def _write(self, target: Path, results: dict):
        local = self.work / "checkpoint-transfer.json"
        write_checkpoint(local, self.config, results, time.time())
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(local, temporary)
        temporary.replace(target)

    def save(self):
        self._write(self.state_path, self.state)

    @contextmanager
    def timed(self, stage: str):
        started = time.monotonic()
        try:
            yield
        finally:
            times = self.state["stage_seconds"]
            times[stage] = times.get(stage, 0.0) + max(0.0, time.monotonic() - started)
            self.save()

    def begin_generation(self, remaining: float):
        self.state["generation_in_flight"] = {"started_utc_seconds": time.time(),
                                               "reserved_seconds": float(remaining)}
        self.save()

    def update_generation_time(self, measured: float):
        self.state["generation_elapsed_seconds"] += float(measured)
        self.state.pop("generation_in_flight", None)
        self.save()

    @property
    def generation_budget_used(self):
        return (self.state["generation_elapsed_seconds"] +
                self.state.get("generation_unresolved_reserved_seconds", 0.0))

    def asset(self, key: str):
        stored = self.state["assets"].get(key)
        if stored is None:
            return None
        clip = dict(stored)
        path = self.root / clip.pop("asset_path")
        if not path.is_file() or pack.sha256_file(path) != clip["audio_sha256"]:
            raise RuntimeError("A completed generated asset is missing or changed; restore the same saved file instead of regenerating.")
        return {**clip, "path": path}

    def put_asset(self, clip: dict):
        path = Path(clip["path"])
        value = {key: value for key, value in clip.items() if key != "path"}
        value.update(asset_path=path.relative_to(self.root).as_posix(), audio_sha256=pack.sha256_file(path))
        self.state["assets"][clip_key(clip)] = value
        self.save()
        return {**clip, "audio_sha256": value["audio_sha256"]}

    def audio_sha256(self, path: Path) -> str:
        """Hash each file version once per runtime; a rewrite changes its stat key."""
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._hashes:
            self._hashes[key] = pack.sha256_file(path)
        return self._hashes[key]

    def score_path(self, clip: dict) -> Path:
        digest = self.audio_sha256(clip["path"])
        logical = hashlib.sha256(clip_key(clip).encode()).hexdigest()
        return self.root / "scores" / digest / (logical + ".json")

    def score(self, clip: dict):
        path = self.score_path(clip)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            environment, config, row = payload["environment"], payload["config"], payload["results"]["row"]
            clip_key(row)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            # A partly written Drive copy is treated as absent. Keep it aside and rescore.
            aside = path.with_name(path.name + f".unreadable-{len(self.state['unreadable_score_checkpoints'])}")
            path.replace(aside)
            self.state["unreadable_score_checkpoints"].append(
                {"clip": clip_key(clip), "checkpoint": aside.relative_to(self.root).as_posix(),
                 "reason": f"{type(exc).__name__}: {exc}", "action": "rescored"})
            self.save()
            print("Unreadable score checkpoint:", clip.get("filename", clip_key(clip)),
                  "It will be scored again.", flush=True)
            return None
        if environment["git_commit"] == "unknown" or environment["git_dirty"] is not False:
            raise RuntimeError(f"Result provenance is not a clean known commit: {path.name}")
        if config != self.config:
            raise ValueError("Score checkpoint belongs to another registration or ruler.")
        if row["audio_sha256"] != self.audio_sha256(clip["path"]) or clip_key(row) != clip_key(clip):
            raise ValueError("Score checkpoint audio or clip identity changed.")
        if clip["source"] == "suno":
            row["owner_record"] = clip.get("owner_record", {})
        if self.resolve_failure(clip_key(clip), row["audio_sha256"]):
            # A kill between the score write and the state write can leave this stale entry.
            self.save()
        return row

    def resolve_failure(self, key: str, digest: str) -> bool:
        """Move a failure of a now-scored clip into the record's history."""
        record = self.state["scoring_failures"].pop(key, None)
        if record is not None:
            self.state["resolved_failures"].append({**record, "clip": key, "scored_audio_sha256": digest})
        return record is not None

    def put_score(self, clip: dict, row: dict):
        key = clip_key(clip)
        row["audio_sha256"] = self.audio_sha256(clip["path"])
        self._write(self.score_path(clip), {"row": row})
        headline = headline_scores(row)
        previous = self.state["scored_audio"].get(key)
        if previous is not None and previous["audio_sha256"] != row["audio_sha256"]:
            # Scores are keyed by audio hash, so a replaced file needs an explicit trace.
            self.state["replaced_after_scoring"].append({
                "clip": key, "source": clip["source"], "filename": clip.get("filename"),
                "previous_audio_sha256": previous["audio_sha256"], "previous_scores": previous["scores"],
                "new_audio_sha256": row["audio_sha256"], "new_scores": headline})
            print("Replaced after scoring:", clip.get("filename", key),
                  "Both versions are listed in the record.", flush=True)
        self.state["scored_audio"][key] = {"audio_sha256": row["audio_sha256"], "scores": headline}
        self.resolve_failure(key, row["audio_sha256"])
        self.save()

    def put_failure(self, clip: dict, exc: Exception):
        key, digest = clip_key(clip), self.audio_sha256(clip["path"])
        previous = self.state["scoring_failures"].get(key, {})
        same = (previous.get("audio_sha256") == digest
                and previous.get("exception_type") == type(exc).__name__)
        self.state["scoring_failures"][key] = {
            "source": clip["source"], "item_id": clip["item_id"], "route": clip["route"],
            "take": clip["take"], "filename": clip["filename"],
            "audio_sha256": digest, "status": "failed",
            "exception_type": type(exc).__name__,
            "reason": f"{type(exc).__name__}: {exc}",
            "counted_runtimes": list(previous.get("counted_runtimes", [])) if same else []}
        self.save()

    def count_failures(self, keys: list[str], runtime: str, limit: int) -> list[str]:
        """Count failures of a finished Suno pass; repeated identical ones become exclusions."""
        excluded = []
        for key in keys:
            record = self.state["scoring_failures"].get(key)
            if record is None or record["status"] != "failed":
                continue
            if runtime not in record["counted_runtimes"]:
                record["counted_runtimes"].append(runtime)
            if len(record["counted_runtimes"]) >= limit:
                record["status"] = "excluded"
                excluded.append(key)
        self.save()
        return excluded

    def failure(self, clip: dict, status: str):
        """Return a stored failure only while it still describes the current file."""
        record = self.state["scoring_failures"].get(clip_key(clip))
        if record is None or record["status"] != status:
            return None
        return record if record["audio_sha256"] == self.audio_sha256(clip["path"]) else None


def registered_arms(cfg: dict) -> dict:
    """Use the E4 arm dictionaries, not merely their names."""
    from experiments.e4_sing.run_sing import arms
    available = {arm["name"]: arm for arm in arms(["clarinet", "oohs"])}
    selected = {name: available[name] for name in ARMS}
    if list(selected.values()) != cfg["arm_definitions"]:
        raise RuntimeError("E4 arm definitions changed from the Stage S registration.")
    return selected


def generate_controls(manifest: dict, store: ResumeStore, cfg: dict) -> list[dict]:
    """Resume saved E4 controls; only generate calls consume the ACE budget."""
    import soundfile as sf
    import torch
    from dacapo import ace
    from dacapo.seeds import run_seeds
    from dacapo.render import render_melody
    from experiments.e4_sing.make_items import GUIDES

    if GUIDES != {"clarinet": 71, "oohs": 53}:
        raise RuntimeError("E4 guide programs differ from the registered controls.")
    definitions = registered_arms(cfg)
    directory = store.root / "assets" / "controls"
    directory.mkdir(parents=True, exist_ok=True)
    budget = Budget(cfg["ace_generation_budget_seconds"], used=store.generation_budget_used,
                    call_limit=cfg["ace_generation_call_limit_seconds"],
                    on_start=store.begin_generation, on_update=store.update_generation_time)
    handler, clips = None, []
    store.state["controls_generation_status"] = "running"
    try:
        for item in manifest["items"]:
            m = pack.melody(item)
            duration = (m.length_beats + cfg["duration_extra_beats"]) * m.spb
            for draw in cfg["draws"]:
                seeds = tuple(run_seeds(cfg["seed_tag"], item["id"], draw))
                if seeds != pack.run_seeds(cfg["seed_tag"], item["id"], draw):
                    raise RuntimeError("E5 v2 run_seeds differs from the registered seed formula.")
                for name, arm in definitions.items():
                    meta = {"source": "ace", "item_id": item["id"], "route": name, "take": TAKES[draw],
                            "filename": f"A_{item['id']}_{name}_{TAKES[draw]}.wav", "dit_seed": seeds[0],
                            "rng_seed": seeds[1], "task": arm["task"],
                            "cover_strength": arm.get("cover_strength", 1.0),
                            "cover_noise": arm.get("cover_noise", 0.0)}
                    saved = store.asset(clip_key(meta))
                    if saved is not None:
                        clips.append(saved)
                        continue
                    path = directory / meta["filename"]
                    if path.exists():
                        # Native generation atomically moves completed audio here.
                        # Recover it even if an interrupted call used the budget.
                        probe_audio(path)
                        meta.update(generation_seconds=None, recovered_completed_output=True)
                        clips.append(store.put_asset({**meta, "path": path}))
                        continue
                    budget.check()
                    if handler is None:
                        with store.timed("ace_model_load"):
                            handler = ace.load("cuda")
                    guide_path = None
                    if "guide" in arm:
                        guide_meta = {"source": "guide", "item_id": item["id"],
                                      "route": arm["guide"], "take": "a"}
                        saved_guide = store.asset(clip_key(guide_meta))
                        if saved_guide is not None:
                            guide_path = saved_guide["path"]
                        else:
                            guide_path = directory / f"{item['id']}_{arm['guide']}_guide.wav"
                            render_melody(m, guide_path, program=GUIDES[arm["guide"]], sr=48000)
                            audio, sr = sf.read(guide_path, always_2d=True)
                            frames = round(duration * sr)
                            audio = np.pad(audio[:frames], ((0, max(0, frames - len(audio))), (0, 0)))
                            sf.write(guide_path, audio, sr, subtype="FLOAT")
                            store.put_asset({**guide_meta, "path": guide_path, "filename": guide_path.name})
                    result = budget.call(
                        ace.generate, handler, path, pack.STYLE, duration, seeds[0], rng_seed=seeds[1],
                        lyrics=item["lyrics"], bpm=int(m.bpm), keyscale=item["key"],
                        steps=cfg["steps"], shift=cfg["shift"], task=arm["task"],
                        src=str(guide_path) if guide_path else None, codes="",
                        cover_strength=meta["cover_strength"], cover_noise=meta["cover_noise"])
                    meta.update(generation_seconds=result["seconds"], denoiser_calls=result["n_denoiser_calls"])
                    clips.append(store.put_asset({**meta, "path": path}))
        store.state["controls_generation_status"] = "complete"
    except BudgetExceeded:
        store.state["controls_generation_status"] = "budget_exceeded"
    except BaseException:
        store.state["controls_generation_status"] = "failed"
        raise
    finally:
        store.save()
        if handler is not None:
            del handler
            gc.collect()
            torch.cuda.empty_cache()
    return clips


def saved_controls(manifest: dict, store: ResumeStore, cfg: dict) -> list[dict]:
    """Read back every saved control in generation order, even after a stop."""
    clips = []
    for item in manifest["items"]:
        for draw in cfg["draws"]:
            for name in ARMS:
                key = "/".join(("ace", item["id"], name, TAKES[draw]))
                try:
                    saved = store.asset(key)
                except RuntimeError as exc:
                    print("Saved control unavailable:", key, exc, flush=True)
                    continue
                if saved is not None:
                    clips.append(saved)
    return clips


def render_ruler(manifest: dict, store: ResumeStore, cfg: dict) -> list[dict]:
    """Known Voice Oohs in a band, with intro, a repeated block and errors."""
    import soundfile as sf
    from dacapo.render import accompaniment, render, render_melody

    directory = store.root / "assets" / "ruler"
    directory.mkdir(parents=True, exist_ok=True)
    clips = []
    roots = {key: tonic - 12 for key, tonic in pack.KEYS}
    for i, item in enumerate(manifest["items"]):
        base = pack.melody(item)
        intro = cfg["intro_seconds_cycle"][i % len(cfg["intro_seconds_cycle"])]
        for variant in cfg["variants"]:
            saved = store.asset(f"ruler/{item['id']}/{variant}/a")
            if saved is not None:
                clips.append(saved)
                continue
            speed = cfg["tempo_ratio"] if variant == "speed_1.03" else 1.0
            shift = cfg["transpose_semitones"] if variant == "transpose_2" else 0
            m = replace(base.transpose(shift), bpm=base.bpm * speed)
            if variant == "two_wrong_notes":
                m = m.with_pitch_errors(cfg["wrong_note_indices"], cfg["wrong_note_offsets"])
            block_s = intro + (base.length_beats + cfg["block_tail_beats"]) * m.spb
            stem = directory / f"R_{item['id']}_{variant}_source.wav"
            band_path = directory / f"R_{item['id']}_{variant}_band.wav"
            render_melody(m.shift(intro / m.spb), stem, program=53, sr=48000)
            render(accompaniment(block_s / m.spb, m.bpm, root=roots[item["key"]] + shift),
                   m.bpm, band_path, sr=48000)
            voice, sr = sf.read(stem, always_2d=True)
            band, band_sr = sf.read(band_path, always_2d=True)
            if sr != 48000 or band_sr != sr or voice.shape[1] != band.shape[1]:
                raise RuntimeError("Ruler renders must use the same registered sample rate and channels.")
            frames = round(block_s * sr)
            mixed = np.zeros((frames, voice.shape[1]), dtype=np.float64)
            mixed[:min(frames, len(voice))] += voice[:frames] * cfg["vocal_gain"]
            mixed[:min(frames, len(band))] += band[:frames] * cfg["accompaniment_gain"]
            mixed *= min(1.0, cfg["peak_cap"] / max(float(np.abs(mixed).max()), 1e-12))
            mixed = np.tile(mixed, (cfg["block_repetitions"], 1))
            path = directory / f"R_{item['id']}_{variant}.wav"
            sf.write(path, mixed, sr, subtype="FLOAT")
            clips.append(store.put_asset({"source": "ruler", "item_id": item["id"], "route": variant,
                          "take": "a", "path": path, "filename": path.name,
                          "expected_start_s": [intro + j * frames / sr for j in range(cfg["block_repetitions"])],
                          "expected_speed_ratio": speed, "expected_transpose_semitones": shift,
                          "synthetic_voice_only": True,
                          "wrong_note_indices": cfg["wrong_note_indices"] if variant == "two_wrong_notes" else []}))
    return clips


class AudioScorer:
    """Pinned CUDA separator and ASR; the same F0 and note tools for all arms."""

    def __init__(self, models: dict, cfg: dict, asr_config: dict):
        import torch
        from importlib.metadata import version
        from demucs.pretrained import get_model
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Do not replace the registered ASR model.")
        for package, expected in (("demucs", "4.1.0"), ("swift-f0", "0.3.0"),
                                  ("basic-pitch", "0.4.0"),
                                  ("transformers", asr_config["transformers_version"])):
            if version(package) != expected:
                raise RuntimeError(f"Use registered {package}=={expected}.")
        if (asr_config["dtype"] != "float16" or asr_config["input_dtype"] != "float32"
                or asr_config["input_sample_rate"] != 16000 or asr_config["channels"] != "mean"
                or asr_config["resample_method"] != "librosa.resample"):
            raise ValueError("Unsupported ASR input preparation; preserve the registered settings.")
        self.models, self.cfg, self.asr_config = models, cfg, asr_config
        self.separator = get_model(models["separator"]).to("cuda").eval()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            models["asr"], revision=models["asr_revision"], torch_dtype=torch.float16,
            **asr_config["model_load"]).to("cuda")
        processor = AutoProcessor.from_pretrained(models["asr"], revision=models["asr_revision"])
        self.asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                            feature_extractor=processor.feature_extractor, torch_dtype=torch.float16,
                            **asr_config["pipeline"])
        self.executor = lib.reference_pool(cfg["search_workers"])

    def close(self):
        import torch
        self.executor.shutdown(wait=True, cancel_futures=True)
        del self.asr, self.separator
        gc.collect()
        torch.cuda.empty_cache()

    def separate(self, path: Path, work: Path) -> tuple[Path, dict]:
        import soundfile as sf
        import torch
        from demucs.apply import apply_model

        sr, channels = int(self.separator.samplerate), int(self.separator.audio_channels)
        mix_path, vocal_path = work / "mixture.wav", work / "vocals.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                        "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_f32le", str(mix_path)], check=True)
        mixture, actual_sr = sf.read(mix_path, always_2d=True, dtype="float32")
        if actual_sr != sr or not len(mixture) or not np.isfinite(mixture).all():
            raise ValueError("The decoded input audio is empty or nonfinite.")
        wave = torch.from_numpy(mixture.T.copy())
        ref = wave.mean(0)
        mean, std = ref.mean(), ref.std(unbiased=ref.numel() > 1).clamp_min(1e-8)
        with torch.inference_mode():
            separated = apply_model(self.separator, ((wave - mean) / std)[None],
                                    device="cuda", shifts=self.cfg["separator_shifts"],
                                    overlap=self.cfg["separator_overlap"], split=True, progress=False)[0]
            voice = separated[self.separator.sources.index("vocals")].cpu() * std + mean
        vocals = voice.numpy().T
        if vocals.shape != mixture.shape:
            raise ValueError("Separator output and mixture are not sample-aligned.")
        sf.write(vocal_path, vocals, sr, subtype="FLOAT")
        return vocal_path, {**lib.voice_present(vocals, mixture, sr), "sample_rate": sr,
                            "duration_seconds": len(mixture) / sr}

    def score(self, clip: dict, items: list[dict], work: Path) -> dict:
        import soundfile as sf
        import librosa
        from dacapo.analysis import f0_swift, notes_basic_pitch, skyline

        work.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        vocal_path, voice = self.separate(clip["path"], work)
        times, f0 = f0_swift(vocal_path, fmin=65.0, fmax=1500.0)
        if len(times) < 2:
            times = np.arange(2) * lib.HOP
            f0 = np.pad(np.asarray(f0), (0, 2 - len(f0)))
        note_events = skyline(notes_basic_pitch(vocal_path, min_note_ms=80.0))
        index = next(i for i, item in enumerate(items) if item["id"] == clip["item_id"])
        intended = items[index]
        others = [item for i, item in enumerate(items) if i != index]
        pitch = lib.score_against_references(pack.notes_seconds(intended),
                                            [pack.notes_seconds(item) for item in others],
                                            times, f0, estimated_notes=note_events, executor=self.executor)
        # The wordless synthetic ruler tests pitch. Its absent lyrics are not an ASR gate.
        transcript = ""
        if clip["source"] != "ruler":
            wave, sr = sf.read(vocal_path, always_2d=True, dtype="float32")
            mono = wave.mean(axis=1)
            target_sr = self.asr_config["input_sample_rate"]
            mono = (librosa.resample(mono, orig_sr=sr, target_sr=target_sr,
                                    **self.asr_config["resample_kwargs"]) if sr != target_sr else mono)
            mono = np.asarray(mono, dtype=self.asr_config["input_dtype"])
            transcript = self.asr({"array": mono, "sampling_rate": target_sr},
                                  generate_kwargs=self.asr_config["generate_kwargs"])["text"]
            (work / "transcript.txt").write_text(transcript)
        words = lib.word_scores(intended["lyrics"], [item["lyrics"] for item in others], transcript)
        row = {key: value for key, value in clip.items() if key not in {"path", "owner_record"}}
        row.update(audio_sha256=pack.sha256_file(clip["path"]), voice=voice,
                   transcript_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
                   pitch=pitch, words=words, scoring_seconds=time.monotonic() - started)
        if clip["source"] == "suno":
            row["owner_record"] = clip["owner_record"]
        if clip["source"] == "ruler":
            fitted = pitch["own"]["strict"]
            row["recovery"] = {
                "start_error_s": min(abs(fitted["start_s"] - start) for start in clip["expected_start_s"]),
                "speed_error": abs(fitted["speed_ratio"] - clip["expected_speed_ratio"]),
                "transpose_error": abs(fitted["transpose_semitones"] - clip["expected_transpose_semitones"]),
            }
            relaxed = pitch["own"]["relaxed"]
            row["relaxed_recovery"] = {
                "start_error_s": (None if relaxed["start_s"] is None else
                                  min(abs(relaxed["start_s"] - start) for start in clip["expected_start_s"])),
                "transpose_error": abs(relaxed["transpose_semitones"] - clip["expected_transpose_semitones"]),
                "duration_ratio": (None if relaxed["start_s"] is None else
                                   (relaxed["end_s"] - relaxed["start_s"]) /
                                   (pack.melody(intended).length_beats * pack.melody(intended).spb)),
            }
        return row


def flatten(row: dict, prefix: str = "") -> dict:
    """Keep one clip per CSV row, including its three selected alignment values."""
    out = {}
    for key, value in row.items():
        name = prefix + key
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        elif isinstance(value, list):
            out[name] = json.dumps(value, separators=(",", ":"))
        else:
            out[name] = value
    return out


def _metrics(row: dict) -> dict[str, float]:
    values = flatten({"pitch": row["pitch"], "words": row["words"]})
    # Alignments are recorded per clip, not treated as aggregate quality scores.
    unwanted = ("start_s", "end_s", "speed_ratio", "transpose_semitones", "reference_frames", "n_mismatched")
    return {key: float(value) for key, value in values.items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
            and not key.endswith(unwanted)}


def paired_summary(rows: list[dict], routes: tuple | list, *, minimum: int = 10) -> dict:
    """Average both returned takes first; bootstrap complete items within route."""
    summary = {}
    for route in routes:
        groups = {}
        for row in rows:
            if row.get("status") == "failed" or row["route"] != route:
                continue
            key = row["item_id"]
            if row["take"] in groups.setdefault(key, {}):
                raise ValueError(f"Duplicate scored take: {route}/{key}/{row['take']}")
            groups[key][row["take"]] = row
        complete = {item: pair for item, pair in groups.items() if set(pair) == set(TAKES)}
        scores = {item: {take: _metrics(pair[take]) for take in TAKES} for item, pair in complete.items()}
        names = sorted(set().union(*[set(values) for pair in scores.values() for values in pair.values()]))
        metrics = {}
        for name in names:
            # A null value in one clip removes only that item from this metric.
            usable = [item for item, pair in scores.items() if all(name in pair[take] for take in TAKES)]
            if not usable:
                metrics[name] = {"median": None, "items": 0, "ci95": None, "item_pair_means": {}}
                continue
            values = [float(np.mean([scores[item][take][name] for take in TAKES])) for item in usable]
            metrics[name] = {"median": float(np.median(values)), "items": len(usable),
                             "ci95": lib.bootstrap_interval(values, statistic="median"),
                             "item_pair_means": dict(zip(usable, values))}
        summary[route] = {"complete_items": len(complete), "complete_item_ids": sorted(complete),
                          "incomplete_item_ids": sorted(set(groups) - set(complete)),
                          "measured": len(complete) >= minimum, "metrics": metrics}
    return summary


def ruler_summary(rows: list[dict]) -> dict:
    """Ceiling and paired error detection use item-level resampling."""
    grouped = {}
    for row in rows:
        if row.get("status") != "failed":
            grouped.setdefault(row["item_id"], {})[row["route"]] = row
    expected = {"clean", "speed_1.03", "transpose_2", "two_wrong_notes"}
    complete = {item: variants for item, variants in grouped.items() if set(variants) == expected}
    if not complete:
        return {"complete_items": 0, "median_npa_ceiling": None, "paired_auc": None,
                "synthetic_voice_only": True, "ruler_validated_on_natural_singing": False}
    paired_wins, recovery, paired_scores = [], [], []
    for variants in complete.values():
        clean = variants["clean"]["pitch"]["own"]["strict"]["npa"]
        wrong = variants["two_wrong_notes"]["pitch"]["own"]["strict"]["npa"]
        paired_scores.append((clean, wrong))
        paired_wins.append(float(clean > wrong) + 0.5 * float(clean == wrong))
        recovery.extend(variants[name]["recovery"] for name in ("clean", "speed_1.03", "transpose_2"))
    all_ceiling_scores = [variants[name]["pitch"]["own"]["strict"]["npa"]
                          for variants in complete.values() for name in ("clean", "speed_1.03", "transpose_2")]
    # Bootstrap whole items and retain their three pristine transformations.
    matrix = np.array([[variants[name]["pitch"]["own"]["strict"]["npa"]
                        for name in ("clean", "speed_1.03", "transpose_2")]
                       for variants in complete.values()])
    rng = np.random.default_rng(20261027)
    selections = rng.integers(0, len(matrix), (2000, len(matrix)))
    medians = np.median(matrix[selections], axis=(1, 2))
    pairs = np.asarray(paired_scores)
    differences = pairs[:, 0, None] - pairs[:, 1][None, :]
    auc = float(np.mean((differences > 0) + .5 * (differences == 0)))
    sampled_pairs = pairs[selections]
    boot_differences = sampled_pairs[:, :, 0, None] - sampled_pairs[:, None, :, 1]
    bootstrap_auc = np.mean((boot_differences > 0) + .5 * (boot_differences == 0), axis=(1, 2))
    return {"complete_items": len(complete), "median_npa_ceiling": float(np.median(all_ceiling_scores)),
            "ceiling_ci95": np.percentile(medians, [2.5, 97.5]).tolist(),
            "paired_auc": auc, "paired_auc_ci95": np.percentile(bootstrap_auc, [2.5, 97.5]).tolist(),
            "paired_auc_definition": "ROC AUC: all clean-versus-corrupt score pairs, half credit for ties; bootstrap matched items",
            "matched_pair_win_rate": float(np.mean(paired_wins)),
            "recovery_max_errors": {key: max(value[key] for value in recovery) for key in recovery[0]},
            "recovery_pass": all(value["start_error_s"] <= .1 and value["speed_error"] <= .010001
                                 and value["transpose_error"] == 0 for value in recovery),
            "synthetic_voice_only": True, "ruler_validated_on_natural_singing": False}


def decide(routes: dict, ruler: dict, *, complete: bool = True, pending: list | tuple = ()) -> dict:
    """Store the preregistered decision privately; the reviewer confirms it."""
    if pending:
        return {"status": "scoring_failures_pending", "kill_rule_3_suno": "unknown",
                "action": "rerun_same_notebook_and_output_folder", "pending_clips": sorted(pending)}
    if not complete:
        return {"status": "incomplete_run", "kill_rule_3_suno": "unknown", "action": "stop_and_report"}
    if ruler.get("complete_items") != 12 or ruler.get("median_npa_ceiling") is None:
        return {"status": "incomplete_ruler", "kill_rule_3_suno": "unknown", "action": "stop_and_report"}
    if ruler["median_npa_ceiling"] < .95:
        return {"status": "ruler_insufficient_pending", "kill_rule_3_suno": "unknown", "action": "stop_and_report"}
    measured = {name: route for name, route in routes.items() if route["measured"]}
    if not measured:
        return {"status": "not_measured", "kill_rule_3_suno": "unknown", "action": "report_missing_routes"}
    passed = [name for name, route in measured.items()
              if route["metrics"]["pitch.own.strict.npa"]["median"] >= .90
              and route["metrics"]["words.wr"]["median"] >= .60]
    return {"status": "already_accurate" if passed else "provisional_negative",
            "kill_rule_3_suno": "triggered" if passed else "provisionally_not_triggered",
            "passing_routes": passed,
            "action": "stop_for_owner_direction" if passed else "recompute_after_stage_0c_before_final_claim"}


def write_csv(path: Path, rows: list[dict]) -> None:
    flat = [flatten(row) for row in rows]
    fields = sorted({key for row in flat for key in row}) or ["source", "item_id", "route", "take"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def assert_result_provenance(paths: list[Path]) -> None:
    for path in paths:
        data = json.loads(path.read_text())
        if data["environment"]["git_commit"] == "unknown" or data["environment"]["git_dirty"] is not False:
            raise RuntimeError(f"Result provenance is not a clean known commit: {path.name}")


def publish_archives(work: Path, output: Path, public_rows: list[dict], private_rows: list[dict],
                     inventory_data: dict, registration: dict, started: float, complete: bool,
                     failure: str | None = None, *, controls_status: str = "pending",
                     stage_seconds: dict | None = None, failures: list | None = None,
                     pending: list | tuple = (), excluded: list | None = None,
                     traces: dict | None = None) -> dict:
    """Archive numeric/text results only. Suno audio and vocals stay outside zips."""
    from dacapo.provenance import write_result

    # Suno traces carry scores, so they stay in the private record.
    traces = traces or {}
    private_traces = {name: [entry for entry in entries if entry["clip"].startswith("suno/")]
                      for name, entries in traces.items()}
    public_traces = {name: [entry for entry in entries if not entry["clip"].startswith("suno/")]
                     for name, entries in traces.items()}

    cfg = registration["config"]
    public, private = work / "public", work / "private"
    public.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    ruler = ruler_summary([row for row in public_rows if row["source"] == "ruler"])
    controls = paired_summary([row for row in public_rows if row["source"] == "ace"], list(ARMS))
    routes = paired_summary(private_rows, pack.ROUTES)
    common = {"experiment": "e7_stage_s", "prereg_sha256": pack.sha256_file(HERE / "prereg_suno.json"),
              "code_sha256": registration["code_sha256"], "manifest_sha256": registration["manifest_sha256"],
              "asr_model": cfg["models"]["asr"], "asr_revision": cfg["models"]["asr_revision"],
              "f0_tool": cfg["models"]["f0"], "separator": cfg["models"]["separator"],
              "separator_package": cfg["models"]["separator_package"], "registered_config": cfg,
              "runtime_model_alignment": registration.get("runtime_model_alignment"),
              "complete": complete}
    public_result = write_result(public / "e7_stage_s_public.json", {**common, "visibility": "public"},
                                 {"controls": controls, "ruler": ruler, "rows": public_rows,
                                  "complete": complete, "failure": failure,
                                  "controls_status": controls_status, **public_traces,
                                  "stage_seconds": {key: value for key, value in (stage_seconds or {}).items()
                                                    if not key.startswith("suno")},
                                  "stage_s_statement": (cfg["privacy"]["public_statement"] if complete else
                                                        cfg["privacy"]["incomplete_statement"]),
                                  "note_f1_status": "unvalidated; excluded from Stage S decision"}, started)
    private_result = write_result(private / "e7_stage_s_private.json", {**common, "visibility": "private"},
                                  {"routes": routes, "rows": private_rows,
                                   "inventory": {key: value for key, value in inventory_data.items()
                                                 if key not in {"clips", "unreadable_paths"}},
                                   "decision_for_reviewer": decide(routes, ruler, complete=complete,
                                                                   pending=pending),
                                   "complete": complete, "failure": failure,
                                   "pending_clips": sorted(pending), "excluded_clips": excluded or [],
                                   **private_traces,
                                   "stage_seconds": stage_seconds or {}, "failed_clips": failures or []}, started)
    write_csv(public / "e7_stage_s_public.csv", public_rows)
    write_csv(private / "e7_stage_s_private.csv", private_rows + [row for row in (failures or []) if row["source"] == "suno"])
    assert_result_provenance([public_result, private_result])
    output.mkdir(parents=True, exist_ok=True)
    created = {}
    for name, folder in (("e7_suno_public.zip", public), ("suno_private.zip", private)):
        target = output / name
        temporary = target.with_name(target.name + ".tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.iterdir()):
                if path.suffix not in {".json", ".csv"}:
                    raise RuntimeError("Only result JSON and CSV files may enter the result archives.")
                archive.write(path, "records/" + path.name)
        temporary.replace(target)
        created[name] = str(target)
    return created


def run(inputs: Path, records_csv: Path, output_dir: Path) -> dict:
    # Inventory uses only decoding tools and precedes CUDA/model imports.
    manifest = pack.load_manifest()
    inputs, output_dir = Path(inputs).resolve(), Path(output_dir).resolve()
    inv = inventory(inputs, Path(records_csv), manifest)
    for name in inv["missing_audio"]:
        print("Missing audio:", name, flush=True)
    for entry in inv["unreadable_audio"]:
        print("Unreadable audio:", entry["filename"], entry["reason"],
              "Download the same returned song again; do not generate a replacement.", flush=True)
    for message in inv["invalid_names"] + inv["warnings"]:
        print("Input notice:", message, flush=True)
    registration = require_gpu_run()
    cfg = registration["config"]
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(HERE / "outputs"):
        raise ValueError("Result outputs in the checkout must stay under e7/outputs.")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}-{time.monotonic_ns()}"
    work = HERE / "outputs" / "suno" / stamp
    work.mkdir(parents=True, exist_ok=False)
    started = time.time()
    result_config = {"experiment": "e7_stage_s_checkpoint", "registered_config": cfg,
                     "asr_model": cfg["models"]["asr"], "asr_revision": cfg["models"]["asr_revision"],
                     "f0_tool": cfg["models"]["f0"], "separator": cfg["models"]["separator"],
                     "separator_package": cfg["models"]["separator_package"],
                     "prereg_sha256": pack.sha256_file(HERE / "prereg_suno.json")}
    store = ResumeStore(output_dir, work, result_config)
    public_rows, private_rows = [], []
    complete, failure, controls_status = False, None, "pending"
    pending, excluded = [], []
    scorer = None
    # Unreadable inventory files are named takes; they are not silently missing.
    unreadable_clips = []
    for entry in inv["unreadable_audio"]:
        name = entry["filename"]
        _, item_id, route, take = Path(name).stem.split("_")
        unreadable_clips.append({"source": "suno", "item_id": item_id, "route": route, "take": take,
                                 "path": Path(inv.get("unreadable_paths", {}).get(name, inputs / name)),
                                 "filename": name, "owner_record": {}, "reason": entry["reason"]})

    def ensure_scorer():
        nonlocal scorer
        if scorer is not None:
            return
        try:
            with store.timed("scorer_model_load"):
                scorer = AudioScorer(cfg["models"], cfg["suno"], cfg["asr"])
        except Exception as exc:
            # A download, version or CUDA setup fault is never a fault of one clip.
            raise RunWideFault(f"The scorer could not start: {type(exc).__name__}: {exc}") from exc

    def score_one(clip):
        row = store.score(clip)
        if row is not None:
            return row
        ensure_scorer()
        key = hashlib.sha256(clip_key(clip).encode()).hexdigest()
        with store.timed(clip["source"] + "_scoring"):
            row = scorer.score(clip, manifest["items"], work / "measure" / key)
        store.put_score(clip, row)
        print("Scored:", clip["source"], clip["item_id"], clip["route"], clip["take"], flush=True)
        return row

    def close_scorer():
        nonlocal scorer
        # Forget the scorer first, so a faulted release is not attempted twice.
        current, scorer = scorer, None
        if current is not None:
            current.close()

    def runtime_healthy() -> bool:
        """Rebuild the scorer and re-score the first saved ruler clip."""
        reference, expected = ruler_clips[0], public_rows[0]
        try:
            close_scorer()
            gc.collect()
            ensure_scorer()
            with store.timed("runtime_probe"):
                row = scorer.score(reference, manifest["items"], work / "runtime_probe")
            difference = abs(headline_scores(row)["npa"] - headline_scores(expected)["npa"])
        except Exception as exc:
            print("Runtime probe failed:", type(exc).__name__, flush=True)
            return False
        print("Runtime probe NPA difference:", round(difference, 4), flush=True)
        return difference <= cfg["suno"]["runtime_probe_npa_tolerance"]

    def publish():
        scored = {clip_key(row) for row in public_rows + private_rows}
        current = {clip_key(clip): store.audio_sha256(clip["path"])
                   for clip in inv["clips"] + unreadable_clips}
        failures = [row for key, row in store.state["scoring_failures"].items()
                    if key not in scored and (row["source"] != "suno" or current.get(key) == row["audio_sha256"])]
        traces = {name: store.state[name] for name in
                  ("replaced_after_scoring", "unreadable_score_checkpoints", "resolved_failures")}
        archives = publish_archives(work, output_dir, public_rows, private_rows, inv,
                                    registration, started, complete, failure,
                                    controls_status=controls_status,
                                    stage_seconds={**store.state["stage_seconds"],
                                                   "ace_generation_calls": store.state["generation_elapsed_seconds"],
                                                   "ace_unresolved_generation_reservation":
                                                   store.state.get("generation_unresolved_reserved_seconds", 0.0)},
                                    failures=failures, pending=pending, excluded=excluded, traces=traces)
        for name, path in archives.items():
            print(f"{name}: {path}", flush=True)
        return archives

    try:
        with store.timed("ruler_render"):
            ruler_clips = render_ruler(manifest, store, cfg["ruler_self_check"])
        suno_clips = [clip for clip in inv["clips"] if store.failure(clip, "excluded") is None]
        blocked = [clip for clip in unreadable_clips if store.failure(clip, "excluded") is None]
        # Build the scorer outside every per-clip handler, so its faults end the run.
        if any(store.score(clip) is None for clip in ruler_clips + suno_clips):
            ensure_scorer()
        for clip in ruler_clips:
            public_rows.append(score_one(clip))
        failed = []
        for clip in blocked:
            # Bytes identical to an earlier scored file keep that score.
            row = store.score(clip)
            if row is not None:
                private_rows.append(row)
                continue
            store.put_failure(clip, UnreadableAudio(clip["reason"]))
            failed.append(clip_key(clip))

        def record_failure(clip, exc):
            store.put_failure(clip, exc)
            failed.append(clip_key(clip))
            print("Failed input:", clip["filename"], type(exc).__name__,
                  "Download the same returned song again, then rerun the same notebook "
                  "and output folder.", flush=True)

        # One faulty file does not censor other inputs or discard complete pairs.
        streak, streak_type = 0, None
        limit = cfg["suno"]["consecutive_failure_abort"]
        for clip in suno_clips:
            try:
                private_rows.append(score_one(clip))
                streak, streak_type = 0, None
            except Exception as exc:
                runtime = run_wide_fault(exc)
                if not runtime:
                    record_failure(clip, exc)
                    streak = streak + 1 if type(exc).__name__ == streak_type else 1
                    streak_type = type(exc).__name__
                if runtime or streak >= limit:
                    # A working runtime reproduces a saved ruler score; then the files are at fault.
                    if not runtime_healthy():
                        cause = (f"{type(exc).__name__}: {exc}" if runtime else
                                 f"{streak} consecutive Suno clips failed with {streak_type}")
                        raise RunWideFault(f"{cause}. The runtime probe also failed.") from exc
                    if runtime:
                        record_failure(clip, exc)
                    streak, streak_type = 0, None
        # Only a finished pass counts toward the registered exclusion rule.
        for key in store.count_failures(failed, stamp, cfg["suno"]["failure_exclusion_runtimes"]):
            print("Excluded take after repeated identical failures:", key, flush=True)
        excluded = sorted(key for key, record in store.state["scoring_failures"].items()
                          if record["source"] == "suno" and record["status"] == "excluded")
        pending = sorted(clip_key(clip) for clip in suno_clips + blocked
                         if store.failure(clip, "failed") is not None)
        complete = not pending and bool(inv["clips"])
        close_scorer()
        if pending:
            print(f"{len(pending)} Suno files are unreadable or failed scoring. The run did not complete. "
                  "Rerun the same notebook and output folder.", flush=True)
        elif not inv["clips"]:
            print("No decodable Suno audio was found. The run did not complete.", flush=True)
        # A kill in the control stage must not withhold a finished decision.
        publish()
        # Controls are independent of the external-system decision.
        stop_controls = False
        try:
            with store.timed("ace_generation_stage"):
                generate_controls(manifest, store, cfg["suno"])
        except KeyboardInterrupt:
            stop_controls = True
            store.state["controls_failure"] = "KeyboardInterrupt"
            store.save()
            print("Controls incomplete: KeyboardInterrupt", flush=True)
        except Exception as exc:
            store.state["controls_failure"] = type(exc).__name__
            store.save()
            print("Controls incomplete:", type(exc).__name__, flush=True)
        generation = store.state["controls_generation_status"]
        controls_status = generation if generation in ("complete", "budget_exceeded") else "generation_incomplete"
        if not stop_controls:
            # Saved controls are scored even when generation stopped early.
            try:
                controls = saved_controls(manifest, store, cfg["suno"])
                if any(store.score(clip) is None for clip in controls):
                    ensure_scorer()
                for clip in controls:
                    try:
                        public_rows.append(score_one(clip))
                    except Exception as exc:
                        if run_wide_fault(exc):
                            raise
                        store.put_failure(clip, exc)
                        controls_status = "scoring_incomplete"
            except (Exception, KeyboardInterrupt) as exc:
                controls_status = "scoring_incomplete"
                store.state["controls_failure"] = type(exc).__name__
                store.save()
                print("Control scoring incomplete:", type(exc).__name__, flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        failure = type(exc).__name__
        complete, pending = False, []
        (work / "failure_private.txt").write_text(str(exc))
        raise
    finally:
        try:
            close_scorer()
        except Exception as exc:
            # A faulted runtime may fail to release memory; the record is still written.
            print("Scorer release failed:", type(exc).__name__, flush=True)
        archives = publish()
        if not complete:
            print("The run did not complete. Rerun the same notebook and output folder.", flush=True)
    return {**archives, "complete": complete}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not run(args.inputs, args.records_csv, args.output_dir)["complete"]:
        # The notebook stops here, so an unfinished result is not downloaded as final.
        raise SystemExit(3)


if __name__ == "__main__":
    main()
