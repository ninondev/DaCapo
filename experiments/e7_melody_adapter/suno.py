"""Run the registered Stage S controls and private external-audio measurement.

Model imports are local to GPU functions. Importing this module does not load
ACE-Step, download weights, read private audio, or start an experiment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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

from . import lib, pack
from .colab_setup import checkout_state, require_e5_source
from .registration import verify_registration


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = {"noguide": (None, 0), "ctx_oohs_k3": ("oohs", 3),
        "ctx_oohs_k6": ("oohs", 6), "ctx_clarinet_k4": ("clarinet", 4)}
TAKES = ("a", "b")
TABLE_FIELDS = ("item_id", "route", "take", "filename", "route_available",
                "actual_option", "model_version", "date_utc", "credits_used", "notes")


class BudgetExceeded(RuntimeError):
    """A registered wall-time limit ended an incomplete run."""


class Budget:
    """Check every clip and interrupt a single overlong call on Linux/Colab."""

    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.started = time.monotonic()
        self.previous_handler = None

    def check(self) -> None:
        if time.monotonic() - self.started >= self.seconds:
            raise BudgetExceeded("Stage S exceeded its registered time budget. Stop and report.")

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            self.previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._expired)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def _expired(self, *_):
        raise BudgetExceeded("Stage S exceeded its registered time budget. Stop and report.")

    def __exit__(self, *args):
        if self.previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous_handler)


def inventory(inputs: Path, table_path: Path, manifest: dict) -> dict:
    """List all missing takes; never choose between duplicate or extra outputs."""
    inputs, table_path = Path(inputs), Path(table_path)
    if not inputs.is_dir() or not table_path.is_file():
        raise FileNotFoundError("Provide the audio directory and the completed recording CSV.")
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
        name = row["filename"].strip()
        key = (row["item_id"].strip(), row["route"].strip(), row["take"].strip())
        if name not in expected or expected[name] != key:
            invalid.append(f"CSV row {index}: invalid item/route/take/filename")
            continue
        if name in table:
            raise ValueError(f"Duplicate recording CSV row for {name}; do not choose a take.")
        table[name] = {field: row[field].strip() for field in TABLE_FIELDS}
    paths = {}
    for path in sorted(inputs.rglob("*")):
        if not path.is_file():
            continue
        if path.name not in expected:
            invalid.append(path.relative_to(inputs).as_posix())
            continue
        if path.name in paths:
            raise ValueError(f"Duplicate audio filename {path.name}; do not choose a take.")
        paths[path.name] = path
    clips = []
    for name in sorted(paths):
        item_id, route, take = expected[name]
        row = table.get(name, {})
        if not row:
            warnings.append(f"Missing CSV row: {name}")
        elif row.get("route_available", "").lower() in {"no", "false", "0", "unavailable"}:
            raise ValueError(f"Audio exists for a route marked unavailable: {name}")
        missing = [field for field in ("actual_option", "model_version", "date_utc", "credits_used")
                   if not row.get(field)]
        if missing:
            warnings.append(f"Missing metadata for {name}: {', '.join(missing)}")
        clips.append({"source": "suno", "item_id": item_id, "route": route, "take": take,
                      "path": paths[name], "filename": name, "owner_record": row})
    missing_audio = sorted(set(expected) - set(paths))
    counts = {route: sum(c["route"] == route for c in clips) for route in pack.ROUTES}
    return {"clips": clips, "missing_audio": missing_audio, "invalid_names": invalid,
            "warnings": warnings, "files_by_route": counts, "table_sha256": pack.sha256_file(table_path)}


def context_pre(source_latents, active_steps: int):
    """Inject dense source frames and keep the decoder's actual mask unchanged."""
    import torch

    source = torch.as_tensor(source_latents)
    if source.ndim != 2 or source.shape[-1] != 64:
        raise ValueError("The encoded guide must have shape [T, 64].")
    if not 1 <= active_steps <= 8:
        raise ValueError("The guide duration must be one through eight denoiser calls.")

    def pre(step, timestep, kwargs):
        if "is_covers" in kwargs and bool(torch.as_tensor(kwargs["is_covers"]).any()):
            raise RuntimeError("Dense Stage S controls require text2music and is_covers=False.")
        if step >= active_steps:
            return kwargs
        original = kwargs["context_latents"]
        if original.ndim != 3 or original.shape[-1] != 128:
            raise ValueError("Expected context_latents [batch, T, 128].")
        changed = original.clone()
        n = min(changed.shape[1], source.shape[0])
        changed[:, :n, :64] = source[:n].to(original.device, original.dtype).unsqueeze(0)
        # Beyond the rendered guide, retain the model's own silence source.
        return {**kwargs, "context_latents": changed}

    return pre


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
    if (weight_dir.name != models["ace_weights_revision"] or not weight_dir.is_dir()
            or (source / "checkpoints").resolve() != weight_dir):
        raise RuntimeError("ACE-Step must use the registered weight snapshot from the prepared runtime.")
    from dacapo import ace
    if "rng_seed" not in inspect.signature(ace.generate).parameters:
        raise RuntimeError("Wait for the reviewed E5 v2 generate(rng_seed=...) change.")
    # The native loader otherwise downloads missing components without our
    # revision pin, or copies source Python over checkpoint Python files.
    # Reject either case before load; this experiment must not patch models.
    from acestep.model_downloader import _check_code_mismatch, check_main_model_exists
    if not check_main_model_exists(weight_dir):
        raise RuntimeError("The pinned ACE snapshot is incomplete. Restore the exact registered revision.")
    if _check_code_mismatch("acestep-v15-turbo", weight_dir):
        raise RuntimeError("Pinned source and checkpoint code differ. Stop and report; do not modify model files.")
    return registration


def write_checkpoint(path: Path, cfg: dict, results: dict, started: float) -> Path:
    """All generation and scoring result JSON files carry clean provenance."""
    from dacapo.provenance import write_result
    result = write_result(path, cfg, results, started)
    assert_result_provenance([result])
    return result


def generate_controls(manifest: dict, directory: Path, budget: Budget, cfg: dict,
                      result_config: dict) -> list[dict]:
    """Render the registered first-k dense contexts on the same seeds per arm."""
    import soundfile as sf
    import torch
    from dacapo import ace
    from dacapo.seeds import run_seeds
    from dacapo.render import render_melody
    from experiments.e4_sing.make_items import GUIDES

    if GUIDES != {"clarinet": 71, "oohs": 53}:
        raise RuntimeError("E4 guide programs differ from the registered controls.")
    directory.mkdir(parents=True, exist_ok=True)
    handler = ace.load("cuda")
    clips = []
    started = time.time()
    try:
        for item in manifest["items"]:
            budget.check()
            m = pack.melody(item)
            duration = (m.length_beats + cfg["duration_extra_beats"]) * m.spb
            guides = {}
            for name, program in GUIDES.items():
                path = directory / f"{item['id']}_{name}_guide.wav"
                render_melody(m, path, program=program, sr=48000)
                audio, sr = sf.read(path, always_2d=True)
                frames = round(duration * sr)
                audio = np.pad(audio[:frames], ((0, max(0, frames - len(audio))), (0, 0)))
                sf.write(path, audio, sr, subtype="FLOAT")
                guides[name] = ace.encode(handler, path)
            for draw in cfg["draws"]:
                expected = pack.run_seeds(cfg["seed_tag"], item["id"], draw)
                dit_seed, rng_seed = run_seeds(cfg["seed_tag"], item["id"], draw)
                if (dit_seed, rng_seed) != expected:
                    raise RuntimeError("E5 v2 run_seeds differs from the registered seed formula.")
                for arm, (guide, k) in ARMS.items():
                    budget.check()
                    path = directory / f"A_{item['id']}_{arm}_{TAKES[draw]}.wav"
                    result = ace.generate(
                        handler, path, pack.STYLE, duration, dit_seed, rng_seed=rng_seed,
                        lyrics=item["lyrics"], bpm=int(m.bpm), keyscale=item["key"],
                        timesignature="4", steps=cfg["steps"], shift=cfg["shift"],
                        task="text2music", codes="", src=None, capture=True,
                        pre=None if guide is None else context_pre(guides[guide], k))
                    context = result["captured"]["context_latents"]
                    if context.shape[-1] != 128:
                        raise RuntimeError("The captured text2music context has the wrong shape.")
                    mask = torch.unique(context[..., 64:].detach().float()).cpu().tolist()
                    clips.append({"source": "ace", "item_id": item["id"], "route": arm,
                                  "take": TAKES[draw], "path": path, "filename": path.name,
                                  "dit_seed": dit_seed, "rng_seed": rng_seed,
                                  "generation_seconds": result["seconds"],
                                  "denoiser_calls": result["n_denoiser_calls"],
                                  "context_mask_values": mask, "task": "text2music"})
                    write_checkpoint(directory / "generation_manifest.json", result_config,
                                     {"clips": [{key: str(value) if isinstance(value, Path) else value
                                                 for key, value in clip.items()} for clip in clips]}, started)
    finally:
        del handler
        gc.collect()
        torch.cuda.empty_cache()
    return clips


def render_ruler(manifest: dict, directory: Path, budget: Budget, cfg: dict) -> list[dict]:
    """Known Voice Oohs in a band, with intro, a repeated block and errors."""
    import soundfile as sf
    from dacapo.render import accompaniment, render, render_melody

    directory.mkdir(parents=True, exist_ok=True)
    clips = []
    roots = {key: tonic - 12 for key, tonic in pack.KEYS}
    for i, item in enumerate(manifest["items"]):
        base = pack.melody(item)
        intro = cfg["intro_seconds_cycle"][i % len(cfg["intro_seconds_cycle"])]
        for variant in cfg["variants"]:
            budget.check()
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
            clips.append({"source": "ruler", "item_id": item["id"], "route": variant,
                          "take": "a", "path": path, "filename": path.name,
                          "expected_start_s": [intro + j * frames / sr for j in range(cfg["block_repetitions"])],
                          "expected_speed_ratio": speed, "expected_transpose_semitones": shift,
                          "synthetic_voice_only": True,
                          "wrong_note_indices": cfg["wrong_note_indices"] if variant == "two_wrong_notes" else []})
    return clips


class AudioScorer:
    """Pinned CUDA separator and ASR; the same F0 and note tools for all arms."""

    def __init__(self, models: dict, cfg: dict):
        import torch
        from importlib.metadata import version
        from demucs.pretrained import get_model
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Do not replace Whisper large-v3 with a smaller model.")
        for package, expected in (("demucs", "4.1.0"), ("swift-f0", "0.3.0"), ("basic-pitch", "0.4.0")):
            if version(package) != expected:
                raise RuntimeError(f"Use registered {package}=={expected}.")
        self.models, self.cfg = models, cfg
        self.separator = get_model(models["separator"]).to("cuda").eval()
        asr = AutoModelForSpeechSeq2Seq.from_pretrained(
            models["asr"], revision=models["asr_revision"], torch_dtype=torch.float16,
            low_cpu_mem_usage=True, use_safetensors=True).to("cuda")
        processor = AutoProcessor.from_pretrained(models["asr"], revision=models["asr_revision"])
        self.asr = pipeline("automatic-speech-recognition", model=asr, tokenizer=processor.tokenizer,
                            feature_extractor=processor.feature_extractor, torch_dtype=torch.float16,
                            device=0, chunk_length_s=30, stride_length_s=5, batch_size=1)

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
        mean, std = ref.mean(), ref.std().clamp_min(1e-8)
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

    def score(self, clip: dict, items: list[dict], work: Path, budget: Budget) -> dict:
        import soundfile as sf
        import librosa
        from dacapo.analysis import f0_swift, notes_basic_pitch, skyline

        budget.check()
        work.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        vocal_path, voice = self.separate(clip["path"], work)
        budget.check()
        times, f0 = f0_swift(vocal_path, fmin=65.0, fmax=1500.0)
        note_events = skyline(notes_basic_pitch(vocal_path, min_note_ms=80.0))
        index = next(i for i, item in enumerate(items) if item["id"] == clip["item_id"])
        intended = items[index]
        others = [item for i, item in enumerate(items) if i != index]
        pitch = lib.score_against_references(pack.notes_seconds(intended),
                                            [pack.notes_seconds(item) for item in others],
                                            times, f0, estimated_notes=note_events)
        budget.check()
        # The wordless synthetic ruler tests pitch. Its absent lyrics are not an ASR gate.
        transcript = ""
        if clip["source"] != "ruler":
            wave, sr = sf.read(vocal_path, always_2d=True, dtype="float32")
            mono = wave.mean(axis=1)
            mono = librosa.resample(mono, orig_sr=sr, target_sr=16000) if sr != 16000 else mono
            transcript = self.asr({"array": mono, "sampling_rate": 16000},
                                  generate_kwargs={"language": "english", "task": "transcribe"})["text"]
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
            if row["route"] != route:
                continue
            key = row["item_id"]
            if row["take"] in groups.setdefault(key, {}):
                raise ValueError(f"Duplicate scored take: {route}/{key}/{row['take']}")
            groups[key][row["take"]] = row
        complete = {item: pair for item, pair in groups.items() if set(pair) == set(TAKES)}
        names = sorted(set.intersection(*[set(_metrics(pair[take]))
                                         for pair in complete.values() for take in TAKES])) if complete else []
        metrics = {}
        for name in names:
            values = [float(np.mean([_metrics(pair[take])[name] for take in TAKES])) for pair in complete.values()]
            metrics[name] = {"median": float(np.median(values)),
                             "ci95": lib.bootstrap_interval(values, statistic="median"),
                             "item_pair_means": dict(zip(complete, values))}
        summary[route] = {"complete_items": len(complete), "complete_item_ids": sorted(complete),
                          "incomplete_item_ids": sorted(set(groups) - set(complete)),
                          "measured": len(complete) >= minimum, "metrics": metrics}
    return summary


def ruler_summary(rows: list[dict]) -> dict:
    """Ceiling and paired error detection use item-level resampling."""
    grouped = {}
    for row in rows:
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


def decide(routes: dict, ruler: dict, *, complete: bool = True) -> dict:
    """Store the preregistered decision privately; the reviewer confirms it."""
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
                     failure: str | None = None) -> dict:
    """Archive numeric/text results only. Suno audio and vocals stay outside zips."""
    from dacapo.provenance import write_result

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
              "complete": complete}
    public_result = write_result(public / "e7_stage_s_public.json", {**common, "visibility": "public"},
                                 {"controls": controls, "ruler": ruler, "rows": public_rows,
                                  "complete": complete, "failure": failure,
                                  "stage_s_statement": cfg["privacy"]["public_statement"],
                                  "note_f1_status": "unvalidated; excluded from Stage S decision"}, started)
    private_result = write_result(private / "e7_stage_s_private.json", {**common, "visibility": "private"},
                                  {"routes": routes, "rows": private_rows,
                                   "inventory": {key: value for key, value in inventory_data.items() if key != "clips"},
                                   "decision_for_reviewer": decide(routes, ruler, complete=complete),
                                   "complete": complete, "failure": failure}, started)
    write_csv(public / "e7_stage_s_public.csv", public_rows)
    write_csv(private / "e7_stage_s_private.csv", private_rows)
    assert_result_provenance([public_result, private_result])
    output.mkdir(parents=True, exist_ok=True)
    created = {}
    for name, folder in (("e7_suno_public.zip", public), ("suno_private.zip", private)):
        target = output / name
        if target.exists():
            raise FileExistsError(f"Do not overwrite an earlier result archive: {target}")
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.iterdir()):
                if path.suffix not in {".json", ".csv"}:
                    raise RuntimeError("Only result JSON and CSV files may enter the result archives.")
                archive.write(path, "records/" + path.name)
        created[name] = str(target)
    return created


def run(inputs: Path, records_csv: Path, output_dir: Path) -> dict:
    registration = require_gpu_run()
    cfg = registration["config"]
    manifest = pack.load_manifest()
    inputs, output_dir = Path(inputs).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(HERE / "outputs"):
        raise ValueError("Result outputs in the checkout must stay under e7/outputs.")
    if any((output_dir / name).exists() for name in ("e7_suno_public.zip", "suno_private.zip")):
        raise FileExistsError("Use a new output directory; earlier result archives must remain intact.")
    inv = inventory(inputs, Path(records_csv), manifest)
    for name in inv["missing_audio"]:
        print("Missing audio:", name, flush=True)
    for message in inv["invalid_names"] + inv["warnings"]:
        print("Input notice:", message, flush=True)
    # Always use an ignored path for provenance-bearing temporary results.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    work = HERE / "outputs" / "suno" / stamp
    work.mkdir(parents=True, exist_ok=False)
    started = time.time()
    public_rows, private_rows = [], []
    complete, failure = False, None
    result_config = {"experiment": "e7_stage_s_checkpoint", "registered_config": cfg,
                     "asr_model": cfg["models"]["asr"], "asr_revision": cfg["models"]["asr_revision"],
                     "f0_tool": cfg["models"]["f0"], "separator": cfg["models"]["separator"],
                     "separator_package": cfg["models"]["separator_package"],
                     "prereg_sha256": pack.sha256_file(HERE / "prereg_suno.json")}
    try:
        with Budget(cfg["suno"]["max_wall_seconds"]) as budget:
            controls = generate_controls(manifest, work / "controls", budget, cfg["suno"], result_config)
            ruler_clips = render_ruler(manifest, work / "ruler", budget, cfg["ruler_self_check"])
            scorer = AudioScorer(cfg["models"], cfg["suno"])
            for clip in ruler_clips + controls + inv["clips"]:
                budget.check()
                key = re.sub(r"[^A-Za-z0-9_-]", "_", Path(clip["filename"]).stem)
                row = scorer.score(clip, manifest["items"], work / "measure" / key, budget)
                target = private_rows if clip["source"] == "suno" else public_rows
                target.append(row)
                # Persist every completed clip before the next expensive operation.
                checkpoint_name = "private_rows.json" if clip["source"] == "suno" else "public_rows.json"
                write_checkpoint(work / checkpoint_name, result_config, {"rows": target, "complete": False}, started)
                print("Scored:", clip["source"], clip["item_id"], clip["route"], clip["take"], flush=True)
            complete = True
    except (Exception, KeyboardInterrupt) as exc:
        # The failure category is public; private filenames or ASR text are not.
        failure = type(exc).__name__
        (work / "failure_private.txt").write_text(str(exc))
        raise
    finally:
        archives = publish_archives(work, output_dir, public_rows, private_rows, inv,
                                    registration, started, complete, failure)
        for name, path in archives.items():
            print(f"{name}: {path}", flush=True)
        if not complete:
            print("Run incomplete. Keep partial records and stop; no Stage S conclusion was made.", flush=True)
    return archives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.records_csv, args.output_dir)


if __name__ == "__main__":
    main()
