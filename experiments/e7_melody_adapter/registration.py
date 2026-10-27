"""Create or verify the Stage S preregistration before any external generation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFIX = "experiments/e7_melody_adapter/"
CODE_FILES = tuple(PREFIX + name for name in (
    "stage_s_lib.py", "pack.py", "suno.py", "registration.py", "stage_s_setup.py",
    "stage_s_run.sh", "requirements-stage-s.txt", "colab_e7_pack.ipynb", "colab_e7_suno.ipynb",
)) + (
    "dacapo/metrics.py", "dacapo/analysis.py", "dacapo/symbolic.py",
    "dacapo/render.py", "dacapo/provenance.py",
)
PREREG = HERE / "prereg_suno.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config() -> dict:
    from .stage_s_lib import SCORING_CONFIG
    from .pack import PROGRAMS, SAMPLE_RATE, SEED, STYLE
    from .stage_s_setup import ORIGINAL_MODEL_CODE_SHA256
    return {
        "experiment": "e7_stage_s", "phase": "preregistered_external_measurement",
        "measurement": SCORING_CONFIG,
        "asr": {
            "transformers_version": "4.57.6", "dtype": "float16",
            "input_sample_rate": 16000, "input_dtype": "float32",
            "channels": "mean", "resample_method": "librosa.resample",
            "resample_kwargs": {"res_type": "soxr_hq", "fix": True, "scale": False},
            "model_load": {"low_cpu_mem_usage": True, "use_safetensors": True},
            "pipeline": {"chunk_length_s": 30, "stride_length_s": 5, "batch_size": 1, "device": 0},
            "generate_kwargs": {"language": "english", "task": "transcribe"},
            "input_source": "htdemucs vocals; decode float32 channels, average to mono, resample to 16000 Hz",
            "fallback": "CUDA only; no smaller or alternate ASR model",
        },
        "pitch_extraction": {
            "method": "dacapo.analysis.f0_swift",
            "package": "swift-f0==0.3.0",
            "fmin_hz": 65.0, "fmax_hz": 1500.0,
            "confidence_min": 0.5,
            "resampling": "10 ms grid from zero; searchsorted timestamp index, clipped at track endpoints, as in dacapo.analysis._to_grid",
        },
        "note_transcription": {
            "method": "dacapo.analysis.notes_basic_pitch",
            "package": "basic-pitch==0.4.0", "backend": "onnx",
            "min_note_ms": 80.0, "skyline": True,
            "use": "unvalidated note-F1 diagnostic only; never the Stage S decision",
        },
        "tool_roles_status": "The three distinct roles in measurement.ruler_roles are provisional H1 examples. Stage 0c must select and validate the evaluator before H1 preregistration.",
        "melodies": {"count": 12, "seed": SEED, "bars": 8, "meter": [4, 4],
                     "bpm_range": [90, 110], "keys": ["C major", "D major", "F major", "G major"],
                     "midi_pitch_range": [57, 76], "notes_range": [24, 40],
                     "lyrics": "one original English syllable per note; no owner-held-out melody"},
        "render": {"programs": PROGRAMS, "sample_rate": SAMPLE_RATE, "reverb": False,
                   "soundfont": "FluidR3_GM; exact SHA256 in records/suno_pack.json"},
        "suno": {
            "f0_method": "swift", "caption": STYLE,
            "routes": {"R1": "MIDI prompt", "R2": "piano audio cover/reference", "R3": "Voice Oohs audio cover/reference"},
            "requests_per_item_route": 1, "retain_both_outputs": True,
            "minimum_complete_item_pairs": 10,
            "pair_aggregation": "arithmetic mean of a and b within item",
            "route_aggregation": "median of item pair means; never pool different routes",
            "npa_threshold": 0.90, "word_recall_threshold": 0.60,
            "ruler_ceiling_minimum": 0.95,
            "bootstrap_draws": 2000, "bootstrap_seed": 20261027,
            "steps": 8, "shift": 3.0, "duration_extra_beats": 8,
            "arms": ["noguide", "ctx_oohs_k3", "ctx_oohs_k6", "ctx_clarinet_k4"],
            "arm_definitions": [
                {"name": "noguide", "task": "text2music"},
                {"name": "ctx_oohs_k3", "task": "cover-nofsq", "guide": "oohs", "cover_strength": 0.375},
                {"name": "ctx_oohs_k6", "task": "cover-nofsq", "guide": "oohs", "cover_strength": 0.75},
                {"name": "ctx_clarinet_k4", "task": "cover-nofsq", "guide": "clarinet", "cover_strength": 0.5},
            ],
            "arm_call": "Use the matching E4 arms dictionary: task and rendered src, cover_strength default 1.0, cover_noise default 0.0; no custom context hook",
            "draws": [0, 1], "seed_tag": "e7-suno",
            "separator_shifts": 0, "separator_overlap": 0.25,
            "search_workers": 2,
            "search_parallelism": "spawn process pool over all twelve references; preserve reference order and exact serial results",
            "scoring_time_limit_seconds": None,
            "stage_order": ["ruler_render_and_score", "suno_score", "ace_generate_and_score"],
            "ace_generation_budget_seconds": 432.0,
            "ace_generation_budget_basis": "96 calls times 3 seconds (upper end of run-1 L4 warm 2-3 seconds per call) times 1.5; active generation calls only, cumulative across resumes; exclude loading, rendering and scoring",
            "interrupted_generation_budget": "Persist the remaining allowance as an in-flight reservation before each generation call. Normal completion replaces it with measured active time. A hard runtime loss leaves timing unknown, so an unresolved reservation consumes the remaining allowance on resume; report this conservative charge separately from measured time. Recover and score any completed output before checking that allowance; never regenerate a completed draw.",
            "resume": "Persistent Drive checkpoints keyed by audio SHA256 and clip identity, bound to this preregistration hash; reuse completed scores and generated audio; retry unreadable or failed original downloads without new service requests",
            "completion": "The Suno decision needs all twelve ruler items and the Suno scoring pass. Incomplete or timed-out ACE controls are reported separately and do not invalidate that decision.",
        },
        "ruler_self_check": {
            "source": "synthetic FluidSynth Voice Oohs, not natural singing",
            "variants": ["clean", "speed_1.03", "transpose_2", "two_wrong_notes"],
            "intro_seconds_cycle": [5, 10, 15, 20], "block_repetitions": 2,
            "tempo_ratio": 1.03, "transpose_semitones": 2,
            "wrong_notes": 2, "wrong_note_offset_semitones": 1,
            "wrong_note_indices": [5, 21], "wrong_note_offsets": [1, -1],
            "block_tail_beats": 8,
            "accompaniment_gain": 0.35, "vocal_gain": 1.0, "peak_cap": 0.95,
            "ceiling": "median strict NPA across clean, speed and transposition variants; exclude corrupted notes",
            "sensitivity": "ROC AUC over twelve clean and twelve two-error NPA scores; bootstrap paired items, retaining both labels; matched-pair win rate is a separate diagnostic; no Stage S AUC gate",
            "recovery_diagnostic": {"strict_start_tolerance_s": 0.1,
                                    "speed_tolerance": 0.010001, "transpose_tolerance_semitones": 0,
                                    "relaxed": "report start and transposition errors without a new gate",
                                    "decision_gate": False},
        },
        "decision": {
            "ruler_insufficient": "ruler median NPA < 0.95: inconclusive, stop and report",
            "not_measured": "no route has at least ten complete item pairs: not measured, no negative claim",
            "already_accurate": "any measured route has median strict-external NPA >= 0.90 and median WR >= 0.60 on that same route",
            "provisional_negative": "measured routes do not pass; re-evaluate the same private files after Stage 0c selects a validated singing ruler",
            "note_f1": "unvalidated diagnostic only in Stage S; excluded from decision",
            "threshold_changes": "only easier positive thresholds with recorded rationale; stricter rules need owner agreement",
        },
        "models": {
            "ace_source_commit": "ca1e85fe9430179831e6bc6be790c332190a3866",
            "ace_weights": "ACE-Step/Ace-Step1.5",
            "ace_weights_revision": "19671f406d603126926c1b7e2adc169acbcade22",
            "original_model_code_sha256": dict(ORIGINAL_MODEL_CODE_SHA256),
            "model_code_alignment": "Download real local_dir files at the fixed revision. Verify both original Python file hashes, then copy only those two files from the fixed source commit. Record original and installed hashes and require an empty upstream mismatch list. Never modify the shared weight cache.",
            "asr": "openai/whisper-large-v3",
            "asr_revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
            "separator": "htdemucs", "separator_package": "demucs==4.1.0",
            "f0": "swift-f0==0.3.0", "cpu_asr_fallback": False,
        },
        "privacy": {
            "public_zip": "ACE-Step controls and synthetic ruler only",
            "private_zip": "Suno numeric CSV/JSON only; retained by the owner privately",
            "audio_in_zips": False,
            "public_statement": "Stage S was evaluated under prereg_suno.json. The conclusion is in the private record.",
            "incomplete_statement": "The run did not complete. No Stage S conclusion was made.",
            "publication": "No Suno score, route outcome, accuracy claim or threshold comparison before owner confirms terms permit publication.",
        },
        "validity": {
            "e5_v2": "required and register.py --check must pass before the GPU run",
            "hash_policy": "verify all code and melody/lyric hashes before scoring; changes require a reviewed addendum marked deviation and reporting original-version results",
            "chance": "all other eleven references, identical search and transcription matching",
            "external_study_comparison": "Do not directly compare deviations with Qwen-Music; references, F0 and alignment differ.",
            "unrun": "Suno service use, GPU generation, separation and Whisper have not run in Cloud.",
        },
        "rescore_after_0c": {
            "trigger": "Stage 0c selects a validated evaluator on held-out separated natural singing.",
            "replace_only": "The separated-vocal F0 extractor and its validated fixed parameters; retain the 10 ms grid. Diagnostic note transcription remains unchanged unless separately registered in a reviewed addendum.",
            "fixed": ["same original audio SHA256", "same twelve melodies and lyrics", "htdemucs and its settings", "ASR model, revision and all settings", "WR normalization", "strict and relaxed alignment searches", "all eleven mismatched references", "pair and item aggregation", "bootstrap seeds and draws", "route eligibility and decision thresholds"],
            "ruler_ceiling_minimum": 0.95,
            "ruler_requirement": "Rerun all twelve synthetic ruler items with the replacement F0 extractor. Median pristine strict NPA below 0.95 is pending, not a negative finding. Stage 0c natural-singing validation is also required.",
            "decision": "Apply the same minimum ten complete item pairs, median strict-external NPA >= 0.90 and median WR >= 0.60 on one route. A positive result triggers the same stop for owner direction. Preserve and report both original and rescored results privately; never select whichever is preferred.",
            "permitted_changes": "Only changes that make the already-accurate condition easier may be added, with rationale in a reviewed deviation addendum. Do not tighten thresholds or eligibility.",
            "record": "Use a separate reviewed rescore addendum and output/checkpoint namespace. Record the original preregistration SHA256, validated ruler identity/version/config, validation record hashes, input audio SHA256 and parent result hashes; never overwrite the original Stage S records.",
        },
    }


def build_registration(registered_utc: str | None = None, root: Path = ROOT) -> dict:
    from .pack import lyrics_hash
    source = root / PREFIX / "melodies_suno.json"
    manifest = json.loads(source.read_text())
    pack_record = root / PREFIX / "records/suno_pack.json"
    return {
        "schema_version": 1,
        "registered_utc": registered_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "awaiting_review_and_merge_before_external_generation",
        "manifest_sha256": digest(source),
        "lyrics_sha256": lyrics_hash(manifest),
        "source_pack_record_sha256": digest(pack_record),
        "code_sha256": {name: digest(root / name) for name in CODE_FILES},
        "config": build_config(),
    }


def verify_registration(path: Path = PREREG, root: Path = ROOT) -> dict:
    from .pack import lyrics_hash
    data = json.loads(path.read_text())
    if set(data.get("code_sha256", {})) != set(CODE_FILES):
        raise ValueError("The registration does not cover the complete runtime code set.")
    for name, expected in data["code_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Registered code changed: {name}. A reviewed addendum is required.")
    source = root / PREFIX / "melodies_suno.json"
    if digest(source) != data["manifest_sha256"]:
        raise ValueError("Registered melody source changed.")
    if lyrics_hash(json.loads(source.read_text())) != data["lyrics_sha256"]:
        raise ValueError("Registered lyrics changed.")
    if digest(root / PREFIX / "records/suno_pack.json") != data["source_pack_record_sha256"]:
        raise ValueError("Registered upload metadata changed.")
    if data["config"] != build_config():
        raise ValueError("Registered definitions differ from the active scoring contract.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--path", type=Path, default=PREREG)
    args = parser.parse_args()
    if args.write:
        if args.path.exists():
            raise SystemExit("Refusing to overwrite a preregistration. Use a reviewed deviation addendum.")
        data = build_registration()
        args.path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print("Wrote Stage S preregistration; owner review and merge must precede Suno generation.")
    else:
        verify_registration(args.path)
        print("Stage S registration hashes and definitions match.")


if __name__ == "__main__":
    main()
