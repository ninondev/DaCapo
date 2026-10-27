"""Prepare an owner-run Colab runtime. Importing this module does no work."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
ACE_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
WHISPER_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
MODEL_CODE_RECORD = "stage_s_model_code.json"
ORIGINAL_MODEL_CODE_SHA256 = {
    "modeling_acestep_v15_turbo.py": "c1ab0dd547124fee7ada449b2b86eae8201dc7d15889932643bbb67e3c982444",
    "configuration_acestep_v15.py": "b89870c5c7a7ce060eb0bcdbb5ffc86b0b1a324ca325a26be552ea1b42496dc5",
}


def require_python_version(version=None) -> None:
    """Accept the default Colab runtime and the CPU development runtime."""
    current = tuple(sys.version_info[:2] if version is None else version[:2])
    if current not in {(3, 12), (3, 13)}:
        raise RuntimeError("Stage S supports Python 3.12 and 3.13. Use the default Colab runtime.")


def preflight_audio_inputs(inputs: Path, manifest_path: Path | None = None) -> dict:
    """List missing or unreadable uploads before any GPU setup or model download."""
    if not inputs.is_dir():
        raise FileNotFoundError("Set INPUTS to the Drive folder with the original downloaded audio.")
    if shutil.which("ffmpeg") is None:
        privilege = [] if os.geteuid() == 0 else ["sudo"]
        if privilege and shutil.which("sudo") is None:
            raise RuntimeError("Install ffmpeg before checking uploaded audio.")
        run(*privilege, "apt-get", "update", "-qq")
        run(*privilege, "env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "-qq",
            "--no-install-recommends", "ffmpeg")
    source = HERE / "melodies_suno.json" if manifest_path is None else manifest_path
    manifest = json.loads(source.read_text())
    expected = {f"S_{item['id']}_{route}_{take}.mp3"
                for item in manifest["items"] for route in ("R1", "R2", "R3") for take in ("a", "b")}
    readable, unreadable, seen = [], [], set()
    for path in sorted(inputs.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".mp3":
            continue
        canonical = path.stem + ".mp3"
        if canonical not in expected:
            continue
        seen.add(canonical)
        reason = ""
        try:
            inspect_audio(path)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            reason = str(exc)
        if reason:
            unreadable.append({"filename": path.name, "reason": reason})
            print(f"Unreadable audio: {path.name}: {reason}. Download the same song again; do not regenerate.")
        else:
            readable.append(path.name)
    missing = sorted(expected - seen)
    for name in missing:
        print("Missing audio:", name)
    return {"missing_audio": missing, "unreadable_audio": unreadable, "readable_audio": readable}


def inspect_audio(path: Path) -> dict:
    """Decode a short prefix without imposing a minimum clip duration."""
    if path.stat().st_size == 0:
        raise ValueError("empty file")
    decoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-t", "1",
         "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
        capture_output=True, timeout=60,
    )
    if decoded.returncode or len(decoded.stdout) < 4:
        raise ValueError("audio cannot be decoded")
    return {"prefix_samples": len(decoded.stdout) // 4}


def download_model_snapshot(weights: Path) -> Path:
    """Materialize the fixed snapshot in writable local files, outside hub cache."""
    from huggingface_hub import snapshot_download

    if weights.is_symlink():
        raise RuntimeError("The model destination must be a real local_dir.")
    downloaded = Path(snapshot_download(
        "ACE-Step/Ace-Step1.5", revision=ACE_REVISION, local_dir=str(weights),
        allow_patterns=["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*",
                        "acestep-5Hz-lm-1.7B/*", "config.json"],
    ))
    if downloaded.resolve() != weights.resolve():
        raise RuntimeError("The snapshot did not materialize in the requested local_dir.")
    if any(path.is_symlink() for path in weights.rglob("*")):
        raise RuntimeError("The local model snapshot contains cache symlinks.")
    return weights


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_code_mismatches(source: Path, weights: Path) -> list[str]:
    """Use the pinned upstream comparison after setup installs its dependencies."""
    sys.path.insert(0, str(source))
    try:
        from acestep.model_downloader import _check_code_mismatch
        return _check_code_mismatch("acestep-v15-turbo", weights)
    finally:
        sys.path.pop(0)


def _alignment_inputs(source: Path, weights: Path, original_hashes: dict | None) -> dict:
    originals = ORIGINAL_MODEL_CODE_SHA256 if original_hashes is None else original_hashes
    if originals != ORIGINAL_MODEL_CODE_SHA256:
        raise RuntimeError("Registered original model-code hashes differ from the reviewed snapshot.")
    if weights.is_symlink() or not weights.is_dir():
        raise RuntimeError("ACE weights must be a real local_dir, not a cache symlink.")
    if any(path.is_symlink() for path in weights.rglob("*")):
        raise RuntimeError("ACE local_dir contains a symlink; do not patch shared cache files.")
    source_files = {name: source / "acestep/models/turbo" / name for name in originals}
    if any(path.is_symlink() or not path.is_file() for path in source_files.values()):
        raise RuntimeError("Pinned source model code is missing or is not a regular file.")
    return {
        "schema_version": 1,
        "source_commit": ACE_COMMIT,
        "weights_revision": ACE_REVISION,
        "original_sha256": dict(originals),
        "installed_sha256": {name: _sha256(path) for name, path in source_files.items()},
    }


def _write_alignment_record(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_model_code_alignment(source: Path, weights: Path, original_hashes: dict | None = None) -> dict:
    """Verify the recorded original revision and the approved source-code copies."""
    expected = _alignment_inputs(source, weights, original_hashes)
    record_path = weights / MODEL_CODE_RECORD
    if not record_path.is_file():
        raise RuntimeError("Prepare the pinned local model copy before GPU scoring.")
    record = json.loads(record_path.read_text())
    if record != {**expected, "complete": True}:
        raise RuntimeError("Model-code alignment metadata changed or setup did not complete.")
    for name, digest in expected["installed_sha256"].items():
        target = weights / "acestep-v15-turbo" / name
        if not target.is_file() or _sha256(target) != digest:
            raise RuntimeError(f"Prepared model code changed: {name}.")
    if _model_code_mismatches(source, weights):
        raise RuntimeError("Model code still differs after the two approved source copies.")
    return record


def prepare_model_code_alignment(source: Path, weights: Path, original_hashes: dict | None = None) -> dict:
    """Validate original bytes, then copy only the two approved source files.

    The pending record is saved before either copy. An interrupted setup may
    resume only if each target is still its original or approved final bytes.
    A completed setup is verified without writing anything again.
    """
    expected = _alignment_inputs(source, weights, original_hashes)
    record_path = weights / MODEL_CODE_RECORD
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get("complete") is True:
            return verify_model_code_alignment(source, weights, original_hashes)
        if record != {**expected, "complete": False}:
            raise RuntimeError("Incomplete model-code alignment metadata changed.")
        for name, original in expected["original_sha256"].items():
            target = weights / "acestep-v15-turbo" / name
            if not target.is_file() or _sha256(target) not in {original, expected["installed_sha256"][name]}:
                raise RuntimeError(f"Interrupted model code changed: {name}.")
    else:
        # Validate both originals before replacing either one.
        for name, original in expected["original_sha256"].items():
            target = weights / "acestep-v15-turbo" / name
            if not target.is_file() or _sha256(target) != original:
                raise RuntimeError(f"Original checkpoint code hash mismatch: {name}. Stop and report.")
        _write_alignment_record(record_path, {**expected, "complete": False})
    for name, installed in expected["installed_sha256"].items():
        target = weights / "acestep-v15-turbo" / name
        if _sha256(target) != installed:
            temporary = target.with_suffix(".py.stage-s-tmp")
            shutil.copyfile(source / "acestep/models/turbo" / name, temporary)
            temporary.replace(target)
    if _model_code_mismatches(source, weights):
        raise RuntimeError("Model code still differs after the two approved source copies.")
    _write_alignment_record(record_path, {**expected, "complete": True})
    return verify_model_code_alignment(source, weights, original_hashes)


def run(*args: str, cwd: Path | None = None, env: dict | None = None) -> None:
    subprocess.run(list(args), cwd=cwd, env=env, check=True)


def checkout_state(root: Path = ROOT) -> str:
    """Reject an uncommitted experiment before downloading or generating anything."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if dirty or len(sha) != 40:
        raise RuntimeError("Use the reviewed, merged, clean main checkout in this runtime.")
    return sha


def verify_registered_files(root: Path = ROOT) -> dict:
    """This check needs only the standard library, before dependency installation."""
    from .registration import CODE_FILES

    prereg = root / "experiments/e7_melody_adapter/prereg_suno.json"
    if not prereg.is_file():
        raise RuntimeError("Wait for the Stage S preregistration to be reviewed and merged.")
    data = json.loads(prereg.read_text())
    if set(data.get("code_sha256", {})) != set(CODE_FILES):
        raise RuntimeError("The preregistration must cover every runtime source file.")
    for name, expected in data["code_sha256"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Registered code changed: {name}. A reviewed addendum is required.")
    for name, field in (("melodies_suno.json", "manifest_sha256"),
                        ("records/suno_pack.json", "source_pack_record_sha256")):
        path = prereg.parent / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != data[field]:
            raise RuntimeError(f"Registered upload source changed: {name}.")
    return data


def require_e5_source(root: Path = ROOT) -> None:
    required = ["dacapo/seeds.py", "dacapo/words.py", "experiments/e6_clock/readout.py",
                "experiments/e5_motif/register.py", "experiments/e5_motif/registration/addendum_2.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError("Wait for E5 v2 to be pushed and registered: " + ", ".join(missing))
    tree = ast.parse((root / "dacapo/ace.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate"]
    if not functions or "rng_seed" not in [arg.arg for arg in functions[0].args.args + functions[0].args.kwonlyargs]:
        raise RuntimeError("E5 v2 generate(rng_seed=...) is required. Wait for its reviewed merge.")


def prepare(mode: str, work_dir: str | Path) -> dict:
    """Install only in the owner's disposable runtime, never during module import."""
    if mode not in {"pack", "gpu"}:
        raise ValueError("mode must be pack or gpu")
    require_python_version()
    checkout_state()
    registration = verify_registered_files()
    if mode == "gpu":
        require_e5_source()
        run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")
    work = Path(work_dir).expanduser().resolve()
    output_root = HERE / "outputs"
    inside_clone = work == ROOT or ROOT in work.parents
    inside_outputs = work == output_root or output_root in work.parents
    if inside_clone and not inside_outputs:
        raise ValueError("Runtime files must be outside the clone or inside e7/outputs.")
    work.mkdir(parents=True, exist_ok=True)
    e7_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT),
                  HF_HOME=str(work / "huggingface"), XDG_CACHE_HOME=str(work / "cache"),
                  NUMBA_CACHE_DIR=str(work / "cache/numba"), MPLCONFIGDIR=str(work / "cache/matplotlib"),
                  OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  E7_PYTHON=sys.executable)
    privilege = [] if os.geteuid() == 0 else ["sudo"]
    if privilege and shutil.which("sudo") is None:
        raise RuntimeError("FluidSynth installation needs an owner-managed system package installation.")
    run(*privilege, "apt-get", "update", "-qq")
    run(*privilege, "env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "-qq",
        "--no-install-recommends", "fluidsynth", "fluid-soundfont-gm", "ffmpeg")
    sf2 = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
    if not sf2.is_file():
        raise RuntimeError("FluidR3_GM.sf2 is missing after system package installation.")
    e7_env["DACAPO_SF2"] = str(sf2)
    pip = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    if mode == "gpu":
        run(*pip, "--index-url", "https://download.pytorch.org/whl/cu128",
            "torch==2.10.0+cu128", "torchaudio==2.10.0+cu128", "torchvision==0.25.0+cu128", env=e7_env)
    else:
        run(*pip, "--index-url", "https://download.pytorch.org/whl/cpu",
            "torch==2.10.0+cpu", "torchaudio==2.10.0+cpu", env=e7_env)
    run(*pip, "-r", str(HERE / "requirements-stage-s.txt"), env=e7_env)
    if mode == "gpu":
        # These are the inference dependencies used by the repository notebook.
        # ACE-Step itself is imported from its exact source checkout; no editable install.
        run(*pip, "transformers==4.57.6", "diffusers==0.37.0", "vector-quantize-pytorch==1.27.15",
            "einops==0.8.1", "loguru==0.7.3", "accelerate==1.12.0", "safetensors==0.7.0",
            "huggingface_hub==0.36.0", "nvidia-ml-py==13.580.126",
            "pytorch-wavelets==1.3.0", "pywavelets==1.9.0", "torchcodec==0.10.0", env=e7_env)
        run(sys.executable, "-c", "import torch; assert torch.cuda.is_available(); "
            "assert torch.cuda.is_bf16_supported(), 'Use a bf16 GPU such as L4 or A100'", env=e7_env)
        run(sys.executable, "experiments/e5_motif/register.py", "--check", cwd=ROOT, env=e7_env)
        ace = work / "ACE-Step-1.5"
        if not ace.exists():
            run("git", "init", "-q", str(ace))
            run("git", "-C", str(ace), "fetch", "--depth", "1",
                "https://github.com/ace-step/ACE-Step-1.5", ACE_COMMIT)
            run("git", "-C", str(ace), "checkout", "--detach", "FETCH_HEAD")
        actual = subprocess.check_output(["git", "-C", str(ace), "rev-parse", "HEAD"], text=True).strip()
        if actual != ACE_COMMIT:
            raise RuntimeError("Existing ACE source has a different commit; use a fresh runtime directory.")
        source_changes = subprocess.check_output(
            ["git", "-C", str(ace), "status", "--porcelain", "--untracked-files=all",
             "--", ".", ":(exclude)checkpoints"], text=True)
        if source_changes:
            raise RuntimeError("ACE source has local changes; use an unchanged pinned source checkout.")
        # local_dir contains real files. Never patch the shared hub cache.
        weights = work / "weights" / ACE_REVISION
        code_record = weights / MODEL_CODE_RECORD
        if not code_record.exists():
            snapshot_code = (
                "from pathlib import Path; "
                "from experiments.e7_melody_adapter.stage_s_setup import download_model_snapshot; "
                f"download_model_snapshot(Path({str(weights)!r}))"
            )
            run(sys.executable, "-c", snapshot_code, env=e7_env)
        expected = registration["config"]["models"]["original_model_code_sha256"]
        alignment_code = (
            "from pathlib import Path; "
            "from experiments.e7_melody_adapter.stage_s_setup import prepare_model_code_alignment; "
            f"prepare_model_code_alignment(Path({str(ace)!r}),Path({str(weights)!r}),{expected!r})"
        )
        alignment_env = {**e7_env, "ACESTEP_REPO": str(ace),
                         "PYTHONPATH": os.pathsep.join((str(ROOT), str(ace)))}
        run(sys.executable, "-c", alignment_code, env=alignment_env)
        link = ace / "checkpoints"
        if link.exists() or link.is_symlink():
            if link.resolve() != weights:
                raise RuntimeError("Existing ACE checkpoints use another source; choose a fresh runtime directory.")
        else:
            link.symlink_to(weights, target_is_directory=True)
        e7_env.update(ACESTEP_REPO=str(ace), ACESTEP_CHECKPOINTS_DIR=str(weights),
                      E7_ACE_WEIGHTS_DIR=str(weights))
    exported = {name: value for name, value in e7_env.items()
                if name in {"PYTHONDONTWRITEBYTECODE", "PYTHONPATH", "HF_HOME", "XDG_CACHE_HOME",
                            "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS", "DACAPO_SF2", "ACESTEP_REPO",
                            "ACESTEP_CHECKPOINTS_DIR", "E7_ACE_WEIGHTS_DIR", "E7_PYTHON"}}
    data = {"mode": mode, "environment": exported, "ace_source_commit": ACE_COMMIT,
            "ace_weights_revision": ACE_REVISION, "whisper_revision": WHISPER_REVISION}
    (work / "runtime.json").write_text(json.dumps(data, indent=2) + "\n")
    return data
