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
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Select a Python 3.12 runtime before running this notebook.")
    checkout_state()
    verify_registered_files()
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
    run(*pip, "-r", str(HERE / "requirements-e7.txt"), env=e7_env)
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
        # Download exact revisions in a separate process after installing dependencies.
        snapshot_code = (
            "from huggingface_hub import snapshot_download; import json; from pathlib import Path; "
            f"p=snapshot_download('ACE-Step/Ace-Step1.5',revision={ACE_REVISION!r},"
            "allow_patterns=['acestep-v15-turbo/*','vae/*','Qwen3-Embedding-0.6B/*',"
            "'acestep-5Hz-lm-1.7B/*','config.json']); "
            f"Path({str(work / 'weights-path.json')!r}).write_text(json.dumps(p))"
        )
        run(sys.executable, "-c", snapshot_code, env=e7_env)
        weights = Path(json.loads((work / "weights-path.json").read_text())).resolve()
        if weights.name != ACE_REVISION:
            raise RuntimeError("The ACE weight cache does not resolve to the registered revision.")
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
