"""Build the Colab notebook for Run A (experiments/run_all.py): the E5 v2 confirmatory retry and the E6
read-out, run unattended on an L4.

  python experiments/build_notebook.py   ->  notebook.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook.ipynb"
REPO_URL = "https://github.com/ninondev/DaCapo"
ACE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
HF_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
# Versions of the measurement stack that the local K0 and the CPU tests ran with (addendum 2, section
# measurement; basic-pitch is installed without its dependencies below).
PINS = ["swift-f0==0.3.0", "onnxruntime==1.31.0", "librosa==1.0.0", "soxr==1.1.0", "mir_eval==0.8.2",
        "pretty_midi==0.2.11.post0", "demucs==4.1.0"]
BASIC_PITCH = "basic-pitch==0.4.0"
# The model-side libraries of the CPU test runs, pinned so a runtime reset reinstalls the same ones (A0 checks them).
ACE_PINS = ["transformers==4.57.6", "diffusers==0.39.0", "accelerate==1.15.0"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def main():
    cells = [
        md(f"""
# Da Capo, Run A

This notebook runs Run A of [Da Capo]({REPO_URL}) on ACE-Step 1.5 (turbo):
the corrected, pre-registered retry of the E5 motif study (addendum 2 in
`experiments/e5_motif/registration/`) and the E6 lyric-alignment read-out.
Every decision rule is applied in code, so the run needs no one watching it.

1. Choose Runtime > Change runtime type > L4 GPU. The E5 registration pins an NVIDIA L4; on another GPU,
   E5 stops at its A0 checks, before Phase 0b.
2. Choose Runtime > Run all and allow Google Drive access once. Drive keeps the ledgers, results, stage
   status and audio in `MyDrive/dacapo_runA`: about 5 GB, but E5 does not start with less than 7.5 GB
   free, which leaves headroom for noisier audio.
3. The run takes about 4 hours: 30 to 40 minutes of installs, downloads and setup (rendering and K0 for
   three timbres), about 2.5 hours of E5, and about 30 minutes of E6 with its model downloads. If the
   runtime resets, choose Run all again: it resumes from Drive at the commit it started with and never
   generates a finished draw twice. Without Drive the run still works but cannot resume, and a lost
   start of E5 v2 must still be reported.
4. The last cell downloads two files: `dacapo_runA_*.zip` (results, logs, a listening subset and the
   blind-listening pack `listening_pack.zip`) and its answer key `listening_key.json`. Give listeners the
   pack only.

Stages: setup (sources, render hashes, K0 per timbre), A0 generation checks, E5 Phase 0b and K1, the E5
confirmatory grid with its quality and accuracy stages, the E6 read-out, packaging. E5 starts only when
the code matches the hashes in addendum 2 and the registered K0 passed for flute; otherwise setup stops
E5 and the E6 read-out still runs. A setup or A0 failure is not a strike; the outcome of E5 v2 is written
once to `e5_terminal.json`.
ACE-Step is pinned to commit `{ACE_COMMIT[:7]}` and the weights to revision `{HF_REVISION[:7]}`.
"""),
        code("""
SCALE = "full"                          # "full" is the registered run; "tiny" only tests the code
STAGES = None                           # None = Run A; the v1 stages, e.g. ["smoke", "e1"], need SCALE = "quick"
REPO_COMMIT = None                      # None = the commit this scale's run recorded on Drive, else the default branch
DRIVE_DIR = "/content/drive/MyDrive/dacapo_runA"
import os, subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
"""),
        code("""
# Google Drive keeps the run state, so Run all can resume after a runtime reset
try:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print("run state in", DRIVE_DIR)
except Exception as exc:
    DRIVE_DIR = None
    print("Drive is not mounted:", exc)
    print("The run continues on the local disk; a runtime reset would lose its state.")
"""),
        code(f"""
# Code. A resumed run checks out the commit its first runtime used (kept on Drive for this scale only), so a
# later push to the default branch cannot change the code mid-run. A pin left by a run that never started
# must match the default branch; set REPO_COMMIT to move to another commit on purpose.
def git(*a):
    return subprocess.run(["git", "-C", "/content/work/repo", *a], check=True, capture_output=True, text=True).stdout.strip()
if not os.path.exists("/content/work/repo/.git"):
    subprocess.run(["git", "clone", "-q", "{REPO_URL}", "/content/work/repo"], check=True)
git("fetch", "-q", "origin")
latest = git("rev-parse", "origin/HEAD")
state_dir = f"{{DRIVE_DIR}}/{{SCALE}}" if DRIVE_DIR else None
commit_file = f"{{state_dir}}/repo_commit.txt" if state_dir else None
started = bool(state_dir) and os.path.exists(f"{{state_dir}}/status.json")
if not REPO_COMMIT and commit_file and os.path.exists(commit_file):
    pin = open(commit_file).read().strip()
    if pin and (started or pin == latest):
        REPO_COMMIT = pin
    elif pin:
        raise SystemExit(f"Drive pins commit {{pin[:7]}} for scale {{SCALE}}, but no {{SCALE}} run has started and the "
                         f"default branch is at {{latest[:7]}}. Delete {{commit_file}} to use the latest code, or set REPO_COMMIT.")
git("checkout", "-q", REPO_COMMIT or latest)
head = git("rev-parse", "HEAD")
if commit_file:
    os.makedirs(state_dir, exist_ok=True)
    open(commit_file, "w").write(head + "\\n")
print("Da Capo HEAD", head, "| default branch", latest[:7], "| resumed run" if started else "| new run")
!git -C /content/work/repo log -1 --format="Da Capo HEAD %h %cd"
!git -C /content/work/repo log -1 --format="addendum 2 last changed in %h %cd" -- experiments/e5_motif/registration/addendum_2.json
"""),
        code(f"""
# Dependencies: FluidSynth and a General MIDI soundfont, ACE-Step inference, the pinned measurement tools.
# Colab's PyTorch stays as it is.
torch_before = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
!apt-get -qq update > /dev/null && apt-get -qq install -y fluidsynth fluid-soundfont-gm > /dev/null
!pip -q install {' '.join(ACE_PINS)} vector-quantize-pytorch einops loguru \\
     soundfile scipy safetensors huggingface_hub mido torchfcpe resampy scikit-learn nvidia-ml-py {' '.join(PINS)}
!pip -q install --no-deps {BASIC_PITCH}
torch_after = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
assert torch_after == torch_before, f"pip changed torch from {{torch_before}} to {{torch_after}}"
if not os.path.exists("/content/work/ACE-Step-1.5"):
    !git clone -q https://github.com/ace-step/ACE-Step-1.5 /content/work/ACE-Step-1.5
!cd /content/work/ACE-Step-1.5 && git checkout -q {ACE_COMMIT} && git log -1 --format="ACE-Step %h %cd"
!fluidsynth --version | head -1
print("torch", torch_after)
"""),
        code(f"""
# Weights at the registered revision: turbo DiT, VAE, text encoder (ACE-Step's loader also checks for the 1.7B LM)
from huggingface_hub import snapshot_download
snapshot_download("ACE-Step/Ace-Step1.5", revision="{HF_REVISION}", local_dir="/content/work/ace_ckpt",
                  allow_patterns=["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*", "acestep-5Hz-lm-1.7B/*", "config.json"])
print("ok")
"""),
        code("""
# Run A. Every stage writes its log; E5 and E6 outcomes are decided by the registered rules in code.
env = dict(os.environ, ACESTEP_REPO="/content/work/ACE-Step-1.5", ACESTEP_CHECKPOINTS_DIR="/content/work/ace_ckpt",
           PYTHONPATH="/content/work/repo", DACAPO_SF2="/usr/share/sounds/sf2/FluidR3_GM.sf2",
           CUBLAS_WORKSPACE_CONFIG=":4096:8", ACESTEP_VAE_DECODE_CHUNK_SIZE="256")
check = subprocess.run([sys.executable, "/content/work/repo/experiments/e5_motif/register.py", "--check"],
                       env=env, capture_output=True, text=True)
print(check.stdout.strip())
args = [sys.executable, "/content/work/repo/experiments/run_all.py", "--scale", SCALE, "--device", "cuda"]
if DRIVE_DIR:
    args += ["--state", DRIVE_DIR]
if STAGES:
    args += ["--stages", ",".join(STAGES)]
p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
p.wait()
print("run_all exit code", p.returncode)
"""),
        code("""
# Outcome summary
import json
root = f"{DRIVE_DIR}/{SCALE}" if DRIVE_DIR else ("/content/work/package" if SCALE == "full" else f"/content/work/package_{SCALE}")
status = json.load(open(f"{root}/status.json"))
for name, st in status.items():
    if name != "_meta":
        print(f"{name:8s}", st.get("state"), {k: st[k] for k in ("paper_level", "reason", "b1_allowed", "labels", "run_b") if k in st})
"""),
        code("""
# Package and download; the blind-listening key is a separate file
import shutil, time
pkg = f"{root}/package" if DRIVE_DIR else root
key = f"{root}/listening_key.json" if DRIVE_DIR else f"/content/work/listening_key{'' if SCALE == 'full' else '_' + SCALE}.json"
name = f"/content/dacapo_runA_{time.strftime('%Y%m%d_%H%M%S')}"
shutil.make_archive(name, "zip", pkg)
print(name + ".zip", round(os.path.getsize(name + ".zip") / 1e6, 1), "MB")
from google.colab import files
files.download(name + ".zip")
if os.path.exists(key):
    files.download(key)
"""),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
