"""Build the Colab notebook for Run B (experiments/run_b.py): B1 and B2 of the lyric clock, run unattended on
one L4 with the run state on Google Drive.

  python experiments/build_notebook_b.py [--commit SHA]   ->  notebook_runB.ipynb

The notebook checks out the registered commit: SHA if given, else the commit that last changed addendum 3 on
the default branch. It installs the same pinned packages as Run A's notebook (experiments/build_notebook.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import build_notebook as A  # noqa: E402  (Run A's pins and helpers)

OUT = ROOT / "notebook_runB.ipynb"
ADDENDUM_3 = "experiments/e5_motif/registration/addendum_3.json"


def cells(commit: str | None) -> list[dict]:
    md, code = A.md, A.code
    return [
        md(f"""
# Da Capo, Run B

This notebook runs Run B of [Da Capo]({A.REPO_URL}) on ACE-Step 1.5 (turbo): the lyric clock, a fixed bias on
the cross-attention that ties each sung word to its written note.

- B1 tests timing alone, exactly as addendum 2 and the B1 code addendum fix it (180 runs).
- B2 tests the clock with a melody guide (addendum 3): a recipe is picked on the 12 E4 lines and tested
  on 24 new lines with 4 draws each.

Every decision rule is applied in code, so the run needs no one watching it.

1. Choose Runtime > Change runtime type > L4 GPU.
2. Choose Runtime > Run all and allow Google Drive access. Drive keeps the ledgers, audio, results and stage
   status in `MyDrive/dacapo_runB` (about 7 GB; setup stops if less than 7.7 GB is free).
   Without Drive the notebook stops before running, because a
   runtime reset would lose the run (Run A ran without Drive by accident).
3. The run is expected to take about 10-13 hours on one L4, after the timing trim that A0 applies; Whisper's
   repetition loops on wordless clips take much of that time. If the runtime resets or Colab stops it, choose
   Run all again: it resumes from Drive at the commit it started with and never generates a finished run twice.
4. The last cell downloads `dacapo_runB_*.zip` (results, ledgers, logs and the 48-clip demo) and, as separate
   files, the blind A/B pack `b2_blind_pack.zip` and its key `b2_blind_key.json`. Give listeners the pack only,
   before they hear the demo.

The code is the registered commit ({'`' + commit[:7] + '`' if commit else 'the commit that last changed addendum 3'}),
ACE-Step is pinned to `{A.ACE_COMMIT[:7]}` and the weights to revision `{A.HF_REVISION[:7]}`.
"""),
        code(f"""
SCALE = "full"                          # "full" is the registered run; "tiny" only tests the code
DRIVE_DIR = "/content/drive/MyDrive/dacapo_runB"
ALLOW_NO_DRIVE = False                  # True runs without Drive: a runtime reset then loses the run
REPO_COMMIT = None                      # None = the commit this scale's run recorded on Drive, else the registered one
REGISTERED = {commit!r}  # None = the commit that last changed addendum 3 on the default branch
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
    from IPython.display import HTML, display
    display(HTML('<h1 style="color:#c00">GOOGLE DRIVE IS NOT MOUNTED</h1>'
                 '<h2>The run state would stay on this runtime and be lost if it resets.</h2>'
                 '<h2>Mount Drive and choose Run all again, or set ALLOW_NO_DRIVE = True in the first cell.</h2>'))
    print("#" * 60, "\\nGOOGLE DRIVE IS NOT MOUNTED:", exc, "\\n" + "#" * 60)
    if not ALLOW_NO_DRIVE:
        raise SystemExit("Run B stops: Drive is not mounted")
    DRIVE_DIR = None
"""),
        code(f"""
# Code: the registered commit. A resumed run checks out the commit its first runtime used (kept on Drive for this
# scale only). A run has started once a B1 or B2 ledger holds a run; a pin left by a run that never started must
# match the registered commit. Set REPO_COMMIT to move to another commit on purpose.
def git(*a):
    return subprocess.run(["git", "-C", "/content/work/repo", *a], check=True, capture_output=True, text=True).stdout.strip()
def has_runs(ledger):
    return os.path.exists(ledger) and any('"run_id"' in line for line in open(ledger))
if not os.path.exists("/content/work/repo/.git"):
    subprocess.run(["git", "clone", "-q", "{A.REPO_URL}", "/content/work/repo"], check=True)
git("fetch", "-q", "origin")
registered = REGISTERED or git("log", "-1", "--format=%H", "origin/HEAD", "--", "{ADDENDUM_3}")
if not registered:
    raise SystemExit("addendum 3 is not on the default branch: Run B is not registered")
state_dir = f"{{DRIVE_DIR}}/{{SCALE}}" if DRIVE_DIR else None
commit_file = f"{{state_dir}}/repo_commit.txt" if state_dir else None
started = bool(state_dir) and any(has_runs(f"{{state_dir}}/{{x}}/ledger.jsonl") for x in ("b1/outputs", "b2/dev", "b2/test"))
if not REPO_COMMIT and commit_file and os.path.exists(commit_file):
    pin = open(commit_file).read().strip()
    if pin and (started or pin == registered):
        REPO_COMMIT = pin
    elif pin:
        raise SystemExit(f"Drive pins commit {{pin[:7]}} for scale {{SCALE}}, but no {{SCALE}} run has generated anything and "
                         f"the registered commit is {{registered[:7]}}. Delete {{commit_file}} to use the registered commit, "
                         f"or set REPO_COMMIT.")
want = REPO_COMMIT or registered
git("checkout", "-q", want)
head = git("rev-parse", "HEAD")
if head != git("rev-parse", want):
    raise SystemExit(f"HEAD is {{head[:7]}}, not the pinned commit {{want[:7]}}")
if commit_file:
    os.makedirs(state_dir, exist_ok=True)
    open(commit_file, "w").write(head + "\\n")
print("Da Capo HEAD", head, "| registered", registered[:7], "| resumed run" if started else "| new run")
!git -C /content/work/repo log -1 --format="Da Capo HEAD %h %cd"
!git -C /content/work/repo log -1 --format="addendum 3 last changed in %h %cd" -- {ADDENDUM_3}
"""),
        code(f"""
# Dependencies: FluidSynth and a General MIDI soundfont, ACE-Step inference, the pinned measurement tools (as Run A).
# Colab's PyTorch stays as it is.
torch_before = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
!apt-get -qq update > /dev/null && apt-get -qq install -y fluidsynth fluid-soundfont-gm ffmpeg > /dev/null
!pip -q install {' '.join(A.ACE_PINS)} vector-quantize-pytorch einops loguru \\
     soundfile scipy safetensors huggingface_hub mido torchfcpe resampy scikit-learn nvidia-ml-py {' '.join(A.PINS)}
!pip -q install --no-deps {A.BASIC_PITCH}
torch_after = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
assert torch_after == torch_before, f"pip changed torch from {{torch_before}} to {{torch_after}}"
if not os.path.exists("/content/work/ACE-Step-1.5"):
    !git clone -q https://github.com/ace-step/ACE-Step-1.5 /content/work/ACE-Step-1.5
!cd /content/work/ACE-Step-1.5 && git checkout -q {A.ACE_COMMIT} && git log -1 --format="ACE-Step %h %cd"
!fluidsynth --version | head -1
print("torch", torch_after)
"""),
        code(f"""
# Weights at the registered revision: turbo DiT, VAE, text encoder (ACE-Step's loader also checks for the 1.7B LM)
from huggingface_hub import snapshot_download
snapshot_download("ACE-Step/Ace-Step1.5", revision="{A.HF_REVISION}", local_dir="/content/work/ace_ckpt",
                  allow_patterns=["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*", "acestep-5Hz-lm-1.7B/*", "config.json"])
print("ok")
"""),
        code("""
# The registrations: addendum 2 (E5 v2 and E6), the B1 code addendum and addendum 3 must all be current
# DISABLE_SAFETENSORS_CONVERSION: the CTC model loads pytorch_model.bin at its pinned revision; this stops
# transformers from also fetching a converted copy in the background
env = dict(os.environ, ACESTEP_REPO="/content/work/ACE-Step-1.5", ACESTEP_CHECKPOINTS_DIR="/content/work/ace_ckpt",
           PYTHONPATH="/content/work/repo", DACAPO_SF2="/usr/share/sounds/sf2/FluidR3_GM.sf2",
           CUBLAS_WORKSPACE_CONFIG=":4096:8", ACESTEP_VAE_DECODE_CHUNK_SIZE="256", DISABLE_SAFETENSORS_CONVERSION="true")
bad = []
for script in ("experiments/e5_motif/register.py", "experiments/e6_clock/register_b1code.py",
               "experiments/e8_clock_sing/register_b2.py"):
    r = subprocess.run([sys.executable, f"/content/work/repo/{script}", "--check"], env=env, capture_output=True, text=True)
    print(script, "->", (r.stdout + r.stderr).strip()[-300:])
    bad += [script] if r.returncode else []
if bad:
    raise SystemExit(f"registration check failed: {bad}")
"""),
        code("""
# Run B. Every stage writes its log; the B1 and B2 outcomes are decided by the registered rules in code.
args = [sys.executable, "/content/work/repo/experiments/run_b.py", "--scale", SCALE, "--device", "cuda"]
if DRIVE_DIR:
    args += ["--state", DRIVE_DIR]
p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
p.wait()
print("run_b exit code", p.returncode)
"""),
        code("""
# Outcome summary
import json
root = f"{DRIVE_DIR}/{SCALE}" if DRIVE_DIR else ("/content/work/runB" if SCALE == "full" else f"/content/work/runB_{SCALE}")
status = json.load(open(f"{root}/status.json"))
for name, st in status.items():
    if name != "_meta":
        print(f"{name:8s}", st.get("state"), {k: st[k] for k in ("verdict", "reasons", "labels", "fallback", "trim", "s_star",
                                                                 "b_star", "note") if st.get(k) is not None})
"""),
        code("""
# Package and download; the blind A/B pack and its key are separate files
import shutil, time
name = f"/content/dacapo_runB_{time.strftime('%Y%m%d_%H%M%S')}"
shutil.make_archive(name, "zip", f"{root}/package")
print(name + ".zip", round(os.path.getsize(name + ".zip") / 1e6, 1), "MB")
from google.colab import files
files.download(name + ".zip")
for extra in ("b2_blind_pack.zip", "b2_blind_key.json"):
    if os.path.exists(f"{root}/{extra}"):
        files.download(f"{root}/{extra}")
"""),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=None, help="the registered commit (default: the last commit of addendum 3)")
    a = ap.parse_args()
    nb = {"cells": cells(a.commit), "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                                 "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(OUT, f"{OUT.stat().st_size / 1e3:.0f} kB", "| commit", a.commit or "resolved from addendum 3 at run time")


if __name__ == "__main__":
    main()
