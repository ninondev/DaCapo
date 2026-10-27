"""Build the Colab notebook that runs this repo's GPU experiments (experiments/run_all.py).

  python experiments/build_notebook.py   ->  notebook.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook.ipynb"
REPO_URL = "https://github.com/ninondev/DaCapo"
ACE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def main():
    cells = [
        md(f"""
# Da Capo GPU tests

This notebook runs the GPU tests of [Da Capo]({REPO_URL}) on ACE-Step 1.5 (turbo) and packs the results into a zip.

1. Choose Runtime > Change runtime type > L4 GPU (A100 also works). Don't use a T4, because it has no bf16.
2. Choose Runtime > Run all. The first run downloads about 10 GB of open model weights.
3. The last cell downloads `dacapo_results_*.zip` with the result files, logs and a few mp3s.

All tests take about 2.5 hours on an L4. For a short trial, set `SCALE = "quick"`.
If the runtime disconnects, choose Run all again. Finished stages are skipped as long as `/content/work` is still there.
ACE-Step is pinned to commit `{ACE_COMMIT[:7]}`.
"""),
        code("""
SCALE = "full"   # "quick" = small trial run (about 15 minutes); "full" = paper scale
STAGES = None    # None = all stages, or e.g. ["smoke", "e1", "e3"]
import os, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
"""),
        code(f"""
# Code
if not os.path.exists("/content/work/repo"):
    !git clone -q --depth 1 {REPO_URL} /content/work/repo
!git -C /content/work/repo log -1 --format="Da Capo %h %cd"
"""),
        code(f"""
# Dependencies: FluidSynth + a General MIDI soundfont, ACE-Step inference, measurement tools (keeps Colab's PyTorch)
!apt-get -qq update > /dev/null && apt-get -qq install -y fluidsynth fluid-soundfont-gm > /dev/null
!pip -q install "transformers>=4.51,<4.58" "diffusers>=0.37" vector-quantize-pytorch einops loguru accelerate \\
     soundfile scipy safetensors huggingface_hub pretty_midi mido mir_eval librosa swift-f0 torchfcpe \\
     onnxruntime resampy scikit-learn nvidia-ml-py
!pip -q install --no-deps basic-pitch
if not os.path.exists("/content/work/ACE-Step-1.5"):
    !git clone -q https://github.com/ace-step/ACE-Step-1.5 /content/work/ACE-Step-1.5
!cd /content/work/ACE-Step-1.5 && git checkout -q {ACE_COMMIT} && git log -1 --format="ACE-Step %h %cd"
"""),
        code("""
# Weights: turbo DiT, VAE, text encoder (ACE-Step's loader also checks for the 1.7B LM, so it is fetched too)
from huggingface_hub import snapshot_download
snapshot_download("ACE-Step/Ace-Step1.5", local_dir="/content/work/ace_ckpt",
                  allow_patterns=["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*", "acestep-5Hz-lm-1.7B/*", "config.json"])
print("ok")
"""),
        code("""
# Run the suite: each stage writes its own log; a failed stage does not stop the next one (a failed smoke test does)
import sys
env = dict(os.environ, ACESTEP_REPO="/content/work/ACE-Step-1.5", ACESTEP_CHECKPOINTS_DIR="/content/work/ace_ckpt",
           PYTHONPATH="/content/work/repo", DACAPO_SF2="/usr/share/sounds/sf2/FluidR3_GM.sf2")
args = [sys.executable, "/content/work/repo/experiments/run_all.py", "--scale", SCALE]
if STAGES:
    args += ["--stages", ",".join(STAGES)]
p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
p.wait()
print("suite exit code", p.returncode)
"""),
        code("""
# Package and download the results
import shutil, time
name = f"/content/dacapo_results_{time.strftime('%Y%m%d_%H%M%S')}"
shutil.make_archive(name, "zip", "/content/work/package")
print(name + ".zip", round(os.path.getsize(name + ".zip") / 1e6, 1), "MB")
from google.colab import files
files.download(name + ".zip")
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
