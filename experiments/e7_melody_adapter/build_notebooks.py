"""Build the two unexecuted Stage S notebooks without touching root notebooks."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


CLONE = '''
import os, sys, subprocess, json
from pathlib import Path
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path("/content/dacapo-e7")
WORK = Path("/content/e7-runtime")
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "main",
                    "https://github.com/ninondev/DaCapo", str(REPO)], check=True)
else:
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("The runtime clone has local edits. Use a fresh Colab runtime.")
    subprocess.run(["git", "-C", str(REPO), "fetch", "origin", "main"], check=True)
    subprocess.run(["git", "-C", str(REPO), "switch", "main"], check=True)
    subprocess.run(["git", "-C", str(REPO), "merge", "--ff-only", "origin/main"], check=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ["PYTHONPATH"] = str(REPO)
if not (REPO / "experiments/e7_melody_adapter/prereg_suno.json").is_file():
    raise RuntimeError("Wait for the preregistration PR to be reviewed and merged into main.")
print("DaCapo commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
from experiments.e7_melody_adapter.stage_s_setup import checkout_state, verify_registered_files, prepare
checkout_state()
verify_registered_files()
'''


def notebook(cells: list[dict], gpu: bool) -> dict:
    metadata = {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        metadata.update(accelerator="GPU", colab={"gpuType": "L4", "provenance": []})
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def build() -> dict[str, dict]:
    pack = notebook([
        md('''
# Stage S: make the upload pack

Use the **default Colab CPU runtime** (Python 3.12 or 3.13). Run this only after the preregistration PR has been
reviewed and merged into main. This notebook does not contact Suno or download song-model weights.
It creates 12 original melody MIDI files, piano and Voice Oohs WAV files, lyric text,
and a recording table. Do not upload any owner-held-out melody.

Rendered outputs stay in the ignored E7 outputs folder. Model caches stay outside the clone.
Dependencies are installed in this disposable
runtime. The existing repository source is never edited. Keep the upload hash JSON and give
it to the reviewer before generating on Suno. Read `suno_protocol.md` for the three routes.
'''),
        code(CLONE),
        code('''
runtime = prepare("pack", WORK)
os.environ.update(runtime["environment"])
PACK_OUTPUT = REPO / "experiments/e7_melody_adapter/outputs/upload-pack"
subprocess.run(["bash", "experiments/e7_melody_adapter/stage_s_run.sh", "pack",
                "--output", str(PACK_OUTPUT)], check=True, env=dict(os.environ))
'''),
        code('''
from google.colab import files
hashes = json.loads((PACK_OUTPUT / "suno_upload_hashes.json").read_text())
assert hashes["environment"]["git_commit"] != "unknown"
assert hashes["environment"]["git_dirty"] is False
checkout_state()
files.download(str(PACK_OUTPUT / "e7_suno_upload.zip"))
files.download(str(PACK_OUTPUT / "suno_upload_hashes.json"))
'''),
        md('''
Keep both files. Give `suno_upload_hashes.json` to the reviewer. Follow the protocol after
the preregistration is merged. Retain both outputs of each single Suno request and record
the actual menu option, displayed model version, date, and credits. Skip unavailable routes.
The GPU scoring notebook additionally requires the reviewed E5 v2 merge and registration.
'''),
    ], False)
    suno = notebook([
        md('''
# Stage S: score on a GPU

Use the **default Colab runtime with an L4 or A100 GPU** (Python 3.12 or 3.13). Run only after the Stage S preregistration and
E5 v2 have both been reviewed and merged. The first guards stop before model downloads if
E5 v2 is missing. This runtime downloads the pinned open ACE-Step and Whisper weights and
the htdemucs separator. No service API or paid endpoint is called by this code.

Scoring has no time stop. Every scored clip is checkpointed in Drive under its audio hash.
Run the same notebook again with the same input and output paths to resume the same
preregistration. It reuses finished scores and generated audio. The ruler is scored first,
then Suno, then ACE-Step controls. Only ACE generation has its registered active-generation
budget; a control timeout does not invalidate a completed ruler-and-Suno decision.

Suno scores stay private. The public zip contains only ACE-Step controls and the synthetic
ruler. Neither zip contains Suno audio or extracted vocals. Do not publish scores or a route
outcome until the owner confirms the terms allow publication. This notebook was not GPU-run
in the development environment.
'''),
        code(CLONE),
        code('''
from google.colab import drive
drive.mount("/content/drive")
# Edit only these input/output locations. Do not edit registered scoring definitions.
INPUTS = Path("/content/drive/MyDrive/e7-suno/audio")
RECORDS_CSV = Path("/content/drive/MyDrive/e7-suno/suno_recording_table.csv")
RESULTS = Path("/content/drive/MyDrive/e7-suno/scored")
# Keep this directory for every resume of this preregistered run.
RESULTS.mkdir(parents=True, exist_ok=True)
if not INPUTS.is_dir() or not RECORDS_CSV.is_file():
    raise FileNotFoundError("Set INPUTS and RECORDS_CSV to the downloaded audio and completed table.")
'''),
        code('''
from experiments.e7_melody_adapter.stage_s_setup import require_e5_source, preflight_audio_inputs
require_e5_source()
preflight_audio_inputs(INPUTS)
runtime = prepare("gpu", WORK)
os.environ.update(runtime["environment"])
# Basic Pitch uses its ONNX path. Install it separately without TensorFlow dependencies.
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "basic-pitch==0.4.0"],
               check=True, env=dict(os.environ))
import inspect
from dacapo.ace import generate
assert "rng_seed" in inspect.signature(generate).parameters, "Wait for E5 v2."
subprocess.run([sys.executable, "experiments/e5_motif/register.py", "--check"], check=True)
from experiments.e4_sing.run_sing import arms
available_arms = {arm["name"] for arm in arms(["clarinet", "oohs"])}
assert {"noguide", "ctx_oohs_k3", "ctx_oohs_k6", "ctx_clarinet_k4"} <= available_arms
'''),
        code('''
checkout_state()
verify_registered_files()
subprocess.run(["bash", "experiments/e7_melody_adapter/stage_s_run.sh", "suno",
                "--inputs", str(INPUTS), "--records-csv", str(RECORDS_CSV),
                "--output-dir", str(RESULTS)], check=True, env=dict(os.environ))
'''),
        code('''
import zipfile
from google.colab import files
# Check every result JSON before sharing either archive. Do not print private scores.
for archive_name in ("e7_suno_public.zip", "suno_private.zip"):
    archive_path = RESULTS / archive_name
    with zipfile.ZipFile(archive_path) as archive:
        result_names = [name for name in archive.namelist() if name.endswith(".json")]
        assert result_names, f"No result JSON in {archive_name}"
        for name in result_names:
            result = json.loads(archive.read(name))
            assert result["environment"]["git_commit"] != "unknown", name
            assert result["environment"]["git_dirty"] is False, name
            assert result["config_hash"], name
    files.download(str(archive_path))
checkout_state()
'''),
        md('''
The scorer prints the two completed zip locations, not Suno scores. Copy `e7_suno_public.zip`
for the public review and keep `suno_private.zip` in the owner's private store. The reviewer
applies the preregistered decision rule. Missing pairs are listed and are not invented.
Keep the Drive output directory and all checkpoints. On a runtime interruption, rerun the
same notebook and the same output path to continue. Unreadable audio must be downloaded
again from the original request; do not generate a replacement song. A control-generation
budget stop is reported separately from ruler and Suno completion. Do not select a
favorable subset or silently change the registered generation budget.
'''),
    ], True)
    return {"colab_e7_pack.ipynb": pack, "colab_e7_suno.ipynb": suno}


def main() -> None:
    for name, data in build().items():
        # Stable cell ids keep regeneration exact without storing execution outputs.
        for i, cell in enumerate(data["cells"]):
            cell["id"] = f"e7-{i:02d}"
        (HERE / name).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        print(name)


if __name__ == "__main__":
    main()
