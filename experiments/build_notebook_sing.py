"""Build the Colab notebook of the sing tool (experiments/sing/sing.py): your melody and words, sung by ACE-Step 1.5
with Run B's recipe S*.

  python experiments/build_notebook_sing.py [--commit SHA]   ->  notebook_sing.ipynb

The notebook installs the same pinned packages as notebook_runB (experiments/build_notebook_b.py, from Run A's
pins), runs on Colab's default runtime (Python 3.12 or 3.13) with a GPU, and refuses the CPU. It checks out SHA
if given, else the default branch as it is when the notebook runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import build_notebook as A  # noqa: E402  (Run A's pins and helpers, as build_notebook_b uses them)

OUT = ROOT / "notebook_sing.ipynb"
PRETTY_MIDI = next(p for p in A.PINS if p.startswith("pretty_midi"))   # the form check needs only this one


def cells(commit: str | None) -> list[dict]:
    md, code = A.md, A.code
    pin = "`" + commit[:7] + "`" if commit else "the default branch as it is now"
    return [
        md(f"""
# Da Capo: sing my melody

Type a melody and one word per note. ACE-Step 1.5 (turbo) sings the words on the melody, a few times, and the
notebook keeps the take that scores best. It uses the recipe that [Da Capo]({A.REPO_URL})'s Run B picked (B2's
S*): a wordless Voice Oohs guide of your melody steers the first 3 of 8 steps, and the lyric clock nudges each
word toward its note.

What to expect. In Run B this recipe reached 14 of 24 new melodies when the best of 4 takes was kept, judged by
Whisper (at least half the notes on pitch and half the words heard, in order). About one in three single takes
got that far (31 of 96). The registered rule did not credit the clock for this: the same guide without the clock
reached 8 of 24, and the gap was too uncertain on 24 melodies. A second recogniser (a wav2vec2 CTC model) heard
the words in only 2 of the 24 kept takes. Not a product, and nothing here is a study.

- One syllable per note, English words, short phrases (up to about 20 seconds). Long words, melismas and fast
  passages are not handled.
- If you will take part in a blind listening test of Da Capo, please do not use this notebook before you rate.

How to use it:

1. Runtime > Change runtime type > an L4 or A100 GPU (Run B used an L4). These need Colab compute units (Colab
   Pro or pay as you go). The free T4 does not work: on GPUs older than compute capability 8.0, ACE-Step switches
   to float16 and another attention kernel, which Run B never used and the tool does not run. The notebook stops
   on a CPU or a T4 before installing anything.
2. Fill in the form below, then Runtime > Run all. The notebook checks your melody and words first. The first
   run then downloads about 14 GB of models (ACE-Step about 9.5 GB, Whisper large-v3 about 3 GB, the CTC model
   about 1.3 GB) and loads them; that time has not been measured on Colab yet. Once the models are loaded, Run B
   on an L4 made and scored a take in about 12 seconds (480 takes in 97 minutes), so 4 takes need about a minute.
3. The last cells show a table of all takes and play the kept one first. The files (WAV, MP3, the guide and
   `sing_result.json` with the settings, seeds, code commit and scores) go to Google Drive if you allow it,
   otherwise they download as a zip.

The code is {pin}; ACE-Step is pinned to `{A.ACE_COMMIT[:7]}` and the weights to revision `{A.HF_REVISION[:7]}`.
"""),
        code(f"""
#@title Your melody and words
#@markdown MELODY: notes separated by spaces, each a name with octave (C4 is middle C; sharps #, flats b), a colon
#@markdown and the length in beats. A beat is a quarter note in 4/4: 1 = quarter note, 2 = half note, 0.5 = eighth
#@markdown note, 1.5 = dotted quarter (3/2 also works). R:1 is a one-beat rest. Bar lines | are ignored, for example
#@markdown `E4:1 G4:1 A4:2 | G4:1 E4:1 D4:2 | C4:2 R:1 D4:1`
MELODY = "C4:1 E4:1 G4:2 A4:1 G4:1 E4:2 D4:2 C4:2"  #@param {{type:"string"}}
#@markdown Or a MIDI file with one monophonic track: its path (on Drive, for example), or tick UPLOAD_MIDI to upload it.
MIDI_FILE = ""  #@param {{type:"string"}}
UPLOAD_MIDI = False  #@param {{type:"boolean"}}
#@markdown LYRIC: one English word per note, separated by spaces, without punctuation (rests take no word).
#@markdown One-syllable words work best.
LYRIC = "Hold my hand and walk with me now"  #@param {{type:"string"}}
#@markdown TEMPO: beats per minute, a whole number from 30 to 300.
TEMPO = 100  #@param {{type:"integer"}}
#@markdown VOICE: the caption asks for a solo female or solo male vocal (Run B only tried female).
VOICE = "female"  #@param ["female", "male"]
#@markdown KEY: auto (chosen from your notes), or a key like D major or F# minor.
KEY = "auto"  #@param {{type:"string"}}
#@markdown DRAWS: how many takes to make; the best one is kept. More takes give a better chance and a longer wait.
DRAWS = 4  #@param {{type:"slider", min:1, max:8, step:1}}
#@markdown SEED: the same seed gives the same takes; change it to get new ones.
SEED = 0  #@param {{type:"integer"}}
#@markdown COMPARE_NO_CLOCK: also make the same takes without the lyric clock, to hear the difference.
COMPARE_NO_CLOCK = False  #@param {{type:"boolean"}}
#@markdown USE_DRIVE: save the files to Google Drive, in DRIVE_DIR. Untick it to download them as a zip at the end.
USE_DRIVE = True  #@param {{type:"boolean"}}
DRIVE_DIR = "/content/drive/MyDrive/dacapo_sing"  #@param {{type:"string"}}
REPO_COMMIT = {commit!r}  # None = the default branch as it is now
"""),
        code("""
# An L4 or A100 GPU is required. On a CPU one take needs many minutes and the scorer is not Run B's; on GPUs older
# than compute capability 8.0 (T4, V100) ACE-Step switches to float16 and eager attention, which Run B never used.
import os, re, subprocess, sys
import torch
from IPython.display import HTML, display
if not torch.cuda.is_available():
    display(HTML('<h2 style="color:#c00">This tool needs a GPU runtime.</h2>'
                 '<p>Runtime &gt; Change runtime type &gt; choose an L4 or A100 GPU, then Runtime &gt; Run all.</p>'))
    raise SystemExit("sing my melody needs a GPU runtime: Runtime > Change runtime type > L4 or A100")
GPU = torch.cuda.get_device_name(0)
if torch.cuda.get_device_capability(0)[0] < 8:
    display(HTML(f'<h2 style="color:#c00">This GPU ({GPU}) cannot run the tool.</h2>'
                 '<p>On a T4 or older GPU, ACE-Step runs in float16 with eager attention, which Run B never used. '
                 'Choose an L4 or A100: Runtime &gt; Change runtime type. These need Colab compute units '
                 '(Colab Pro or pay as you go).</p>'))
    raise SystemExit(f"{GPU} is older than compute capability 8.0: choose an L4 or A100 (Runtime > Change runtime type)")
print("GPU", GPU, "| Python", sys.version.split()[0], "| torch", torch.__version__)
if sys.version_info[:2] not in ((3, 12), (3, 13)):
    print("note: the notebook was written for Colab's default runtime (Python 3.12 or 3.13)")
"""),
        code(f"""
# The Da Capo code
def git(*a):
    return subprocess.run(["git", "-C", "/content/work/repo", *a], check=True, capture_output=True, text=True).stdout.strip()
if not os.path.exists("/content/work/repo/.git"):
    subprocess.run(["git", "clone", "-q", "{A.REPO_URL}", "/content/work/repo"], check=True)
git("fetch", "-q", "origin")
git("checkout", "-q", REPO_COMMIT or "origin/HEAD")
if not os.path.exists("/content/work/repo/experiments/sing/sing.py"):
    raise SystemExit("this commit of Da Capo has no experiments/sing/sing.py")
print("Da Capo", git("rev-parse", "HEAD"))
"""),
        code(f"""
# Check the melody and the words now, before the long install
!pip -q install {PRETTY_MIDI}
if MIDI_FILE.startswith("/content/drive") and not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
if UPLOAD_MIDI:
    from google.colab import files
    up = files.upload()
    if not up:
        raise SystemExit("no MIDI file uploaded")
    MIDI_FILE = os.path.abspath(next(iter(up)))
SING = "/content/work/repo/experiments/sing/sing.py"
melody = ["--midi", MIDI_FILE] if MIDI_FILE else ["--melody", MELODY]
form = ["--lyric", LYRIC, "--bpm", str(TEMPO), "--voice", VOICE, "--draws", str(DRAWS), "--seed", str(SEED), "--key", KEY]
r = subprocess.run([sys.executable, SING, "check", *melody, *form], capture_output=True, text=True,
                   env=dict(os.environ, PYTHONPATH="/content/work/repo"))
print(r.stdout + r.stderr)
if r.returncode:
    raise SystemExit("fix the melody or the words in the form above, then Run all again")
"""),
        code("""
# Google Drive keeps the files (optional)
OUT_DIR = "/content/sing_outputs"
if USE_DRIVE:
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        OUT_DIR = DRIVE_DIR
    except Exception as exc:
        print("Drive is not mounted:", exc, "\\nThe files stay on this runtime; the last cell downloads them.")
os.makedirs(OUT_DIR, exist_ok=True)
print("files go to", OUT_DIR)
"""),
        code(f"""
# Dependencies, exactly as notebook_runB: FluidSynth and a General MIDI soundfont, ACE-Step inference, the pinned
# measurement tools. Colab's PyTorch stays as it is.
torch_before = torch.__version__
!apt-get -qq update > /dev/null && apt-get -qq install -y fluidsynth fluid-soundfont-gm ffmpeg > /dev/null
!pip -q install {' '.join(A.ACE_PINS)} vector-quantize-pytorch einops loguru \\
     soundfile scipy safetensors huggingface_hub mido torchfcpe resampy scikit-learn nvidia-ml-py {' '.join(A.PINS)}
!pip -q install --no-deps {A.BASIC_PITCH}
torch_after = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
assert torch_after == torch_before, f"pip changed torch from {{torch_before}} to {{torch_after}}"
if not os.path.exists("/content/work/ACE-Step-1.5"):
    !git clone -q https://github.com/ace-step/ACE-Step-1.5 /content/work/ACE-Step-1.5
!cd /content/work/ACE-Step-1.5 && git checkout -q {A.ACE_COMMIT} && git log -1 --format="ACE-Step %h %cd"
"""),
        code(f"""
# Weights at Run B's revision: turbo DiT, VAE, text encoder (ACE-Step's loader also checks for the 1.7B LM)
from huggingface_hub import snapshot_download
snapshot_download("ACE-Step/Ace-Step1.5", revision="{A.HF_REVISION}", local_dir="/content/work/ace_ckpt",
                  allow_patterns=["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*", "acestep-5Hz-lm-1.7B/*", "config.json"])
print("ok")
"""),
        code("""
# Sing: generate the takes, score each with B2's scorer (Whisper large-v3 at Run B's revision), keep the best.
# Library warnings such as "You should probably TRAIN this model" are normal.
env = dict(os.environ, ACESTEP_REPO="/content/work/ACE-Step-1.5", ACESTEP_CHECKPOINTS_DIR="/content/work/ace_ckpt",
           PYTHONPATH="/content/work/repo", DACAPO_SF2="/usr/share/sounds/sf2/FluidR3_GM.sf2")
args = [sys.executable, SING, "run", *melody, *form, "--out", OUT_DIR, "--device", "cuda"]
if COMPARE_NO_CLOCK:
    args.append("--compare")
RUNDIR = None
p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
    m = re.match(r"(?:results in|\\d+ takes? in) (.+)$", line.strip())
    if m:
        RUNDIR = m.group(1)
p.wait()
WHY = {1: "the input was refused: read the line starting with refused: above",
       2: "a setup problem: read the line starting with setup problem: above",
       3: "no take was made: read the FAILED lines above"}
if p.returncode or not RUNDIR or not os.path.exists(f"{RUNDIR}/sing_result.json"):
    why = WHY.get(p.returncode, f"exit code {p.returncode}")
    raise SystemExit(f"sing.py stopped: {why}" + (f". The full logs are in {RUNDIR}/logs" if RUNDIR else ""))
"""),
        md("""
How to read the scores. q runs from 0 to 1: the vocal gate times the smaller of two shares, the notes sung on
pitch and the words Whisper heard in order. A take with q of 0.5 or more had at least half the notes in tune and
half the words heard (B2's bar); 0.75 was the strict bar. "Voice found" means a singing voice was detected. The
words column is Whisper's opinion alone. In Run B a second recogniser agreed in only 2 of the 24 kept takes, so
listen to the takes yourself.
"""),
        code("""
# All takes, the kept one first (q = vocal gate x the smaller of notes right and words recognised)
import html, json
from IPython.display import Audio, HTML, display
sys.path.insert(0, "/content/work/repo/experiments/sing")
import sing
doc = json.load(open(f"{RUNDIR}/sing_result.json"))
def label(t):
    if t["run_id"] == doc["kept"].get("clock"):
        return " (kept)"
    return " (best without clock)" if t["run_id"] == doc["kept"].get("no clock") else ""
rows = "".join(
    f"<tr><td>{t['take']}{label(t)}</td><td>{t['recipe']}</td><td>{t['q']:.2f}</td>"
    f"<td>{t['notes_right']}</td><td>{t['words_recognised']}</td><td>{'yes' if t['gate'] else 'no'}</td>"
    f"<td>{html.escape((t['whisper_text'] or '').strip() if t['status'] == 'scored' else t['status'])}</td></tr>"
    for t in sing.ordered(doc))
display(HTML("<table><tr><th>take</th><th>recipe</th><th>q</th><th>notes right</th><th>words recognised</th>"
             f"<th>voice found</th><th>Whisper heard</th></tr>{rows}</table>"))
for w in doc["warnings"]:
    print("warning:", w)
for t in sing.ordered(doc):
    f = t.get("mp3") or t.get("wav")
    if f:
        display(HTML(f"<b>take {t['take']}, {t['recipe']}{label(t)}</b>: q {t['q']:.2f}, "
                     f"notes {t['notes_right']}, words {t['words_recognised']}"))
        display(Audio(filename=f"{RUNDIR}/{f}"))
display(HTML("<b>the wordless guide</b>"))
display(Audio(filename=f"{RUNDIR}/{doc['guide']['file']}"))
print("files in", RUNDIR)
"""),
        code("""
# Without Drive: download the files as a zip
import shutil
if not RUNDIR.startswith("/content/drive"):
    z = shutil.make_archive(f"/content/{os.path.basename(RUNDIR)}", "zip", RUNDIR)
    from google.colab import files
    files.download(z)
"""),
    ]


def notebook(commit: str | None) -> dict:
    return {"cells": cells(commit), "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                                 "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=None, help="the Da Capo commit to check out (default: the default branch)")
    a = ap.parse_args()
    OUT.write_text(json.dumps(notebook(a.commit), indent=1) + "\n")
    print(OUT, f"{OUT.stat().st_size / 1e3:.0f} kB", "| commit", a.commit or "the default branch at run time")


if __name__ == "__main__":
    main()
