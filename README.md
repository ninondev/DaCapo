# Da Capo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninondev/DaCapo/blob/main/notebook.ipynb)

English | [中文](README.zh-CN.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/hero_dark.svg">
  <img alt="Diagram: in ACE-Step 1.5, a melody sent through the planner's 5 Hz plan tokens has the right pitch class in 29% of frames. The same melody given to the renderer as dense audio scores 91%, as high as the input itself (90%)." src="figures/hero_light.svg" width="100%">
</picture>

I studied music for five years as a kid (amateur grade exams, grade 6 or 7 out of 10) and stopped in middle school. AI song models got me interested again. I wanted one to sing a melody I wrote. The song sounded good, but the melody wasn't mine, and when I asked for my chorus to come back later in the song, it usually didn't.

This repo is my attempt to find out why. It studies one open model, ACE-Step 1.5, and measures where a melody gets lost inside it.

## Who did what

- Me: choosing the question; running the GPU tests in Colab; deciding to move the tests to a GPU and to build melody control inside the renderer; asking for each main study's rules to be written down before it runs, and for the final claims to come from people listening blind.
- Another AI: suggested ACE-Step 1.5 and the overall plan.
- Coding agents: designed the tests, wrote the code, ran the CPU tests, worked out the numbers, and wrote this README and the log.

## How the model works

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (MIT license) has two parts. A language model, the planner, writes a rough plan of the song as tokens, 5 per second. A flow-matching model, the renderer, turns that plan plus the lyrics and a short text prompt into audio in 8 denoising steps. The renderer can also take real audio as dense context instead of a plan.

To score a melody, the tests use pitch-class accuracy: the share of time where the note name (C, D, E and so on, octave ignored) matches the written notes.

## What the tests found

1. The plan loses most of the melody. Through the planner's tokens, only 29% of frames have the right note name, where random guessing gets about 10%. That's with tokens made from the melody's own audio, an easy case for any planner. The same melody given to the renderer as dense audio scores 91%, as high as the input itself (90%).
2. The loss is in the tokenizer. A linear probe reads the note name from audio latents at the same 5 Hz rate with 0.919 accuracy. From the plan tokens it gets 0.575.
3. In these tests the model sings the words or the melody, not both. With lyrics but no melody it sings 66% of the words. When it's guided with a wordless version of the melody, the notes become right (86% to 100%) and the words drop to 3% or less.

So I decided to build control inside the renderer, where the melody survives. The code can look at each of the 8 denoising steps, read or change what the model is making, and switch the text, lyrics or audio it listens to. It adds no extra model calls.

## Results

Every number here comes from a file in [`results/`](results/), and [`figures/make_figures.py`](figures/make_figures.py) draws every chart from those files. The tests are still small, so each section gives its sample size.

### Where the melody gets lost

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e1_melody_paths_dark.svg">
  <img alt="Bar chart of melody pitch-class accuracy, pooled over 80, 120 and 160 BPM, at shift 3.0 and 1.0: input itself 0.90 and 0.90; dense 25 Hz latents as context 0.91 and 0.91; VAE round trip 0.90 and 0.90; 5 Hz plan tokens 0.29 and 0.31; plan tokens with new instrumentation 0.31 and 0.32; no audio condition 0.10 and 0.11." src="figures/e1_melody_paths_light.svg" width="100%">
</picture>

The agents made 16 melodies with known notes, each a piano or flute line over chords, bass and drums, at 80, 120 and 160 BPM. Each one went through every path of the model, and the tests measured how much melody came out.

Dense audio keeps the note names as well as the measuring tool can see. Its note F1 is 0.72 against 0.78 for the original, about the same loss as encoding and decoding the audio once. The plan tokens keep much less. Every output is also scored against the other melodies at the same tempo, and after removing that chance level, the tokens keep 28% (95% CI 22% to 34%) of the melody information. Their note F1 is 0.12.

<sub>16 melodies × 3 tempos, one seed, NVIDIA L4, 8-step turbo model. Shift is a sampler setting: 3.0 is what the ACE-Step docs recommend for turbo, and the default 1.0 gives the same result. The "no audio" bar uses only 6 different generations per setting. A smaller CPU test the agents ran earlier (4 melodies per tempo) gave the tokens 0.51 at 90 BPM, and the GPU run didn't repeat that, so this page reports the GPU numbers. Code: [`experiments/e1_melody_paths`](experiments/e1_melody_paths/).</sub>

### Why it gets lost

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e1b_probe_dark.svg">
  <img alt="Bar chart of linear-probe pitch-class accuracy on held-out melodies, plan tokens versus audio latents at the same 5 Hz rate: all frames 0.575 vs 0.919; pitch steady within the frame 0.645 vs 0.982; note changes within the frame 0.412 vs 0.773; unseen lead instrument 0.504 vs 0.846. The majority-class baseline is 0.125." src="figures/e1b_probe_light.svg" width="100%">
</picture>

The low frame rate isn't the whole reason. When a note stays the same for a full 200 ms frame, the probe reads it from the audio latents almost perfectly (0.982) but from the plan tokens only at 0.645. Reading the separate code digits inside each token does a little better (0.748). The tokens also mix the instrument sound into the plan: over whole clips, changing only the lead instrument changes about as many tokens as changing the melody (73% vs 71% of frames). Encoding the same audio twice already changes about 15%.

<sub>48 melodies × 3 lead instruments. The probe learns on 42 melodies and is tested on the other 6 (720 frames, one split). Code: [`experiments/e1_melody_paths`](experiments/e1_melody_paths/).</sub>

### Words or melody

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e4_tradeoff_dark.svg">
  <img alt="Scatter plot of 21 settings (20 guides and no guide): notes sung at the written pitch against lyric words recognised. Without a guide, 0.03 of notes and 0.66 of words. Settings that reach 0.86 or more of the notes keep at most 0.03 of the words. Only 2 of 378 clips get at least half of both right." src="figures/e4_tradeoff_light.svg" width="100%">
</picture>

The agents asked the model to sing one lyric line on a melody, one note per one-syllable word. Without a guide it doesn't know the melody, so its notes match only by chance (3%), but it sings 66% of the words. Then they guided it with the melody sung or played without words, either as context for the first k of the 8 steps or as a noisy starting point (SDEdit, a known method used here as a baseline). Every setting that got the notes right (86% or more) lost the words (3% or less). Only 2 of 378 test clips got at least half of both.

The working guess is that melody and lyrics have to be guided together, one syllable at a time. That's the idea behind the lyric clock (E6), which isn't built yet.

<sub>12 melodies (3 for trying settings, 9 for testing) × 21 settings × 2 seeds. Words come from a Whisper large-v3 transcript. This test is exploratory: its settings weren't registered before the run. Code: [`experiments/e4_sing`](experiments/e4_sing/).</sub>

### Measuring is hard

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e0_ruler_dark.svg">
  <img alt="Dot plot of pitch-tracker accuracy on a perfectly played melody, alone versus with a loud band: piano lead 0.95 vs 0.09, synth lead 0.95 vs 0.55, voice-like lead 0.93 vs 0.59." src="figures/e0_ruler_light.svg" width="100%">
</picture>

Before testing the model, the agents tested the measuring tools on audio where every note is known. Pitch trackers are very good on a solo melody and bad once a loud band plays. With a voice-like melody in a full mix, one or two wrong notes are hard to catch (best AUC 0.65 to 0.78). So with a band, every score is compared with the same score on the original audio, and singing is only scored in songs where the prompt asked for no instruments.

<sub>9 melodies per condition in the chart; the wrong-note tests use 6 melodies per condition. Code: [`experiments/e0_ruler`](experiments/e0_ruler/).</sub>

## Working on now

The hooks live in [`dacapo/ace.py`](dacapo/ace.py). At each of the 8 denoising steps they show what the model expects the final audio to be, and an experiment can read it, change it, or swap the text, lyrics or audio part way through. If a hook changes nothing, the output is exactly the same as without it (difference 0.0 on CPU and GPU), so any change in the output comes from the edit itself.

The main study (E5) asks whether the model can develop a motif. The test plays it a pattern like A B A, or a motif and the same motif moved up, and leaves a gap for the model to fill. Then it checks what the model put in the gap: a copy, a transposition, a sequence or an analogy, compared with shuffled melodies and wrong timing as controls. As I asked, the hypotheses, the stop rules and the code version were written down before the first run ([registration](experiments/e5_motif/registration/prereg.json)).

That first run failed its own quality check. Only 49% of the gaps had one clear melody line at a normal volume, and the rule required 75%, so by that rule it says nothing about motifs either way. The agents found two mistakes in their setup that can explain it: the model got the wrong task instruction for filling a gap, and every song started from the same noise. The result and a correction are in [`v1_outcome.json`](experiments/e5_motif/registration/v1_outcome.json). The fixed run will be registered before it starts.

The second study (E6) is the lyric clock. It will guide the attention heads that connect lyrics to audio, so each syllable lands on its note while the melody guide keeps the pitch.

## Status

| | Question | Status |
|---|---|---|
| E0 | How good are the measuring tools? | Done |
| E1 | How much melody survives each path through the model? | Done on GPU: 16 melodies × 3 tempos, two shift settings |
| E1b | What is in the plan tokens, and where is the loss? | Done on GPU: 48 melodies, one test split |
| E2 | Can the code hook every denoising step without changing the output? | Done: exactly the same output on CPU and GPU |
| E3 | Can the pitch be read from the audio latents for any instrument? | Exploratory: 0.88 on new instruments (chance 0.03); a special subspace is no better than the full latent |
| E4 | Can it sing the right notes and clear words? | Exploratory: 12 melodies × 21 settings × 2 seeds |
| E5 | Can it develop a motif? | Run 1 failed its quality check; the fixed run is next |
| E6 | Can a lyric clock time each syllable? | Planned |

[`LOG.md`](LOG.md) has every run, including the failures and the mistakes.

## How the tests are run

The measuring tools are tested on known notes before the model is. For the main studies, the hypotheses and stop rules are written down before the run (I asked for this), and when something changes, a new file is added instead of editing the old one. Bad results get published too. Every method tested will get the same number of tries, with "generate many times and keep the best" as a baseline. No language model judges the music; final claims will come from people listening blind (I asked for this too). Each number in this repo can be traced to a result file, and the audio can be made again from code with fixed seeds (the same device gives the same result, but CPU and GPU can differ).

## Run it

The easiest way is the Colab button at the top. Pick an L4 runtime and choose Run all. The notebook installs everything, downloads the open ACE-Step weights (about 10 GB), runs all the GPU tests (about 2.5 hours on an L4, plus install and download), and gives you a zip with the results.

To run the measuring tools on your own computer (macOS or Linux):

```bash
git clone https://github.com/ninondev/DaCapo && cd DaCapo
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,transcription]"
uv pip install --no-deps basic-pitch        # ONNX model, so no TensorFlow
# FluidSynth + a General MIDI soundfont:
#   Ubuntu: sudo apt install fluidsynth fluid-soundfont-gm
#   macOS:  brew install fluid-synth, then export DACAPO_SF2=/path/to/FluidR3_GM.sf2
pytest -q                                    # 13 tests
python experiments/e0_ruler/run.py --quick   # test the measuring tools (~3 min on CPU)
python figures/make_figures.py               # redraw all charts from results/
```

Tests that run the song model need [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) installed as well. [`experiments/run_all.py`](experiments/run_all.py) lists the exact command for each step.

<details>
<summary>Files</summary>

```
dacapo/          the library: melodies, rendering, pitch tracking, scores, and ace.py (hooks into ACE-Step)
experiments/     one folder per study (e0 to e5), plus run_all.py, which the notebook calls
results/cpu/     result files from CPU runs
results/gpu_run1/ result files from the first GPU run (NVIDIA L4)
figures/         charts and the script that draws them
tests/           unit tests and a render-to-analysis test
LOG.md           every run, in order
notebook.ipynb   the Colab notebook
```

</details>

## Plan

- [x] Test the measuring tools on known notes
- [x] Find where the melody gets lost inside an open song model
- [x] Hooks on every denoising step, checked to be exact on CPU and GPU
- [x] Map the trade-off between right notes and clear words (exploratory)
- [ ] Motif study: the fixed run, registered before it starts
- [ ] Lyric clock (E6)
- [ ] More seeds, and a blind listening test with musicians
- [ ] arXiv paper and a demo page with audio

## Credits and license

This project studies [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) and uses [SwiftF0](https://pypi.org/project/swift-f0/), [FCPE](https://pypi.org/project/torchfcpe/), [CREPE](https://github.com/maxrmorrison/torchcrepe), [Basic Pitch](https://github.com/spotify/basic-pitch), [Whisper](https://github.com/openai/whisper), [mir_eval](https://github.com/craffel/mir_eval), [FluidSynth](https://www.fluidsynth.org/) and the FluidR3_GM soundfont. Thanks to everyone who shares open models and tools.

To cite it:

```bibtex
@misc{dacapo2026,
  title  = {Da Capo: Measuring and Steering Musical Structure in AI Song Models},
  author = {ninondev},
  year   = {2026},
  url    = {https://github.com/ninondev/DaCapo}
}
```

MIT license, © 2026 ninondev ([LICENSE](LICENSE)).
