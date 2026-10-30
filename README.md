# Da Capo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninondev/DaCapo/blob/main/notebook.ipynb)

English | [中文](README.zh-CN.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/hero_dark.svg">
  <img alt="Diagram: in ACE-Step 1.5, the model's tokenizer turns a melody's audio into 5 Hz tokens (the planner language model is not run), and the renderer's output from those tokens has the right pitch class in 29% of frames. The same melody given to the renderer as dense 25 Hz audio latents for all 8 steps scores 91%, as high as the input itself (90%)." src="figures/hero_light.svg" width="100%">
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

1. The plan loses most of the melody. Through the 5 Hz plan tokens, only 29% of frames have the right note name, where random guessing gets about 10%. That's with tokens made from the melody's own audio, an easy case for any planner. The same melody given to the renderer as dense audio scores 91%, as high as the input itself (90%).
2. Much of the loss is in the tokenizer. A linear probe reads the note name from audio latents at the same 5 Hz rate with 0.919 accuracy. From the plan tokens it gets 0.575, which is still more than the 29% the renderer sings from them, so the renderer's use of the tokens loses some too.
3. In these tests the model sings the words or the melody, not both. With lyrics but no melody it sings 66% of the words. When it's guided with a wordless version of the melody, the notes become right (86% to 100%) and the words drop to 3% or less. A later test with a lyric clock did better by Whisper's count (see Results), but a second recogniser doesn't confirm the words.

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
  <img alt="Scatter plot of 21 settings (20 guides and no guide): notes at the written pitch in the whole output (no vocal gate) against lyric words recognised. Without a guide, 0.03 of notes and 0.66 of words. Settings that reach 0.86 or more of the notes keep at most 0.03 of the words. Only 2 of 378 clips get at least half of both right." src="figures/e4_tradeoff_light.svg" width="100%">
</picture>

The agents asked the model to sing one lyric line on a melody, one note per one-syllable word. Without a guide it doesn't know the melody, so its notes match only by chance (3%), but it sings 66% of the words. Then they guided it with the melody sung or played without words, either as context for the first k of the 8 steps or as a noisy starting point (SDEdit, a known method used here as a baseline). Every setting that got the notes right (86% or more) lost the words (3% or less). Only 2 of 378 test clips got at least half of both.

The working guess is that melody and lyrics have to be guided together, one syllable at a time. That's the idea behind the lyric clock, tested in Run B below.

<sub>12 melodies (3 for trying settings, 9 for testing) × 21 settings × 2 seeds. Words come from a Whisper large-v3 transcript. This test is exploratory: its settings weren't registered before the run. Code: [`experiments/e4_sing`](experiments/e4_sing/).</sub>

### Can it develop a motif?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e5_motif_dark.svg">
  <img alt="Dot plot with 95% CIs over 36 melodies, 0 is chance. How well the gap matches the answer the pattern asks for: after A B A, 0.47 (0.40 to 0.54), threshold 0.25; after A B and A moved up 2, up 5 or down 3 semitones, -0.02, 0.00 and 0.02, threshold 0.15; analogy (A, A+5, B), 0.01, threshold 0.15. How well it matches a plain copy of B: after A B C, 0.11 (0.06 to 0.15); in the three transposed patterns, 0.13, 0.10 and 0.10; after silence, -0.03. After the sequence A, A+2, A+4, the gap matched the next step A+6 in 0 of 144 draws, where 20% was needed." src="figures/e5_motif_light.svg" width="100%">
</picture>

This is the main study (E5). The test plays a pattern of three short melodies, 4 seconds each, leaves a gap from 12 to 16 s, and lets the model fill it (repaint). After A B A, the pattern asks for B. Each gap is scored by how well its notes match the answer, minus how well they match a shuffled answer with the same rhythm and the same notes, so 0 is chance. The hypotheses and thresholds are in the [pre-registration](experiments/e5_motif/registration/prereg.json). The first run failed its own quality check ([`v1_outcome.json`](experiments/e5_motif/registration/v1_outcome.json)), so this is run 2, registered in [addendum 2](experiments/e5_motif/registration/addendum_2.json) before it started. Before run 2, a check of the measuring tool failed once. I chose to rebuild the melodies, as the registration says, instead of changing the check ([`LOG.md`](LOG.md) has the details).

- Repeat (H1, supported): after A B A, the gap's score for B was 0.47 (95% CI 0.40 to 0.54). The registered bar was 0.25.
- Repeat because of the pattern (H2, supported): after A B C, nothing asks for B, and the score for B was only 0.11 (0.06 to 0.15). The difference, 0.37 (0.31 to 0.42), clears the registered 0.15. It also holds in the gaps that passed the quality check (0.40, 0.33 to 0.48). So the model repeats B because the pattern calls for it, not only because B was played 8 seconds earlier.
- Transposition (H3, not supported): after A B and then A moved up 2 semitones, the answer is B moved up 2. The gap didn't match it: -0.02, 0.00 and 0.02 for +2, +5 and -3 semitones, where 0.15 was needed. It matched plain, unmoved B a little more (0.13, 0.10 and 0.10).
- Sequence (H4, not supported): after A, A+2, A+4, the gap never matched the next step A+6 (0 of 144 draws, 20% needed).
- Analogy (H5, not supported): after A, A+5, B, the gap didn't match B+5: 0.01 (-0.02 to 0.03), where 0.15 was needed.

So the model can continue a pattern it heard by repeating a part of it unchanged, and the pattern decides what it repeats. Transposition, sequence and analogy all stayed far below their registered thresholds. The largest of those effects, the -3 transposition compared with its A B C baseline, was 0.03 (0.005 to 0.05), where 0.15 was needed. The melodies are instrumental flute lines with no singing, filled by repaint, in one model. The scores come from a pitch tracker. People will listen to the clips blind before I claim anything about how they sound.

<sub>36 new melodies × 8 conditions × 4 draws = 1152 gaps, NVIDIA L4, commit 83d46e7. Phase 0b picked the setting first (repaint, flute, 88 of 112 gaps passed, 84 needed). 804 of the 1008 gaps outside the silence condition passed the quality check (756 needed). Verdicts use all draws, as registered, and the CIs come from resampling melodies. Code: [`experiments/e5_motif`](experiments/e5_motif/).</sub>

### Do the lyric heads follow the words?

ACE-Step 1.5 uses seven attention heads to find when each lyric word is sung. The read-out (E6) checks whether they really follow the words before the lyric clock is built on them. Of 88 a cappella clips, 76 had singing (44 needed) and 50 could be scored, with at least 4 words matched to Whisper (20 needed). In a typical clip, the heads put each word's start 0.21 s from Whisper's time (0.25 s allowed). Random sets of 7 heads were off by a median of 1.95 s, and the real heads beat the typical random set in 49 of 50 clips (98%, where 75% was needed). So the registered rule allows the lyric clock (B1). Two cautions: Whisper and a second aligner (MMS) disagree with each other by a median of 0.39 s on these clips, so 0.21 s is within the noise of the reference. And six of the seven heads are the six best single heads, but the seventh (layer 3, head 11) is off by 2.41 s on its own.

This shows the heads follow the timing of words in finished audio. It doesn't show they can control it. B1 tested that under its own rule, and at the registered strength the clock didn't move the timing (see below). ACE-Step may have been trained to line these heads up with the words, so beating random heads is a sanity check, not a finding.

<sub>88 clips: 12 lyric lines from E4 × 4 seeds and 4 verses × 2 voices × 5 seeds, 8-step turbo model, NVIDIA L4. Whisper large-v3 word times are the reference. The read-out is exploratory; its rule decides only whether B1 runs. Code: [`experiments/e6_clock`](experiments/e6_clock/).</sub>

### Can the lyric clock make it sing the words on the notes?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/b2_clock_dark.svg">
  <img alt="Dot plot, share of 24 new melodies where the kept draw (best of 4) gets at least half the notes and at least half the words right. Words recognised by Whisper: clock on the 7 lyric heads 14 of 24, past the capability bar of 0.50; no clock with the same guide 8; clock on 7 random heads 8; clock with the words on the wrong notes 7; no clock with a stronger guide (first 4 steps) 3. Words recognised by a second recogniser (CTC) on the same draws: 2, 3, 1, 2 and 1." src="figures/b2_clock_light.svg" width="100%">
</picture>

The lyric clock adds a fixed amount (beta) to the attention scores, before the softmax, that the seven lyric heads give each word while that word's note plays. At beta 8, a word that had a tenth of a head's attention gets almost all of it. Run B tested the clock twice, each time under a rule registered before the run.

On its own (B1), the clock didn't fix the timing. Without a melody guide the model sings these lines early: in a typical clip the words start 3.6 s before their notes. At the registered strength, beta 2, the timing error fell by 30% or more in 2 of the 12 lines, where 9 were needed. Word recall and singing held, and random heads did nothing. A stronger clock (beta 4) cut the error by 30% or more in 6 lines, at some cost in word recall (0.59 to 0.52), and the words still started a median 2.9 s early. That wasn't the registered test.

With a melody guide (B2), registered code first picked one clocked recipe out of 20, from runs on the 12 E4 lines: a wordless Voice Oohs version of the melody as context for the first 3 of the 8 steps, and the clock at beta 8. That recipe then ran on 24 new lines and melodies, 4 draws each, and the draw that scored best on notes and words together was kept. On 14 of the 24 melodies the kept draw had at least half the notes right and Whisper recognised at least half the words, in order (12 were needed). With the same guide and no clock it was 8. The clock on random heads got 8, and the clock with each word's window moved to the wrong note got 7. By the mean score of the kept draw (0.52 with the clock on the lyric heads, 0.35 on random heads, 0.39 with the windows moved, 0.39 with no clock), the registered labels call the clock's effect head-specific and timing-specific. Labels don't change the verdict.

What limits this result:
- The rule credits the clock only if the share of melodies that pass beats the best recipe without the clock by at least 0.20 (5 of 24 melodies), and the lower end of the gap's 95% bootstrap interval is above 0. The gap was 0.25 (6 melodies), but the lower end was exactly 0. The registered verdict is "Capability reached, clock not credited". Outside the rule, the mean score of the kept draw is higher with the clock: +0.13 (95% CI 0.06 to 0.20). That score also counts Whisper's words and uses the same best-of-4 pick.
- A second speech recogniser (a wav2vec2 CTC model), on the same kept draws, passes 2 of 24 with the clock and 3 with the guide alone. On the kept draws it rates the clock a little below the guide alone (word recall -0.08); over all 96 clips of each arm the two are close (0.18 and 0.16). It reads every arm poorly (0.05 to 0.18), so it doesn't settle the question, but it doesn't confirm Whisper either. The words result rests on Whisper.
- The kept draw is the best of 4 by the same score that decides whether it passes. That is the registered pipeline. With a single draw the clocked recipe passes on 32% of melodies (14% without the clock).
- The margin is small. Two of the 14 melodies pass with no note to spare: one note fewer on every clip would leave exactly 12, and two fewer would leave 10. At the strict bar (three quarters of both), the clock and the guide alone each reach 2 of 24.
- A timing check before the run overestimated its length about sixfold, because Whisper looped on the check's clips. So the registered trim dropped the clarinet guide at k = 3 from the recipe search (25 of 30 recipes ran) and the no-guide arm from the test, although the full design would have fit. The no-guide arm was a reference, not a comparator, so no verdict depends on it.

People will hear the 24 pairs blind (each melody's kept clip with the clock and without it), asked whether the melody is right and the words are clear, before I claim anything about the words. That test is exploratory under addendum 3 and won't change the registered verdict.

<sub>B1: 12 lines × 5 arms × 3 draws. B2: Stage D 25 recipes × 12 lines × 3 draws; Stage T 5 arms × 24 new lines × 4 draws, 480 clips, none lost. Notes are scored on the separated vocal with a pitch tracker (chance about 0.10). Words are matched, in order, against a Whisper large-v3 transcript, and Whisper isn't given the lyric. NVIDIA L4, commit bd61a9f. Rules: B1 in [addendum 2](experiments/e5_motif/registration/addendum_2.json) with the [B1 code addendum](experiments/e5_motif/registration/addendum_2_b1code.json); B2 in [addendum 3](experiments/e5_motif/registration/addendum_3.json). Code: [`experiments/e6_clock`](experiments/e6_clock/), [`experiments/e8_clock_sing`](experiments/e8_clock_sing/).</sub>

### Measuring is hard

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e0_ruler_dark.svg">
  <img alt="Dot plot of pitch-tracker accuracy on a perfectly played melody, alone versus with a loud band: piano lead 0.95 vs 0.09, synth lead 0.95 vs 0.55, voice-like lead 0.93 vs 0.59." src="figures/e0_ruler_light.svg" width="100%">
</picture>

Before testing the model, the agents tested the measuring tools on audio where every note is known. Pitch trackers are very good on a solo melody and bad once a loud band plays. With a voice-like melody in a full mix, one or two wrong notes are hard to catch (best AUC 0.65 to 0.78). So with a band, every score is compared with the same score on the original audio, and singing is only scored in songs where the prompt asked for no instruments.

<sub>9 melodies per condition in the chart; the wrong-note tests use 6 melodies per condition. Code: [`experiments/e0_ruler`](experiments/e0_ruler/).</sub>

## Working on now

The hooks live in [`dacapo/ace.py`](dacapo/ace.py). At each of the 8 denoising steps they show what the model expects the final audio to be, and an experiment can read it, change it, or swap the text, lyrics or audio part way through. If a hook changes nothing, the output is exactly the same as without it (difference 0.0 on CPU and GPU), so any change in the output comes from the edit itself.

Run A finished on 2026-10-28, and its results are above. The motif study (E5) found literal repetition of a pattern, and no transposition, sequence or analogy at the registered sizes. It goes on to its next phases, which ask which layers of the model make the copy and whether pointing the model at B can make it copy on purpose. Their design will be written down in a new addendum before they run, as the registration says.

Run B finished on 2026-10-30, and its results are above. The lyric clock alone didn't fix the timing (B1). With a melody guide, the chosen recipe reached the registered bar by Whisper's count (B2), but the clock wasn't credited and a second recogniser didn't confirm the words.

Run C, a registered replication of B2 on 96 new melodies, is built and waits for a GPU session ([addendum 4](experiments/e5_motif/registration/addendum_4.json), [`notebook_runC.ipynb`](notebook_runC.ipynb)). With four times as many melodies, it should settle whether the clock helps.

The blind listening tests come from the runs: 60 motif clips from Run A and 24 singing pairs from Run B, which people will hear without knowing which condition each one comes from. I asked for this. Until they're done, the motif result rests on pitch-tracker scores and the singing result on the pitch tracker and Whisper.

## Status

| | Question | Status |
|---|---|---|
| E0 | How good are the measuring tools? | Done |
| E1 | How much melody survives each path through the model? | Done on GPU: 16 melodies × 3 tempos, two shift settings |
| E1b | What is in the plan tokens, and where is the loss? | Done on GPU: 48 melodies, one test split |
| E2 | Can the code hook every denoising step without changing the output? | Done: exactly the same output on CPU and GPU |
| E3 | Can the pitch be read from the audio latents for any instrument? | Exploratory: 0.88 on new instruments (chance 0.03); a special subspace is no better than the full latent |
| E4 | Can it sing the right notes and clear words? | Exploratory: 12 melodies × 21 settings × 2 seeds |
| E5 | Can it develop a motif? | Run 2 done on GPU: it repeats part of a pattern unchanged (H1, H2 supported), but shows no transposition, sequence or analogy at the registered thresholds (H3 to H5 not supported). Run 1 failed its quality check |
| E6 | Can a lyric clock time each syllable? | Read-out done on GPU: the lyric heads put word starts a median 0.21 s from Whisper's. The clock alone (B1) is not supported: it cut the timing error enough in 2 of 12 lines, where 9 were needed |
| E7 | How well do Suno and ACE-Step sing a melody you give them? | Registered; waiting for the Suno songs |
| E8 | With a melody guide, can the lyric clock sing new melodies with the right notes and recognisable words? (stage B2) | Done on GPU: 14 of 24 new melodies pass by Whisper (12 needed), so the capability bar is reached, but the clock isn't credited (the gap's 95% interval starts at 0) and by a second recogniser only 2 pass. Blind test next |

[`LOG.md`](LOG.md) has every run, including the failures and the mistakes.

## How the tests are run

The measuring tools are tested on known notes before the model is. For the main studies, the hypotheses and stop rules are written down before the run (I asked for this), and when something changes, a new file is added instead of editing the old one. Bad results get published too. Every method tested will get the same number of tries, with "generate many times and keep the best" as a baseline. No language model judges the music; final claims will come from people listening blind (I asked for this too). Each number in this repo can be traced to a result file, and the audio can be made again from code with fixed seeds (the same device gives the same result, but CPU and GPU can differ).

## Run it

The easiest way is the Colab button at the top. Pick an L4 runtime, choose Run all, and allow Google Drive access once. The notebook installs everything, downloads the open ACE-Step weights (about 10 GB) and runs Run A: the fixed motif study and the lyric read-out. That takes about 4 hours on an L4, install and download included. At the end you get a zip with the results, and a separate file with the answers for the blind listening test. The run state is kept on Drive (about 5 GB; E5 wants 7.5 GB free before it starts), so after a runtime reset, Run all continues where it stopped. The GPU tests from run 1 can still run with `--stages`. Run B, the lyric clock, has its own notebook, [`notebook_runB.ipynb`](notebook_runB.ipynb); its stages took about 5 hours on an L4. To try the B2 recipe on a melody of your own, [`notebook_sing.ipynb`](notebook_sing.ipynb) sings one word per note with the lyric clock and keeps the best of a few takes; it is a tool, not a study, and it needs an L4 or A100 ([`experiments/sing`](experiments/sing/)).

To run the measuring tools on your own computer (macOS or Linux):

```bash
git clone https://github.com/ninondev/DaCapo && cd DaCapo
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,transcription]"
uv pip install --no-deps basic-pitch        # ONNX model, so no TensorFlow
# FluidSynth + a General MIDI soundfont:
#   Ubuntu: sudo apt install fluidsynth fluid-soundfont-gm
#   macOS:  brew install fluid-synth, then export DACAPO_SF2=/path/to/FluidR3_GM.sf2
pytest -q -m "not slow"                      # unit tests; plain pytest -q also runs the end-to-end tests
python experiments/e0_ruler/run.py --quick   # test the measuring tools (~3 min on CPU)
python figures/make_figures.py               # redraw all charts from results/
```

Tests that run the song model need [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) installed as well. [`experiments/run_all.py`](experiments/run_all.py) lists the exact command for each step.

<details>
<summary>Files</summary>

```
dacapo/          the library: melodies, rendering, pitch tracking, scores, and ace.py (hooks into ACE-Step)
experiments/     one folder per study (e0 to e9), plus run_all.py, run_b.py and run_c.py, which the notebooks call
results/cpu/     result files from CPU runs
results/gpu_run1/ result files from the first GPU run (NVIDIA L4)
results/gpu_run2/ result files from Run A: the motif study and the lyric read-out (NVIDIA L4)
results/gpu_run3/ result files from Run B: the lyric clock alone (B1) and with a melody guide (B2) (NVIDIA L4)
figures/         charts and the script that draws them
tests/           unit tests and a render-to-analysis test
LOG.md           every run, in order
notebook.ipynb   the Colab notebook for Run A
notebook_runB.ipynb the Colab notebook for Run B
notebook_sing.ipynb a tool: sing your own melody with the B2 recipe
notebook_runC.ipynb the Colab notebook for Run C
```

</details>

## Plan

- [x] Test the measuring tools on known notes
- [x] Find where the melody gets lost inside an open song model
- [x] Hooks on every denoising step, checked to be exact on CPU and GPU
- [x] Map the trade-off between right notes and clear words (exploratory)
- [x] Motif study: register the fixed run before it starts
- [x] Run the fixed motif study and the lyric read-out on the GPU (Run A)
- [ ] Blind listening test of the motif clips from Run A
- [ ] Motif study, next phases: which layers make the copy (new addendum first)
- [x] Lyric clock alone (B1) and with a melody guide (B2) on the GPU (Run B)
- [ ] Blind listening test of the 24 singing pairs from Run B
- [ ] Run C: B2's recipes on 96 new melodies (registered, waiting for a GPU session)
- [ ] Measure Suno and ACE-Step on the same 12 melodies (E7, Stage S)
- [ ] Train a melody adapter inside the renderer (E7), unless Stage S shows it is already solved
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
