# Log

Every run, in the order I did it: what I asked, how I ran it, where the result file is, what it showed, and what went wrong. Result files are written by `dacapo.provenance.write_result` and keep their settings hash, environment and run time.

## 2026-10-26: CPU setup

I started on a 4-core x86 CPU (AVX-512 BF16, 15 GB RAM, no GPU) with Python 3.12 and CPU builds of PyTorch. Tools: FluidSynth with FluidR3_GM, SwiftF0 0.3.0, FCPE 0.0.4, torchcrepe, and Basic Pitch through its ONNX model.

I didn't use a source-separation model, so absolute note-level scores only come from clean renders. Full-mix scores (E1) are always read next to the same score on the input mix.

E0 ran before my first commit, so its result files say `git_commit: unknown`. The other CPU files point to commits from my development history, which is not part of this repo, and the GPU run says `unknown`. I shortened the platform strings in the CPU files to OS, architecture and libc.

## E0: testing the measuring tools (CPU, done)

Question: how big is the measuring error on audio where I know every note, and can it catch k wrong notes?

I ran `experiments/e0_ruler/run.py` with seeds 0 and 1: 9 melodies × 18 conditions (part A), and 6 melodies × 5 conditions × k = 0, 1, 2 or 4 wrong notes (part B). It took 668 s. Results: `results/cpu/e0_ruler.json` (and a short version, `results/cpu/e0_ruler_quick.json`).

On a solo line a single wrong note is easy to catch. In a full mix every score depends on the instrument, and with a voice-like melody over a band one or two wrong notes are hard to catch (best AUC 0.65 to 0.78, 6 melodies per condition). So absolute note-level scores need clean renders or separated stems. I'd call this evidence moderate.

What went wrong: CREPE (full) ran at about 0.25× real time, so I stopped the first run and switched to SwiftF0 and FCPE. SwiftF0 makes octave errors on high flute notes (the integration test covers this). Chroma gets confused by trumpet overtones (0.52).

## E1: how much melody survives each path (CPU, first try)

Question: if the planner wrote the right 5 Hz codes, how much of the melody would the renderer give back?

Setup: `acestep-v15-turbo`, 8 steps, seed 1234, CPU bf16, `audio_cover_strength = 1.0`, no language model (codes taken from the reference audio), ACE-Step commit `ca1e85f`. 4 eight-bar melodies at 90 and 150 BPM, piano or flute over chords, bass and drums, through 5 paths. Result: `results/cpu/e1_melody_paths.json`.

Dense context and the VAE round trip matched the input (pitch-class accuracy about 0.91). The codes got 0.51 at 90 BPM and 0.34 at 150 BPM. With only 4 melodies per tempo this is weak evidence. I also used the library default `shift = 1.0`, while the ACE-Step docs recommend 3.0 for turbo. The GPU run below replaces this one and tests both settings.

## E1b: what the plan tokens hold (CPU)

I made 24 four-bar melodies × 3 lead instruments, plus 3 changed versions of 6 melodies (`probe_stimuli.py`), then ran `ace_codes.py` and `probe.py`. Result: `results/cpu/e1b_probe.json`.

Encoding the same audio twice gives the same codes in 88% of frames, because the encoder samples. Changing only the lead instrument changes about as many codes as changing the melody. A linear probe reads pitch class much better from 5 Hz pooled latents than from the codes. The GPU run below uses more melodies and a different split.

## E2: hooks on the denoising steps

`dacapo/ace.py` wraps the DiT decoder. It can read or edit the model's guess of the clean latents at every step, swap the conditioning part way through, or call the denoiser more than once per step.

CPU smoke test: an edit that changes nothing gives exactly the same output (max difference 0.0); swapping the lyrics part way through changes the output; re-running the same conditioning in a combine step gives the same output.

What went wrong: my first version did the clean-latent math in bf16 and wasn't exact. It runs in float32 now.

## E4: sing my melody, one-melody pilot (CPU)

One melody, so only a direction: `results/cpu/e4_pilot.json`. The expected trade-off between right notes and clear words showed up. The GPU sweep replaces it.

## E5: registering the motif study

Before any model run, the measuring check passed: my reference renders were all sorted into the right hypothesis, and one wrong note is caught at AUC 0.93 (`results/cpu/e5_ruler_gate.json`).

I registered the study at 2026-10-26T17:14:21Z in `experiments/e5_motif/registration/prereg.json`. The file also lists the one CPU trial I ran before registering (pilot melody mel00, left out of all confirmatory tests).

## Renaming the package

The package was called `ryuichi` and became `dacapo`. No logic changed and all tests passed. Addendum 1 (`registration/addendum_1.json`) records the new code hashes instead of editing the registration. It says the rename came before any confirmatory run, and that is wrong: the confirmatory run had already finished (see E5 v1 below). That run used the code from before the rename, whose hashes are the registered ones.

## 2026-10-26: GPU run 1 (NVIDIA L4)

Seven of eight stages finished in 144 minutes. The melody and lyrics swap stage crashed. I didn't record install and download time. Result files are in `results/gpu_run1/`.

### E1 on GPU (done)

16 melodies × 80, 120 and 160 BPM × 5 paths, one seed, at shift 3.0 and 1.0. All 480 generations ran.

Pitch-class accuracy at shift 3.0, pooled, with 95% bootstrap CIs over melodies: input 0.897 [0.843, 0.945], dense context 0.910 [0.866, 0.951], VAE round trip 0.896, codes 0.294 [0.250, 0.336], codes with new instrumentation 0.308, no audio 0.103. Shift 1.0 moves no path by more than 0.02.

I also score each output against the other melodies at the same tempo. After taking out that chance level, the codes keep 28% [22%, 34%] of the melody information, and dense context keeps all of it. Note F1 through the codes is 0.12 (input 0.78).

There's no large tempo effect: raw codes accuracy at 160 vs 80 BPM differs by −0.0015 [−0.051, 0.044]. The chance-corrected differences lean negative, but this many melodies can't settle it.

The CPU value at 90 BPM (0.51) didn't come back. VAE precision (bf16 on GPU), random streams and the tempo grid all differ between the runs, and I don't know the cause yet. The CPU value at 150 BPM (0.335) sits at the upper edge of the GPU 95% CI at 160 BPM (0.282 [0.226, 0.335]).

The no-audio baseline depends only on the prompt, the length and the seed, so it rests on 6 different generations per setting. Its confidence interval means nothing.

### E1b on GPU (done)

48 melodies × 3 lead instruments; the probe learned on 42 melodies and was tested on 6 (720 frames, one split).

Pitch-class accuracy, codes vs pooled latents: all frames 0.575 vs 0.919, steady frames 0.645 vs 0.982, frames with a note change 0.412 vs 0.773, unseen lead instrument 0.504 vs 0.846. The majority baseline is 0.125.

The drop from the CPU value (0.71) comes with a new test split whose classes are more balanced. The token-change rates count whole clips, including frames after the melody ends, so they understate the change inside the melody for both changes. Encoding the same audio twice changes about 15% of tokens. I didn't save the codes file in this run, so there are no confidence intervals yet.

### E3: reading pitch from the audio latents (exploratory)

Chromatic scales in 30 General MIDI instruments, with 6 instruments held out (2688 test frames).

32-way pitch accuracy on the held-out instruments: 0.882 with a 16-dimensional discriminant subspace, 0.880 with 31, and 0.879 with nearest class mean in the full 64-dimensional latent (chance 0.031). Pitch class: 0.920. So a special subspace adds nothing over the full latent. Choir "aahs" is the weakest instrument (0.583), probably because of its slow attack, not a real failure.

Editing the latents toward a target melody (8 melodies): replacing the 31 discriminant coordinates reaches 0.61 of the target frames, against 0.08 with no edit. I haven't run a random-subspace control yet.

### E4: sing my melody, sweep (exploratory)

12 melodies (3 for trying settings, 9 held out), one note per one-syllable word, 21 settings × 2 seeds, 504 generations.

Without a guide the model gets the lyrics but not the melody: 3% of notes at the written pitch, which is chance, and 66% of the words recognised. Settings that put 86% or more of the notes at pitch keep at most 3% of the words. Only 2 of 378 held-out clips get at least half of the notes and half of the words right.

The "oohs" guide released after one step raises note accuracy by 0.046, about half a note per line (the clarinet guide: +0.010). Neither survives correction for 20 comparisons.

Whisper makes up words on wordless output (106 of 504 transcripts have WER above 1), so I trust word recall more than WER. Basic Pitch often reports a fake low note near 49 Hz, so its polyphony count can't measure accompaniment. Starting from a noised guide is prior art (arXiv 2610.32211); I use it as a baseline.

### E5 v1: first confirmatory run of the motif study (failed its quality check)

Outcome and correction: `experiments/e5_motif/registration/v1_outcome.json`.

Only 59 of 120 non-null gaps (49%) passed the registered quality check, and I required 75%. One of the two seeds gave quiet, mostly unvoiced gaps for every melody. By my registered rule this is a gap-filling failure, and it says nothing about motif development.

What I found: the repaint runs passed the text-to-music task instruction instead of the repaint one. Within a seed, every item started from the same noise, so the whole grid had only two independent noise draws, and I chose the caption on the same draw. Also, the registered K1 rule named flute, synth-voice and clarinet captions, but my Phase 0 compared three flute captions.

I recorded the registered scores anyway (CI_lit 0.159 [0.096, 0.232]); they aren't evidence of anything.

### Swap stage (crashed, fixed)

Melody swaps failed because two guides were 372 and 373 latent frames long, and the DiT's `torch.cat` refused them. `swap_conditioning` in `dacapo/ace.py` now crops or pads the swapped-in context by at most two frames and refuses anything bigger. `experiments/e2_hooks/check_swap_lengths.py` reproduces the crash and checks the fix. The stage has to run again.

### Smoke test, a cappella pilot, cost

GPU smoke test: identity 0.0, combine 0.0 (16 denoiser calls), and the swap changes the output. Its short clips were almost silent, so I'll add a loudness check. CPU and GPU give different audio for the same seed.

A cappella pilot (3 captions × 3 seeds): 3 of 6 clips captioned "a cappella" are clean single-voice singing, and one seed gave no vocals under any caption. So scoring the pitch of raw outputs needs a check that there's a voice at all.

A warm generation takes about 2 to 3 s on the L4. Scoring took longer than generating: E4 scoring with Whisper, SwiftF0 and Basic Pitch took 68 minutes.

### To fix before the next run

Use the repaint instruction for gap filling, give each item its own seed, seed the global random stream before the encoder samples, set ACE-Step's optional wavelet step explicitly, and add a loudness check to the smoke test. The scorer should compute every registered E5 number. Each stage should save its codes, latents and audio, and each result file should record the code commit.

## 2026-10-27: new folder layout

I moved the files into a simpler layout: the library is in `dacapo/` (the hooks moved from `experiments/e2_trajectory/ace_lab.py` to `dacapo/ace.py`), the experiment folders got plainer names (`e1_melody_paths`, `e2_hooks`, `e3_pitch_readout`, `e5_motif`), and all kept result files are in `results/cpu/` and `results/gpu_run1/` with shorter names. The E5 registration files moved to `experiments/e5_motif/registration/` without any change to their content. No numbers changed.
