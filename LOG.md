# Log

Every run, in order: the question, how it ran, where the result file is, what it showed, and what went wrong. Coding agents ran the CPU tests and wrote this log. I ran the GPU tests in Colab. Result files are written by `dacapo.provenance.write_result` and keep their settings hash, environment and run time.

## 2026-10-26: CPU setup

The agents started on a 4-core x86 CPU (AVX-512 BF16, 15 GB RAM, no GPU) with Python 3.12 and CPU builds of PyTorch. Tools: FluidSynth with FluidR3_GM, SwiftF0 0.3.0, FCPE 0.0.4, torchcrepe, and Basic Pitch through its ONNX model.

No source-separation model was used, so absolute note-level scores only come from clean renders. Full-mix scores (E1) are always read next to the same score on the input mix.

E0 ran before the first commit, so its result files say `git_commit: unknown`. The other CPU files point to commits from the development history, which is not part of this repo, and the GPU run says `unknown`. The platform strings in the CPU files were shortened to OS, architecture and libc.

## E0: testing the measuring tools (CPU, done)

Question: how big is the measuring error on audio where every note is known, and can it catch k wrong notes?

`experiments/e0_ruler/run.py` ran with seeds 0 and 1: 9 melodies × 18 conditions (part A), and 6 melodies × 5 conditions × k = 0, 1, 2 or 4 wrong notes (part B). It took 668 s. Results: `results/cpu/e0_ruler.json` (and a short version, `results/cpu/e0_ruler_quick.json`).

On a solo line a single wrong note is easy to catch. In a full mix every score depends on the instrument, and with a voice-like melody over a band one or two wrong notes are hard to catch (best AUC 0.65 to 0.78, 6 melodies per condition). So absolute note-level scores need clean renders or separated stems. This evidence is moderate.

What went wrong: CREPE (full) ran at about 0.25× real time, so the agents stopped the first run and switched to SwiftF0 and FCPE. SwiftF0 makes octave errors on high flute notes (the integration test covers this). Chroma gets confused by trumpet overtones (0.52).

## E1: how much melody survives each path (CPU, first try)

Question: if the planner wrote the right 5 Hz codes, how much of the melody would the renderer give back?

Setup: `acestep-v15-turbo`, 8 steps, seed 1234, CPU bf16, `audio_cover_strength = 1.0`, no language model (codes taken from the reference audio), ACE-Step commit `ca1e85f`. 4 eight-bar melodies at 90 and 150 BPM, piano or flute over chords, bass and drums, through 5 paths. Result: `results/cpu/e1_melody_paths.json`.

Dense context and the VAE round trip matched the input (pitch-class accuracy about 0.91). The codes got 0.51 at 90 BPM and 0.34 at 150 BPM. With only 4 melodies per tempo this is weak evidence. This run also used the library default `shift = 1.0`, while the ACE-Step docs recommend 3.0 for turbo. The GPU run below replaces this one and tests both settings.

## E1b: what the plan tokens hold (CPU)

The agents made 24 four-bar melodies × 3 lead instruments, plus 3 changed versions of 6 melodies (`probe_stimuli.py`), then ran `ace_codes.py` and `probe.py`. Result: `results/cpu/e1b_probe.json`.

Encoding the same audio twice gives the same codes in 88% of frames, because the encoder samples. Changing only the lead instrument changes about as many codes as changing the melody. A linear probe reads pitch class much better from 5 Hz pooled latents than from the codes. The GPU run below uses more melodies and a different split.

## E2: hooks on the denoising steps

`dacapo/ace.py` wraps the DiT decoder. It can read or edit the model's guess of the clean latents at every step, swap the conditioning part way through, or call the denoiser more than once per step.

CPU smoke test: an edit that changes nothing gives exactly the same output (max difference 0.0); swapping the lyrics part way through changes the output; re-running the same conditioning in a combine step gives the same output.

What went wrong: the first version did the clean-latent math in bf16 and wasn't exact. It runs in float32 now.

## E4: singing a given melody, one-melody pilot (CPU)

One melody, so only a direction: `results/cpu/e4_pilot.json`. The expected trade-off between right notes and clear words showed up. The GPU sweep replaces it.

## E5: registering the motif study

Before any model run, the measuring check passed: the reference renders were all sorted into the right hypothesis, and one wrong note is caught at AUC 0.93 (`results/cpu/e5_ruler_gate.json`).

The study was registered at 2026-10-26T17:14:21Z in `experiments/e5_motif/registration/prereg.json`. The file also lists the one CPU trial run before registering (pilot melody mel00, left out of all confirmatory tests).

## Renaming the package

The package was called `ryuichi` and became `dacapo`. No logic changed and all tests passed. Addendum 1 (`registration/addendum_1.json`) records the new code hashes instead of editing the registration. It says the rename came before any confirmatory run, and that is wrong: the confirmatory run had already finished (see E5 v1 below). That run used the code from before the rename, whose hashes are the registered ones.

## 2026-10-26: GPU run 1 (NVIDIA L4)

I ran this one in Colab. Seven of eight stages finished in 144 minutes. The melody and lyrics swap stage crashed. Install and download time wasn't recorded. Result files are in `results/gpu_run1/`.

### E1 on GPU (done)

16 melodies × 80, 120 and 160 BPM × 5 paths, one seed, at shift 3.0 and 1.0. All 480 generations ran.

Pitch-class accuracy at shift 3.0, pooled, with 95% bootstrap CIs over melodies: input 0.897 [0.843, 0.945], dense context 0.910 [0.866, 0.951], VAE round trip 0.896, codes 0.294 [0.250, 0.336], codes with new instrumentation 0.308, no audio 0.103. Shift 1.0 moves no path by more than 0.02.

Each output is also scored against the other melodies at the same tempo. After taking out that chance level, the codes keep 28% [22%, 34%] of the melody information, and dense context keeps all of it. Note F1 through the codes is 0.12 (input 0.78).

There's no large tempo effect: raw codes accuracy at 160 vs 80 BPM differs by −0.0015 [−0.051, 0.044]. The chance-corrected differences lean negative, but this many melodies can't settle it.

The CPU value at 90 BPM (0.51) didn't come back. VAE precision (bf16 on GPU), random streams and the tempo grid all differ between the runs, and the cause isn't known yet. The CPU value at 150 BPM (0.335) sits at the upper edge of the GPU 95% CI at 160 BPM (0.282 [0.226, 0.335]).

The no-audio baseline depends only on the prompt, the length and the seed, so it rests on 6 different generations per setting. Its confidence interval means nothing.

### E1b on GPU (done)

48 melodies × 3 lead instruments; the probe learned on 42 melodies and was tested on 6 (720 frames, one split).

Pitch-class accuracy, codes vs pooled latents: all frames 0.575 vs 0.919, steady frames 0.645 vs 0.982, frames with a note change 0.412 vs 0.773, unseen lead instrument 0.504 vs 0.846. The majority baseline is 0.125.

The drop from the CPU value (0.71) comes with a new test split whose classes are more balanced. The token-change rates count whole clips, including frames after the melody ends, so they understate the change inside the melody for both changes. Encoding the same audio twice changes about 15% of tokens. The codes file wasn't saved in this run, so there are no confidence intervals yet.

### E3: reading pitch from the audio latents (exploratory)

Chromatic scales in 30 General MIDI instruments, with 6 instruments held out (2688 test frames).

32-way pitch accuracy on the held-out instruments: 0.882 with a 16-dimensional discriminant subspace, 0.880 with 31, and 0.879 with nearest class mean in the full 64-dimensional latent (chance 0.031). Pitch class: 0.920. So a special subspace adds nothing over the full latent. Choir "aahs" is the weakest instrument (0.583), probably because of its slow attack, not a real failure.

Editing the latents toward a target melody (8 melodies): replacing the 31 discriminant coordinates reaches 0.61 of the target frames, against 0.08 with no edit. A random-subspace control hasn't been run yet.

### E4: singing a given melody, sweep (exploratory)

12 melodies (3 for trying settings, 9 held out), one note per one-syllable word, 21 settings × 2 seeds, 504 generations.

Without a guide the model gets the lyrics but not the melody: 3% of notes at the written pitch, which is chance, and 66% of the words recognised. Settings that put 86% or more of the notes at pitch keep at most 3% of the words. Only 2 of 378 held-out clips get at least half of the notes and half of the words right.

The "oohs" guide released after one step raises note accuracy by 0.046, about half a note per line (the clarinet guide: +0.010). Neither survives correction for 20 comparisons.

Whisper makes up words on wordless output (106 of 504 transcripts have WER above 1), so word recall is more trustworthy here than WER. Basic Pitch often reports a fake low note near 49 Hz, so its polyphony count can't measure accompaniment. Starting from a noised guide is prior art (arXiv 2610.32211); it's used here as a baseline.

### E5 v1: first confirmatory run of the motif study (failed its quality check)

Outcome and correction: `experiments/e5_motif/registration/v1_outcome.json`.

Only 59 of 120 non-null gaps (49%) passed the registered quality check, and the rule required 75%. One of the two seeds gave quiet, mostly unvoiced gaps for every melody. By the registered rule this is a gap-filling failure, and it says nothing about motif development.

What the agents found: the repaint runs passed the text-to-music task instruction instead of the repaint one. Within a seed, every item started from the same noise, so the whole grid had only two independent noise draws, and the caption was chosen on the same draw. Also, the registered K1 rule named flute, synth-voice and clarinet captions, but Phase 0 compared three flute captions.

The registered scores are recorded anyway (CI_lit 0.159 [0.096, 0.232]); they aren't evidence of anything.

### Swap stage (crashed, fixed)

Melody swaps failed because two guides were 372 and 373 latent frames long, and the DiT's `torch.cat` refused them. `swap_conditioning` in `dacapo/ace.py` now crops or pads the swapped-in context by at most two frames and refuses anything bigger. `experiments/e2_hooks/check_swap_lengths.py` reproduces the crash and checks the fix. The stage has to run again.

### Smoke test, a cappella pilot, cost

GPU smoke test: identity 0.0, combine 0.0 (16 denoiser calls), and the swap changes the output. Its short clips were almost silent, so a loudness check comes next. CPU and GPU give different audio for the same seed.

A cappella pilot (3 captions × 3 seeds): 3 of 6 clips captioned "a cappella" are clean single-voice singing, and one seed gave no vocals under any caption. So scoring the pitch of raw outputs needs a check that there's a voice at all.

A warm generation takes about 2 to 3 s on the L4. Scoring took longer than generating: E4 scoring with Whisper, SwiftF0 and Basic Pitch took 68 minutes.

### To fix before the next run

Use the repaint instruction for gap filling, give each item its own seed, seed the global random stream before the encoder samples, set ACE-Step's optional wavelet step explicitly, and add a loudness check to the smoke test. The scorer should compute every registered E5 number. Each stage should save its codes, latents and audio, and each result file should record the code commit.

## 2026-10-27: new folder layout

The files moved into a simpler layout: the library is in `dacapo/` (the hooks moved from `experiments/e2_trajectory/ace_lab.py` to `dacapo/ace.py`), the experiment folders got plainer names (`e1_melody_paths`, `e2_hooks`, `e3_pitch_readout`, `e5_motif`), and all kept result files are in `results/cpu/` and `results/gpu_run1/` with shorter names. The E5 registration files moved to `experiments/e5_motif/registration/` without any change to their content. No numbers changed.

## 2026-10-27: E5 v2 and the E6 read-out, ready for the GPU

The agents put the fixed E5 run (run v2) and the E6 lyric read-out into one GPU run, Run A, which the notebook runs from start to end without anyone watching. Addendum 2 (`experiments/e5_motif/registration/addendum_2.json`) lists every rule, every change from the registration, the CPU runs made before it, and the hashes the code checks before it starts.

The confirmatory grid gets 36 new melodies (v2c00-v2c35, `manifest_v2.json`) that share no unit with v1. Phase 0b runs on the 14 v1 melodies (`manifest_p0b.json`). Both manifests record the sha256 of every rendered source, and `manifest_v2.json` also of every hypothesis render in the three K0 timbres.

The E6 read-out asks whether the seven attention heads that ACE-Step 1.5 uses for its lyric timestamps find the start of each sung word. It runs 88 a cappella clips and compares those heads with 200 random sets of heads, using Whisper large-v3 word times as the reference. The lyric clock (stage B1) is built only if the read-out passes the rule in the addendum.

The agents tested the whole chain on a CPU with `run_all.py --scale tiny --device cpu`. It runs every Run A stage once with one or two generations each, on pilot melodies mel00 and mel01, with seeds of their own tags, and keeps its run state in folders the full scale never reads. All six stages finished in about 25 minutes. Phase 0b triggered K1 and the E6 read-out blocked B1, as they should on a test this small. A second run from a clean state gave the same audio, byte for byte. Test-scale outputs aren't kept in the repo.

A correction to the folder-layout entry above: `v1_outcome.json` did change when it moved, but only in its two path references.

## 2026-10-27: review fixes before the GPU run

Three reviews of the Run A code found ways a runtime reset or a broken environment could end the registered study. These now stop the run without a strike. A crash, a refusal or five generation errors in a row stop before the quality stage, so they use no resume pass and freeze no missing draw as a gate failure. A generation cut off by a reset no longer counts as an attempt. A phase whose quality stage has frozen is never generated again: Run all reads its outcome back and goes on to scoring. A0 runs again when the addendum or the runtime changes, and setup stops when the measurement packages, the registered files in git or the free space on Drive are not as they should be.

The quality stage now checks integrity before it runs the trackers, so a void attempt has no quality decision until the terminal outcome exists, and attempt 2 needs an amendment that links the failed checks to every file the fix changed. The quality and score stages also check the hashes of the registered data (manifests, seed table, K0 files). H4 now also needs the registered comparison against null draws. E5 audio is written as 16-bit FLAC, so the format no longer depends on the torchaudio in the Colab image, and the run needs about 4 GB on Drive instead of 10. Confirmatory audio leaves the run only after the terminal outcome, and the blind-listening key is a file of its own.

## 2026-10-27: K0 failed, so the melodies were rebuilt

Before any model run, K0 ran on the first set of v2 melodies (generator seed 202) with the registered procedure: tau 0.7, ties counted as errors. Classification passed in all three timbres (0.976 to 0.983, threshold 0.95). The 1-wrong-note AUC failed in all three: flute 0.848, synth voice 0.861, clarinet 0.806, where 0.9 is required.

The cause was in the procedure, not in the tool. The registered draw picks the wrong note among all notes of B, but the tool only scores notes inside the gap minus 0.4 s at each edge, and B's first note starts at 12.0 s. So 2 or 3 notes of each B are never scored. When the wrong note lands on one of them, the wrong render scores exactly like the right one, and even a perfect tool would expect an AUC near 0.86. v1 passed (0.93) because only 2 of its 14 wrong notes landed there, where about 4 were expected.

After a failed K0, the registration says to redesign the units before any model run. The other option was to change the check so wrong notes come only from scored notes. I chose to rebuild the melodies, because that is what the registration says. In the new melodies the A, B, C and B2 units keep every note inside the scored window (generator seed 303). The ids and run seeds stay the same, since no model run had used them. The first melodies and their K0 files are kept unchanged in `experiments/e5_motif/k0_first_run/`. This choice and its exact procedure were pushed to GitHub (commit 1c91106) before K0 ran again, and the rule was that K0 runs once more and a second flute failure ends E5 v2.

K0 on the rebuilt melodies passed in all three timbres, and no wrong note landed outside the scored window. Flute: classification 0.983 (1663 of 1692 rows), 1-wrong-note AUC 0.965, 2-wrong 1.0. Synth voice: 0.987, 0.996 and 1.0. Clarinet: 0.985, 0.944 and 0.999. Every render matched its hash in the manifest. Results: `experiments/e5_motif/results/e5_ruler_gate_p73.json`, `_p54.json` and `_p71.json`. Run 2 can start.

What went wrong: the first start of the rerun was stopped by a tool time limit after about 10 minutes, before it wrote any result or printed anything. It was started again with the same code and files.

## 2026-10-27: E7 Stage S, registered

E7 asks whether a song model can sing a melody the user gives it, with clear words. Before anything is built, Stage S measures where things stand. It uses 12 new 8-bar melodies with lyrics. Suno sings them through up to three input routes (MIDI, a piano recording, a wordless "oohs" recording), and ACE-Step 1.5 sings them with four settings from E4. The same tools score everything. Because Suno songs have intros, repeats and small changes of tempo or key, the scorer searches for the start, tempo and key first, and every score is also computed against the 11 other melodies to remove lucky matches.

The rules, the hashes of the scoring code and the decision rule are in `experiments/e7_melody_adapter/prereg_suno.json`, registered before any song was generated. If Suno already sings the given melodies accurately on one route (median note accuracy at least 0.90 and median word recall at least 0.60), the plan changes. Otherwise the next step is a melody adapter trained inside the renderer. Suno's scores stay private until its terms of use are checked.

Two rounds of review before registration fixed problems that would have stopped the scoring halfway through the Suno songs, refused to load the model, or let a run with broken files report a result. Details are in `experiments/e7_melody_adapter/LOG.md`.
