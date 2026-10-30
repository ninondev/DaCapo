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

## 2026-10-28: independent review before the GPU run

Before the GPU run, new review agents that had not seen the design work read the registered E5 v2 and E6 code against the registration, ran its tests, and ran small CPU generations where reading the code was not enough. They found no problem that would make the run invalid. The instruction that reaches the model, the repaint window, the noise per melody and draw, the 416 seeds, the gates and the statistics all match the registration, and the quality gate reproduces the 59 passing gaps of v1.

One confirmed problem was in the notebook, which is not a registered file. The commit pin it keeps on Drive was shared by every scale, so a pin left by an earlier test could have made the full run use old code. The pin is now kept per scale, and a pin from a run that never started must match the latest code or the notebook stops. The smaller findings are noted here and handled after the run:
- The per-job check doesn't re-test the decode settings (chunk 256, no CPU fallback, no out-of-memory). Each job records them, so they will be checked in the ledger after the run.
- The labelled listening examples in the package overlap the blind listening pack. Nobody should hear the labelled examples before doing the blind test.
- K0 rounds its AUC to 4 decimals before the 0.9 test. The lowest registered AUC is 0.944, so no result changes.
- The H3 summary leaves out one registered label ("below threshold, CI excludes 0"). It will be added in the write-up.

## 2026-10-28: Run A on the GPU (E5 v2 and the E6 read-out)

I ran Run A in Colab on an NVIDIA L4 with commit `83d46e7`, the code registered in addendum 2. Run all started at 09:43:35Z and finished at about 12:37Z, 173 minutes in all: setup 15.4 min, A0 1.6 min, Phase 0b 71.1 min, the confirmatory grid with quality and scoring 65.1 min (scoring alone took 177 s), the E6 read-out 19.3 min and packaging 0.8 min. Install and download time before that isn't recorded. The registered outputs are in `results/gpu_run2/` (its README lists them). The ledgers, logs, design and provenance files, the grid's quality file, the audio and the blind-listening pack are kept outside the repo, and the listening key is a separate file.

### Setup and A0

Setup found the registration intact (`registration_ok`, no uncommitted files) and rendered every source with the hash in its manifest. The scorer reproduced every stored v1 number. K0 on the Colab renders passed in all three timbres: flute classification 0.9829 and 1-wrong-note AUC 0.9653, synth voice 0.987 and 0.9961, clarinet 0.9852 and 0.9444, with no render mismatch. All 24 hard A0 checks passed, including the decode chunk of 256 with no CPU or out-of-memory decode. A repeat and a run in a fresh process gave bit-identical latents and audio (max difference 0.0).

### Phase 0b (K1 and the cell choice)

1344 repaints on the pilot melodies mel00-mel13, 112 gaps per cell. Three cells passed K1 (84 of 112 needed): repaint with the source as rendered, for flute (88), clarinet (88) and synth voice (84). The other nine cells passed 10 to 65 gaps. Repaint with the source cut at 18 s passed 39 to 43, text2music as rendered 60 to 65, and text2music cut at 18 s 10 to 14. Flute and clarinet tied at 88, and the registered tie order (flute before synth voice before clarinet) chose `rep-rend-flute`. The whole grid ran in that cell.

### E5 v2: the confirmatory grid

1152 repaints: 36 new melodies (v2c00-v2c35) × 8 conditions × 4 draws. None was missing or lost.

Quality passed. Q1: 804 of 1008 non-null gaps passed the gate (0.798; 756 needed). Q2: 246 of 252 cells had at least 2 passing draws (0.976; 189 needed). Pass rates per condition: lit 0.875, ctl 0.778, rel+2 0.771, rel+5 0.778, rel-3 0.771, seq 0.8125, ana 0.799. Lit passed more often than ctl (exact McNemar 17 against 3, p = 0.0026), but the gap between the rates (0.097) is under the 0.10 that would add the "differential gating" label.

Paper level: `supported:H1,H2`. All numbers are the registered estimator on ALL draws (36 melodies), with 95% bootstrap CIs over melodies; PASS is the gate-passing draws.
- H1, supported. CI_lit 0.471 [0.405, 0.536], threshold 0.25. PASS 0.498 [0.426, 0.570] (label: PASS meets the threshold).
- H2, supported. CI_lit - CI_ctl 0.365 [0.306, 0.425] in ALL and 0.403 [0.327, 0.479] in PASS, threshold 0.15, 36 common melodies each. CI_ctl alone is 0.106 [0.063, 0.152].
- H3, not supported for any k. RI+2 -0.021 [-0.043, 0.001], RI+5 0.002 [-0.024, 0.028], RI-3 0.020 [-0.002, 0.041], threshold 0.15. RI_k - RI_k(X-ctl): -0.011 [-0.034, 0.014], 0.017 [-0.009, 0.042], 0.028 [0.005, 0.050]. RI_k - LI_k is negative for every k: -0.152 [-0.221, -0.089], -0.096 [-0.158, -0.040], -0.082 [-0.136, -0.030]. LI_k is 0.131 [0.076, 0.193], 0.098 [0.051, 0.151] and 0.103 [0.057, 0.150], so the gap leans toward the unmoved B. Every LI_k is under 0.15, so the registered "literal-lag copier" label doesn't apply to any k. PASS: RI -0.012, 0.012 and 0.025; RI_k - RI_k(X-ctl) 0.000 [-0.028, 0.028], 0.019 [-0.013, 0.051] and 0.029 [0.003, 0.054].
- H4, not supported. Rates in seq: T+6A 0.000, T+5A 0.000, T+7A 0.000, T+4A (copy-left) 0.035, T+2A (lag) 0.132, D 0.042, none 0.792, tie 0.000. T+6A came up in 0 of 144 draws, and 0 of 144 under both NULL and X-ctl, so both tests give p = 1 (sign test with no untied melodies; binomial with p0 = 0.5/144). PASS: 0 of 117.
- H5, not supported. ANA 0.008 [-0.018, 0.034], threshold 0.15. PASS 0.007 [-0.025, 0.038] (label: PASS does not meet the threshold).
- K2 not triggered: on v2c00-v2c05, CI_lit 0.520 and CI_ctl 0.077, and it needs both under 0.10.
- K3 not triggered: the H2 difference is far above 0.05 in ALL and in PASS.

The label "below threshold, CI excludes 0" is registered for a negative benchmark, which this run isn't. The earlier entry said it would be added in the write-up, so for the record: it fits only the H3 clause RI-3 - RI-3(X-ctl), 0.028 [0.005, 0.050] in ALL and 0.029 [0.003, 0.054] in PASS. CI_null is -0.025 [-0.044, -0.007].

Exploratory, from the per-draw classification at tau 0.7: 74 of 144 lit gaps were classified as B, against 12 of 144 ctl gaps.

What it means: the model continues A B A with B, and that depends on the pattern, not only on the 8 s lag. It doesn't carry a transposition over to B, extend a sequence, or complete an analogy. These are instrumental flute lines filled by repaint in one model, scored by a pitch tracker, and the blind listening test hasn't been done yet.

### Checks after the run

Agents rechecked the run from the files. `e5_terminal.json` holds the sha256 of `e5v2_main.json`. All 18 code hashes and 14 data hashes in the result files equal addendum 2, and the repo at `83d46e7` matches them. Both design files name the sha256 of `addendum_2.json`, and the first generation (10:00:53Z) came after `addendum_utc` (2026-10-27T18:08:52Z). The ledgers have one pass and one attempt each: Phase 0b 1344 started and 1344 done, the grid 1152 and 1152, with no missing, extra, duplicate or lost job. All 2496 seed pairs match the committed table and its sha256 derivation, Phase 0b and the grid share none, and no reserved seed was used. Every quality-file wav hash equals the ledger's.

A separate recompute, both with the repo's `motif_stats.summarize` and with an independent numpy version, reproduced every stored number and verdict, to within 1.7e-16. It used numpy 2.5.4 locally where Colab used 2.1.3.

### E6: the lyric read-out (B1 allowed)

88 clips planned, 88 generated and scored, 0 failed. Every condition of the registered rule passed:
- Vocal gate: 76 of 88 clips (0.864; 44 needed).
- Evaluable clips: 50 (20 needed): 37 from the E4 lines and 13 from the verses. Another 26 vocal clips had fewer than 4 words matched to Whisper.
- Median over clips of the per-clip median onset error against Whisper: 0.21 s (0.25 s allowed).
- The alignment heads beat the median of the 200 random sets in 49 of 50 clips (0.98; 75% needed). The random sets' median error was 1.95 s, and none of the 200 did as well as the alignment set.
- ACE-Step's own `get_lyric_timestamp` worked on every clip, and both aligners reproduced it.

Descriptive numbers, outside the rule: 95% cluster-bootstrap intervals of 0.795 to 0.932 for the vocal rate, 0.18 to 0.22 s for the error and 0.944 to 1.0 for the beat rate. The heads' word starts lag Whisper's by a median of 0.20 s (signed), and 65% of matched words are within 0.25 s. Re-noising seeds 43 and 44 give 0.20 and 0.21 s. MMS forced alignment differs from Whisper by a median of 0.39 s on these clips, and from the heads by 0.21 s. Six of the seven alignment heads are the six best single heads (0.10 to 0.22 s); the seventh, layer 3 head 11, is at 2.41 s. The E4 lines give 0.20 s and the verses 0.24 s. So B1, the lyric clock, can run in Run B. The read-out shows the heads follow word timing in finished audio, not that they control it.

### What went wrong

Nothing stopped the run, but four small things are worth recording.
- The run state was on the Colab disk, not on Drive (`durable: false` in the status file). Addendum 2 asks that every start be reported, and this was the only one: one pass, attempt 1, no interruption.
- The per-run check doesn't test the decode settings, as the review before the run noted. The ledgers show all 2496 runs with decode chunk 256, no CPU decode and no out-of-memory line, and the logs agree.
- Null gaps weren't silent this time. Their level had a median of -21.7 dBFS (-46.4 to -12.8), and 21 of 144 were classified as D, the bar after the gap. In v1, 0 of 12 were voiced. Addendum 2 expected silent null gaps, which would make the NULL comparison in H4 easy to pass. Here it made no difference, because T+6A was 0 in every condition it was compared with.
- Some result files have no `git_commit` field. The hashes above tie them to the run, and every file that has the field names `83d46e7`.

The labelled listening examples in the package still overlap the blind pack (60 clips). Nobody should hear them before doing the blind test.

## 2026-10-28 and 2026-10-29: Run B registered

Run B has two parts. B1 is the lyric clock on its own, with no melody guide, under the rule addendum 2 fixed before Run A. The clock adds a fixed bias (beta) to the attention scores, before the softmax, of the seven alignment heads in layers 2 to 6, so each lyric word's tokens get more weight in the frames of that word's note. The code for B1 was registered in the B1 code addendum (`addendum_2_b1code.json`, 2026-10-28T20:56:49Z) after a CPU test showed that the hook at beta 0 leaves the output unchanged. B2 is the clock together with a melody guide. It was registered in addendum 3 (`addendum_3.json`, 2026-10-29T04:50:31Z) before any full-scale B2 generation. Before that, the CPU code tests had run the tiny scale, including two runs on the test item b2t00 in early passes, with their own seeds and Whisper small.en. Nobody listened to them, and addendum 3 lists them.

B2 asks whether one fixed recipe can sing new melodies with the right notes and recognisable words. A clip is joint when its vocal-gate result times the smaller of NPA_v (notes right on the separated vocal) and WR (lyric words Whisper recognises, in order) is at least 0.5. Stage D picks one clocked recipe (S*) and one unclocked recipe (B*) on the 12 E4 lines. Stage T runs them, with controls, on 24 new lines and melodies, 4 draws each, and keeps the draw with the highest score. J is the share of the 24 melodies whose kept draw is joint. The clock gets credit (P1) if J(S*) beats the best unclocked recipe by at least 0.20 and the bootstrap's 2.5th percentile of that gap is above 0. The bootstrap resamples the melodies. The capability (P2) needs J(S*) of at least 0.50.

I approved one Colab L4 session and, before registration, accepted an estimate of 10 to 13 hours.

## 2026-10-29 and 2026-10-30: Run B on the GPU (B1 and B2)

I ran Run B in Colab on an NVIDIA L4 with commit `bd61a9f`, the code registered in the B1 code addendum and addendum 3. It started at 21:01:47Z and decide finished at 01:44:50Z, 283 minutes of stages in all: setup 0.9 min, A0 4.5, B1 26.0, Stage D 154.0, selection 0.2, Stage T 96.9 and decide 0.6. The run state was on Google Drive, and every stage ran once with no resume. The registered outputs are in `results/gpu_run3/` (its README lists them). The ledgers, logs, audio and blind pack are kept outside the repo, and the blind key is a separate file.

### Setup and A0

Setup found all three registrations current, rendered the 72 guides again with their registered hashes, and found the seed table and soundfont as registered. A0 passed all 15 hard checks. With the hook attached at beta 0, the output was bit-identical to the run without the hook, both for text2music and for the Voice Oohs cover at k = 4, so the registered fallback was not needed. Every hooked run made 40 hook calls.

A0 also sets the registered timing trim. It projected 32.6 hours for the full design, so it dropped the five clarinet configs at k = 3 from Stage D (29.4 h) and then the no-guide arm from Stage T (29.1 h). The trim was fixed 26 minutes before Stage D began. The projection was far too high: the whole run took 4.7 hours. The agents traced it to A0's scoring times. Three of A0's five guided clips were the Voice Oohs cover at k = 4 (no hook; hook at beta 0, an identical WAV; and beta 2, a different WAV), and Whisper looped on all three ("oh, oh, ...", 222 chunks each), so the median guided scoring time was 49.9 s instead of about 5 s. The registered allowance for Whisper loops was then added on top. At the run's measured costs (8.7 s per text2music run; 10.3 s and 12.1 s per guided run in Stage D and Stage T), A0's formula would have projected 6.2 to 6.6 hours for the full design, under the 7.0-hour limit, so no trim would have applied. The trim followed the registered rule, but it means clarinet at k = 3 was never tested and Stage T has no no-guide reference arm. The no-guide arm was a reference for words and the vocal gate, not a comparator, so no verdict depends on it. Whether clarinet at k = 3 would have changed S* or B* can't be known; the clarinet recipes that ran were joint on at most 3 of the 12 development lines, against S*'s 9.

### B1: the clock alone (not supported)

180 runs: 12 E4 lines × 5 arms (no hook; beta 1, 2 and 4 on the alignment heads; beta 2 on random heads) × 3 draws. All were generated and scored. Without a guide the model sings these lines early: in a typical clip the words start 3.6 s before their written notes.

The registered rule, at beta 2:
1. At least a 30% cut in timing error in 9 of the 12 lines. Not met: 2 lines (0.41 and 0.77), with a median cut of 0.047. The closest miss was 0.28.
2. Word recall and the share of clips with singing drop by 0.10 at most. Met: recall 0.595 to 0.600, singing 0.944 to 0.972.
3. Random heads at beta 2 help less than half as much. Met: median cut -0.001, against 0.024 allowed.

Verdict: not supported. The run's label "timing not moved in 9 items" is a fixed phrase in the code for a failed condition 1; in fact 10 of the 12 lines were not improved. Its label "head-specific" only records that condition 3 held: random heads cut the error less (median -0.001) than the alignment heads (0.047). With so small a main effect, it says little.

Outside the rule: beta 1 improved 1 line (median cut 0.003), and beta 4 improved 6 (median cut 0.229, 95% interval over lines 0.008 to 0.365), while recall fell to 0.525 and the words still started a median 2.9 s early. So a stronger clock moves the timing somewhat, but beta 2 alone doesn't.

### B2 Stage D and the selection

900 runs: 25 recipes (guide Voice Oohs or clarinet, k = 3, 4 or 5, clock none, A2, A4, A8 or W2, minus the trimmed clarinet k = 3) × 12 lines × 3 draws. In the clock names, A is the seven alignment heads, W is all 80 heads of layers 2 to 6, and the number is beta. 690 of 900 clips passed the vocal gate. 13 of the 20 clocked recipes passed the 0.60 gate rate needed to be chosen.

S* is `oohs_k3_A8`: the Voice Oohs guide for the first 3 of 8 steps and the clock at beta 8 on the alignment heads. It was joint on 9 of the 12 lines, against 6 for the next recipe (`oohs_k4_A8`), so no tie-break was needed. B* is `oohs_k3_none`, the same guide without the clock (2 of 12). Three Stage T arms are therefore the same recipe (S*_b0, ctx_oohs_k3 and B*), and Stage T merged them as registered. The selection file was written at 00:07:20Z, its hash is the first line of the Stage T ledger, and the first Stage T generation followed at 00:07:40Z.

### B2 Stage T and the decision

480 runs: 5 arms × 24 new lines × 4 draws, none missing or failed. P0, the integrity conditions, held.

| Arm | J (of 24) | Joint clips (of 96) | Mean NPA_v | Mean WR | Median word-start error |
|---|---|---|---|---|---|
| S*: clock on the alignment heads, beta 8 | 14 | 31 | 0.44 | 0.50 | 0.42 s |
| Same guide, no clock (S*_b0 = ctx_oohs_k3 = B*) | 8 | 13 | 0.45 | 0.32 | 1.73 s |
| Clock on random heads (S*_rand) | 8 | 11 | 0.44 | 0.28 | 1.86 s |
| Clock with the windows rotated (S*_shuf) | 7 | 9 | 0.41 | 0.31 | 4.49 s |
| Stronger guide, no clock (ctx_oohs_k4) | 3 | 5 | 0.75 | 0.07 | 3.75 s |

NPA_v by chance (scored against the other test melodies) is about 0.10 in every arm. The word-start error is the median over clips with at least 4 matched words.

- P1, clock credit, failed. The gap is J(S*) - J(S*_b0) = 0.583 - 0.333 = 0.25, above the 0.20 needed. But the bootstrap's 2.5th percentile is exactly 0, and the rule needs it above 0. 286 of the 10,000 resamples had a gap of 0 or less (2.9%). The exact bootstrap gives 3.0%, and seeds 0 to 19 and every percentile method also give 0, so this is not resampling noise. For reference, 1.0% of resamples were below 0, so the gap sits exactly on the rule's boundary. Against the same guide without the clock, S* won 8 melodies, lost 2 (b2t14 and b2t22) and tied 14.
- P2, capability, held: 14 of 24 melodies, where 12 were needed.

Verdict: "Capability reached, clock not credited."

The registered labels:
- Strict capability (kept q of 0.75 or more on at least 12 of the 24 melodies): no, 2 of 24. The same guide without the clock also has 2.
- Head-specific clock: yes. Mean kept q is 0.517 for S*, 0.353 with random heads and 0.385 without the clock.
- Timing-specific: yes. The lower bound of S* minus S*_shuf in mean kept q is 0.049 (mean 0.126).
- Timbre copy: no. None of S*'s 96 clips was vowels only.
- Words cut: no. Recall is 0.503 in the first half of the line and 0.489 in the second.
- Frontier moved: no. NPA_v stays the same (-0.006), and WR rises by 0.174, short of the 0.20 needed.
- Dev futility: no. On the development lines S* ranked above B* (joint on 9 lines against 2).
- Independent ASR: P2 fails with the second recogniser. With the CTC model's word recognition on the same kept draws, S* is joint on 2 of 24 melodies.

The paired contrast of mean kept q against the same guide without the clock is +0.131 (95% interval 0.064 to 0.200). Best of 1, 2, 3 and 4 draws gives J of 0.32, 0.47, 0.53 and 0.58 for S* (averaged over all subsets of draws), against 0.14, 0.23, 0.29 and 0.33 without the clock.

What it means: with the clock, the kept draw has at least half the notes right on the vocal and Whisper recognises at least half the words, in order, for 14 of 24 new melodies. By mean kept q, a secondary measure outside the rule, S* is ahead of the same guide without the clock (+0.131, 95% interval 0.064 to 0.200), though q also rests on Whisper. By the registered J rule the gain can't be credited to the clock: the gap of 0.25 clears the 0.20 needed, but its bootstrap 2.5th percentile is 0, not above it. The second recogniser doesn't back up the words: it finds a joint clip for 2 of S*'s 24 kept draws and 3 of the unclocked arm's. On the kept draws it rates S* a little below the unclocked arm (WR -0.08, 9 melodies up and 13 down). Over all 96 seed-paired clips S* is slightly ahead (WR +0.02). The CTC model reads every arm poorly (mean WR 0.05 to 0.18), and Whisper and CTC agree least on S*'s clips (correlation 0.53, against 0.61 to 0.79). So the capability rests on Whisper large-v3 alone, and the blind listening test has to decide whether people hear the words. Whisper got no lyric prompt, and none of S*'s joint transcripts contains a known hallucination phrase.

The margin is small. Two of S*'s 14 joint melodies (b2t12 and b2t18) have no spare note: with one note fewer right on every clip, J(S*) would be exactly 12 of 24, and with two fewer, 10.

### Checks after the run

Agents rechecked the run from the files and wrote their own code from the registered text. They didn't use the repo's decision code for the main computation.
- Provenance: all 89 files registered in the three addenda hash to their registered values at `bd61a9f`, the tree was clean, every result file that records a commit names `bd61a9f`, and every ledger row carries the registered code hashes. `register_b2.py --check` passed on Colab during setup (`setup_check.json`) and passes here too, with stand-ins for two audio packages it imports but doesn't call.
- Ledgers: every planned run appears exactly once, in the registered order: B1 180, Stage D 900, Stage T 480. Every hooked run made 40 hook calls with every biased key inside the lyric, and every run without a hook made none. The heads, beta and window rotation match the registered sets. Within each line, draw and guide, every cover run used the same guide latent, whatever its k. All seeds match the registered table. Whisper ran at its pinned revision on every clip, and the CTC model at its pinned revision on every B2 clip. To make sure the checker can fail, they injected 20 faults into a copy of the ledgers, and it raised all 21 expected flags.
- B1: an independent recompute matched every number in `b1_result.json`. Under 12 other defensible readings of the rule, beta 2 improves 2 or 3 lines, never close to 9.
- B2: independent recomputes of the selection and of the decision matched every stored number, with one exception: one ineligible recipe's mean differs by 1e-6 because of rounding order.
- The blind pack: each clip is the demo clip of the run its key names, the pair order follows the registered seed, and nothing in the files (names, tags, durations, sizes, loudness) shows which side is which.

### What went wrong

Nothing stopped the run. These are worth recording:
- The A0 time projection was six times too high, as described above. The trim was applied as registered, but it removed clarinet at k = 3 and Stage T's no-guide arm when the full design would have fit.
- B1's label "timing not moved in 9 items" reads wrongly when fewer than 9 lines improve; 10 of 12 were not improved.
- `b2_selection.json` marks every unclocked recipe as eligible, including two below the 0.60 gate rate. The rule puts no gate on B*, so this is only how the field is named.
- ACE-Step copied its own `configuration_acestep_v15.py` into the checkpoint before the weights were hashed. The model file it copied matches the registered hash; the configuration file isn't in the registered list, so its content wasn't checked.
- Ledger rows carry no time stamps. The log lines show the selection was written before Stage T began.
- The WAV and latent files are on Drive, not in the package, so their hashes couldn't be checked again here.
- The word matcher in `dacapo/words.py` is the one the registration names, but it isn't always the longest in-order match. On 12 to 13 of the 480 clips a longest match finds 1 or 2 more words. J and the verdict don't change.

The labelled demo clips in the result package show which recipe each clip comes from. Nobody should hear them before doing the blind test.

## 2026-10-30: the listening analysis, registered before any rating

Nobody had rated either blind pack yet, so the agents wrote down how the ratings will be analysed and registered it (`experiments/listening/registration/listening_plan.json`, with the code and tests hashed). Two independent reviews came first, and they changed three things.

- The first Run A pack named its clips with a seed in the public code, so anyone could rebuild its key. The same 60 clips were renamed in an order from the operating system's random source (pack v2), and the first pack won't be handed to raters.
- In the B2 pack the side of each pair was a coin toss, so the number of pairs with the clock on side A isn't 12. A plain test would let a listener who just likes clip A tip the result. The registered test is conditional on that number and compares how often A was picked when the clock was on A with how often it was picked when the clock was on B, so a side preference can't move it.
- The keys stay in the owner's private repository. Their published fingerprints use a random salt, because a 24-pair key is short enough to search.

Ratings close when the owner says no more sheets are coming, at the latest on 2026-11-30. The agents check every sheet without the key first, then commit the exclusions and the sheets' hashes, and only then open the keys.
