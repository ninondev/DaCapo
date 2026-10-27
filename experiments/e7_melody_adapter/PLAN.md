# E7 Melody Adapter ExecPlan

Read this file first at the start of every task. Continue from Progress.
Reply to the owner in Chinese. Use simple English in repository files.
Do not put the owner's personal background in the repository.

## Progress

- [x] Saved the task base in `.git/e7_base`.
- [x] Read the task brief, root README and log, E4 code, and `dacapo/ace.py`.
- [x] Fetched `origin/main`. It matches local HEAD at
  `3c0c259d53251bd1992f1330e74f93776d20a0af`.
- [x] Checked the five E5 v2 prerequisite files. All are absent at that commit.
- [x] Audit and complete the pre-existing untracked E7 draft.
- [x] Finish S1: fixed melodies, source records, and CPU upload notebook.
- [x] Finish S2: the Suno operation protocol.
- [x] Finish S3: freeze the full scoring contract and file hashes.
- [x] Finish S4 implementation: GPU controls, synthetic ruler, scoring, and exports.
- [x] Run `pytest experiments/e7_melody_adapter/tests` and record all counts.
- [x] Run `bash experiments/e7_melody_adapter/precommit_check.sh`.
- [ ] Owner creates the PR. Claude reviews and merges the preregistration.
- [ ] Owner runs the pack notebook and Suno protocol after that merge.
- [ ] Wait for E5 v2 to be pushed and registered before GPU scoring.
- [ ] Claude applies S5 using the private result package.
- [ ] H1 Stage 0 through Stage 5 remain future work, subject to the gates below.

This task delivers the implementation of the brief's first task, steps 1-6.
It does not run Suno, GPU generation, separation, Whisper, training, or H1 gates.

## Surprises & Discoveries

- The task started with an untracked E7 draft. Useful work must be preserved and
  audited. Existing draft files are not evidence that a GPU run succeeded.
- Local HEAD and fetched `origin/main` are the commit listed in Progress.
  `dacapo/seeds.py`, `dacapo/words.py`, `experiments/e6_clock/readout.py`,
  `experiments/e5_motif/registration/addendum_2.json`, and
  `experiments/e5_motif/register.py` are absent there.
- The current `dacapo.ace.generate` has no `rng_seed` argument. GPU scoring must
  refuse to run on this checkout. Do not repair that protected file here.
- The absence of E5 v2 permits CPU preparation. It does not permit the GPU run.
- The actual Cloud image differs from the brief. Initially there is no
  FluidSynth executable, no soundfont at `/usr/share/sounds/sf2/FluidR3_GM.sf2`,
  and no `/opt/ACE-Step-1.5` source copy. Python measurement dependencies are
  available in `.venv`. System package installation is not permitted.
- The exact FluidR3 soundfont and FluidSynth CLI were obtained from Debian
  packages into a local prefix under ignored E7 outputs. No fallback soundfont
  or protected-file change is used. Do not set `ACESTEP_REPO` in Cloud.

## Decision Log

- 2026-10-27: Continue the existing untracked E7 draft. All source changes stay
  under `experiments/e7_melody_adapter/`.
- 2026-10-27: Use the required lyric-authorship provenance field only. Do not add
  authorship badges or commit trailers. This narrow field is required by S1.
- 2026-10-27: Keep E5 v2 as a hard GPU prerequisite. Its absence is not a reason
  to alter protected code or to omit CPU preparation.
- 2026-10-27: Use the vanilla Side-Step trainer first when H1 starts. Its original
  eight turbo timesteps and lack of CFG dropout match the first training plan.
  The single trainer retry is `train_fixed --cfg-ratio 0`.
- 2026-10-27: The Stage S strict search defines speed as performed tempo divided
  by reference tempo. A speed of 1.03 means reference times are divided by 1.03.
  A positive transpose means the performed pitch is higher than the reference.
- 2026-10-27: Stage S uses its own frozen lyric matcher. H1 will switch to the
  E5 v2 word helpers and record that change in the E7 log.
- 2026-10-27: H1 evaluator and selector names are provisional until Stage 0c.
  Stage S uses SwiftF0. It does not validate a singing note-transcription ruler.
- 2026-10-27: Use the local Debian runtime prefix for mandatory CPU rendering
  because system installation is unavailable. Keep the CLI and exact FluidR3
  font in ignored outputs, and preserve the soundfont SHA256 in provenance.
- 2026-10-27: Assert the four E4 arm names in the notebook, but infer with
  `text2music` and `is_covers=False`. On the first k denoiser calls, replace only
  the 64 source channels and preserve the captured live mask. Do not use E4's
  cover-nofsq task instruction; training and inference must share text2music.
- 2026-10-27: Strict-external alignment optimizes full-note absolute-pitch frame
  accuracy over 100 ms starts, all 21 speeds, and all 25 shifts. Keep two start
  candidates per speed/shift, then refine the best sixteen globally over plus
  or minus 100 ms at 10 ms steps. NPA, note-F1, and octave diagnostics reuse
  the selected transform. Wrong references receive the identical search.
- 2026-10-27: Relaxed DTW charges every consumed reference coordinate. A (1,2)
  step averages two output costs for one reference frame; a (2,1) step sums
  both reference costs. Normalize by non-rest reference frame count. Word
  recall uses longest common subsequence, preserving order and multiplicity.

- 2026-10-27: Pin matching torch and torchaudio 2.10.0 builds. Use CUDA 12.8
  and torchvision 0.25.0 for the GPU notebook. Use CPU builds for the pack.
  Basic Pitch is installed separately with --no-deps.
- 2026-10-27: Freeze SwiftF0 bounds at 65-1500 Hz, confidence at 0.5, and
  the existing 10 ms resampling rule. Use Basic Pitch ONNX with 80 ms minimum
  notes and skyline for the unvalidated note-F1 diagnostic.
- 2026-10-27: Hash the read-only metrics, analysis, symbolic, rendering and
  provenance dependencies as well as E7 runtime files. Changes need a reviewed
  deviation. Do not hash the old ace.py API, which must receive E5 v2 first.
- 2026-10-27: Synthetic detection AUC means ROC AUC over clean and corrupted
  clips. Bootstrap whole clean/error item pairs. Report matched-pair win rate
  separately. Start, speed and transposition recovery are diagnostics only;
  they do not add a gate beyond the registered NPA ceiling.
- 2026-10-27: Use the upstream E5 seed helper for GPU generation only after
  checking equality with the frozen local formula. Intermediate result JSON
  files also use write_result and clean provenance.

- 2026-10-27: Reviewed the pinned upstream conditioning and loader source.
  The native loader can download unpinned missing assets or overwrite checkpoint
  Python from the source tree. Require a complete pinned snapshot and matching
  model code before loading. A mismatch stops the run without modifying files.

Small choices are made here and work continues. Do not ask the owner to resolve
routine details. Stop only for the explicit stop conditions in this plan.

## Goal and evidence

The goal is a change creators can hear: write eight bars and lyrics, then hear
that melody sung accurately in a full arrangement with clear words. A wrapper,
a leaderboard score, or a selected demo is not the goal. Do not promise success.
A clear negative result is also a completed result.

Stage S first measures Suno and untrained ACE-Step 1.5 on the same twelve
melodies with the same ruler. If Suno already meets the rule, stop for the owner.
H1 then tests a melody LoRA in the MIT-licensed ACE-Step 1.5 DiT decoder.
Training data comes from the model itself. Evaluation uses verifiable notes.

Read these existing sources before implementation work: `README.md`, root
`LOG.md`, `experiments/e4_sing/`, and `dacapo/ace.py`. They are read-only here.
The inherited evidence in the brief is:

- E1, `results/gpu_run1/e1_melody_paths_shift3.json`: instrumental dense 25 Hz
  context gives pitch-class accuracy 0.910, versus 0.897 on the input. The 5 Hz
  codes preserve about 28 percent of melody information. Work in the decoder.
- E4, `results/gpu_run1/e4_sweep.json`: generated four-bar a cappella references.
  `ctx_oohs_k6` gives NPA 1.000 and word recall 0.028. No guide gives NPA 0.032
  and word recall 0.659. Only 2 of 378 clips reach both NPA and WR of 0.5.
  Whisper can invent words without vocals: 106 of 504 transcripts have WER > 1.
- E0, `results/cpu/e0_ruler.json`: Basic Pitch skyline note-F1 is 0.42 on
  voice-like solo audio. Detecting two wrong notes in mixtures gives AUC
  0.75-0.78, below 0.8. Separated real singing has not been validated.

The brief cites Claude's source review dated 2026-10-27. These are context,
not new measurements from this task:

- Qwen-Music, arXiv 2607.11699: audio-reference cover melody deviation was
  2.00 semitones for Suno V5.5, 1.87 for Suno V5, and 1.89 for MiniMax Cover.
  Its F0 tool, reference audio, and DTW differ from this study. Do not compare
  our numbers directly or claim an improvement of a given number of semitones.
- Suno Studio 2.0, dated 2026-08-13, can use MIDI as a generation prompt. No
  official vocal melody accuracy was supplied in the brief.
- YuE2, arXiv 2609.33757, accepts ABC scores and reports edit accuracy 84.17.
  Its weights are CC BY-NC 4.0. It was not tested on human-written scores in
  the cited review. Use only for measurement, never for training here.
- SongEcho, arXiv 2602.19976, and MPEcho, arXiv 2607.26698, add F0 conditions
  from reference vocals to ACE-Step v1, not ACE-Step 1.5.

## Work boundaries and delivery

The repository is `https://github.com/ninondev/DaCapo`. E5 is owned by Claude.
Start the next task from the updated main branch, after reading this plan.
At the start of each task run:

```bash
test -f .git/e7_base || git rev-parse HEAD > .git/e7_base
```

Every new or changed source file must be under `experiments/e7_melody_adapter/`.
Tests belong in `tests/test_e7_*.py` under this directory. Never change, add,
or delete files in these protected locations:

- All of `dacapo/`, root `tests/`, and `results/**`.
- `experiments/e5_motif/`, `experiments/e6_clock/`, `experiments/e4_sing/`,
  `experiments/run_all.py`, and `experiments/build_notebook.py`.
- Root `README.md`, `README.zh-CN.md`, `LOG.md`, `pyproject.toml`, `.gitignore`,
  `notebook.ipynb`, and `AGENTS.md`.

E5 registration hashes protected files. A one-byte change can invalidate it.
Do not run `experiments/e5_motif/register.py` except with `--check`.
Do not run `experiments/e4_sing/make_items.py` main. Do not run editable package
installation, `uv add`, or `uv lock`. Never add attribution trailers or marks
to commit messages, PR descriptions, or source files.

Do not commit binary files, images, MIDI, audio, checkpoints, or datasets.
Store melodies as JSON. Third-party audio, annotations, and MIDI stay outside
the repository. Retained JSON records belong under `records/` in this directory.
The local `.gitignore` must cover `outputs/`, `*.pt`, `*.safetensors`, `*.ckpt`,
`*.bin`, `*.zip`, `*.mid`, `*.wav`, `*.mp3`, `*.ogg`, `*.opus`, and `*.m4a`.
Extra cache and generated-file ignore rules are allowed in this directory.

Run `precommit_check.sh` before ending every task. It must:

1. Use `$E7_BASE`, then `.git/e7_base`, then `origin/main` as the baseline.
   With no baseline, print the required setup command and exit 2.
2. Require every path from `git diff --name-only <base>` and
   `git ls-files --others --exclude-standard` to begin with the E7 prefix.
3. Reject each file over 1 MB. Reject each nonempty file that fails
   `grep -Iq .`. Check both changed and untracked files.
4. Run `python experiments/e5_motif/register.py --check` if that file exists.
   A nonzero result is a stop condition. Never rewrite the registration.

Run `pytest experiments/e7_melody_adapter/tests`. Missing FluidSynth,
soundfont, `swift_f0`, or `librosa` must fail tests, not skip them. Only a test
that needs the actual model may skip when `ACESTEP_REPO` is absent. Use fake
tensors following the approach in
`/opt/ACE-Step-1.5/acestep/training_v2/make_test_fixtures.py` where useful.
The owner reply must state passed, skipped, and failed counts separately.

Cloud work is CPU implementation, tests, and documentation. All model generation,
vocal separation, Whisper, and training are written for the owner's GPU runtime.
Use `run_stage.sh <stage>` for every GPU stage. Colab notebooks call that script.
This also supports a later 12 GB RTX 4080 Laptop runtime if Colab is insufficient.

The owner creates the PR in Codex and gives it to Claude for review and merge.
The owner does not merge it. Do not start using a main-branch Colab link before
that notebook is merged. The link form is:
`https://colab.research.google.com/github/ninondev/DaCapo/blob/main/experiments/e7_melody_adapter/<name>.ipynb`.

GPU notebooks emit result zips. The owner gives them to Claude. Claude checks
them and commits the public `records/*.json`. Suno records follow the private
policy below. Never claim a GPU path was tested merely because CPU tests pass.

## Provenance, versions, and seeds

Record every external model and dataset license and source URL in
`records/licenses.json`. Research-only data such as CC BY-NC may be measured.
It may not train a model or enter the repository. Stop if research use is banned.

Colab must change to the clone root before imports and result writing. Keep
ACE-Step source, weights, Drive inputs, and outputs outside the clone, or in
ignored E7 `outputs/`. Use `dacapo.provenance.write_result` to write result JSON
to `outputs/` first. Put those JSON files under a `records/` path inside result
zips. At notebook end assert every result has a known `environment.git_commit`
and `environment.git_dirty is False`. Every result also records the ASR model
and revision, F0 tool, separator, and configuration hash.

Cloud source records such as `suno_pack.json` and `licenses.json` may correctly
have `git_dirty: true`. Their config includes the generator script SHA256.
A test must regenerate their contents exactly, with their source-record
provenance fixed rather than replacing it with a new runtime timestamp.

Pin the following in the requirements or explicit notebook installation:

| Component | Fixed value |
| --- | --- |
| ACE-Step 1.5 source | `ca1e85fe9430179831e6bc6be790c332190a3866` |
| `ACE-Step/Ace-Step1.5` weights | `19671f406d603126926c1b7e2adc169acbcade22` |
| `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| Separator | `htdemucs`, `demucs==4.1.0` |
| F0 packages | `swift-f0==0.3.0`, `torchfcpe==0.0.4`, `torchcrepe==0.0.24` |
| Metrics and audio | `mir_eval==0.8.2`, `pretty_midi==0.2.11.post0`, `librosa==1.0.0` |
| Basic Pitch | Separate `pip install --no-deps basic-pitch==0.4.0` |

Do not put Basic Pitch as a dependency line in `requirements-e7.txt`; that can
pull TensorFlow. Explain the separate no-dependencies command in a comment.

Until E5 v2 exists, use this exact local seed formula:

```python
d = sha256(f'{tag}|{item_id}|{draw}'.encode()).digest()
dit_seed = int.from_bytes(d[0:4], 'little') & 0x7FFFFFFF
rng_seed = int.from_bytes(d[4:8], 'little') & 0x7FFFFFFF
if rng_seed == dit_seed:
    rng_seed ^= 1
```

Allowed tags are `e7-suno`, `e7-smoke`, `e7-data`, `e7-dev`, `e7-test`, and
`e7-listen`. Generation-data and evaluation seeds must not overlap. For a given
item and draw every arm uses the same pair. When E5 v2 lands, first rerun the
tests on its new main. Add an exact equivalence test against
`dacapo.seeds.run_seeds`, then use that helper. Do not silently change Stage S
registered code; any post-registration change needs the addendum process.

## Shared measurement contract

Freeze definitions and parameters in each preregistration before the relevant
run. Code hashes bind the Stage S implementation to `prereg_suno.json`.

**Separation and voice presence.** Use htdemucs vocals. Voice-present requires
vocal-stem energy share at least 0.2 and at least 1.0 second of nonoverlapping
50 ms vocal frames with RMS at least -40 dBFS. These three limits come from
E5 v2 `experiments/e6_clock/readout.py` (`VOCAL_SHARE_MIN`,
`VOCAL_ACTIVE_MIN_S`, `VOCAL_ACTIVE_DBFS`). Do not use its fourth `PEAK_MIN`
condition for external audio; the original generation peak is unavailable.

**Lyrics.** WR is reference-word recall from fixed-revision Whisper large-v3 on
separated vocals. Refuse scoring without CUDA. Do not substitute a smaller ASR
model. Stage S freezes its own matching function. H1 must instead use
`dacapo.words.lyric_words`, `norm_word`, and `match_words`; record that transition
in `LOG.md`. Match the same transcript against each of the other eleven lyrics
and report their mean and correct-minus-wrong WR. Repeats, added lyrics, and
hallucinations on instrumental sections can inflate recall.

**Strict pitch.** Use `dacapo.metrics.note_pitch_accuracy` for NPA and
`note_metrics` for note-F1. Preserve note order and relative timing at the
selected tempo; no local tempo warping or structure edits. H1 arms permit at
most 0.5 seconds of global lag, use absolute pitch, and do not transpose.

All Stage S audio, including ACE-Step controls, uses strict-external alignment.
Search the entire output for one start, one tempo ratio in 0.90-1.10 at 0.01
steps, and one whole-semitone transpose from -12 through +12. Choose these once
for each output/reference pairing and put them in CSV. Freeze the coarse/fine
algorithm in the preregistration. Use precisely the same search for all eleven
wrong references. Report chance means and correct-minus-wrong scores. Also
report octave-tolerant NPA on the same alignment as a diagnostic only. The
wrapper is in `lib.py`; do not alter the library's fixed-lag NPA function.

**Relaxed pitch.** Draw reference MIDI pitch at 10 ms and remove reference rest
frames. Convert output vocal F0 to MIDI pitch at 10 ms and keep its unvoiced
frames. Their fixed DTW cost is 3 semitones. Run subsequence DTW for each global
transpose -12 through +12 and take the smallest normalized cost. Only steps
(1,1), (1,2), and (2,1) are allowed. The matched duration is 0.5-2 times the
non-rest reference duration, to frame quantization. Cost is absolute semitone
error without octave folding, normalized by reference frame count.

Report the matched segment's mean absolute deviation in semitones and the share
of reference notes whose matched output median pitch is within 50 cents. Missing
notes count as wrong. Report octave-folded diagnostics separately. Perform the
identical full search for all eleven wrong melodies. For accuracy report correct
minus their mean; for deviation report their mean minus correct. The best segment
is selected using the right melody, so the absolute score is optimistic. The
adjusted result is the main comparison. Qwen-Music has the same deviation unit
but different references and tools; direct numeric claims are not allowed.

**H1 primary score.** Gated note-F1 is zero if voice-present is false or WR is
below `WR_min`; otherwise it is eval-ruler note-F1. Average by item.
Stage 1 sets `WR_min` to the 30th percentile of base no-guide full-arrangement
WR, using no test melody. Plain note accuracy already saturated in E4 while
the words vanished. Label tool, eval ruler, and best-of-N selector must all
differ. Test at least that selector and evaluator are different.

## Stage S: fixed-melody external measurement

### S1: source pack

Use `dacapo.symbolic` and a fixed seed to make twelve original eight-bar,
4/4 melodies. Tempo is 90-110 BPM; keys are C, D, F, and G major; range is
A3-E5 (MIDI 57-76); each melody has 24-40 notes. Use mostly scale steps, a few
leaps, and repeated motifs. Write simple singable English lyrics, mostly
one-syllable words, with one syllable per note. The required lyric provenance
must identify its author in records. Do not use the owner's future test set.

`melodies_suno.json` stores pitch, start and duration in beats, BPM, key, and
lyrics. `records/suno_pack.json` records its SHA256, the lyrics SHA256, and
render settings including program, sample rate, and soundfont SHA256. It does
not contain rendered audio or Suno output.

`colab_e7_pack.ipynb` runs on CPU after merge to main. It renders piano program
0 and Voice Oohs program 53 at 48 kHz, writes MIDI with `Melody.to_midi`, and
adds lyric text and the CSV record sheet. It emits `e7_suno_upload.zip` and
`suno_upload_hashes.json`, containing each upload file's SHA256. The owner
downloads both and gives the hash record to Claude.

### S2: owner operation

Keep the English protocol in `suno_protocol.md` and give the instructions in
Chinese in the owner reply. Wait for Claude to confirm the preregistration is
merged before any Suno generation.

- R1: use a MIDI prompt in Suno Studio, the supplied lyrics, and the exact style
  `pop, full band, female vocal`.
- R2: upload the piano WAV with the available cover/reference-audio feature,
  using the same lyrics and style.
- R3: upload the Voice Oohs WAV, otherwise using R2's settings.

Current menu names are not known. Skip any unavailable route and record the
actual options shown, displayed model version, date, and credits used. Request
once per item and available route. Keep both returned songs; never pick one.
Name them `S_<melody_id>_<route>_<a|b>.mp3` and place them in one Drive folder.
Keep the filled CSV with them. Only upload these twelve study melodies. Do not
upload owner-held-out melodies to any website or app before Stage 4. The owner
must read Suno's terms and confirm whether evaluation results may be published.

### S3: preregistration and decision

Before external generation, write `prereg_suno.json` with a UTC timestamp,
melody/lyric hashes, all Stage S measurement definitions and parameters,
routes, controls, thresholds, decision rule, analysis, and scoring-code hashes.
Hash at least the scoring source, scoring notebook, and `run_stage.sh`. The
scoring notebook checks hashes before scoring. Mismatch must stop it.
After registration, use a new reviewed addendum marked `deviation` for a scoring
change, retain the old registration, and still report original-version results.

A route is measured only with both returned songs on at least ten items.
First average a and b for each item. Then take the median across complete
items. A positive rule is met if at least one measured route has median
strict-external NPA at least 0.90 and median WR at least 0.60 on that same route.
Report every route. If none is measured, the outcome is not measured, not
inaccurate, and Suno kill rule 3 remains unknown. Only changes that make a
positive decision easier may be made with recorded rationale. Making the rule
stricter requires the owner's agreement.

Stage S uses NPA for this decision because singing note-F1 is not yet validated.
Mark all Stage S note-F1 as an unvalidated-ruler diagnostic; it cannot decide.

### S4: GPU notebook and scripts

At notebook start require `dacapo.ace.generate` to accept `rng_seed` and require
`register.py --check` to return zero. E5 v2 and its addendum must exist. Otherwise
stop and tell the owner to wait for Claude. Call `arms(['clarinet', 'oohs'])`
only in the GPU notebook and assert the four required arms exist.

Read the Drive audio and completed record sheet. Validate names and quantities;
list missing songs without aborting the whole analysis or inventing results.
Generate ACE-Step controls for the same twelve melodies with `noguide`,
`ctx_oohs_k3`, `ctx_oohs_k6`, and `ctx_clarinet_k4`. Use the exact shared style,
lyrics, BPM, and keyscale. Preregister duration as melody length plus two bars.
Read `GUIDES` from E4 without running its main: clarinet 71 and oohs 53 at
48 kHz. Use two `e7-suno` seed pairs per item, matched across arms. Apply the
same strict-external, relaxed, WR, voice, and chance measures as for Suno.
The main comparison is correct-minus-wrong.

Ruler self-check uses rendered Voice Oohs, not natural singing. Mix with
accompaniment after a 5-20 second intro and repeat the whole section once.
Also test tempo times 1.03 and transpose +2. Separate vocals, score, and check
recovery of start, global tempo, and transpose. Report median NPA as the ruler
ceiling. Make variants with two wrong notes and report paired detection AUC.
Freeze the F0 tool and settings before running; Stage S uses SwiftF0.

Write one CSV row per clip and summary JSON with item-level bootstrap 95 percent
intervals. Split numeric results into `e7_suno_public.zip` for ACE-Step and the
synthetic ruler, and `suno_private.zip` for Suno. Neither zip contains Suno audio
or separated vocals. The L4 estimate is about one hour, not a measurement.
Stop if the registered GPU budget exceeds 1.5 times its limit; do not silently
extend it or select a favorable completed subset.

### S5: private review and continuation

Claude applies the preregistration to the results. Before the owner confirms
the terms permit publication, the only public outcome statement is:
"Stage S was evaluated under prereg_suno.json. The conclusion is in the private
record." Do not put Suno scores, route outcomes, accuracy conclusions, or
threshold comparisons in public PLAN, LOG, records, or the PR. Suno result
records belong only in the owner's private repository. Do not use the quoted
statement before an evaluation has actually happened.

If synthetic-ruler median NPA is below 0.95, the outcome is ruler insufficient,
pending. Stop and report; do not start H1. If Suno meets the positive rule, kill
rule 3 is triggered: stop for the owner to choose H7 (one-note edits), H2 (motif
development), or H1 with an open-source parity goal. The last option needs a
kill-rule-3 deviation in `prereg_h1.json`. A measured negative is provisional.
H1 may then proceed, but after Stage 0c picks a validated ruler, rescore the same
private Suno files. If that reaches the positive rule, stop in the same way.

## H1 roadmap and gates

At each phase end, use `write_result` for `records/stageN_*.json` through the
ignored output workflow. Update this plan and the E7 log, including failures.
Run precommit checks. Give the owner a Chinese summary of at most ten lines:
work done, key numbers, gate status, next step, and required owner action.
Then add the learning material described below. Total GPU budget is about
32 hours. Stop any phase beyond 1.5 times its stated budget.

Before any Stage 0 GPU work, main must contain E5 v2 `addendum_2.json` and the
registration check must pass. Until then do independent CPU preparation and
report that Claude's E5 v2 push and registration are pending.

### Stage 0: cheap checks, at most 3 GPU hours

- 0a: if networking permits, spend at most 30 minutes on relevant work after
  2026-10-27, checking kill rule 3. If not, log that it was not done and relies
  on Claude's 2026-10-27 review.
- Before 0b, write `prereg_h1_stage01.json` and have Claude merge it. Specify
  ruler candidates, separator, dataset, vocal levels, 0c rule, Stage 1 labels
  and gate, and smoke-run settings. Later changes need deviation addenda.
- 0b: measure L4 peak VRAM, seconds per step, and per-clip generation time for
  turbo bf16, r32 self+cross-attention LoRA, batch 1, 30-second audio, gradient
  checkpointing, AdamW8bit, and encoder offload. Estimate 12 GB feasibility.
- Smoke run: at most one hour, exploratory only. Generate eight `e7-smoke`
  clips, use temporary SwiftF0 labels, train about 300 steps, and check whether
  note scores rise on their own melodies. This diagnoses masks, hook behavior,
  and frame alignment. It must not support any gate.
- 0c: use at least 30 real-singing clips with 32 annotated notes each. Prefer
  English. Candidate datasets are GTSinger, Children's Song Dataset, M4Singer,
  and Opencpop. Download in the notebook and record license and URL.
- For each clip make k=2 by shifting two notes by plus/minus one semitone with
  pyworld or a phase vocoder. Make k=0 with the identical vocoder and zero shift
  so processing artifacts are balanced. Mix at two or more preregistered vocal
  levels, then separate with htdemucs, never ACE-Step's own separation.
- Select the ruler on half the clips. On the other half report paired AUC and
  bootstrap 95 percent intervals. At every vocal level the selected ruler's
  AUC point estimate must be at least 0.8. Otherwise stop to repair the ruler.
  Rescore the private Stage S audio with the selected ruler.
- Prepare `melodies/` templates, validation, and seal scripts. The owner
  instructions are Chinese: eight original bars, English lyrics, one syllable
  per note, declared key, meter, and tempo; accept MuseScore MusicXML or MIDI
  plus lyric text. Use JSON for repository melody files.

### Stage 1: data pipeline and label noise, at most 3 GPU hours

Download public-domain poetry, for example Project Gutenberg, in the notebook.
Select fixed-seed passages of 2-4 lines and 24-40 syllables. Record URLs and
SHA256. Generate 200 thirty-second `text2music` songs using `e7-data`, a fixed
list of full-band vocal styles, and report the voice-present rate.

Labels follow separation, the label F0 tool, 25 Hz frames, semitone quantization,
median filtering with voicing retained, plain-timbre rendering, VAE encoding,
and `src_latents` insertion into `.pt`. Label tool and eval ruler must differ.
Run the same pipeline on known-answer 0c real-singing mixtures and rendered
voice-like instruments with accompaniment. Report voiced-frame RPA and voicing
recall. Set `WR_min` from no-guide full-band WR without test melodies.

Gate: real-singing-mixture RPA at least 0.85, voicing recall at least 0.8, and
generated voice-present at least 60 percent. If it fails, try at most two
alternatives written down in advance. If it still fails, stop and report.

### H1 preregistration, before Stage 2

Before Stage 2 training results, write `prereg_h1.json` with a UTC timestamp.
It includes selected label and eval tools, shared definitions, `WR_min`, all
baselines, budgets, seed table, main caption, thresholds, data splits, analyses,
kill rules, claim levels, and listening design. Give it to Claude for merge.
Never overwrite it after registration; changes use a new deviation addendum.

Before submitting this H1 preregistration, stop and explain in Chinese the two
changes to kill rule 1: the gated metric and the listening design. The owner
must agree. Current authorization for Stage S does not replace that later step.

Kill rules:

1. At day 30, about 2026-11-25 from a 2026-10-26 research start, stop the adapter
   and publish a negative result if held-out human-melody gated note-F1 is less
   than the best training-free baseline plus 0.15 at equal sample budget, or if
   primary listening preference for adapter over base is below 40 percent,
   excluding ties. Either condition is enough.
2. If two wrong notes among 32 cannot be detected in separated vocals with
   AUC at least 0.8, repair the ruler first.
3. If a publicly available system with a commercial-use license reaches at
   least 90 percent note-F1 on user melodies at comparable quality, switch to
   H7 or H2. Stage S is the Suno check, with its stated NPA substitution.
4. If verifiable-reward post-training is no better than best-of-N at equal
   compute, publish a negative result. Register this rule only; do not do
   reward training in this work.

Claim levels:

1. Pass kill rule 1.
2. Positive result: the lower bound of the paired bootstrap 95 percent interval
   for adapter minus best baseline exceeds 0.15, and primary listening
   preference is at least 50 percent.
3. Mission-level evidence: report the share of test clips with note-F1 at least
   0.9 and WR no more than 0.1 below no-guide.

The owner supplies 36 melodies to Claude privately. Before Stage 3 training,
Claude runs the fixed-seed seal script: six become dev JSON under
`melodies/dev/`; thirty remain private with only SHA256 in
`records/testset_seal.json`. Before Stage 4, test melodies may undergo format
validation only. No generation, tuning, or scoring on them. Stages 0-2 may
continue before all owner melodies exist. Stage 3 must wait for them.

### Stage 2: overfit about 100 clips, at most 2 GPU hours

Perform three identity tests: writing the original captured context back gives
maximum absolute difference zero; a loaded LoRA at scale zero differs only
within bf16 noise; writing melody latents without LoRA changes the output.
Train route-A melody LoRA (r32, self+cross attention) and a silence-source
control LoRA on identical data and steps with identical mask values.

Compare base no-guide, `routeA_ctx_noLoRA`, control LoRA, and melody LoRA on the
training clips' own melodies and on swapped conditions (item j melody with
item i text and lyrics). In both cases melody LoRA gated note-F1 must exceed
all three other arms by at least 0.30. Its WR may be at most 0.10 below base.
Control LoRA voice-present rate and WR must be within 0.05 of base.

On failure, inspect the pitfalls below. Retry once with the other trainer.
Then allow one route-B retry: a zero-initialized Conv1d from K to 2048 channels,
kernel 2, stride 2, reads a timbre-free piano roll and adds to decoder `proj_in`
output via a forward hook. Save it separately. If that fails, stop and report.

### Stage 3: data at most 10 hours, training at most 8 hours

Make 1,000 clips first, with resumable shards. Train fresh melody and control
LoRAs. Select checkpoints and settings only on the owner's six dev melodies
and the read-only E4 manifest melodies. Training-free arms all use base model:
no-guide, `routeA_ctx_noLoRA`, `ctx_clarinet_k1`, `ctx_oohs_k1`, `ctx_oohs_k2`,
`ctx_oohs_k3`, `ctx_oohs_k4`, `ctx_clarinet_k4`, `sde_clarinet_n0.17`, and
`ctx_oohs_k6`.

On dev at equal sampling budget, adapter gated note-F1 must reach best baseline
plus 0.15. At most two dev-selected changes are allowed after a failed gate,
such as 3,000 clips, more steps, or rank. If it still fails, stop and report
that dev failed and test remained sealed. If Colab cannot support the runtime
or memory needs, stop, recommend the 12 GB laptop option, and provide local
run instructions rather than silently changing the study.

### Stage 4: held-out test and listening, at most 6 GPU hours

Run the thirty test melodies with melody LoRA, control LoRA, no-guide,
`routeA_ctx_noLoRA`, and the best dense guide selected on dev. Each item and arm
gets four fixed seeds and two or three fixed arrangement captions. Freeze the
main caption in the preregistration. Primary N=1 uses the first seed and main
caption. Use item-level paired bootstrap 95 percent intervals. Also report
best-of-N for N=1,2,4 with distinct selector and evaluator. Report per-clip
generation time for every arm.

Report the rescored Stage S reference separately, with its different melody
set stated. It is not part of the H1 kill-rule comparison. Keep Suno numbers
private under S5. If Colab can fit YuE2, measure it on the same test melodies
under CC BY-NC 4.0. Do not train with it.

Primary listening compares adapter with base no-guide on unheard `e7-listen`
melodies, with matched captions, lyrics, and seeds. Ask which sounds better as
a recording, ignoring the melody. Obtain at least 85 non-tie judgments; for
example, sixty pairs and two listeners. With only the owner, use 110 pairs and
state that there is one listener. This is the listening input to kill rule 1.

Secondary listening uses the owner's thirty test melodies and asks which is
more worth keeping as a song. The owner can recognize the melodies, so mark
it as not blind and report it separately. For both packs, match loudness,
randomize names and left/right order, strip metadata, hash and commit the
answer key before listening, keep the key from listeners, and provide a simple
response sheet. After returned responses, report preference, kill rule 1, and
the achieved claim level. Then stop.

### Stage 5: release, only after claim level 2 and owner agreement

Do not start release automatically. The owner and Claude decide license and
release method first. If authorized, provide a 12 GB inference script, a
Gradio demo (optionally on Hugging Face Spaces), weights, and clear usage.
Select demo clips by rules or seeds fixed in advance, never by listening for
the best ones.

## ACE-Step implementation notes

Use the existing `dacapo.symbolic` Note and Melody APIs for JSON-to-MIDI and
reference notes, plus `dacapo.render`, `dacapo.analysis`, `dacapo.metrics`, and
`dacapo.provenance.write_result`. Import `dacapo.ace` only inside functions that
run with the real model. Its top-level code reads `ACESTEP_REPO` and imports
ACE-Step. Do not import it at `lib.py` top level or in ordinary CPU tests.
The same restriction applies to E4 `run_sing.py`; its `arms()` is called only
from the GPU notebook.

At the pinned source, the decoder supports LoRA or LoKr, not full SFT. See
`docs/en/Large_Scale_SFT_Training_Guide.md`. Vanilla trainer is
`acestep/training_v2/cli/train_vanilla.py`. Fixed trainer is
`python -m acestep.training_v2.cli.train_fixed`; it uses logit-normal time and
normally 15 percent CFG dropout, so explicitly use `--cfg-ratio 0` for the retry.
Documentation suggests about 7 GiB with checkpointing and batch 1, and
AdamW8bit, encoder offload, rank 32-64 for 10-16 GB. These are not measurements.

Each training `.pt` has `context_latents [T,128]`, split into 64 source channels
and 64 chunk-mask channels. `build_context_latents` in
`acestep/training/dataset_builder_modules/preprocess_context.py` accepts source
latents already. For route A, run Side-Step preprocessing, then use the E7
wrapper to replace `.pt` contexts with rendered melody VAE source latents.
Do not alter ACE-Step source or model files.

Check these four pitfalls before interpreting a failure:

1. Do not guess chunk-mask values. Side-Step records 1.0 while auto masks can
   be assigned 2.0 into a possibly boolean tensor. Capture a real text2music
   context with `StepHook(capture=True)` and use its observed mask values for
   both training and inference.
2. Cover tasks or code hints can replace source context with 5 Hz codes. Use
   `text2music`, keep `is_covers=False`, and inject via StepHook.
3. Training text task instructions must match inference text2music. Check
   `acestep.constants.TASK_INSTRUCTIONS`; E5 v1 had the wrong instruction.
4. StepHook wraps `model.decoder.forward`. After PEFT injection it wraps a
   `PeftModel`. Verify this with Stage 2 identity tests, not assumptions.

## Honesty, stopping, and owner handoff

Do not use a language model as ground truth or judge. Separate contributions
from melody conditioning, extra singing data, no-training context, sampling,
and external tools. Manual selection is not permitted in metric results.
Compare equal sample budgets, tune only on dev, and run test once. Record
failures, bugs, and abandoned attempts as carefully as successes in the E7
log's Failures section. Every reported number must be reproducible from result
JSON and fixed seeds. State the achieved claim level and avoid inflated claims.

Stop and report, without expanding scope, when any of these applies:

- A gate still fails after its allowed retries.
- Stage S reaches its positive rule or has an insufficient ruler.
- Work needs a protected-file edit, or the E5 registration check fails.
- A phase exceeds 1.5 times its GPU budget.
- Kill rule 3 is triggered, or a license bans the needed research use.
- The next action needs the owner: Suno generation, Colab execution, new
  melodies, H1 kill-rule-1 agreement, listening, or the day-30 decision.

At this task's handoff, explain in Chinese what changed and confirm all changes
are under E7. State pytest passed/skipped/failed counts and the precommit result.
Tell the owner to create the PR for Claude, wait for preregistration merge, run
the pack notebook, and follow the Suno protocol. The scoring notebook also
waits for E5 v2. Include this exact opening for future tasks:

```text
Read experiments/e7_melody_adapter/PLAN.md first and follow every rule in it. Continue from Progress. Reply to the owner in Chinese.
```

Each phase-end Chinese reply also includes learning material, not a repository
lesson file: a plain-language explanation of at most 300 Chinese characters
without formulas, three questions with reference answers at the end, and one
15-minute exercise naming a file, function, number to change, command, and
expected result. Tie it to the current code. Suitable topics include note
names, semitones, intervals, keys, meter, motifs and transformations using
`Melody`, `transpose`, and `with_pitch_errors`; means, medians, intervals,
chance level, AUC, frame shapes and training intuition; and sample rate,
spectrograms, and fundamental frequency. Keep personal background out of files.

## Outcomes & Retrospective

The first task's implementation is complete. Final CPU pytest: 64 passed,
0 skipped, 0 failed in 19.62 seconds. This includes real FluidSynth rendering,
SwiftF0 alignment, byte-identical source-record regeneration, notebook checks,
context-hook tensors, private/public export boundaries, and registration guards.
The upload package test verified 49 assets: 24 WAV, 12 MIDI, 12 lyric text files,
and one 72-row recording CSV. Those binary outputs remain ignored.

Preregistration hashes match the frozen code and definitions. Precommit passes
for all 24 delivered text files, each under 1 MB and under E7. No tracked file
outside E7 changed. E5 registration is absent, so its check was not run and is
not reported as passed. E5 v2 is still required before GPU scoring.

No GPU or Suno result exists. The Stage S decision has not been made. Stop here
for the owner to create the PR and have the registration reviewed and merged.
Only after merge may the owner run the CPU upload notebook and Suno protocol.
The GPU notebook additionally waits for E5 v2. H1 has not started.

## Ideas not done

H1 training and later stages wait for the Stage S decision. Future ideas such
as humming input, Suno one-note edits, reward training, and changing models are
not authorized by this first task. Keep them here until separately chosen.
Do not start H7, H2, deployment, a demo, or release because it seems useful.
