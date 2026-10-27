# E7 Melody Adapter ExecPlan

Read this file first at the start of every task. Continue from Progress.
Reply to the owner in Chinese. Use simple English in repository files.
Do not put the owner's personal background in the repository.
Do not put assistant product names, real names, personal email addresses, or
other private information in repository files, commit messages, or PR text.
Use "the agents" or a role such as "the reviewing agent" for attribution.
Paper citations may retain author names. Exact model and tool identifiers
remain experiment inputs; they are needed to identify the measured systems.

## Progress

- [x] Saved the task base in `.git/e7_base`.
- [x] Read the task brief, root README and log, E4 code, and `dacapo/ace.py`.
- [x] Fetched `origin/main`. The checked base is
  `3c0c259d53251bd1992f1330e74f93776d20a0af`.
- [x] Checked the five E5 v2 prerequisite files. All are absent at that commit.
- [x] Audit and complete the pre-existing untracked E7 draft.
- [x] Finish S1: fixed melodies, source records, and CPU upload notebook.
- [x] Finish S2: the Suno operation protocol.
- [x] Finish S3: freeze the full scoring contract and file hashes.
- [x] Finish S4 implementation: GPU controls, synthetic ruler, scoring, and exports.
- [x] Run `pytest experiments/e7_melody_adapter/tests` and record all counts.
- [x] Run `bash experiments/e7_melody_adapter/precommit_check.sh`.
- [x] Opened PR #1 on the requested branch after the first implementation.
- [x] Complete the thirteen review fixes described in this plan.
- [x] Regenerate source records and the Stage S preregistration before any run.
- [x] Run the full CPU suite with no skips and the boundary/hash checks.
- [x] Add the repair as a new commit for PR #1 with the configured identity.
- [x] Second review repair: completion rules, runtime faults, lyric compounds,
  E4 control call, ACE budget, locked control code, metric aggregation, and
  checkpoint traces. Records and the preregistration were regenerated.
- [x] Third review repair: unreadable files block completion, a runtime probe
  separates runtime faults from bad files, empty metric summaries, the budget
  reservation charge, and a failed scorer release. The preregistration was
  regenerated.
- [ ] The reviewing agent reviews and merges the revised preregistration.
- [ ] Owner runs the pack notebook and Suno protocol after that merge.
- [ ] Wait for E5 v2 to be pushed and registered before GPU scoring.
- [ ] The reviewing agent applies S5 using the private result package.
- [ ] H1 Stage 0 through Stage 5 remain future work, subject to the gates below.

The first implementation is commit 59a8365 in PR #1. The current task repairs
that implementation before external generation. Append commits on the same
branch; do not rewrite history, push to main, or merge the PR. The owner has
authorized setting the required repository-local submission identity. Verify
that identity before committing; keep identity details out of source files.
This task does not run Suno, GPU generation, separation, Whisper, training,
or H1 gates.

## Surprises & Discoveries

- The task started with an untracked E7 draft. Useful work must be preserved and
  audited. Existing draft files are not evidence that a GPU run succeeded.
- The initial base and fetched `origin/main` are the base commit in Progress.
  The first implementation was committed later as 59a8365.
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
- 2026-10-27: Review correction: Stage S controls use the actual E4 arm
  dictionaries. The earlier custom text2music hook did not implement those
  controls. H1 route A remains text2music with its own context hook.
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
  deviation. The first version left ace.py out because it still needed E5 v2.
  That reason no longer holds; see the second review decisions below.
- 2026-10-27: Synthetic detection AUC means ROC AUC over clean and corrupted
  clips. Bootstrap whole clean/error item pairs. Report matched-pair win rate
  separately. Start, speed and transposition recovery are diagnostics only;
  they do not add a gate beyond the registered NPA ceiling.
- 2026-10-27: Use the upstream E5 seed helper for GPU generation only after
  checking equality with the frozen local formula. Intermediate result JSON
  files also use write_result and clean provenance.

- 2026-10-27: Review correction: the two pinned checkpoint Python files differ
  from the pinned source by design. The earlier mismatch rejection blocked all
  GPU runs. Materialize the snapshot, verify its original hashes, then copy
  only the two approved files from the pinned source before loading. Record
  both original and replacement hashes; never modify the shared cache.

### Review repair decisions, before external generation

On a full runtime loss during an ACE call, exact active time cannot be recovered.
Persist a reservation of the remaining allowance before each call. Normal
completion replaces it with measured time; an unresolved reservation consumes
the remaining allowance on restart. Record this conservative charge separately
from measured time. Recover completed output audio before checking the budget
or loading the model. Hash saved guides before reuse as well as output clips.

The owner explicitly authorizes a new preregistration for this unrun study.
Keep the old version in Git history; do not rewrite that history. The repaired
registration gets a fresh UTC time and hashes. This pre-generation repair is
not a silent change to an already executed study. After the new registration
is merged, changes to its locked files require a reviewed deviation addendum.

1. Score the complete synthetic ruler first, then Suno, then ACE controls.
   Scoring has no wall-time stop. Save each completed audio-hash result and
   generated artifact on Drive under the same registration identity. Resuming
   reuses them. Record actual stage times. Use a spawn-based process pool for
   all twelve reference searches and test exact serial/parallel equality.
   ACE generation alone first had 432 seconds: 96 draws times the previous
   run's upper warm-call estimate of 3 seconds, times 1.5. The second review
   raised it to 925 seconds for the longer Stage S clips. Count active generation
   calls across resumes, not installation, model loading, or scoring. An ACE
   budget stop is reported separately and cannot cancel a complete Suno/ruler
   decision. The earlier unmeasured 5400-second total stop is removed.
2. Stage S uses `stage_s_lib.py`, `stage_s_setup.py`, `stage_s_run.sh`, and
   `requirements-stage-s.txt`. Lock these, `suno.py`, `pack.py`,
   `registration.py`, both Stage S notebooks, and the read-only measurement
   dependencies listed in `registration.CODE_FILES`. H1 uses separate future
   files such as `h1_lib.py`, `run_h1.sh`, and `requirements-h1.txt`.
3. Download the exact weight revision with `local_dir`, using real files.
   Before replacing checkpoint code, verify the original SHA256 values:
   `modeling_acestep_v15_turbo.py` is
   `c1ab0dd547124fee7ada449b2b86eae8201dc7d15889932643bbb67e3c982444`;
   `configuration_acestep_v15.py` is
   `b89870c5c7a7ce060eb0bcdbb5ffc86b0b1a324ca325a26be552ea1b42496dc5`.
   Copy only those files from the pinned `acestep/models/turbo/` source.
   Then require an empty mismatch list. Stop on an unexpected original hash
   or source revision. A resumed setup verifies the recorded replacement too.
4. Accept Python 3.12 and 3.13. Tell the owner to use the default Colab runtime;
   do not require a manual Python version change.
5. Check nonzero size and decodability before GPU work. Record unreadable
   inputs separately and request a new download of the same song, never a
   new generation. Score decodable short clips normally. Record per-Suno
   scoring failures and continue with the next file; preserve completed rows.
6. Read the actual four arm dictionaries from E4 at GPU execution time.
   Forward task, guide source, cover strength, and cover noise through
   `dacapo.ace.generate`. Keep the registered shared captions, seeds, BPM,
   keyscale, and duration. The guided Stage S arms use `cover-nofsq`.
7. Replace number words and easily varied spellings in the supplied lyrics
   before registration, while keeping one syllable per note. Freeze a small
   equivalence table for number forms and spellings such as gray/grey and
   till/til. Regenerate the melody/lyric source hashes.
8. Freeze ASR waveform preparation, resampling, input dtype, model dtype,
   pipeline parameters, language, task, and transformers version. Assert the
   active ASR configuration rather than relying on changing library defaults.
9. After Stage 0c, replace only the selected F0 evaluator for rescoring the
   same private inputs. Keep alignment, ASR, lyrics, stems, sampling budget,
   thresholds, and aggregation fixed. Recheck the synthetic-ruler NPA ceiling
   of at least 0.95. The same route-level NPA 0.90 and WR 0.60 rule applies.
   A positive rescore triggers the stop rule; inadequate calibration remains
   pending. Record the evaluator and a reviewed addendum. Any changed rule may
   only make a positive conclusion easier, not harder.
10. The recording table supplies BPM, key, and the shared style, plus fields
    for actual entered values. Fill tempo/key controls when available. Define
    Drive locations, full file names, and route availability values in the
    protocol. Accept extension case variants without choosing between files.
11. Call the silence-penalized relaxed objective DTW cost. Separately report
    voiced aligned-frame absolute pitch deviation and the unvoiced-frame
    fraction. A compatibility `deviation_semitones` field denotes penalized
    cost, never a pure voiced pitch error. No cross-study numeric comparison.
12. Retain the current Stage S voice measure and label it as its own
    definition. Vocal energy is divided by mixture energy over all channels;
    active RMS uses all samples and channels in each 50 ms frame. This is not
    an assertion of equivalence to the E6 stem-sum/mono algorithm.
13. Write the public "Stage S was evaluated" statement only when the relevant
    measurement is complete. Otherwise write "the run did not complete".
    Keep every Suno score, route outcome, and decision private.

Small choices are made here and work continues. Do not ask the owner to resolve
routine details. Stop only for the explicit stop conditions in this plan.

### Second review decisions, before external generation

The owner has not generated anything on Suno, so the preregistration was
written again with a new UTC time and new hashes. These rules replace the
matching parts above.

- 2026-10-27: The scorer is built once before the ruler and Suno passes,
  outside the per-clip handler, whenever a clip has no score checkpoint. A
  model download, version check or CUDA setup fault ends the run as incomplete.
- 2026-10-27: Runtime fault classes are BrokenProcessPool, MemoryError, torch
  out-of-memory and accelerator errors, and messages that name CUDA, CUBLAS,
  cuDNN or out of memory. Such a fault on a Suno clip, or three consecutive
  Suno failures of one exception type, starts a runtime probe. The run releases
  the scorer, builds a new one, and scores the first ruler clip again without
  saving it. If the probe raises or its strict NPA differs from the saved score
  by more than 0.1, the run ends incomplete and the faulting clip is not a clip
  failure. Otherwise the faulting clips are clip failures and the pass
  continues, so a file that always fails can reach the exclusion rule.
- 2026-10-27: The run is complete only when every Suno inventory file has a
  score, apart from registered exclusions, and at least one decodable Suno file
  exists. A file that is unreadable at inventory is pending, like a failed
  clip. This includes a take that was scored earlier and is unreadable now;
  bytes identical to an earlier scored file keep that score. An open failure or
  unreadable file gives the status `scoring_failures_pending`, the incomplete
  public statement, and exit code 3. The owner reruns the same notebook and
  folder. A clip that is unreadable, or fails with the same exception type,
  with the same audio hash in three finished passes becomes a listed
  exclusion. A new file hash starts a new count. Failures in a run that
  stopped early do not count.
- 2026-10-27: Results are written right after the Suno pass and again after the
  controls, so a kill in the control stage keeps the finished decision.
- 2026-10-27: Lyrics avoid split compounds that a transcriber joins. "sun rise"
  became "sun climb" and "Some day" became "Each day". A test checks adjacent
  word pairs against a list of common closed compounds.
- 2026-10-27: The control call no longer passes a time signature. E4 left it
  at the empty default, and the registration now says so.
- 2026-10-27: The ACE generation budget is 925 seconds. The warm part is
  96 calls times 3 seconds scaled by 26.09 over 14 seconds (the longest Stage S
  control over the E4 clip length) times 1.5, which is 805 seconds. One 120-second
  cold-call allowance is added. Each call has a 120-second limit. A lost runtime
  is charged that bounded call allowance, not the whole remainder. The alarm
  sets a flag because ACE catches broad exceptions, and the budget stop is
  raised after the call whenever the flag is set. Saved controls are scored
  even when generation stops early.
- 2026-10-27: `registration.CODE_FILES` now also covers `dacapo/__init__.py`,
  `dacapo/ace.py`, `dacapo/seeds.py`, `experiments/e4_sing/run_sing.py`, and
  `experiments/e4_sing/make_items.py`. The control path imports all of them.
- 2026-10-27: Each route metric uses the complete item pairs where both takes
  have a value, and reports that item count. Strict NPA and WR are always
  numeric, so the decision still uses every complete pair.
- 2026-10-27: A metric with no complete pair that has both values reports a
  null median, zero items, a null interval and no item means.
- 2026-10-27: Each generation call adds only its own measured time. An
  unresolved reservation is charged once and kept apart from measured time.
- 2026-10-27: An unreadable score checkpoint is set aside, logged, and scored
  again. A failure entry is moved to a resolved list when its clip has a score.
  When a scored file's audio hash changes, the record lists both hashes and both
  headline scores. Suno entries stay in the private record.

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

The brief cites the reviewing agent's source review dated 2026-10-27. These are context,
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

The repository is `https://github.com/ninondev/DaCapo`. E5 is owned by the reviewing agent.
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
This repair delivery requires the complete test suite to pass with no skips.

Cloud work is CPU implementation, tests, and documentation. All model generation,
vocal separation, Whisper, and training are written for the owner's GPU runtime.
The Stage S notebooks call `stage_s_run.sh <stage>`. H1 will use its own
`run_h1.sh`, code, and requirements. Never extend or repurpose a locked Stage S
file for H1. Keep GPU operations callable from scripts so later work can also
use a 12 GB RTX 4080 Laptop if Colab is insufficient.

PR #1 is open. Append the repair commit and give the PR to the reviewing
agent. The owner does not merge it. Do not use a main-branch Colab link before
that notebook is merged. The link form is:
`https://colab.research.google.com/github/ninondev/DaCapo/blob/main/experiments/e7_melody_adapter/<name>.ipynb`.

GPU notebooks emit result zips. The owner gives them to the reviewing agent,
who checks them and commits the public `records/*.json`. Suno records follow the private
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

Do not put Basic Pitch as a dependency line in `requirements-stage-s.txt`; that can
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

**Separation and voice presence.** Use htdemucs vocals. Stage S has its own
registered voice-present definition: summed vocal-stem energy divided by summed
mixture energy over all channels must be at least 0.2. Nonoverlapping 50 ms
frames with RMS at least -40 dBFS must total at least 1.0 second. Compute RMS
over all samples and channels of each frame, including a final partial frame.
The numeric thresholds were borrowed from E5 v2, but this is not its E6
stem-sum energy and mono-average RMS algorithm. Do not use the E6 `PEAK_MIN`
condition for external audio; the original generation peak is unavailable.
H1 must register its chosen definition separately after E5 v2 is available.

**Lyrics.** WR is reference-word recall from fixed-revision Whisper large-v3 on
separated vocals. Refuse scoring without CUDA. Do not substitute a smaller ASR
model. Stage S freezes its own matching function and small word-equivalence
table. Normalize digits 0-12 to their English words, gray to grey, and til to
till in both lyrics and transcripts. Keep LCS word matching and multiplicity.
Freeze transformers 4.57.6, Whisper float16, and float32 waveform input. Average
vocal channels to mono and use librosa resampling to 16 kHz. The ASR pipeline
uses 30-second chunks, 5-second stride, batch size 1, and CUDA device 0.
Generation kwargs set language to English and task to transcribe. Assert these
settings at runtime; do not replace the model or alter defaults silently.
H1 must instead use `dacapo.words.lyric_words`, `norm_word`, and `match_words`; record that transition
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
Stage S wrapper is in `stage_s_lib.py`; do not alter the library's fixed-lag
NPA function. H1 wrappers belong in separate files.

**Relaxed pitch.** Draw reference MIDI pitch at 10 ms and remove reference rest
frames. Convert output vocal F0 to MIDI pitch at 10 ms and keep its unvoiced
frames. Their fixed DTW cost is 3 semitones. Run subsequence DTW for each global
transpose -12 through +12 and take the smallest normalized cost. Only steps
(1,1), (1,2), and (2,1) are allowed. The matched duration is 0.5-2 times the
non-rest reference duration, to frame quantization. Cost is absolute semitone
error without octave folding, normalized by reference frame count.

Report the normalized DTW cost, which includes the fixed silence penalty.
Also report the mean absolute deviation on voiced aligned frames and the
unvoiced aligned-frame fraction. Each reference frame has unit total weight,
shared between its assigned output frames; use that same weighting for both
quantities. Pure voiced deviation is null when no voiced assignment exists. Name these separately so the cost is never
presented as pure voiced pitch deviation. Report the share of reference notes
whose matched output median pitch is within 50 cents; missing notes are wrong.
Report octave-folded diagnostics separately. Perform the
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
one-syllable words, with one syllable per note. Avoid numeric forms and easily
varied spellings in the source lyrics. The required provenance says the lyrics
were written by the agents. Do not use the owner's future test set.

`melodies_suno.json` stores pitch, start and duration in beats, BPM, key, and
lyrics. `records/suno_pack.json` records its SHA256, the lyrics SHA256, and
render settings including program, sample rate, and soundfont SHA256. It does
not contain rendered audio or Suno output.

`colab_e7_pack.ipynb` runs on the default Colab CPU runtime after merge to main.
Python 3.12 and 3.13 are supported. It renders piano program
0 and Voice Oohs program 53 at 48 kHz, writes MIDI with `Melody.to_midi`, and
adds lyric text and the CSV record sheet. It emits `e7_suno_upload.zip` and
`suno_upload_hashes.json`, containing each upload file's SHA256. The owner
downloads both and gives the hash record to the reviewing agent.

### S2: owner operation

Keep the English protocol in `suno_protocol.md` and give the instructions in
Chinese in the owner reply. Wait for the reviewing agent to confirm the
preregistration is merged before any Suno generation.

- R1: use a MIDI prompt in Suno Studio, the supplied lyrics, and the exact style
  `pop, full band, female vocal`.
- R2: upload the piano WAV with the available cover/reference-audio feature,
  using the same lyrics and style.
- R3: upload the Voice Oohs WAV, otherwise using R2's settings.

Current menu names are not known. Skip any unavailable route and record the
actual options shown, displayed model version, date, and credits used. Request
once per item and available route. Keep both returned songs; never pick one.
Name them `S_<melody_id>_<route>_<a|b>.mp3`; the extension is case-insensitive.
Use `/content/drive/MyDrive/e7-suno/audio` for audio and
`/content/drive/MyDrive/e7-suno/suno_recording_table.csv` for the filled sheet.
The sheet supplies `bpm`, `key`, and `style`. Fill tempo and key controls when
available, then record the actual entered values and settings. Record
`route_available` as `yes`, `no`, or `unknown`. Keep result checkpoints in the
Drive output directory specified by the notebook. Only upload these twelve
study melodies. Do not upload owner-held-out melodies to any website or app before Stage 4. The owner
must read Suno's terms and confirm whether evaluation results may be published.

### S3: preregistration and decision

Before external generation, write `prereg_suno.json` with a UTC timestamp,
melody/lyric hashes, all Stage S measurement definitions and parameters,
routes, controls, thresholds, decision rule, analysis, and scoring-code hashes.
`registration.CODE_FILES` is the exact locked runtime dependency set. The
Stage S files are separate from future H1 code. The scoring notebook checks
all registered hashes before scoring, and a mismatch stops the run. After
`prereg_suno.json` is merged, any change to a file in CODE_FILES requires a
reviewed addendum marked `deviation`. Retain the original registration and
report original-version results when any exist. The owner has explicitly
authorized replacing the initial unrun registration for this repair, before
external generation; the original remains in Git history.

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
stop and tell the owner to wait for the reviewing agent. Use the default Colab
GPU runtime with Python 3.12 or 3.13. Import E4 `arms(['clarinet', 'oohs'])`
inside GPU-only code and assert the four required dictionaries match the
registration. Ordinary CPU tests never import the real model through E4.

Before any GPU model work, read the Drive audio and completed record sheet.
Validate names, quantities, nonzero file size, and decodability. List missing
and unreadable songs separately; exclude only unreadable inputs from clips.
Ask for a new download of the same song, never a new generation. Keep decodable
short clips. Catch each Suno scoring failure, save its reason, and continue.

The execution order is complete ruler, Suno scoring, then ACE controls.
Generate ACE-Step controls for the same twelve melodies with `noguide`,
`ctx_oohs_k3`, `ctx_oohs_k6`, and `ctx_clarinet_k4`. Use the exact shared style,
lyrics, BPM, and keyscale. Preregister duration as melody length plus two bars.
Forward each actual E4 arm's task, source guide, cover strength, and cover noise
through `dacapo.ace.generate`. No-guide uses text2music; guided arms use
cover-nofsq with cover strengths 3/8, 6/8, and 4/8 respectively. Do not replace
these controls with a different context-hook implementation.
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
or separated vocals. Persist each finished clip and generated control asset
on Drive with its audio SHA256 and registration identity. On a rerun, verify
and reuse matching completed work. Do not regenerate or rescore completed
clips. A different preregistration must not consume those checkpoint results.

Scoring has no wall-time limit. Record actual times for each stage. The ACE
generation allowance is 925 seconds of active generation across resumes.
The basis is in the second review decisions. Setup, loading,
rendering, and scoring time are reported separately. If controls exhaust their
allowance, preserve their results and report the incomplete control stage.
That cannot invalidate an otherwise complete ruler and Suno decision.

### S5: private review and continuation

The reviewing agent applies the preregistration to the results. Before the
owner confirms the terms permit publication, a complete evaluation uses only:
"Stage S was evaluated under prereg_suno.json. The conclusion is in the private
record." An incomplete evaluation instead says "the run did not complete".
Do not put Suno scores, route outcomes, accuracy conclusions, or threshold
comparisons in public PLAN, LOG, records, or the PR. Suno result
records belong only in the owner's private repository. Do not use the quoted
statement before an evaluation has actually happened.

If synthetic-ruler median NPA is below 0.95, the outcome is ruler insufficient,
pending. Stop and report; do not start H1. If Suno meets the positive rule, kill
rule 3 is triggered: stop for the owner to choose H7 (one-note edits), H2 (motif
development), or H1 with an open-source parity goal. The last option needs a
kill-rule-3 deviation in `prereg_h1.json`. A measured negative is provisional.
H1 may then proceed, but after Stage 0c picks a validated ruler, rescore the same
private Suno files under `rescore_after_0c`. Replace only the selected F0
estimator, keeping ASR, separation, audio hashes, all search parameters,
thresholds, and aggregation fixed. Record the evaluator and its revision in a
reviewed addendum. Require the same synthetic-ruler NPA ceiling of 0.95 before
using the same 0.90 NPA and 0.60 WR decision. Insufficient calibration remains
pending; a positive result stops in the same way. Never make the rule stricter
as part of rescoring. If a rule changes, it may only make a positive decision
easier, with the reason recorded.

## H1 roadmap and gates

At each phase end, use `write_result` for `records/stageN_*.json` through the
ignored output workflow. Update this plan and the E7 log, including failures.
Run precommit checks. Give the owner a Chinese summary of at most ten lines:
work done, key numbers, gate status, next step, and required owner action.
Then add the learning material described below. Total GPU budget is about
32 hours. Stop any phase beyond 1.5 times its stated budget.

Before any Stage 0 GPU work, main must contain E5 v2 `addendum_2.json` and the
registration check must pass. Until then do independent CPU preparation and
report that the reviewing agent's E5 v2 push and registration are pending.

### Stage 0: cheap checks, at most 3 GPU hours

- 0a: if networking permits, spend at most 30 minutes on relevant work after
  2026-10-27, checking kill rule 3. If not, log that it was not done and relies
  on the reviewing agent's 2026-10-27 review.
- Before 0b, write `prereg_h1_stage01.json` and have the reviewing agent merge it. Specify
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
kill rules, claim levels, and listening design. Give it to the reviewing agent for merge.
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

The owner supplies 36 melodies to the reviewing agent privately. Before Stage 3 training,
the reviewing agent runs the fixed-seed seal script: six become dev JSON under
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

Do not start release automatically. The owner and the reviewing agent decide license and
release method first. If authorized, provide a 12 GB inference script, a
Gradio demo (optionally on Hugging Face Spaces), weights, and clear usage.
Select demo clips by rules or seeds fixed in advance, never by listening for
the best ones.

## ACE-Step implementation notes

Use the existing `dacapo.symbolic` Note and Melody APIs for JSON-to-MIDI and
reference notes, plus `dacapo.render`, `dacapo.analysis`, `dacapo.metrics`, and
`dacapo.provenance.write_result`. Import `dacapo.ace` only inside functions that
run with the real model. Its top-level code reads `ACESTEP_REPO` and imports
ACE-Step. Do not import it at `stage_s_lib.py` top level or in ordinary CPU
tests. Import E4 `run_sing.py` and its `arms()` only inside GPU execution
functions or the GPU notebook, since that module imports the real model.

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
Do not patch ACE-Step source or model files for H1. The sole approved Stage S
setup operation is the verified two-file code copy described above, into the
materialized local weight snapshot. It is not an adapter or training change.

Check these four H1 pitfalls before interpreting a failure:

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
- An H1 phase exceeds 1.5 times its GPU budget. Stage S scoring has no time
  stop; its separately reported ACE generation stage has the 925-second cap.
- Kill rule 3 is triggered, or a license bans the needed research use.
- The next action needs the owner: Suno generation, Colab execution, new
  melodies, H1 kill-rule-1 agreement, listening, or the day-30 decision.

At this task's handoff, explain in Chinese what changed and confirm all changes
are under E7. State pytest passed/skipped/failed counts and the precommit result.
Tell the owner that the existing PR has a new commit for review. Wait for
preregistration merge before running the pack notebook and Suno protocol.
The scoring notebook also waits for E5 v2. Include this exact opening for future tasks:

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

The initial implementation had 64 passed, 0 skipped, and 0 failed CPU tests.
It verified 49 upload assets and stayed inside the E7 boundary. Independent
review nevertheless found runtime and scientific defects. Those counts are
historical; they do not validate this repair or the unrun GPU pipeline.

The thirteen review repairs passed integrated validation: **92 passed,
0 skipped, 0 failed**, in 24.62 seconds. Source records regenerate byte for
byte. The repaired registration was frozen at `2026-10-27T17:56:14Z`, covers
fourteen runtime/dependency files, and passes its hash/configuration check.
The precommit guard passes for 25 text paths, all under E7 and below 1 MB.
The repaired implementation is an additional commit for PR #1; history is
preserved. Runtime tests cover interruption, hard reset, file failures, E4
forwarding, and ASR settings without claiming a real model measurement.
No GPU, Suno, or H1 measurement was run during this repair. E5 v2 is still
required before GPU scoring. External generation must wait for review and
merge of the repaired registration.

The second review found that one scoring fault could still produce a complete
result with a verdict. The repair passed **112 passed, 0 skipped, 0 failed**
E7 tests. The root suite gave 204 passed, 3 skipped (missing ACE-Step source
and tokenizer, outside E7), and 0 failed. The new registration was frozen at
`2026-10-27T19:21:45Z` and covers nineteen files. The E5 check and the E7
precommit guard pass. No GPU, Suno, or H1 measurement ran.

## Ideas not done

H1 training and later stages wait for the Stage S decision. Future ideas such
as humming input, Suno one-note edits, reward training, and changing models are
not authorized by this first task. Keep them here until separately chosen.
Do not start H7, H2, deployment, a demo, or release because it seems useful.
