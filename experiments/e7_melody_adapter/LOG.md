# E7 work log

## 2026-10-27: Initial Stage S implementation (historical)

The first task prepares a fixed-melody Suno measurement and matching ACE-Step
controls. It does not generate Suno songs or run GPU models in Cloud.
The work is restricted to `experiments/e7_melody_adapter/`.

The base commit is `3c0c259d53251bd1992f1330e74f93776d20a0af`.
The task saved its baseline in `.git/e7_base`. Fetched `origin/main` is the same
commit. The initial workspace already contained an untracked E7 draft.
The draft was audited and completed instead of discarded.

The following E5 v2 files are absent at the base commit:
`dacapo/seeds.py`, `dacapo/words.py`, `experiments/e6_clock/readout.py`,
`experiments/e5_motif/register.py`, and
`experiments/e5_motif/registration/addendum_2.json`.
The current `dacapo.ace.generate` also lacks `rng_seed`.
These findings block GPU scoring. They do not block CPU preparation.

The Cloud image also differs from the brief: it initially lacks the FluidSynth
executable, the expected system soundfont, and the read-only ACE-Step source
copy. Python dependencies are in `.venv`. System apt installation is unavailable.
The exact FluidR3 font and CLI were obtained from Debian packages into an
ignored E7 output prefix. No alternative soundfont was substituted.

The delivery contains twelve original eight-bar melodies and lyric
source records, a CPU upload-pack notebook, an English Suno protocol, a frozen
Stage S registration, strict and relaxed pitch metrics, lyric matching, chance
controls, GPU runtime scripts, and separate public/private numeric exports.
Detailed definitions and every later H1 gate are in `PLAN.md`.

Stage S uses its own frozen lyric matcher. No switch to `dacapo.words` has
happened. H1 must record that switch here after E5 v2 is available.

The first implementation used E4 arm names for a custom text2music context
hook. Review found that this did not implement the E4 controls. The repair
below replaces that path with the actual E4 task, source, strength, and noise
settings. This old implementation is not the current Stage S protocol.

### Validation

Initial CPU pytest: **64 passed, 0 skipped, 0 failed**, in 19.62 seconds.
The suite includes 22 scoring tests, 9 source/upload-pack tests, 11 boundary
checks, 7 notebook/registration tests, and 15 runtime orchestration tests.
All 49 upload assets were rendered and checked, including 24 WAV files at
48 kHz. Source melody and metadata files regenerate byte for byte.
Both notebooks regenerate exactly and contain no execution output.

Precommit passes: 24 text files, all under E7, each below 1 MB. No protected
tracked file changed. The baseline is the commit above. E5 register.py is
absent, so its check is unavailable and GPU work stays blocked.
`registration --check` confirms all frozen code, dependency and input hashes.

Tests used `.venv/bin/python`, with the local FluidSynth prefix on PATH and
DACAPO_SF2 pointing to its exact FluidR3_GM.sf2. All runtime test artifacts and
caches stay under ignored outputs. Missing required render or analysis tools
are test failures, never skips.
No GPU runtime, Suno service, separated-vocal ruler ceiling, ASR result, or H1
gate has been measured in this task. No performance claim follows from the
CPU implementation tests.

### Failures

The existing main branch cannot run the required GPU protocol because the E5
v2 files and independent RNG seed API are absent. The implementation must
fail closed at that guard, with instructions to wait for the reviewing agent. No protected
file is changed to bypass it.

The precommit-guard tests first produced ten passes and one fixture error.
Pytest derived an over-1-MB parameter ID from the oversized-file case, which
made the temporary test path invalid. Short explicit parameter IDs fixed the
fixture. The targeted rerun passed all eleven guard tests. This was a test
fixture failure, not a repository boundary failure.

The initial strict note-F1 implementation clipped a note that began before
the chosen passage. That could invent an onset at the passage boundary. The
fix uses absolute reference and transcription times with the chosen global
alignment. A regression test checks that a carried-in note earns no onset.

The initial ruler summary called matched-pair win probability AUC. It now
computes standard ROC AUC and resamples complete clean/error item pairs for
its interval. Matched-pair win probability is a separate diagnostic. A test
uses pairs where these two quantities differ. These fixes precede the first
preregistration; no study output was available when they were made.

The initial notebook setup mixed incompatible torch and torchaudio versions.
It now pins the matching 2.10.0 family and keeps Basic Pitch's no-dependency
installation separate. The disposable GPU setup has not run here.

The first loader guard rejected every source/checkpoint code mismatch. Review
confirmed that the two pinned revisions intentionally differ in two Python
files, so this guard made the GPU notebook unusable. The approved repair
verifies the original hashes and then performs the native two-file source
sync into a materialized local snapshot, with provenance and resume checks.

A self-check construction test initially compared floating-point times with
exact equality. Its assertion now uses a numeric tolerance. This was a test
assertion issue; the generated timing arithmetic did not need a change.

Record any later implementation failures here, including their trigger, fix,
and rerun. Do not record unrun GPU behavior as a
successful result. No study gate has failed or passed yet.

### Initial handoff

The first implementation was committed as 59a8365 and opened as PR #1. The
review below occurs before any external generation or GPU measurement. The
first test count does not establish that the owner-run pipeline was usable.

## 2026-10-27: Pre-generation review repair

The owner authorized all thirteen review fixes and a fresh preregistration
before generating study outputs. Continue on the existing PR branch, append
commits, and preserve its history. Do not push to main or merge. Repository
identity was initially mismatched; no repair commit was made with that identity.
The owner then authorized setting the specified local identity in the Cloud
checkout. Verify it again before committing. Do not put identity details or
private information in source files, commit messages, or the PR description.

The repaired execution and registration contract is:

1. Finish the synthetic ruler, then score Suno, then generate and score ACE
   controls. Persist completed work on Drive by registration and audio hash.
   Resume without repeating finished generation or scoring. Search the twelve
   references with a spawn-based process pool and require exact serial parity.
   Remove the unmeasured 5400-second scoring stop. Record actual stage times.
   ACE generation alone has 432 active seconds, from 96 draws times the earlier
   measured upper warm-call estimate of 3 seconds times 1.5. A control timeout
   is reported without cancelling an already complete Suno/ruler decision.
2. Move Stage S execution into stage_s_lib.py, stage_s_setup.py, stage_s_run.sh,
   and requirements-stage-s.txt. Freeze only the Stage S runtime dependency
   set. Future H1 work uses separate files, so it cannot break Stage S hashes.
3. Materialize the pinned weight revision with local_dir. Verify both original
   checkpoint Python hashes from the repair brief, copy the two approved files
   from the pinned source, and require no mismatch afterwards. Record old and
   new hashes and verify the same state when setup resumes. Never edit a
   shared download-cache target or silently accept a different original file.
4. Accept Python 3.12 and 3.13 and use the default Colab runtime.
5. Check audio size and decoding before GPU work. List unreadable files and
   ask for the same song to be downloaded again. Preserve decodable short
   clips. Record each Suno scoring failure and continue with remaining files.
6. Obtain the actual four E4 arm dictionaries inside GPU execution code and
   forward their task, source guide, cover strength, and noise. Guided controls
   use cover-nofsq. The H1 text2music plan remains separate.
7. Remove number forms and easily varied spellings from source lyrics before
   re-registration. Freeze a small lexical equivalence table and regenerate
   all changed lyric and melody-source hashes.
8. Register and assert the complete ASR preparation, dtype, pipeline, language,
   task, and transformers-version settings.
9. Register Stage 0c rescoring as a change to the F0 evaluator only. Keep the
   same audio, separation, ASR, searches, aggregation, and thresholds. Require
   the same 0.95 synthetic-ruler ceiling before applying the same route-level
   NPA 0.90 and WR 0.60 rule. Any allowed rule change can only make a positive
   conclusion easier. Record the chosen evaluator in a reviewed addendum.
10. Supply BPM, key, and style in the recording table and record actual inputs.
    Document Drive paths, complete names, route availability values, and
    case-insensitive file extensions in the protocol.
11. Separate the silence-penalized DTW objective from voiced aligned-frame
    pitch deviation and unvoiced-frame fraction. Do not present penalized
    cost as a pure pitch-error estimate.
12. Explicitly label the retained voice calculation as Stage S-specific:
    vocal/mixture energy over all channels and frame RMS over all samples and
    channels. It is not the E6 stem-sum/mono calculation.
13. Use the public evaluated statement only for a complete measurement. Use
    "the run did not complete" otherwise. Public files still contain no Suno
    score, route outcome, or threshold comparison.

The owner also replaced named assistant attribution with "the agents" or
"the reviewing agent". Repository prose, commit messages, and PR text contain
no assistant product names or private identity details. Exact experimental
model/tool identifiers and permitted paper citations remain intact.

### Repair validation

Final repair pytest: **92 passed, 0 skipped, 0 failed**, in 24.62 seconds.
This includes 26 scoring, 11 source/upload-pack, 11 boundary, 21 delivery,
13 runtime, and 10 ASR tests. Serial/process-pool parity covers all twelve
references. Runtime tests cover interruptions, persisted hard-reset budgets,
unchanged score reuse, per-file failures, E4 forwarding, model-code alignment,
supported Python versions, fixed ASR settings, and public/private boundaries.
The upload test renders and verifies all 49 assets and all 72 recording rows.
Source records regenerate byte for byte; both notebooks regenerate exactly.
The new preregistration timestamp is `2026-10-27T17:56:14Z`. Its fourteen code
hashes and complete definitions pass `registration --check`. Precommit passes:
25 text paths, all under E7, each below 1 MB, with no protected-file changes.
No real GPU generation, source separation, Whisper, or Suno measurement has
run during this repair. Mocked runtime checks do not establish model quality.

### Failures and limits

Independent review reproduced the problems above despite 64 initial passing
tests. The total time budget could end before Suno finished, generic file
locks blocked future H1 work, and the strict checkpoint-code guard rejected
the supplied pinned revisions. The custom controls also differed from their
E4 names. These were implementation and measurement-design defects, not
negative findings about the measured systems.

The second review found three resume edges: an orphaned completed output was
checked after the budget, saved guides lacked hashes, and a hard runtime reset
could lose an active call's budget charge. Recovery now precedes model loading
and budget checks, guides use the asset hash store, and persisted in-flight
reservations conservatively consume uncertain time separately from measured
time. Tests cover output recovery at exhausted capacity and modified guides.
An ASR audit also found that single-sample normalization produced NaN through
the sample standard deviation. That case now uses a finite variance; a real
one-sample WAV/ffmpeg/torch regression passes. Normal longer clips retain the
existing normalization. The full rerun above includes all these corrections.
E5 v2 remains absent at the checked main commit; its seed API and registration
are still required before the owner GPU run.

### Next handoff

The repair is delivered as an additional commit for existing PR #1, with no
history rewrite or main-branch change. The next step is review of that PR.
Do not start external generation until the repaired preregistration is
reviewed and merged. The owner then runs the
CPU upload notebook and gives its hash JSON to the reviewing agent.
The GPU notebook waits for E5 v2 and its passing registration check. Preserve
Drive checkpoints and rerun the same registered notebook after interruption.
No Suno score or conclusion becomes public until the owner confirms the terms
permit publication. No Stage S or H1 result has been decided in this task.

## 2026-10-27: Second pre-generation review repair

A second review ran the real `run()`, checkpoint and archive code with a fake
scorer. It found that a scoring fault could still end with `complete` set and a
verdict. Example: on a resumed runtime the scorer was first built inside the
per-clip handler. One failed model download was then logged as 24 Suno clip
failures, and the run still said "Stage S was evaluated". A CUDA fault partway
through the pass did the same, and a later retry of the same files could flip
the verdict. The owner has not generated anything on Suno yet, so the fixes
went into a new preregistration. The work stays under `experiments/e7_melody_adapter/`.

What changed:

1. Completion and runtime faults (`suno.run`, `suno.run_wide_fault`,
   `suno.decide`, `ResumeStore.count_failures`). The scorer is built once,
   before the ruler and Suno passes and outside the per-clip handler. A start
   fault ends the run as incomplete. CUDA errors, out-of-memory errors, a
   broken process pool, and three consecutive failures of one exception type
   also end the run; they are not stored as clip failures. After the Suno pass
   the run is complete only if every decodable clip has a score, apart from
   registered exclusions. Otherwise the decision is `scoring_failures_pending`,
   the public record uses the incomplete statement, the console asks for a
   rerun of the same notebook and folder, and the command exits with code 3.
   A clip that fails the same way in three finished passes becomes a listed
   exclusion. An empty input folder is also incomplete. Results are written
   right after the Suno pass and again at the end.
2. Lyrics (`pack.LYRICS`). "sun rise" became "sun climb" and "Some day" became
   "Each day", one syllable per note as before. The melody source, upload
   record and license record were regenerated with a new snapshot.
3. Controls (`suno.generate_controls`). The unregistered `timesignature="4"`
   argument is gone. E4 never passed it.
4. Budget (`suno.Budget`, `suno.saved_controls`). The alarm sets a flag, and the
   budget stop is raised after the call whenever the flag is set. Each call has
   a 120-second limit, and a lost runtime is charged only that bounded amount.
   The budget is now 925 seconds, rebased on the 26.09-second longest Stage S
   control. Saved controls are scored even when generation stops early.
5. Locked code (`registration.CODE_FILES`). Added `dacapo/__init__.py`,
   `dacapo/ace.py`, `dacapo/seeds.py`, and the two E4 files the controls use.
6. Aggregation (`suno.paired_summary`). Each metric uses the complete item pairs
   that have a value for it and reports the item count.
7. Checkpoints (`ResumeStore.score`, `ResumeStore.put_score`). An unreadable
   score file is set aside, logged and scored again. A stale failure is moved
   to a resolved list when a score exists. A scored file whose hash later
   changes leaves a `replaced_after_scoring` entry with both hashes and both
   headline scores. Suno entries stay private.

### Second repair validation

E7 pytest: **112 passed, 0 skipped, 0 failed**, in about 37 seconds. New tests
cover a scorer start fault on resume, a sticky CUDA fault, one failed clip,
consecutive failures, the three-pass exclusion, empty inputs, the exit code,
early publication with saved controls, the budget alarm flag, the bounded
reservation, aggregation with null values, unreadable and stale checkpoints,
replaced files, the locked control code, the budget arithmetic, and lyric
compounds. The old test that expected a complete run with one failed clip now
expects a pending run.

The root suite (`pytest -q -m "not slow"`) gave 204 passed, 3 skipped and
0 failed. The skips need the ACE-Step source or a tokenizer and are outside E7.
The new preregistration time is `2026-10-27T19:21:45Z`. It covers nineteen
files, and `registration --check` passes. The E5 `register.py --check` passes.
The precommit guard passes for 25 paths, all under E7.

### Second repair limits

No GPU generation, separation, Whisper, or Suno measurement ran. A file that
is unreadable during one inventory is still only listed as unreadable, so a
transient read error can drop an already scored take from that run. A missing
ruler asset still stops a resume, and the ruler stems are still written to
Drive. The new budget numbers are estimates from E4 timing, not Stage S
measurements.

## 2026-10-27: Third pre-generation review repair

A third review ran the real `run()` code with a fake scorer and found three
problems. First, a file that was unreadable at inventory did not block
completion. With three R3 songs still syncing in Drive, the run said
`complete`, gave `provisional_negative`, and wrote "Stage S was evaluated".
After the same songs were downloaded again, the verdict became
`already_accurate`. Second, a metric with no complete pair that had both values
crashed the archive step, so no record was written. Third, three adjacent
files that always failed tripped the consecutive-failure abort on every rerun,
so they never reached the exclusion rule and the run could never complete. The
owner has not generated anything on Suno, so the fixes went into a new
preregistration.

What changed:

1. Unreadable files (`suno.inventory`, `suno.run`, `suno.UnreadableAudio`).
   Each unreadable file is now a pending take with its own failure record. A
   take that was scored earlier and is unreadable now is pending as well; bytes
   identical to an earlier scored file keep that score. The status is
   `scoring_failures_pending`, the public record uses the incomplete statement,
   and the command exits with code 3. The same file hash unreadable in three
   finished passes becomes a listed exclusion, like a scoring failure.
2. Empty metric summaries (`suno.paired_summary`). A metric with no usable
   pair reports a null median, zero items, a null interval and no item means.
   This applies to Suno routes and ACE control arms.
3. Runtime probe (`suno.run`). A CUDA or memory fault on a Suno clip, or three
   consecutive failures of one exception type, no longer ends the run at once.
   The run releases the scorer, builds a new one, and scores the first ruler
   clip again without saving it. If the probe raises or its strict NPA differs
   from the saved score by more than 0.1, the run ends incomplete and the clip
   is not recorded as a failure. Otherwise the faulting clips are recorded as
   clip failures and the pass continues. A file that always fails, or always
   runs out of memory, can then reach the exclusion rule.

The review's test file also showed two smaller faults, fixed here as well:

4. Budget (`suno.Budget`, `ResumeStore.update_generation_time`). Each call now
   reports only its own measured time. Before, a lost call's reservation was
   added to measured time on every resume, so it was charged again each time.
5. Scorer release (`suno.run`). If releasing the scorer raises after a CUDA
   fault, the record is still written.

### Third repair validation

E7 pytest: **120 passed, 0 skipped, 0 failed**. New tests cover unreadable files
at inventory, a scored take that later becomes unreadable, an unreadable file
excluded after three passes, adjacent failing files that pass the probe and
become exclusions, an out-of-memory file, a probe outside the tolerance, a
metric with no usable pair for routes and controls, a failed scorer release,
and the reservation charge across resumes. Two older tests now model a runtime
that stays broken, so the probe fails as well.

The root suite (`pytest -q -m "not slow"`) gave 204 passed, 3 skipped and
0 failed. The skips need the ACE-Step source or a tokenizer and are outside E7.
The new preregistration time is `2026-10-27T19:43:25Z`. It covers nineteen
files, and `registration --check` passes. The E5 `register.py --check` passes.
The precommit guard passes for 25 paths, all under E7.

### Third repair limits

No GPU generation, separation, Whisper, or Suno measurement ran. The 0.1 probe
tolerance is a choice, not a measurement of GPU run-to-run variation. A
runtime that fails the probe every time still blocks completion, which is
intended. A missing file, unlike an unreadable one, is still allowed and only
listed.

## Upload pack rendered (2026-10-28)

The owner ran the CPU pack notebook on Colab (Python 3.13.18, FluidSynth 2.3.4) at commit f40132d with a clean checkout. Its hash record is in `records/suno_upload_hashes.json`. The same pack rendered again on the development machine from the same commit gave identical SHA256 values for all 49 files (12 MIDI files, 12 lyric files, 24 WAV files and the recording table), so the files sent to Suno are exactly the registered ones. No Suno song existed yet.
