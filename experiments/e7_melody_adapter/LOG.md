# E7 work log

## 2026-10-27: Stage S implementation

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

The E4 arm names define the controls, but the E7 wrapper uses text2music with
`is_covers=False`. Early-step guidance replaces only the 64 source channels
and preserves the observed live mask. This avoids the older E4 cover task
instruction while keeping the requested guide lengths.

### Validation

Final CPU pytest: **64 passed, 0 skipped, 0 failed**, in 19.62 seconds.
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
fail closed at that guard, with instructions to wait for Claude. No protected
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

Pinned upstream source review confirmed the bool chunk-mask path and the
text2music cover flag. It also found native automatic missing-weight downloads
and checkpoint Python sync. The runtime checks snapshot completeness and code
agreement before loading, so it cannot silently replace the registered model.

A self-check construction test initially compared floating-point times with
exact equality. Its assertion now uses a numeric tolerance. This was a test
assertion issue; the generated timing arithmetic did not need a change.

Record any later implementation failures here, including their trigger, fix,
and rerun. Do not record unrun GPU behavior as a
successful result. No study gate has failed or passed yet.

### Next handoff

After CPU checks pass, the owner creates a PR for Claude to review and merge.
Suno generation starts only after Claude confirms the preregistration is on
main. The owner then runs `colab_e7_pack.ipynb`, downloads both outputs, gives
the upload hashes to Claude, and follows `suno_protocol.md`.
`colab_e7_suno.ipynb` additionally waits for the E5 v2 merge and registration.
Once E5 v2 lands, first rerun tests on that main and verify local seed parity.
No Suno score or conclusion is public until the owner confirms the terms
permit publication. No Stage S decision has been made yet.
