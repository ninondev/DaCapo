# Stage S Suno protocol

Use only the twelve melodies in `melodies_suno.json`. Do not use any melody
from the owner's future held-out test set. Do not upload those test melodies
to Suno or any other website or app before Stage 4.

## Before generating

1. Create the implementation PR and give it to Claude for review and merge.
   Wait for Claude to confirm `prereg_suno.json` is merged into main. Do not
   generate study songs before that confirmation.
2. Read Suno's current user terms. Decide whether they permit public evaluation
   results. Until the owner confirms this, every Suno score and conclusion
   stays private. This protocol does not make a legal determination.
3. Open the merged
   [CPU pack notebook](https://colab.research.google.com/github/ninondev/DaCapo/blob/main/experiments/e7_melody_adapter/colab_e7_pack.ipynb).
   Use a Python 3.12 CPU runtime and run its cells in order.
4. Download `e7_suno_upload.zip` and `suno_upload_hashes.json`. Give the hash
   JSON to Claude. Extract the zip locally. It contains MIDI, piano WAV,
   Voice Oohs WAV, lyrics, and a recording table for all twelve melodies.

Use the files exactly as rendered. Do not retune the melodies, change their
lyrics, or select favorable passages. Keep the source hash record with the run.

## Three routes

The current Suno menu names are not known. Use the feature actually available
in the interface. Record its exact name and settings. If a route is unavailable,
skip it and record that fact; do not silently replace it with another method.

| Route | Melody input | Operation |
| --- | --- | --- |
| R1 | MIDI | In Suno Studio, use the MIDI as a generation prompt. |
| R2 | Piano WAV | Upload it using the available cover or reference-audio feature. |
| R3 | Voice Oohs WAV | Use the same type of operation as R2. |

For all three routes, paste that melody's supplied lyrics. Use the exact style:

```text
pop, full band, female vocal
```

Do not alter the style or lyrics between routes. Use the supplied melody tempo
and key when the interface permits them, and record any unavailable controls.

## Requests, downloads, and records

Request **once** for every melody and available route. Keep **both** songs
returned by that request. Do not choose the better song, request extra draws,
edit the audio, or replace a failed song without reporting it.

For each request, fill in the CSV record sheet. Record the actual menu option,
displayed model version, date, and credits spent. Also record route availability,
missing output, errors, and any setting that differs from this protocol.
Do not guess a model version that the interface does not show.

Download both audio files and name them:

```text
S_<melody_id>_<route>_a.mp3
S_<melody_id>_<route>_b.mp3
```

Use the melody IDs supplied in the pack and `R1`, `R2`, or `R3` exactly.
Assign `a` and `b` by returned order, not quality. Put all audio files in one
Google Drive folder and keep the completed CSV nearby. The scoring notebook
lets you set the audio folder, CSV path, and output directory.

A full run is twelve melodies times three routes times two songs, or 72 files.
Unavailable routes and missing songs are allowed and must be listed. A route
needs both songs on at least ten melodies to count as measured. No measured
route means not measured, not a negative finding.

## GPU scoring and review

Wait for Claude to push and register E5 v2. Its registration check must pass,
and `dacapo.ace.generate` must support `rng_seed`. Only then run the merged
[GPU scoring notebook](https://colab.research.google.com/github/ninondev/DaCapo/blob/main/experiments/e7_melody_adapter/colab_e7_suno.ipynb).
Choose the documented GPU runtime, set the Drive paths, and run in order.
The notebook verifies the registered code and source hashes. A mismatch stops
scoring and needs a reviewed deviation addendum. Do not bypass that guard.

The notebook also makes matched ACE-Step controls and a synthetic Voice Oohs
ruler self-check. Voice Oohs is a synthesized instrument, not real singing.
The one-hour L4 estimate is unmeasured. Report a runtime-budget stop instead
of extending the study or selecting a subset.

Keep the exports separate:

- `e7_suno_public.zip`: ACE-Step controls and synthetic-ruler numeric results.
- `suno_private.zip`: Suno numeric scores, for the owner's private storage and
  Claude's private review only.

Neither result zip includes Suno audio or separated vocals. Keep original
audio privately in Drive so the validated Stage 0c ruler can rescore the same
files later. Give the public and private packages to Claude through their
appropriate destinations. Claude applies the registered decision rule.

Do not publish Suno scores, route outcomes, accuracy conclusions, or threshold
comparisons until the owner confirms the terms permit that publication. Do not
start H1 from a missing, incomplete, or insufficient-ruler result.
