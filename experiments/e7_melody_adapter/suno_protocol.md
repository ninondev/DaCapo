# Stage S Suno protocol

Use only the twelve melodies in `melodies_suno.json`. Do not use any melody
from the owner's future held-out test set. Do not upload those test melodies
to Suno or any other website or app before Stage 4.

## Before generating

1. Give the implementation PR to the reviewing agent. Wait for confirmation
   that the revised `prereg_suno.json` is merged into main. Do not generate
   study songs before that confirmation.
2. Read Suno's current user terms. Decide whether they permit public evaluation
   results. Until the owner confirms this, every Suno score and conclusion
   stays private. This protocol does not make a legal determination.
3. Open the merged
   [CPU pack notebook](https://colab.research.google.com/github/ninondev/DaCapo/blob/main/experiments/e7_melody_adapter/colab_e7_pack.ipynb).
   Use the default Colab runtime with a CPU and run its cells in order. Python
   3.12 and 3.13 are supported; do not change the default Python version.
4. Download `e7_suno_upload.zip` and `suno_upload_hashes.json`. Give the hash
   JSON to the reviewing agent. Extract the zip locally. It contains MIDI,
   piano WAV, Voice Oohs WAV, lyrics, and a recording table for all melodies.

Use the files exactly as rendered. Do not retune the melodies, change their
lyrics, or select favorable passages. Keep the source hash record with the run.
Discard any earlier unregistered upload pack: the revised lyrics and hashes
must match this preregistration.

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

The CSV pre-fills `bpm`, `key`, and `style` with the reference values. When the
interface offers tempo and key inputs, enter those values. Record what was
actually entered in `actual_bpm`, `actual_key`, and `actual_style`. Write
`unavailable` when an input does not exist. Record any difference in `notes`.
Do not alter the style or lyrics between routes or guess a hidden setting.

## Requests, downloads, and records

Request **once** for every melody and available route. Keep **both** songs
returned by that request. Do not choose the better song, request extra draws,
edit the audio, or replace a failed song by generating a new song.

For each request, fill in the CSV record sheet. Record the actual menu option,
displayed model version, date in UTC, and credits spent. Do not guess a model
version that the interface does not show. The `route_available` field accepts:

- `yes`: the route is available and was used.
- `no`: the route is not available; explain why in `notes`.
- `unknown`: availability has not been checked. This is the blank-table default.

Download both audio files. Use the complete filename pattern
`S_<melody_id>_<route>_<take>.mp3`, where the melody ID is `mel00` through `mel11`,
the route is `R1`, `R2`, or `R3`, and the take is `a` or `b`. For example:

```text
S_mel00_R1_a.mp3
S_mel00_R1_b.mp3
```

Assign `a` and `b` by returned order, not quality. Keep the melody ID, route,
and take spelling exact. Extensions are case-insensitive: `.mp3`, `.MP3`, and
mixed-case forms are accepted. Match the actual filename in the CSV.

Use these Google Drive locations, or change all corresponding notebook paths:

```text
MyDrive/e7-suno/audio/                       downloaded audio
MyDrive/e7-suno/suno_recording_table.csv      completed recording table
MyDrive/e7-suno/scored/                      results and persistent checkpoints
```

In Colab they begin with `/content/drive/`. Put all study audio in the audio
folder. Keep the same `scored` folder when resuming a session. Do not delete
its checkpoints or retained generated controls between sessions.

A full run is twelve melodies times three routes times two songs, or 72 files.
Unavailable routes and missing songs must be listed. A route needs both songs
on at least ten melodies to count as measured. No measured route means not
measured, not a negative finding.

Before GPU work the notebook lists missing, empty, and unreadable audio. If a
file is unreadable, download **the same returned song** again and replace that
file. Do not request a new generation. An unreadable file keeps the run
incomplete, with no decision, until it is replaced or excluded. Wait until
Drive has finished syncing before you rerun. Decodable short clips are still
scored. If one clip fails during scoring, its reason is recorded and the other
clips continue. The run is then not complete and has no decision. Download
that same returned song again, then rerun the notebook with the same paths to
retry it. If the same file is unreadable, or fails with the same error, in
three finished passes, it is listed as an excluded take in the private record
and no longer blocks completion.

A runtime fault, such as a failed model download, stops the run as incomplete.
Rerun the same notebook and output folder. After a CUDA or memory error, or
three failed songs in a row, the notebook checks itself: it restarts the scorer
and scores one saved ruler clip again. If that check fails, the run stops as
incomplete. If it passes, the songs are treated as failed files and the run
goes on.

## GPU scoring, resuming, and review

Wait for the reviewing agent to push and register E5 v2. Its registration check
must pass, and `dacapo.ace.generate` must support `rng_seed`. Only then run the
merged [GPU scoring notebook](https://colab.research.google.com/github/ninondev/DaCapo/blob/main/experiments/e7_melody_adapter/colab_e7_suno.ipynb).
Use the default Colab runtime with an L4 or A100 GPU. Leave its default Python
version unchanged, set the Drive paths above, and run the cells in order.
The notebook calls `stage_s_run.sh` for Stage S work.

The notebook checks registered code and source hashes. A mismatch stops scoring
and needs a reviewed deviation addendum. Do not bypass that guard. Stage S
files stay separate from future H1 code so H1 development does not break this
registered measurement.

The order is synthetic ruler first, Suno second, then ACE-Step generation and
control scoring. Voice Oohs is a synthesized instrument, not real singing.
Scoring has no fixed wall-time cutoff. Each completed score is checkpointed
in Drive using the audio hash and the same preregistration. Rerun the same
notebook with the same inputs and `scored` directory after a disconnect; it
reuses completed scores and generated controls. It records actual stage times.
Only ACE generation has the registered budget. An incomplete control run is
reported separately and does not erase a completed Suno/ruler decision.
The result zips are written right after the Suno pass and again after the
controls. Saved controls are scored even when generation stops early.

Keep the exports separate:

- `e7_suno_public.zip`: ACE-Step controls and synthetic-ruler numeric results.
- `suno_private.zip`: Suno numeric scores, for private storage and private review.

Neither result zip includes Suno audio or separated vocals. Keep original audio
privately in Drive so the validated Stage 0c ruler can rescore the same files.
Give the public and private packages to the reviewing agent through their
appropriate destinations. The reviewing agent applies the registered rule.

Do not publish Suno scores, route outcomes, accuracy conclusions, or threshold
comparisons until the owner confirms the terms permit that publication. Do not
start H1 from a missing, incomplete, or insufficient-ruler result.
