# GPU run 3 (Run B)

I ran this in Colab on an NVIDIA L4 from 2026-10-29 21:01Z to 2026-10-30 01:45Z (decide finished at 01:44:50Z; packaging followed). The code was commit `bd61a9f`, registered in the B1 code addendum and in addendum 3 before the run. Run B has two parts: B1, the lyric clock on its own (no melody guide), and B2, the clock together with a melody guide. The files here are the run's own outputs, copied byte for byte.

| File | What it is |
|---|---|
| `setup_check.json` | Setup: the registration checks for addendum 2, the B1 code addendum and addendum 3, the soundfont, the 72 guides rendered again, and the seed table |
| `setup_model.json` | The model facts setup recorded: attention backend, determinism flags, ACE-Step commit, the weights' sha256, and that the model loaded with the registered alignment heads and lyric keys |
| `a0_report.json` | A0: the GPU checks before any study run (determinism, the hook at beta 0, 40 hook calls, one guide latent across k) and the registered timing trim |
| `b1_setup.json` | B1's setup: runtime and determinism checks, code hashes, and each line's layout and the note windows the clock uses |
| `b1_result.json` | B1: the registered rule, its verdict, every arm and all 180 scored clips |
| `b2_selection.json` | B2 Stage D: the full development table (25 recipes × 12 lines) and the two recipes it chose (S*, with the clock, and B*, without) |
| `b2_result.json` | B2 Stage T: the integrity checks (P0), the rule (P1, P2), the labels, every arm, the paired contrasts, the demo by rule and all 480 scored clips |
| `b2_summary.json` | B2's verdict, labels and J per arm, in short |
| `status.json` | Each stage's state and minutes. `package` reads "running" because the file was written while packaging ran; the package log ends with state ok |

Some Whisper transcripts in `b1_result.json` and `b2_result.json` are Whisper's known hallucinated credit lines (for example "transcript Emily Beynon"), which name real people. They are model output and are kept unedited so the files match the run. In B2, 35 of the 39 such clips count as wordless, and none matches more than a tenth of the lyric.

The ledgers, logs, audio, the labelled demo clips and the blind A/B pack are kept outside the repo. The answer key for the blind test is kept apart from all of them. `LOG.md` describes the run and the checks made on it afterwards.
