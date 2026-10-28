# GPU run 2 (Run A)

I ran this in Colab on an NVIDIA L4 on 2026-10-28. The code was commit `83d46e7`. Run A has two parts: the fixed motif study (E5 v2) and the lyric read-out (E6). The files here are the run's own outputs, copied byte for byte.

| File | What it is |
|---|---|
| `e5v2_setup.json` | Setup checks: registration hashes, source renders, measurement packages, K0 for flute, synth voice and clarinet |
| `e5v2_a0.json` | A0: the generation checks on the GPU before any study run (instruction, noise, decode settings, determinism) |
| `e5v2_phase0b.json` | Phase 0b: gaps that passed the quality gate in each of the 12 cells, and the cell it chose |
| `e5v2_p0b_quality.json` | Phase 0b quality decision with every gap's gate values |
| `e5v2_main.json` | The confirmatory grid: the quality decision, every registered index and verdict (H1 to H5, K2, K3), and all 1152 scored rows |
| `e5_terminal.json` | The E5 v2 end state: paper level `supported:H1,H2`, and the sha256 of `e5v2_main.json` |
| `e6_readout.json` | E6 read-out: summary and all 88 clips |
| `e6_gate.json` | E6 decision: B1 (the lyric clock) is allowed |
| `e6_scores.json` | E6 per-clip scores against Whisper and MMS |

Some files have no `git_commit` field. They are tied to the run in other ways. `e5_terminal.json` stores the sha256 of `e5v2_main.json`. The design files store the sha256 of `addendum_2.json`, and every file that has the field says `83d46e7`.

The ledgers, logs, design and provenance files, the other quality file, the audio and the blind-listening pack are kept outside the repo. The answer key for the blind test is kept apart from all of them. `LOG.md` describes the run.
