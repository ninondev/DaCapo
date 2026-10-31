# GPU run 4 (Run C)

I ran this in Colab on an NVIDIA L4. The code was commit `0c97266`, registered in addendum 4 before the run. Run C (C1) tests B2's two recipes again, fixed in advance, with the random-head and moved-window controls, on 96 new melodies: 1,536 runs. The files here are the run's own outputs, copied byte for byte.

| File | What it is |
|---|---|
| `setup_check.json` | Setup (third session): the registration checks for addendum 2, the B1 code addendum, addendum 3 and addendum 4, the soundfont's sha256, and the lists of mismatches, all empty, for the 98 guides rendered again, the seed table and the package pins |
| `setup_model.json` | The model facts setup recorded: attention backend, determinism flags, the runtime fingerprint, ACE-Step commit, the weights' sha256, and that ACE-Step's own lyric keys equal the registered token check |
| `a0_report.json` | A0: the GPU checks before any Run C run (the hook at beta 0 bit-identical to no hook, 40 hook calls, one guide latent, the clock changing the output, the scorer) and the time projection |
| `c1_result.json` | C1: the integrity checks (P0), the rule (R1, R2), the labels, every arm, the paired contrasts, the pooled estimate with B2, the demo by rule and all 1,536 scored clips |
| `c1_summary.json` | C1's verdict, labels and J per arm, in short |
| `status.json` | Each stage's state and its minutes summed over the sessions, as written by the last of three sessions. `package` reads "running" because the file was written while packaging ran |

Three Colab sessions touched the run state. The first (2026-10-30, from 09:07Z) ran setup, A0, all 1,536 generations and scores (09:10Z to 14:33Z), decide and package. A second session on the same Colab machine (17:34Z) and a third on a fresh machine with the same runtime fingerprint (2026-10-31, 04:09Z) ran only setup, decide and package: they generated and scored nothing, reused A0, and decide printed the same result each time. The files here come from the third session's package; `a0_report.json` is the first session's A0 report, which the later sessions reused.

The per-clip transcripts in `c1_result.json`, read together with the demo list, would tell a listener which clip of a C1 blind pair has the clock. People who will rate the C1 pack should not read this file first.

The ledgers, logs, audio, the labelled demo clips and the blind A/B pack are kept outside the repo. The answer key for the blind test is kept apart from all of them. `LOG.md` describes the run and the checks made on it afterwards.
