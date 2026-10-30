"""Write or check experiments/e5_motif/registration/addendum_4.json, the registration of Run C (C1): the text, the
sha256 of the C1 code, of the B2 code and other code it reuses (with the notebook's package pins and the check of
addendum 2), of its data (the lines, manifest, guides, tokenizer check, seed table and power simulation, and the
earlier records it reads) and of its tests, with the sha256 of addendum 2, the B1 code addendum and addendum 3,
which stay unchanged.

  python experiments/e9_clock_replicate/register_c1.py              write it; refuses on a seed collision, a broken
                                                                     line rule, a stale manifest or a failed check
  python experiments/e9_clock_replicate/register_c1.py --out PATH   write it to PATH instead (a draft for review)
  python experiments/e9_clock_replicate/register_c1.py --check [--addendum PATH]   exit 1 if it is not current

Run it as the last step before the commit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E8_DIR = ROOT / "experiments" / "e8_clock_sing"
for p in (ROOT, HERE, E8_DIR, ROOT / "experiments" / "e6_clock"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import plan_b2 as P2  # noqa: E402
import plan_c1 as P  # noqa: E402
import readout as R  # noqa: E402
import register_b2 as REG3  # noqa: E402

REG_DIR = REG3.REG_DIR
ADDENDUM = REG_DIR / "addendum_4.json"
REG2, REG_B1, REG_3 = REG3.REG2, REG3.REG_B1, REG3.ADDENDUM
E9 = "experiments/e9_clock_replicate/"
E8 = "experiments/e8_clock_sing/"
CODE = tuple(E9 + f for f in ("plan_c1.py", "make_items_c1.py", "run_c1.py", "score_c1.py", "decide_c1.py",
                             "package_c1.py", "power_c1.py", "register_c1.py")) + ("experiments/run_c.py",
                                                                                   "experiments/build_notebook_c.py")
# B2's code, the shared code B2 registered, the notebook pins (build_notebook.py) and the check of addendum 2
# (e5_motif/register.py), which no earlier addendum hashes
REUSED = tuple(dict.fromkeys(REG3.REUSED + tuple(E8 + f for f in ("plan_b2.py", "make_items_b2.py", "run_b2.py", "score_b2.py",
                                                                  "decide_b2.py", "package_b2.py", "register_b2.py"))
                             + ("experiments/build_notebook.py", "experiments/e5_motif/register.py")))
DATA = tuple(E9 + f for f in ("lines_c1.txt", "manifest_c1.json", "guides_c1.json", "tok_check_c1.json", "seeds_c1.json",
                             "power_c1.json")) + (
    "experiments/e4_sing/manifest.json", E8 + "lines_b2.txt", E8 + "manifest_b2.json", E8 + "guides_b2.json",
    E8 + "tok_check_b2.json", E8 + "seeds_b2.json", "experiments/e7_melody_adapter/melodies_suno.json",
    "results/gpu_run3/b2_result.json", "results/gpu_run3/b2_selection.json")
TESTS = E9 + "tests"
POWER_SIMS = 2000
TITLE = "Addendum 4: C1 of Run C, a registered replication of B2's fixed recipes on new melodies"
KEYS = ("addendum_utc", "title", "refers_to", "earlier_addenda_sha256", "custody", "sections", "fill_ins", "plan",
        "code_sha256", "data_sha256", "tests_sha256")

CUSTODY = ("This addendum is committed and pushed before Run C and before any Run C generation. It changes nothing "
           "that addendum 2, the B1 code addendum or addendum 3 registers, and it names their sha256; the B2 code it "
           "reuses is imported unchanged and hashed here again. That push is the external timestamp.")

SECTIONS = [
    ("question", "B2 (Run B, addendum 3) ended 'Capability reached, clock not credited'. With one recipe fixed on "
                 "development lines, 4 draws and the automatic selector, S* (the Voice Oohs guide for the first 3 of 8 "
                 "steps and the lyric clock at beta 8 on the seven alignment heads) sang 14 of 24 new melodies with at "
                 "least half the notes right on the vocal stem and at least half the words recognised in order, against "
                 "8 of 24 for the same guide without the clock. The gap, Delta_J = 0.25, cleared B2's 0.20 bar, but the "
                 "2.5th percentile of its item bootstrap was exactly 0, and the rule needed it above 0. 24 melodies were "
                 "too few to settle the clock's effect. C1 asks two questions on 96 new lines and melodies, with the "
                 "same two recipes fixed in advance and no selection stage: does the clock help at all (R1), and does "
                 "the capability hold (R2)?"),
    ("design", "One stage, C1 test: the four arms of B2's Stage T that rest on S* (S*, S*_b0, S*_rand and S*_shuf) on 96 "
               "new items c1t00-c1t95, 4 draws each, in the order item, draw, arm: 96 x 4 x 4 = 1,536 runs. The recipes "
               "are B2's selection, taken from b2_selection.json and parsed by plan_b2.parse, so nothing is chosen on "
               "Run C data. There is no timing trim; the design is fixed. A0 runs first and must pass. A run that "
               "cannot finish is decided on what exists, under the missing rule of section rule."),
    ("generation", "Exactly B2's (addendum 3, section generation): acestep-v15-turbo at ca1e85f with the weights of "
                   "addendum 2, 8 steps, shift 3.0, DCW off, batch 1, no language model, sdpa attention; caption 'a "
                   "cappella, solo female vocal, no instruments, no drums, dry studio recording, clear lyrics'; bpm 100; "
                   "keyscale the item's key; vocal language en; lyrics '[Verse]\\n' + line; task cover-nofsq with "
                   "audio_cover_strength 3/8, cover_noise_strength 0 and the item's Voice Oohs guide as source; the E5 v2 "
                   "determinism settings (setup_determinism, VAE decode chunk 256) and the global RNG pinned to rng_seed "
                   "right before each run; 16-bit PCM WAV through soundfile, pred_latents and target_latents kept as "
                   "float32 .npy. Generation calls B2's own run_b2.generate_run, unchanged."),
    ("items", "Test: c1t00-c1t95 of experiments/e9_clock_replicate/manifest_c1.json, built by make_items_c1.py from the 96 "
              "lines of lines_c1.txt by make_items_b2's procedure: E4's rhythm() and pitches() imported unchanged, numpy "
              "default_rng(33), keys C, D, F and G in turn, 100 BPM, a 4-beat lead-in, 16 beats, MIDI 57-76, each note "
              "sounding for 0.9 of its length, one note per word. The lines were written by the agents under addendum "
              "3's line rules, checked in code with make_items_b2.line_problems and its syllable count: 11-13 words; "
              "every word one syllable, letters only and capitalised as in E4's lines; every word 1-2 Qwen3 tokens; the "
              "tokenizer check passes with s = 9, the token words equal to the lyric words and as many words as notes; no "
              "word from the registered homophone list; none of thank, you, watch, subscribe. Three rules are added: no "
              "line from E4, E6, B2 (lines_b2.txt) or E7 (the lines of melodies_suno.json); no two Run C lines sharing "
              "more than 6 words; and no Run C line sharing more than 6 words with a line of E4, E6, B2 or E7. Shared "
              "words are counted with repeats and case-insensitive. No Run C melody repeats another Run C, B2 or E4 "
              "melody. A0 uses item00 and the tiny scale item01 of the E4 manifest."),
    ("guides", "FluidSynth 2.3.4 with FluidR3_GM.sf2 (sha256 74594e8f4250680adf590507a306655a299935343583256f3b722c48a1"
               "bc1cb0), 48 kHz, gain 0.6, reverb and chorus off, program 53 (Voice Oohs), rendered by "
               "make_items_b2.render_one, unchanged. One guide per item: c1t00-c1t95, item00 and item01, 98 WAVs. Their "
               "sha256 are in guides_c1.json and in section plan.guides_sha256; item00's and item01's equal guides_b2.json. "
               "They are not in git; Run C renders them again in setup and stops on any mismatch."),
    ("seeds", "C1 test: (dit_seed, rng_seed) = dacapo.seeds.run_seeds('runC-test', item_id, r), r = 0-3, shared by the "
              "four arms of an (item, draw). A0: run_seeds('runC-a0', 'item00', 0). The tiny test scale uses runC-a0-tiny "
              "and runC-test-tiny on item01, draw 0. seeds_c1.json holds every pair. register_c1.py checks that no new "
              "DiT seed or pin equals another new one or any earlier one: every source plan_b2.earlier_seeds() lists (E4, "
              "E5 v1 and its reserved seeds, the E5 v2 table and tiny tags, E6, the E6 acceptance seeds, B1 and its tiny "
              "runs, the CPU test, 42, 43, 44, 20261027, 20261029 and the E7 Stage S tags) and every pair of B2's "
              "seeds_b2.json at both scales. On a collision it refuses and never re-draws. There was no collision."),
    ("arms", "The recipes, exactly as B2 ran them (plan_b2.config and parse). 1. S* = oohs_k3_A8: the Voice Oohs guide, "
             "cover for the first 3 of 8 steps, the clock at beta 8 on head set A (L2h6, L3h10, L3h11, L4h3, L5h8, L5h9, "
             "L6h8). 2. S*_b0 = oohs_k3_none: the same without the hook; the context manager is not entered. 3. S*_rand = "
             "oohs_k3_R8: S*'s guide, k and beta on readout.random_head_sets()[3 + r] for draw r. 4. S*_shuf = "
             "oohs_k3_A8_shuf: S*'s guide, k, heads and beta, with the windows rotated so that word j's tokens get the "
             "window of word (j + floor(n/2)) mod n. The clock is dacapo/attn_bias.py of the B1 code addendum, "
             "unchanged, with B2's windows (plan_b2.spans): forward pre-hooks on decoder.layers[L].cross_attn for "
             "L = 2-6 at all 8 steps; word j's tokens get +beta in query frames whose centre lies in [onset_j, "
             "onset_(j+1)), the last word's window running to its note end."),
    ("a0", "A0 runs on the GPU on item00 with the runC-a0 seeds, before any C1 generation: S*_b0 once, S* once, and "
           "oohs_k3 with the hook attached at beta 0 on set A. It requires every run to finish with 8 denoiser calls and "
           "its pre-normalisation peak; the beta 0 run bit-identical to S*_b0 (torch.equal on pred_latents); 40 hook "
           "calls with every biased key below n_lyr in each hooked run and none in S*_b0; one guide latent for the three "
           "runs; S*'s pred_latents different from S*_b0's; and the registered scorer on S*'s and S*_b0's clips with "
           "the pinned Whisper and CTC revisions. A failure stops the run before any C1 generation; B2's fallback (the "
           "hook at beta 0 on every unclocked run) does not apply in Run C. A0 records the runtime it ran on (fill-in "
           "runtime). A0 runs once: a later runtime reuses its report while the report, its inputs and the code still "
           "hash to what A0 recorded and the session runs on the same runtime; before any C1 generation, a session on "
           "another runtime runs A0 again."),
    ("measures", "One hashed scorer: score_c1.py wraps B2's score_b2.B2Scorer (on clock_score.py), imported unchanged, "
                 "so every B2 measure is computed exactly as in addendum 3, section measures: the vocal gate; NPA_v (E4's "
                 "note pitch accuracy on SwiftF0 of the htdemucs vocal stem); WR (lyric words matched in order by "
                 "dacapo.words.match_words to the word chunks of Whisper large-v3 at revision "
                 "06f233fe06e710322aca913c1bc4249a0d71fce1, fp16, greedy, English, word timestamps, on the full mix); "
                 "q = gate x min(NPA_v, WR); joint q >= 0.5; strict q >= 0.75; the selector (largest q; ties to higher WR, "
                 "higher NPA_v, then the lower draw); bo4_i, the q of the kept draw; J_i = 1 if bo4_i >= 0.5; J(arm), the "
                 "mean of J_i over the 96 items. Secondary, as B2: mix NPA, octave-tolerant NPA, chance NPA, E4's multiset "
                 "WR and capped WER, onset errors, recall by position, gate rate, vowel-only and wordless rates, the "
                 "guide-copy index, bleed, the CTC WR on the mix (facebook/wav2vec2-large-960h-lv60-self at "
                 "54074b1c16f4de6a5ad59affb4caa8f2ea03a119, fp32, greedy) and best-of-n. New secondary measures: the CTC WR "
                 "on the htdemucs vocal stem (the same model, revision and decoding, on the stem B2's scorer separates) "
                 "and q_ctc_stem = gate x min(NPA_v, CTC WR on the stem). Chance NPA compares each clip with the 23 other "
                 "melodies of its group of 24 (c1t00-c1t23, c1t24-c1t47, c1t48-c1t71, c1t72-c1t95), as many as in B2."),
    ("rule", "decide_c1.py applies the rule to C1 test only. Values are rounded to 1e-6 before comparison. Bootstrap: a "
             "cluster bootstrap over items, 10,000 resamples, numpy default_rng(0).integers(0, 96, (10000, 96)), the draws "
             "of an item kept together, all arms resampled together, numpy's linear percentiles. P0, integrity; all must "
             "hold: the code, data, guide, soundfont and seed-table hashes match addendum 4 at generation (every run row "
             "records them and the sha256 of the guide it used) and at decision; the last setup rendered the guides again "
             "with the registered soundfont and found the installed packages at the notebook's pins; the four "
             "registration checks (addendum 2, the B1 code addendum, addendum 3 and addendum 4, each run as its own "
             "--check) passed at the last setup and pass again at decision, so the files the earlier addenda register "
             "are unchanged; A0 is recorded, passed and ran on the registered code, and every run row names its report; "
             "every run row matches its planned run (item, draw, config, guide, k, clock, heads, beta, rotation, hook, "
             "task, cover strength and seeds); every hooked run made 40 hook calls with max biased key < n_lyr, and "
             "every unhooked run made none; within each (item, draw), every run received the same guide latent (sha256 "
             "of target_latents); every scored clip used the pinned Whisper revision without a test flag, and every "
             "generated clip has its pre-normalisation peak; at most 5% of the 1,536 planned runs (76) are missing, "
             "failed after 3 attempts with the same seeds, or unscored, and these count as q = 0, gate failed, WR 0, "
             "NPA 0 and CTC WR 0. R1, the clock effect: Delta_J = J(S*) - J(S*_b0); R1 holds if the bootstrap 2.5th "
             "percentile of Delta_J is above 0. R2, capability: J(S*) >= 0.50, a joint kept draw for at least 48 of "
             "the 96 melodies. Verdicts: 'Replicated: the clock helps and the capability holds' if P0, R1 and R2 hold; "
             "'Clock helps, capability not reached' if P0 and R1 hold and R2 fails; 'Capability holds, clock effect not "
             "shown' if P0 and R2 hold and R1 fails; 'Not replicated' if P0 holds and R1 and R2 both fail, naming both; "
             "'Technical' if P0 fails, and no effect is claimed either way. R1 drops B2's 0.20 size bar as a condition, "
             "deliberately and before any Run C generation: Run C asks whether the clock helps at all, which B2 could "
             "not settle. The bar stays as a label, so B2's criterion can still be read off."),
    ("labels", "Labels never change a verdict. B2's size bar: Delta_J >= 0.20 (and, read with R1, B2's P1 against S*_b0). "
               "Strict capability: S* has bo4 >= 0.75 on at least 48 of the 96 items. Head-specific clock, as in B2: mean "
               "bo4 q(S*) - mean bo4 q(S*_rand) >= 0.5 x (mean bo4 q(S*) - mean bo4 q(S*_b0)), else 'nonspecific bias', "
               "and 'no clock effect to attribute' if mean bo4 q(S*) <= mean bo4 q(S*_b0). Timing-specific, as in B2: the "
               "95% lower bound (the bootstrap 2.5th percentile) of mean_i [bo4 q(S*) - bo4 q(S*_shuf)] is above 0; "
               "'emphasis, not timing' if J(S*_shuf) - J(S*_b0) >= 0.20 and that bound is at most 0; else 'timing not "
               "shown'. Timbre copy: S*'s vowel-only rate over its 384 planned clips is at least 0.30. Words cut: S*'s "
               "second-half recall is at least 0.20 below its first-half recall. Frontier moved: mean NPA_v(S*) >= mean "
               "NPA_v(S*_b0) - 0.10 and mean WR(S*) >= mean WR(S*_b0) + 0.20. Independent ASR: R2 recomputed with the CTC "
               "WR on the mix, and with the CTC WR on the vocal stem, on the same kept draws (kept by Whisper's q). "
               "Listeners: filled later from the Run C blind pack, exploratory."),
    ("analysis", "Reported and exploratory: for each of the four arms, J; joint and strict clips out of 384; mean bo4 q and "
                 "mean per-clip q; NPA_v, mix NPA, WR and gate rate; vowel-only and wordless rates; onset errors and "
                 "recall by position; guide-copy index and bleed rate; CTC WR on the mix and on the stem; best-of-n curves "
                 "for n = 1-4; per-item kept draws. Paired contrasts with 95% intervals: mean bo4 q and J of S* against "
                 "each other arm. A pooled estimate of Delta_J and of the mean bo4 q difference of S* against S*_b0 over "
                 "B2's 24 test items (results/gpu_run3/b2_result.json) and Run C's 96, with a cluster bootstrap over the "
                 "120 items (default_rng(0).integers(0, 120, (10000, 120))), labelled exploratory."),
    ("power", None),   # power_text(), from power_c1.json
    ("package", "Every planned clip, failures included, with its metrics, in item order. Results, ledgers, logs and a "
                "demo by rule: for each of the 96 items, S*'s kept draw next to S*_b0's kept draw, nothing picked by ear. "
                "A blind A/B pack of 24 of these pairs asks 'melody right' and 'words clear'. Its items and their order "
                "are the first 24 of numpy default_rng(20261031).permutation over the 96 demo items in item order; which "
                "clip of a pair is A comes from secrets.randbelow(2) and is written only to the key, so the code and the "
                "seed do not reveal it; packaging again keeps the sides of an existing key. The pack and its key are "
                "separate files outside the package, and listeners get the pack before they hear the demo. Its items "
                "are Run C items, so it shares none with B2's pack. The pack is exploratory."),
    ("success", "'Replicated' requires P0, R1 and R2. In plain terms: on 96 melodies it has not seen, with the recipe fixed "
                "before the test, 4 draws and the automatic selector, ACE-Step 1.5 turbo with the clock sings at least "
                "half of them with at least half the notes right on the vocal stem and at least half the words recognised "
                "in order, and the clock's gain over the same guide without it has a bootstrap 2.5th percentile above 0."),
    ("disclosures", "Run C was designed after B2's results were known: the recipes, the arms and the choice to replicate "
                    "come from B2's outcome, and the power simulation uses B2's per-item outcomes. Nothing in Run C is "
                    "chosen on Run C data. The 96 lines were written by the agents after Run B. Six drafts failed a rule "
                    "checked in code (four in make_items_b2.line_problems, two in the tokenizer check, each a first word "
                    "of 3 tokens) and three were reworded for plainer words; all were rewritten before registration, and "
                    "the manifest was rebuilt from seed 33 after the last change. A review then found that c1t48 shared 8 "
                    "words with a B2 line; it was reworded to share at most 6 with any earlier line, that limit became a "
                    "rule checked in code, and its melody, guide and seeds did not change. No Run C item has been "
                    "generated. R1 drops B2's 0.20 "
                    "size bar as a condition (section rule). A0 adds one hard check to B2's list, that beta 8 changes "
                    "S*'s output, and has no fallback: if the hook at beta 0 is not bit-identical to no hook, the run "
                    "stops. Chance NPA uses the 23 other melodies of each item's group of 24, so a clip is compared with as "
                    "many melodies as in B2. run_c.py generates and then scores the items in blocks of 24 in plan order, so "
                    "a session that ends early leaves at most one block unscored; the run order within a block is the "
                    "plan's. Three reviews of the build came before registration, and their fixes are in the code and "
                    "in this text: setup's comparison of the rendered guides called a missing function and would have "
                    "stopped every registered session at setup; a failed attempt now runs in a fresh process; an attempt "
                    "whose process died is counted; generation refuses a runtime other than A0's; setup checks the "
                    "package pins; and decide runs the four registration checks again. The owner decided on 2026-10-30 to build and register Run C now and to run it only when "
                    "the owner approves a Colab L4 session of about 6 hours: at Run B's measured Stage T cost (12.1 s per "
                    "run, generation and scoring) the 1,536 runs take about 5.2 hours, and with install, setup, A0, the "
                    "model loads of the four blocks, decide and package about 5.9 hours. A stopped session resumes from "
                    "Drive. Before registration, the tiny test chain (experiments/run_c.py --scale tiny --device cpu) ran "
                    "on this machine's CPU to test the code, with the tiny seed tags: each pass made 2 A0 runs and 2 C1 "
                    "test runs on the dev item item01 and scored them with Whisper small.en and the pinned CTC model on "
                    "the CPU; every verdict was 'test only'. Two passes from a clean state on the final code gave identical "
                    "audio, latents, ledgers and results apart from time stamps, timings and process ids. A third pass "
                    "was killed during C1 generation and resumed: the cut-off run was generated again byte for byte, "
                    "its killed attempt did not count against it, A0 was reused, nothing was scored twice, and the "
                    "results equalled the clean passes'. Nobody listened to "
                    "the clips; only the logged measures were read, to check the code, and the audio was deleted. No Run C setting changes after the first Run C "
                    "generation. Any deviation goes into the ledger and the write-up."),
    ("run", "experiments/run_c.py runs every stage and resumes: setup, a0, c1test, decide, package. setup checks every "
            "hash (addendum 2, the B1 code addendum, addendum 3, addendum 4), renders the 98 guides again and compares "
            "their sha256, checks the seed table and the free space, and loads the model to record sdpa, the alignment "
            "heads, the determinism flags, the ACE-Step commit, the weights' sha256 and the lyric keys of all 98 items "
            "through ACE-Step's own path. a0 runs A0 (section a0). c1test generates and then scores the 96 items in "
            "four blocks of 24, in plan order; each stage keeps an append-only JSONL ledger; a run is done when its WAV "
            "and latents hash to its ledger row, and a clip is scored when its score row names the WAV's sha256; a "
            "failed run gets at most 3 attempts with the same seeds, each in a fresh process (fill-in attempts), and a "
            "clip whose scoring raised is scored again once. A gen process refuses to generate on a runtime other than "
            "A0's (fill-in runtime). A heartbeat file in the state folder, written every 30 s, keeps a second "
            "orchestrator from working on the same run state; the processes run_c.py starts end with it. Stop rules: a setup or A0 failure before any C1 generation stops the run (package still collects "
            "the logs); the same failure on a resumed run stops generation, and decide and package still run; once C1 "
            "generation has started, decide always runs and writes 'Technical' when P0 fails; when C1 generation is "
            "incomplete, run_c.py says how many runs exist and that Run all resumes. --state DIR keeps the "
            "run state on Google Drive. The tiny scale runs every stage with 1-2 runs on the dev item item01, on the "
            "CPU, with Whisper small.en and its own seed tags; its verdicts are 'test only'. The full scale refuses the "
            "CPU and test flags. Registered outputs go to results/gpu_run4/; audio, the demo and the pack stay outside "
            "the repository."),
]

FILL_INS = [
    ("names", "Configs keep B2's names: oohs_k3_A8 is the Voice Oohs guide at k = 3 with beta 8 on set A; R8 marks beta "
              "8 on random set 3 + r; the suffix _shuf marks the rotated windows. Each C1 run carries one arm label."),
    ("missing", "A planned clip that is missing, failed after 3 attempts or not scored counts as q 0, gate failed, WR 0, "
                "NPA_v 0, mix NPA 0, CTC WR 0 on the mix and on the stem (score_c1.MISSING). The vowel-only, wordless "
                "and bleed rates and the guide-copy index are over scored clips; the timbre-copy label divides S*'s "
                "vowel-only clips by all 384 planned clips."),
    ("positions", "As in addendum 3: the first half of a line is its first floor(n/2) words, the second half the rest; "
                  "the last word is word n."),
    ("ctc_stem", "The vocal stem is clock_score.Scorer.measure's: htdemucs, mono float32 at the separator's 44.1 kHz. It is "
                 "resampled to 16 kHz with librosa and decoded by greedy argmax, exactly as score_b2 decodes the mix."),
    ("best_of_n", "Best-of-n averages, over items and over every subset of n of the 4 draws, the largest q (and the share of "
                  "items whose largest q is at least 0.5), as decide_b2.best_of_n computes it."),
    ("strict", "The strict-capability label needs bo4 >= 0.75 on at least half of the items, rounded up: 48 of 96."),
    ("attempts", "A failed run's attempts run in separate gen processes, at most 3, with the same seeds. Each attempt "
                 "first appends a 'started' row to the ledger. An attempt that started and never finished (its process "
                 "died) is free the first time and counts as failed from the second time on, so a runtime reset costs a "
                 "run nothing and a run that kills its process cannot block the rest. A gen process stops (exit 4) when "
                 "failures in a row span two items, which points to an error that sticks to the process or the GPU, and "
                 "run_c.py starts a fresh one; a block stops for the session after two such stops or 8 gen processes, "
                 "and Run all resumes it."),
    ("runtime", "A0 records the runtime fingerprint: the fields of dacapo.ace.runtime_facts listed in plan.runtime_keys "
                "(the GPU and its SM count, CUDA, cuDNN, the torch, diffusers, transformers, accelerate, safetensors and "
                "numpy versions, the attention kernel, the model dtypes and the VAE encode chunk). Every gen process "
                "compares its own with A0's and, at the full scale, refuses to generate when they differ, unless "
                "DACAPO_ALLOW_RUNTIME_CHANGE=1 (ALLOW_RUNTIME_CHANGE in the notebook); runs made that way record the "
                "change, and decide reports how many there are. Within an (item, draw), a change of runtime still fails "
                "P0 if the guide latents differ. setup checks the installed packages against the notebook's pins "
                "(experiments/build_notebook.py, hashed here; plan.environment) in the measuring process and in "
                "ACE-Step's, and the full scale stops on a difference. The notebook stops unless the GPU is an NVIDIA "
                "L4."),
    ("a0", "A0's hard checks stop the run on failure. If the beta 0 run differs from S*_b0, A0 runs S*_b0 again and "
           "records whether the difference comes from nondeterminism; the check has failed either way. A0 records a time "
           "projection from its medians and from Run B's cost; no trim follows from it."),
    ("rounding", "Values are rounded to 1e-6 before every comparison, as in E6, B1 and B2. Percentiles are numpy's linear "
                 "interpolation. The bootstrap index of the rule is the same matrix for every arm and every contrast."),
]


def power_text(power: dict) -> str:
    c, b = power["cells"], power["b2"]
    o, h, s, n = c["observed"], c["halved"], c["halved_s_down"], c["null"]
    return (f"power_c1.py simulates the rule from B2's per-item kept-draw outcomes of S* and S*_b0 ({b['wins']} melodies "
            f"won by S*, {b['losses']} lost, {b['ties']} tied), drawing {power['n_items']} items with replacement, "
            f"{power['sims_per_cell']:,} simulated runs per cell, with the registered bootstrap index. At B2's observed "
            f"effect (Delta_J 0.25): P(R1) {o['P_R1']}, P(R2) {o['P_R2']}, P(R1 and R2) {o['P_R1_and_R2']}, "
            f"P(Delta_J >= 0.20 and R1) {o['P_size_bar_and_R1']}. With the effect halved and J(S*) as observed (a win "
            f"becomes a tie with probability 3/8): P(R1) {h['P_R1']}, P(R2) {h['P_R2']}, P(Delta_J >= 0.20 and R1) "
            f"{h['P_size_bar_and_R1']}. With the effect halved by lowering S*: P(R1) {s['P_R1']}, P(R2) {s['P_R2']}. "
            f"With no effect, R1 holds in {100 * n['P_R1']:.1f}% of simulated runs. The cells are in power_c1.json and in section "
            f"plan.power.")


def sections() -> list[tuple[str, str]]:
    power = P.read_json(HERE / "power_c1.json")
    return [(k, power_text(power) if t is None else t) for k, t in SECTIONS]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hashes(files) -> dict[str, str | None]:
    return {f: sha256(ROOT / f) if (ROOT / f).exists() else None for f in files}


def pins() -> dict[str, dict[str, str]]:
    """The package pins the notebook installs (experiments/build_notebook.py): "main" for the measuring process (the
    measurement stack, and transformers for Whisper and the CTC model), "ace" for ACE-Step's process."""
    spec = importlib.util.spec_from_file_location("dacapo_notebook_pins", ROOT / "experiments" / "build_notebook.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)

    def split(xs):
        return dict(x.split("==", 1) for x in xs)
    ace = split(A.ACE_PINS)
    return {"main": {**split(A.PINS), **split([A.BASIC_PITCH]), "transformers": ace["transformers"]}, "ace": ace,
            "ace_commit": A.ACE_COMMIT, "hf_revision": A.HF_REVISION}


def runtime_files() -> tuple[str, ...]:
    """The files every C1 generation row records and decide compares with this addendum."""
    return CODE + REUSED + DATA


def test_files() -> list[str]:
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / TESTS).glob("*.py"))


def earlier_record() -> dict:
    return {str(p.relative_to(ROOT)): sha256(p) for p in (REG2, REG_B1, REG_3)}


def plan_record() -> dict:
    import decide_c1 as D

    guides = P.read_json(P.GUIDES_JSON)
    tok = P.read_json(P.TOK_JSON)
    power = P.read_json(HERE / "power_c1.json")
    runs = P.test_runs("full")
    return {"test_items": [it["id"] for it in P.test_items()], "lines": P.LINES.read_text().splitlines(),
            "arms": [{"label": lab, "config": cid, **{k: v for k, v in P2.parse(cid).items() if k != "config"}}
                     for lab, cid in P.ARMS],
            "order": "item, draw, arm", "draws": list(P.DRAWS), "n_runs": len(runs),
            "random_heads": {str(P2.RANDOM_FIRST + r): {str(k): v for k, v in R.random_head_sets()[P2.RANDOM_FIRST + r].items()}
                             for r in P.DRAWS},
            "head_set_A": {str(k): v for k, v in R.ALIGN.items()},
            "a0_runs": [{"name": n, "config": c, "beta": b, "why": w, "scored": n in P.A0_SCORED["full"]}
                        for n, c, b, w in P.A0_RUNS], "a0_item": P.A0_ITEM["full"],
            "tags": P.TAGS, "seed_table_sha256": sha256(P.SEEDS_JSON),
            "guides_sha256": guides["sha256"], "soundfont_sha256": guides["soundfont_sha256"],
            "fluidsynth": guides["fluidsynth"],
            "lyric_keys": {k: {"n_lyr": v["n_lyr"], "start": v["start"], "keys": v["keys"]} for k, v in tok["items"].items()},
            "chance_group": P.GROUP, "block_items": P.BLOCK_ITEMS,
            "environment": pins(), "runtime_keys": list(P.RUNTIME_KEYS),
            "attempts": {"max_attempts": R.MAX_ATTEMPTS, "gen_processes_per_block": P.GEN_PASSES,
                         "sticky_stops_per_block": P.STICKY_TRIPS, "died_attempts_free": 1},
            "rule": {"j_min": D.J_MIN, "r1": "bootstrap 2.5th percentile of J(S*) - J(S*_b0) > 0", "joint": 0.5,
                     "strict": 0.75, "size_bar_label": D.SIZE_BAR, "max_missing_share": D.MAX_MISSING_SHARE,
                     "max_missing_runs": int(D.MAX_MISSING_SHARE * len(runs)), "n_boot": D.N_BOOT,
                     "boot_seed": D.BOOT_SEED, "boot_index": "numpy default_rng(0).integers(0, 96, (10000, 96))",
                     "digits": 6, "verdicts": D.VERDICTS},
            "time": {"stage_t_seconds_run_b": P.STAGE_T_SECONDS_B2, "fixed_hours": round(P.FIXED_HOURS, 6),
                     "runs_hours": round(len(runs) * P.STAGE_T_SECONDS_B2 / 3600, 2),
                     "session_hours": round(P.projected_hours(P.STAGE_T_SECONDS_B2, len(runs),
                                                              len(P.blocks("full", P.BLOCK_ITEMS))), 2)},
            "tiny": {"items": list(P.TINY["test_items"]), "draws": list(P.TINY["test_draws"]),
                     "labels": list(P.TINY["test_labels"]), "a0": list(P.A0_TINY)},
            "package": {"blind_seed": 20261031, "n_pairs": 24, "sides": "secrets.randbelow(2)"},
            "b2_result_sha256": sha256(P.B2_RESULT),
            "power": {"path": E9 + "power_c1.json", "sims_per_cell": power["sims_per_cell"], "b2": power["b2"],
                      "cells": power["cells"]}}


def checks() -> dict[str, list[str]]:
    """What must hold before the addendum is written (each value lists problems; empty = held)."""
    import make_items_c1 as M

    out: dict[str, list[str]] = {}
    lines = M.read_lines()
    out["line_rules"] = [f"{k}: {v}" for k, v in M.rule_problems(lines).items()]
    man = M.build(lines)
    out["manifest_rebuilds"] = [] if P.C1_MANIFEST.read_text() == M.manifest_text(man) else [
        "manifest_c1.json differs from a fresh build"]
    out["melodies"] = M.melody_problems(man)
    tok = P.read_json(P.TOK_JSON)
    out["tok_check"] = [] if tok.get("passed") else [k for k, v in tok["items"].items() if v["problems"]]
    if set(tok["items"]) != set(P.all_items()):
        out["tok_check"].append("tok_check_c1.json does not cover the 98 items")
    out["tok_check"] += M.b2_tok_problems(tok)
    guides = P.read_json(P.GUIDES_JSON)
    out["guides"] = [] if set(guides["sha256"]) == set(P.all_items()) else ["guides_c1.json does not cover the 98 items"]
    out["guides"] += M.b2_guide_problems(guides)
    out["seeds"] = P.seed_problems()
    if P.read_json(P.SEEDS_JSON) != json.loads(json.dumps(P.seed_table())):
        out["seeds"].append("seeds_c1.json differs from plan_c1.seed_table()")
    power = P.read_json(HERE / "power_c1.json")
    out["power"] = [] if power.get("sims_per_cell") == POWER_SIMS else [f"power ran {power.get('sims_per_cell')} sims"]
    if power.get("source_sha256") != sha256(P.B2_RESULT):
        out["power"].append("power_c1.json was not simulated from the current b2_result.json")
    out["recipes"] = recipe_problems()
    out["earlier_addenda"] = earlier_problems()
    return out


def recipe_problems() -> list[str]:
    """S* and S*_b0 are B2's selection, and the four arms are B2's Stage T arms for that selection."""
    sel = json.loads((ROOT / "results" / "gpu_run3" / "b2_selection.json").read_text())
    out = [] if (sel["s_star"], sel["b_star"]) == (P.ARMS[0][1], P.ARMS[1][1]) else ["the recipes differ from B2's selection"]
    b2 = {lab: cid for lab, cid in P2.label_configs(P2.test_arms(P2.parse(sel["s_star"]), P2.parse(sel["b_star"]))).items()}
    out += [f"{lab} differs from B2's {b2.get(lab)}" for lab, cid in P.ARMS if b2.get(lab) != cid]
    return out


def earlier_problems() -> list[str]:
    """Addendum 2, the B1 code addendum and addendum 3 are current, and so are the files they register."""
    out = []
    for name, script in (("addendum 2", ROOT / "experiments" / "e5_motif" / "register.py"),
                         ("the B1 code addendum", ROOT / "experiments" / "e6_clock" / "register_b1code.py"),
                         ("addendum 3", E8_DIR / "register_b2.py")):
        try:
            rc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, cwd=ROOT).returncode
        except Exception as exc:
            rc = f"did not run: {exc!r}"
        if rc != 0:
            out.append(f"{name}: {script.name} --check {'fails' if isinstance(rc, int) else rc}")
    out += REG3.registration_problems()
    return out


def computed() -> dict:
    return {"refers_to": [str(p.relative_to(ROOT)) for p in (REG2, REG_B1, REG_3)], "earlier_addenda_sha256": earlier_record(),
            "custody": CUSTODY, "sections": [{"id": k, "text": t} for k, t in sections()],
            "fill_ins": [{"id": k, "text": t} for k, t in FILL_INS], "plan": plan_record(),
            "code_sha256": hashes(CODE + REUSED), "data_sha256": hashes(DATA), "tests_sha256": hashes(test_files())}


def form_problems(add: dict) -> list[str]:
    """Addendum 4 holds exactly its keys, its title and a UTC time stamp (the hashes cannot show an added key)."""
    out = [f"addendum 4 has an unexpected key {k!r}" for k in sorted(set(add) - set(KEYS))]
    out += [f"addendum 4 has no {k!r}" for k in KEYS if k not in add]
    if add.get("title") != TITLE:
        out.append("addendum 4's title differs")
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", str(add.get("addendum_utc", ""))):
        out.append("addendum 4's addendum_utc is not a UTC time stamp")
    return out


def registration_problems(add: dict | None = None) -> list[str]:
    """Run-time check (setup, generation, decision): addendum 4 has its form, the code and data hash to it, the three
    earlier addenda to the copies it names, and addendum 3's files (with the B1 and E6 files it checks) are current."""
    try:
        add = add or json.loads(ADDENDUM.read_text())
    except (OSError, ValueError):
        return [f"no addendum 4 ({ADDENDUM.relative_to(ROOT)})"]
    reg = {**add.get("code_sha256", {}), **add.get("data_sha256", {})}
    now = hashes(runtime_files())
    out = form_problems(add)
    out += [f"{f} differs from addendum 4" for f in runtime_files() if reg.get(f) is None or now[f] != reg[f]]
    named = add.get("earlier_addenda_sha256", {})
    out += [f"{f} differs from the copy addendum 4 names" for f, h in earlier_record().items() if named.get(f) != h]
    out += REG3.registration_problems()
    return out


def registered_runtime(add: dict | None = None) -> dict:
    add = add or json.loads(ADDENDUM.read_text())
    reg = {**add.get("code_sha256", {}), **add.get("data_sha256", {})}
    return {f: reg.get(f) for f in runtime_files()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--addendum", default=str(ADDENDUM), help="--check: the file to check")
    ap.add_argument("--out", default=str(ADDENDUM), help="write: where to write (a draft path for review)")
    a = ap.parse_args()
    bad = {k: v for k, v in checks().items() if v}
    new = computed()
    if a.check:
        try:
            cur = json.loads(Path(a.addendum).read_text())
        except (OSError, ValueError):
            cur = {}
        stale = sorted(k for k in new if new[k] != cur.get(k)) + (form_problems(cur) if cur else ["no addendum 4"])
        print("addendum 4:", "current" if not (stale or bad) else f"stale {stale} {bad}")
        return 1 if stale or bad else 0
    if bad:
        print("addendum 4 not written:", json.dumps(bad, indent=1))
        return 1
    out = {"addendum_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "title": TITLE, **new}
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    shown = dest.relative_to(ROOT) if dest.resolve().is_relative_to(ROOT) else dest
    print(f"wrote {shown}: {len(new['code_sha256'])} code files, {len(new['data_sha256'])} data files, "
          f"{len(new['tests_sha256'])} tests, addendum_utc {out['addendum_utc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
