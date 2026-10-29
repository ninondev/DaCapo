"""Write or check experiments/e5_motif/registration/addendum_3.json, the registration of B2: the text, the
sha256 of the B2 code, of the code it reuses, of its data (the lines, manifest, guides, tokenizer check,
seed table and power simulation) and of its tests, with the B1 code addendum's hashes listed again.

  python experiments/e8_clock_sing/register_b2.py           write it; refuses on a seed collision, a broken
                                                             line rule, a stale manifest or a failed check
  python experiments/e8_clock_sing/register_b2.py --check   exit 1 if the written file is not current

Run it as the last step before the commit. The B1 code addendum is committed and pushed first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "e6_clock"))
import plan_b2 as P  # noqa: E402
import readout as R  # noqa: E402

REG_DIR = ROOT / "experiments" / "e5_motif" / "registration"
ADDENDUM = REG_DIR / "addendum_3.json"
REG2 = REG_DIR / "addendum_2.json"
REG_B1 = REG_DIR / "addendum_2_b1code.json"
E8 = "experiments/e8_clock_sing/"
CODE = tuple(E8 + f for f in ("plan_b2.py", "make_items_b2.py", "run_b2.py", "score_b2.py", "select_b2.py",
                             "decide_b2.py", "package_b2.py", "register_b2.py", "power_b2.py")) + ("experiments/run_b.py",)
REUSED = ("dacapo/__init__.py", "dacapo/ace.py", "dacapo/seeds.py", "dacapo/words.py", "dacapo/metrics.py",
          "dacapo/render.py", "dacapo/symbolic.py", "dacapo/analysis.py", "dacapo/provenance.py", "dacapo/attn_bias.py",
          "dacapo/motif_integrity.py", "experiments/e6_clock/clock_score.py", "experiments/e6_clock/readout.py",
          "experiments/e6_clock/run_b1.py", "experiments/e4_sing/make_items.py", "experiments/e4_sing/run_sing.py",
          "experiments/e4_sing/score_sing.py", "experiments/e5_motif/stage.py", "experiments/e5_motif/v2_config.py")
DATA = tuple(E8 + f for f in ("lines_b2.txt", "manifest_b2.json", "guides_b2.json", "tok_check_b2.json", "seeds_b2.json",
                             "power_b2.json")) + ("experiments/e4_sing/manifest.json", "results/cpu/runb_cpu_test.json")
TESTS = E8 + "tests"
SHARED_B1 = ("dacapo/attn_bias.py", "experiments/e6_clock/clock_score.py")
POWER_SIMS = 2000

CUSTODY = ("This addendum is committed and pushed before Run B and before any B2 generation. The B1 code addendum "
           "(addendum_2_b1code.json) is committed and pushed first; addendum 3 follows it. That push is the external "
           "timestamp. Addendum 3 changes nothing that addendum 2 or the B1 code addendum registers.")

SECTIONS = [
    ("question", "Given a melody guide and a lyric, ACE-Step 1.5 turbo sings either the notes or the words and rarely "
                 "both: 4 of 504 E4 clips had NPA >= 0.5 and word recall >= 0.5. B2 asks whether the lyric clock lets the "
                 "model sing new melodies with the right notes and recognisable words. The clock is a pre-softmax bias "
                 "that ties each word's lyric tokens to its written note window. The claim is about a pipeline a user "
                 "could run: one recipe fixed on development lines, the user's melody rendered as a guide, the user's "
                 "lyric, 4 draws, and an automatic selector that keeps the draw with the largest q. The selector needs "
                 "only the melody and the lyric. The rationale is a hypothesis. Without a guide the model sings these "
                 "lines early and fast. Under a moderate guide, the guide's note times and the model's own lyric timing "
                 "disagree, and E4's outputs settle this by dropping the words. The clock moves the lyric attention onto "
                 "the guide's note times, so the two conditions point the same way."),
    ("design", "Stage D (development) picks one clocked recipe, S*, and one unclocked recipe, B*, on the 12 E4 lines. "
               "Stage T (test) runs S*, its controls and the unclocked comparators on 24 new lines with new melodies and "
               "new seeds, 4 draws each, and decides. Registered code (select_b2.py) computes the selection from the "
               "Stage D rows, and the selection is fixed before Stage T starts. B1 runs first, as addendum 2 and the B1 "
               "code addendum fix it; B2 takes nothing from B1's results."),
    ("generation", "acestep-v15-turbo at ca1e85f with the weights of addendum 2, 8 steps, shift 3.0, DCW off, batch 1, "
                   "no language model, sdpa attention. Caption 'a cappella, solo female vocal, no instruments, no drums, "
                   "dry studio recording, clear lyrics'; bpm 100; keyscale the item's key; vocal language en; lyrics "
                   "'[Verse]\\n' + line. Text2music runs are 14.0 s. Guided runs use task cover-nofsq, "
                   "audio_cover_strength k/8, cover_noise_strength 0 and the guide WAV as source, so their length follows "
                   "the guide. The E5 v2 determinism settings apply (setup_determinism, VAE decode chunk 256), and the "
                   "global RNG is pinned to rng_seed right before each run. Clips are written by soundfile as 16-bit PCM "
                   "WAV; pred_latents and target_latents are kept as float32 .npy."),
    ("items", "Dev: item00-item11 of experiments/e4_sing/manifest.json (registered in addendum 2). Test: b2t00-b2t23 of "
              "experiments/e8_clock_sing/manifest_b2.json, built by make_items_b2.py from the 24 lines of lines_b2.txt "
              "with E4's rhythm() and pitches() imported unchanged, numpy default_rng(22), keys C, D, F and G in turn, "
              "100 BPM, a 4-beat lead-in, 16 beats, MIDI 57-76, each note sounding for 0.9 of its length, one note per "
              "word. The lines were written by the agents. The line rules are checked in code before registration: "
              "11-13 words; every word one syllable (make_items_b2.syllables), letters only and capitalised as in E4's "
              "lines (the first word and I); no line from E4 or E6; every word 1-2 Qwen3 tokens; the tokenizer check "
              "passes (the header gives s = 9, the token words equal the lyric words, and the word count equals the "
              "note count); no word from the registered homophone list (sea, see, night, knight, one, won, sun, son, "
              "hear, here, road, rode, whole, hole, by, buy, new, knew, right, write, red, read, tale, tail, blue, blew, "
              "week, weak, made, maid, meet, meat); none of thank, you, watch, subscribe. A failing line is rewritten "
              "before registration, never after."),
    ("guides", "FluidSynth 2.3.4 with FluidR3_GM.sf2 (sha256 74594e8f4250680adf590507a306655a299935343583256f3b722c48a1"
               "bc1cb0, as in E5 v2), 48 kHz, gain 0.6, reverb and chorus off (dacapo/render.py), program 53 (Voice Oohs) "
               "and program 71 (clarinet). One guide per item and timbre: 36 x 2 = 72 WAVs, E4's guides included because "
               "they were never stored. Their sha256 are in guides_b2.json and in section plan.guides_sha256. They are not "
               "in git; Run B renders them again in setup and stops on any mismatch. item00's Voice Oohs guide is the "
               "guide the CPU test rendered."),
    ("seeds", "Stage D: (dit_seed, rng_seed) = dacapo.seeds.run_seeds('runB2-dev', item_id, r), r = 0, 1, 2. Stage T: "
              "run_seeds('runB2-test', item_id, r), r = 0-3. GPU A0: run_seeds('runB2-a0', 'item00', r), r = 0, 1. The "
              "tiny test scale uses the tags runB2-dev-tiny, runB2-test-tiny and runB2-a0-tiny. All arms and configs of "
              "an (item, draw) share the pair. seeds_b2.json holds every pair. register_b2.py checks that no new DiT seed "
              "or pin equals another new one or any earlier one: E4 {1, 2}; E5 v1 {1001, 2002} and the reserved "
              "{1001, 2002, 20261027}; the E5 v2 seed table and every E5 v2 tiny tag; E6 61000 + 10i + r and "
              "62000 + 100v + 10w + r with their +900000 pins; E6 acceptance seeds 6000 and 6001 with their pins; B1 "
              "63000 + 10i + r with their pins; the CPU test, 6002 and 906002; 42, 43, 44, 20261027 and 20261029; and "
              "the seed tags of E7 Stage S. It also checks that dev and test share no seed. On a collision it refuses "
              "and never re-draws. There was no collision."),
    ("intervention", "The clock is dacapo/attn_bias.py of the B1 code addendum, unchanged: forward pre-hooks on "
                     "decoder.layers[L].cross_attn for L = 2-6, at all 8 steps, in both the cover and the non-cover "
                     "phase. Word j's tokens (keys 9 + token index; tok_check_b2.json lists them per item) get +beta in "
                     "query frames whose centre, 0.08q + 0.04 s, lies in [onset_j, onset_(j+1)); the last word's window "
                     "runs to its note end. Untouched: lead-in frames, frames after the last note, the [Verse] tag, the "
                     "header, timbre and caption keys, and all other heads. Head sets: A = the seven alignment heads "
                     "(L2h6, L3h10, L3h11, L4h3, L5h8, L5h9, L6h8); W = all 16 heads of layers 2-6. Beta 2, 4 and 8 are "
                     "exact in bf16. 'No hook' means the context manager is not entered (see the fallback in section "
                     "run). For a head that gives a word probability p in its window, beta raises it to "
                     "p e^beta / (p e^beta + 1 - p): from p = 0.10, 0.45 at beta 2, 0.86 at beta 4 and 0.997 at beta 8."),
    ("stage_d", "30 configs: guide {oohs, clarinet} x k {3, 4, 5} x clock {none, A2, A4, A8, W2}, where A4 means head "
                "set A at beta 4. 12 items x 3 draws gives 1,080 runs, in the order item, draw, config. A Stage D run "
                "that still fails after 3 attempts, or is not scored, counts as q = 0 and gate failed."),
    ("selection", "select_b2.py, a pure function of the Stage D rows. For config c and item i, bo3_ci is the largest q "
                  "over the 3 draws, and J_dev(c) is the number of items with bo3_ci >= 0.5. A clocked config is "
                  "eligible if its vocal-gate rate over its 36 clips is at least 0.60. S* is the eligible clocked config "
                  "with the highest key (J_dev, mean_i bo3_ci, mean per-clip q); remaining ties go, in order, to fewer "
                  "steered heads, smaller beta, smaller k, and oohs before clarinet. B* is the unclocked config with the "
                  "highest value of the same key. If no clocked config is eligible, Stage T is not run, and B2 is 'not "
                  "supported' with the label 'clock silences the voice'. select writes b2_selection.json with S*, B*, the "
                  "full dev table and the sha256 of the Stage D ledgers and of the code; the file has no time stamp, so "
                  "the same rows give the same file. select writes the file's sha256 as the first line of the Stage T "
                  "ledger. Stage T refuses to start without that line, and every Stage T row carries the hash. select is "
                  "idempotent and refuses to change once a Stage T row exists. decide recomputes the selection from the "
                  "Stage D rows and treats a mismatch as a P0 failure."),
    ("stage_t", "Each arm gets 4 draws on each of the 24 test items. Arms that coincide are merged, which leaves 5-8 arms "
                "and 480-768 runs, in the order item, draw, arm. 1. S*. 2. S*_b0: S*'s guide and k, no hook; it isolates "
                "the clock. 3. S*_rand, only if S* steers set A: S*'s guide, k and beta on E6 random set 3 + r for draw r "
                "(readout.random_head_sets()[3 + r]). 4. S*_shuf: S*'s guide, k, heads and beta, with the windows rotated "
                "so that word j's tokens get the window of word (j + floor(n/2)) mod n: the same heads and nearly the "
                "same number of biased pairs, with the wrong timing. 5. ctx_oohs_k3, no hook: E4's best mean best-of-2 q. "
                "6. ctx_oohs_k4, no hook: E4's arm with the most joint clips. 7. noguide, no hook: the reference for words "
                "and the vocal gate, not a comparator. 8. B*, if it is none of the above."),
    ("measures", "One hashed scorer (score_b2.py on clock_score.py) computes everything; nothing is scored by hand. "
                 "Vocal gate: E6's, unchanged (htdemucs vocal stem at least 20% of stem energy, at least 1.0 s of its 50 "
                 "ms frames at -40 dBFS or more, pre-normalisation peak at least 0.01; a missing peak fails). NPA_v: E4's "
                 "note_pitch_accuracy (dacapo/metrics.py through experiments/e4_sing/score_sing.py pitch_scores) on "
                 "SwiftF0 of the htdemucs vocal stem: confidence >= 0.5, 65-1500 Hz, one global lag within +-0.5 s from "
                 "frame_metrics, the median f0 over each note's central 60%, +-50 cents, absolute octave; unvoiced notes "
                 "count as wrong. WR: lyric words matched in order (dacapo.words.match_words) to the word chunks of "
                 "Whisper large-v3, divided by the number of lyric words; Whisper runs at revision "
                 "06f233fe06e710322aca913c1bc4249a0d71fce1, fp16, greedy, English, with word timestamps, on the full mix, "
                 "and the scorer raises on the CPU and on any other model or revision. q = gate x min(NPA_v, WR). A clip "
                 "is joint if q >= 0.5 and strict if q >= 0.75. Item outcome: bo4_i is the q of the draw the selector "
                 "keeps; J_i = 1 if bo4_i >= 0.5; J(arm) is the mean of J_i over the 24 items. Selector: keep the draw "
                 "with the largest q; ties go to higher WR, then higher NPA_v, then the lower draw index. Secondary "
                 "measures, reported and never in the rule: NPA on the mix, octave-tolerant NPA, and NPA against the "
                 "other melodies of the set (23 for test, 11 for dev: chance); E4's multiset WR and capped WER; onset "
                 "error, the median |Whisper word onset - written onset| over matched words, signed and absolute, for "
                 "clips with at least 4 matched words; ordered recall in the first half, the second half and on the last "
                 "word; gate rate; the vowel-only rate; the wordless rate; the guide-copy index; instrument bleed; WR from "
                 "the pinned CTC model facebook/wav2vec2-large-960h-lv60-self at revision "
                 "54074b1c16f4de6a5ad59affb4caa8f2ea03a119 (fp32, greedy) on the mix; best-of-n curves for n = 1-4."),
    ("rule", "decide_b2.py applies the rule to Stage T only. Values are rounded to 1e-6 before comparison. Intervals "
             "come from a cluster bootstrap over items: 10,000 resamples, numpy default_rng(0), draws kept within items, "
             "all arms resampled together. P0, integrity; all of these must hold: the code, manifest, guide, soundfont "
             "and seed-table hashes match addendum 3 at generation (every run row records them, and the sha256 of the "
             "guide it used) and at decision, when setup has rendered the guides again with the registered soundfont, "
             "dacapo/attn_bias.py and experiments/e6_clock/clock_score.py match the B1 code addendum, and the addendum "
             "2 files are unchanged; the selection recomputed from the Stage D rows equals b2_selection.json, its hash "
             "is the first line of the Stage T ledger, written before the first Stage T generation, and every Stage T "
             "row carries it; the CPU beta 0 test passed before registration; the GPU A0 check is recorded, it ran on the "
             "registered code, and every B2 row follows its report; if it found a difference, the registered fallback "
             "ran: every unclocked B2 run, in Stage D and Stage T, has the hook attached at beta 0 on set A (B1's "
             "no-hook arm stays as registered); every hooked run made 40 hook calls with max biased key < n_lyr, and "
             "every unhooked run made none; within each (item, draw, guide), every cover run received the same guide "
             "latent (sha256 of target_latents), whatever its k; every scored clip, in Stage D and Stage T, used the "
             "pinned Whisper revision, and every generated clip has its pre-normalisation peak; at most 5% of "
             "planned Stage T runs are missing, failed after 3 attempts with the same seeds, or unscored, and these "
             "count as q = 0. P1, clock credit; both must hold: Delta_J = J(S*) - max_b J(b) >= 0.20, where b ranges "
             "over the unclocked guided arms run in Stage T (S*_b0, ctx_oohs_k3, ctx_oohs_k4 and B*), so the comparator "
             "is the best of them on the test data; and the 2.5th percentile of the bootstrap Delta_J, with the max "
             "retaken in every resample, is above 0. P2, capability: J(S*) >= 0.50, a joint clip for at least 12 of "
             "the 24 melodies. Verdicts: 'B2 supported: the lyric clock gives the capability' if P0, P1 and P2 hold; "
             "'Clock helps, capability not reached' if P0 and P1 hold and P2 fails; 'Capability reached, clock not "
             "credited' if P0 and P2 hold and P1 fails; 'Not supported' if P0 holds and both P1 and P2 fail, with the "
             "failed conditions named; 'Technical' if P0 fails, and no effect is claimed either way."),
    ("labels", "Labels never change a verdict. Strict capability: S* has bo4 >= 0.75 on at least 12 of 24 items. "
               "Head-specific clock (only if S* steers set A): mean bo4 q(S*) - mean bo4 q(S*_rand) >= 0.5 x (mean bo4 "
               "q(S*) - mean bo4 q(S*_b0)); otherwise 'nonspecific bias'. Timing-specific: the 95% lower bound of "
               "mean_i [bo4 q(S*) - bo4 q(S*_shuf)] is above 0; 'emphasis, not timing' if J(S*_shuf) - max_b J(b) >= 0.20 "
               "and that bound is at most 0. Timbre copy: S*'s vowel-only rate over its 96 clips is at least 0.30. Words "
               "cut: S*'s second-half recall is at least 0.20 below its first-half recall. Dev futility: S* does not rank "
               "above B* on the dev selection key; Stage T runs anyway. Frontier moved: mean NPA_v(S*) >= mean "
               "NPA_v(S*_b0) - 0.10 and mean WR(S*) >= mean WR(S*_b0) + 0.20. Independent ASR: P2 recomputed with the CTC "
               "model's WR on the same selected draws."),
    ("analysis", "For every arm: J; joint and strict clips out of 96; mean bo4 q and mean per-clip q; NPA_v, mix NPA, WR "
                 "and gate rate; vowel-only and wordless rates; onset errors and recall by position; guide-copy index "
                 "and bleed rate; CTC WR; best-of-n curves. Paired contrasts with 95% intervals: mean bo4 q and J of S* "
                 "against each other arm. S* - S*_b0 is the clock's effect at a fixed recipe; S* - S*_rand and "
                 "S* - S*_shuf are the specificity contrasts. The dev table is reported in full with the selection."),
    ("reporting", "Every planned clip, failures included, with its metrics, in item order. Demo by rule: for each of the "
                  "24 test items, the S* draw the selector keeps, next to the primary comparator's kept draw. The primary "
                  "comparator is the unclocked guided arm with the highest J on test; ties go to mean bo4 q, then to the "
                  "arm order of section stage_t. All 48 clips are published. A short reel, if one is made, is the 6 items "
                  "with the highest S* q (ties go to item order), labelled 'top 6 of 24 by rule', with the full set "
                  "linked. A blind A/B pack of the same 24 pairs asks 'melody right' and 'words clear'. The pair order "
                  "comes from numpy default_rng(20261029); which clip of a pair is A comes from the operating system's "
                  "random source and is written only to the key, so the code and the seed do not reveal it, and "
                  "packaging again keeps the sides of the existing key. The pack and its key are separate files "
                  "outside the package, and listeners get the pack before they hear the demo. The pack is "
                  "exploratory. There is no owner demo."),
    ("success", "'B2 supported' requires P0, P1 and P2. In plain terms, with one recipe fixed before the test, 4 draws "
                "and the automatic selector, ACE-Step 1.5 turbo sings at least half of 24 unseen melodies with at least "
                "half the notes right on the vocal stem and at least half the words recognised in order, and does so "
                "for at least 5 more melodies than the best unclocked recipe chosen with hindsight. The labels say "
                "whether the gain belongs to the alignment heads, to the timing, or to neither."),
    ("disclosures", "E4's results chose the guide range, the k range and the comparators. No E6 read-out clip, onset or "
                    "table enters B1 or B2; E6 figures appear only as background. The owner chose to include W2 in the "
                    "Stage D grid; there is no owner demo. The owner approved one Colab L4 session of about 5-6 hours on "
                    "the spec's time estimate, which left out Whisper's repetition loops on wordless clips. With the "
                    "loop allowance of section run, the run is expected to take about 10-13 hours after the registered "
                    "trim, and a run that Colab stops resumes from Drive in a new session. On 2026-10-29, before "
                    "registration, the owner accepted this estimate instead of a smaller design. Before registration, the "
                    "B2-type generations were these. The CPU layout test (item00, seed 6002) was not scored and not "
                    "listened to. The tiny test chain (experiments/run_b.py --scale tiny --device cpu) ran several "
                    "passes on this machine's CPU to test the code, with the tiny seed tags. Each pass made 2 A0 runs and "
                    "2 Stage D runs on item00, and 2 Stage T runs: on the test item b2t00 (Voice Oohs k4, with and "
                    "without the clock) in the passes before review, and on the dev item item01 since then, so the tiny "
                    "scale no longer touches a test item. Each pass also made B1's two tiny runs (see the B1 code "
                    "addendum). The clips were scored with Whisper small.en and the pinned CTC model on the CPU, and every "
                    "verdict was 'test only'. Nobody listened to them; their transcripts were read only to check the "
                    "code, no setting or rule changed because of what they contained, and they were deleted. A review "
                    "noted Whisper loops in some of these transcripts; the loop allowance added after it rests only on "
                    "Run A's E4 and E6 records. No B2 setting changes after the first B2 generation, apart from the "
                    "registered timing trim (section run). Any deviation goes into the ledger and the write-up."),
    ("run", "experiments/run_b.py runs every stage and resumes: setup, a0, b1, b2dev, select, b2test, decide, package. "
            "setup checks every hash (addendum 2, the B1 code addendum, addendum 3), runs readout.py check-gate with "
            "E6_RES=results/gpu_run2 (B1 needs it), checks the soundfont, renders the 72 guides again and compares their "
            "sha256, checks the seed table, and loads the model to record sdpa, the alignment heads, the determinism "
            "flags, the ACE-Step commit, the weights' sha256 and the lyric keys of all 36 items through ACE-Step's own "
            "path. A0 runs on the GPU with the runB2-a0 seeds on item00: text2music without the hook twice "
            "(determinism, recorded), text2music with the hook at beta 0 on set A, Voice Oohs k4 without the hook, with "
            "the hook at beta 0 and with beta 2, and Voice Oohs k3 and k5 without the hook. It requires 8 denoiser "
            "calls, the peak, 40 hook calls with every key below n_lyr, equal lyric rows of the encoder states at step "
            "0 (cover) and step 4 (non-cover), one guide latent across the cover runs of k3, k4 and k5, a change of "
            "pred_latents at beta 2, and the registered scorer on all A0 clips (Whisper revision checked). If either "
            "beta 0 run differs from its no-hook run (torch.equal on pred_latents), the registered fallback applies to "
            "every unclocked B2 run, and B1 gets a note. Timing: for guided and for text2music runs separately, A0 "
            "takes the median generation seconds (leaving out the first run, which warms up the GPU) plus the median "
            "scoring seconds, and adds an allowance for Whisper's repetition loops ('oh, oh, ...' up to its 448-token "
            "limit): 46 s on 25% of guided runs and on 8% of text2music runs. On Run A's L4 such a clip took about 50 "
            "s to score against a median of 4.1 s (results/gpu_run2/e6_scores.json), and loops hit 22 of 96 guided "
            "E4 clips at k3 and k4 and 6 of 88 E6 text2music clips. A0 projects hours = 35/60 + (text2music runs x "
            "text2music seconds + guided runs x guided seconds) / 3600, where B1's 180 runs and Stage T's noguide arm "
            "are text2music and Stage D and the other seven Stage T arms are guided (Stage T at its largest, 8 arms). "
            "Past 7.0 h, A0 drops the five clarinet-k3 configs (-180 runs), and if still past 7.0 h, noguide from "
            "Stage T (-96 runs); the trim is fixed before Stage D. A0 runs once: a later runtime reuses its report "
            "while the report, its inputs and the code still hash to what A0 recorded. Stop rules: a setup or A0 "
            "failure before any B1 or B2 generation stops the run (package still collects the logs); the same "
            "failure on a resumed run stops generation, and B1's decision, decide and package still run; a B1 "
            "problem does not stop B2 unless the shared code fails its hash check; Stage T needs the selection line "
            "in its ledger; once B1 and B2 are under way, decide always runs and writes 'Technical' when P0 fails. "
            "Resume: each stage keeps an append-only JSONL ledger; a run is done when its WAV and latents hash to "
            "its ledger row, and a clip is scored when its score row names the WAV's sha256; a failed run gets at "
            "most 3 attempts with the same seeds. The tiny scale runs every stage with 1-2 runs each, on the CPU, "
            "with Whisper small.en and its own seed tags; its Stage T runs on the dev item item01, and its verdicts "
            "are 'test only'. The full scale refuses the CPU and test flags. Registered outputs go to "
            "results/gpu_run3/; audio and packs stay outside the repository."),
]

FILL_INS = [
    ("names", "Configs are named guide_kK_clock: oohs_k3_A2 is the Voice Oohs guide at k = 3 with beta 2 on set A; "
              "R2 marks beta 2 on random set 3 + r, and the suffix _shuf the rotated windows; noguide is text2music. "
              "Stage T arms carry the labels of section stage_t; a merged run carries all of its labels."),
    ("missing", "A planned clip that is missing, failed after 3 attempts or not scored counts as q 0, gate failed, WR 0, "
                "NPA_v 0, mix NPA 0 and CTC WR 0. The vowel-only, wordless and bleed rates and the guide-copy index are "
                "reported over scored clips; the timbre-copy label divides S*'s vowel-only clips by all 96 planned clips. "
                "A missing, failed or unscored clip is not vowel-only; a scored clip with an empty transcript is."),
    ("positions", "The first half of a line is its first floor(n/2) words, the second half the rest; the last word is "
                  "word n. Recall at a position set uses the same in-order matching (clock_score.ordered_recall)."),
    ("text", "Vowel-only: the transcript has no word, or at least half of its normalised words fully match "
             "(o+h*|a+h+|(la)+|m+|h+m+). Wordless: vowel-only, or the transcript is one or more of the registered E4 "
             "hallucination strings (thank you for watching, transcript emily beynon, satsang with mooji, thank you, oh "
             "my god, bf watch tv, bells chiming, oh god, oh yeah, so, um) with only vowel words besides. E4's capped "
             "WER compares lower-case lyric words with the transcript (score_sing compared the capitalised words)."),
    ("audio", "Guide-copy index: the largest Pearson correlation, over lags within +-0.5 s, between the log-mel "
              "spectrograms (16 kHz, 64 mels, 10 ms hop) of the vocal stem and the guide, on the frames where the guide "
              "is within 40 dB of its loudest frame; noguide clips are compared with the item's Voice Oohs guide as a "
              "reference level. Bleed: the non-vocal stems hold more than half of the stem energy. CTC: greedy argmax "
              "decoding of the mix resampled to 16 kHz mono."),
    ("best_of_n", "Best-of-n averages, over items and over every subset of n of the 4 draws, the largest q (and the "
                  "share of items whose largest q is at least 0.5)."),
    ("labels", "The head-specific label needs a positive clock effect: if mean bo4 q(S*) <= mean bo4 q(S*_b0), it reads "
               "'no clock effect to attribute'. The timing bound is the 2.5th percentile of the same item bootstrap; when "
               "neither 'timing-specific' nor 'emphasis, not timing' applies, the label reads 'timing not shown'. Dev "
               "futility compares (J_dev, mean bo3, mean q) only."),
    ("selection", "decide compares S*, B*, the config list, the trim and the dev table recomputed from the Stage D rows "
                  "with b2_selection.json."),
    ("a0", "A0's hard checks stop the run on failure; the determinism repeat and the timing are recorded. Seconds per "
           "run, for guided and for text2music runs separately, are the median generation seconds of that kind's A0 "
           "runs (without the first A0 run) plus the median scoring seconds of its A0 clips, plus the loop allowance "
           "of section run; a kind A0 did not run takes the other kind's medians. Every B2 row records the sha256 of "
           "the A0 report it followed, and the report records the code it ran on."),
    ("rounding", "Values are rounded to 1e-6 before every comparison, as in E6 and B1. Percentiles are numpy's linear "
                 "interpolation."),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hashes(files) -> dict[str, str | None]:
    return {f: sha256(ROOT / f) if (ROOT / f).exists() else None for f in files}


def runtime_files() -> tuple[str, ...]:
    """The files every B2 generation row records and decide compares with this addendum."""
    return CODE + REUSED + DATA


def test_files() -> list[str]:
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / TESTS).glob("*.py"))


def b1code_record() -> dict:
    b1 = json.loads(REG_B1.read_text())
    return {"path": str(REG_B1.relative_to(ROOT)), "sha256": sha256(REG_B1), "addendum_utc": b1.get("addendum_utc"),
            "code_sha256": b1["code_sha256"], "tests_sha256": b1["tests_sha256"],
            "cpu_test": {k: b1["cpu_test"][k] for k in ("path", "sha256", "passed", "guide_sha256")}}


def plan_record() -> dict:
    guides = P.read_json(P.GUIDES_JSON)
    tok = P.read_json(P.TOK_JSON)
    power = P.read_json(HERE / "power_b2.json")
    return {"dev_items": [it["id"] for it in P.dev_items()], "test_items": [it["id"] for it in P.test_items()],
            "lines": P.LINES.read_text().splitlines(),
            "stage_d_configs": [c["config"] for c in P.dev_configs()], "n_stage_d_runs": len(P.dev_runs()),
            "stage_t_arm_order": list(P.ARM_ORDER), "comparators": list(P.COMPARATORS),
            "a0_runs": [{"name": n, "draw": r, "config": c, "beta": b, "why": w} for n, r, c, b, w in P.A0_RUNS],
            "head_sets": {"A": {str(k): v for k, v in R.ALIGN.items()}, "W": "all 16 heads of layers 2-6",
                          "R": {str(3 + r): {str(k): v for k, v in R.random_head_sets()[3 + r].items()} for r in P.TEST_DRAWS}},
            "tags": P.TAGS, "seed_table_sha256": sha256(P.SEEDS_JSON),
            "guides_sha256": guides["sha256"], "soundfont_sha256": guides["soundfont_sha256"],
            "fluidsynth": guides["fluidsynth"],
            "lyric_keys": {k: {"n_lyr": v["n_lyr"], "start": v["start"], "keys": v["keys"]} for k, v in tok["items"].items()},
            "trim": {"budget_hours": P.BUDGET_HOURS, "fixed_hours": round(P.FIXED_HOURS, 6), "b1_runs": P.N_B1,
                     "loop_extra_s": P.LOOP_EXTRA_S, "loop_rate": dict(P.LOOP_RATE)},
            "tiny": {"dev": [list(x) for x in P.TINY["dev"]], "test_items": list(P.TINY["test_items"])},
            "power": {"path": E8 + "power_b2.json", "sims_per_cell": power["sims_per_cell"], "cells": power["cells"]}}


def checks() -> dict[str, list[str]]:
    """What must hold before the addendum is written (each value lists problems; empty = held)."""
    import make_items_b2 as M

    out: dict[str, list[str]] = {}
    lines = M.read_lines()
    old = M.earlier_lines()
    out["line_rules"] = [f"{ln}: {p}" for ln in lines if (p := M.line_problems(ln, old))]
    if len(lines) != 24:
        out["line_rules"].append(f"{len(lines)} lines")
    out["manifest_rebuilds"] = [] if P.B2_MANIFEST.read_text() == M.manifest_text(M.build(lines)) else [
        "manifest_b2.json differs from a fresh build"]
    tok = P.read_json(P.TOK_JSON)
    out["tok_check"] = [] if tok.get("passed") else [k for k, v in tok["items"].items() if v["problems"]]
    if set(tok["items"]) != set(P.all_items()):
        out["tok_check"].append("tok_check_b2.json does not cover the 36 items")
    out["seeds"] = P.seed_problems()
    if P.read_json(P.SEEDS_JSON) != json.loads(json.dumps(P.seed_table())):
        out["seeds"].append("seeds_b2.json differs from plan_b2.seed_table()")
    guides = P.read_json(P.GUIDES_JSON)
    b1 = json.loads(REG_B1.read_text())
    out["cpu_test"] = [] if b1["cpu_test"]["passed"] else ["the CPU test did not pass"]
    if guides["sha256"]["item00"]["oohs"] != b1["cpu_test"]["guide_sha256"]:
        out["cpu_test"].append("item00's Voice Oohs guide differs from the CPU test's guide")
    if sha256(ROOT / b1["cpu_test"]["path"]) != b1["cpu_test"]["sha256"]:
        out["cpu_test"].append("results/cpu/runb_cpu_test.json differs from the B1 code addendum")
    power = P.read_json(HERE / "power_b2.json")
    out["power"] = [] if power.get("sims_per_cell") == POWER_SIMS else [f"power ran {power.get('sims_per_cell')} sims"]
    out["b1_code_addendum"] = b1_problems()
    out["addendum_2"] = e6_problems()
    return out


def b1_problems() -> list[str]:
    try:
        rc = subprocess.run([sys.executable, str(ROOT / "experiments" / "e6_clock" / "register_b1code.py"), "--check"],
                            capture_output=True, text=True, cwd=ROOT).returncode
    except Exception as exc:
        return [f"register_b1code.py --check did not run: {exc!r}"]
    out = [] if rc == 0 else ["register_b1code.py --check fails"]
    reg = json.loads(REG_B1.read_text())["code_sha256"]
    out += [f"{f} differs from the B1 code addendum" for f in SHARED_B1 if sha256(ROOT / f) != reg.get(f)]
    return out


def e6_problems() -> list[str]:
    reg = R.registered_hashes(REG2)
    now = R.code_hashes()
    return [] if reg == now else [f"{f} differs from addendum 2" for f in now if reg is None or reg.get(f) != now[f]]


def computed() -> dict:
    return {"refers_to": [str(REG2.relative_to(ROOT)), str(REG_B1.relative_to(ROOT))],
            "addendum_2_sha256": sha256(REG2), "b1_code_addendum": b1code_record(), "custody": CUSTODY,
            "sections": [{"id": k, "text": t} for k, t in SECTIONS], "fill_ins": [{"id": k, "text": t} for k, t in FILL_INS],
            "plan": plan_record(), "code_sha256": hashes(CODE + REUSED), "data_sha256": hashes(DATA),
            "tests_sha256": hashes(test_files())}


def registration_problems(add: dict | None = None) -> list[str]:
    """Run-time check (setup, generation, decision): the code and data hash to addendum 3, the shared B1 files
    to the B1 code addendum, the E6 files to addendum 2, and both earlier addenda to the copies addendum 3 names."""
    try:
        add = add or json.loads(ADDENDUM.read_text())
    except (OSError, ValueError):
        return [f"no addendum 3 ({ADDENDUM.relative_to(ROOT)})"]
    reg = {**add.get("code_sha256", {}), **add.get("data_sha256", {})}
    now = hashes(runtime_files())
    out = [f"{f} differs from addendum 3" for f in runtime_files() if reg.get(f) is None or now[f] != reg[f]]
    if sha256(REG2) != add.get("addendum_2_sha256"):
        out.append("addendum_2.json differs from the copy addendum 3 names")
    if sha256(REG_B1) != add.get("b1_code_addendum", {}).get("sha256"):
        out.append("addendum_2_b1code.json differs from the copy addendum 3 names")
    b1 = json.loads(REG_B1.read_text())["code_sha256"]
    out += [f"{f} differs from the B1 code addendum" for f in SHARED_B1 if now.get(f) != b1.get(f)]
    out += e6_problems()
    return out


def registered_runtime(add: dict | None = None) -> dict:
    add = add or json.loads(ADDENDUM.read_text())
    reg = {**add.get("code_sha256", {}), **add.get("data_sha256", {})}
    return {f: reg.get(f) for f in runtime_files()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bad = {k: v for k, v in checks().items() if v}
    new = computed()
    if a.check:
        try:
            cur = json.loads(ADDENDUM.read_text())
        except (OSError, ValueError):
            cur = {}
        stale = sorted(k for k in new if new[k] != cur.get(k))
        print("addendum 3:", "current" if not (stale or bad) else f"stale {stale} {bad}")
        return 1 if stale or bad else 0
    if bad:
        print("addendum 3 not written:", json.dumps(bad, indent=1))
        return 1
    out = {"addendum_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "title": "Addendum 3: B2 of Run B, the lyric clock with a melody guide (clocked cover)", **new}
    ADDENDUM.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {ADDENDUM.relative_to(ROOT)}: {len(new['code_sha256'])} code files, {len(new['data_sha256'])} data "
          f"files, {len(new['tests_sha256'])} tests, addendum_utc {out['addendum_utc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
