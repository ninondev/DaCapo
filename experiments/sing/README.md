# Sing my melody

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninondev/DaCapo/blob/main/notebook_sing.ipynb)

Give it a melody and one word per note, and ACE-Step 1.5 (turbo) sings those words on that melody. It makes a few
takes, scores each one the way Run B did, and keeps the best. The recipe is the one Run B picked for B2, called
S* (`oohs_k3_A8`): a wordless Voice Oohs guide of your melody steers the first 3 of the 8 sampling steps, and the
lyric clock adds a fixed bias on seven cross-attention heads so each word leans toward the time of its note.

The notebook is [`notebook_sing.ipynb`](../../notebook_sing.ipynb) (Colab, an L4 or A100 GPU required). The code
is [`sing.py`](sing.py).

## What it is and what it is not

It is a small creator tool, the first and simplest version of "give it your melody and your words, and it sings
them". It is not a study. Nothing in this folder is registered, its scores are not evidence about the lyric clock,
and it is not a product.

In Run B this recipe reached 14 of 24 new melodies by Whisper: with the best of 4 takes kept, 14 melodies had at
least half the notes on pitch and half the words heard in order (q >= 0.5). Taken one at a time, 31 of 96 takes got
that far, about one in three. The same guide without the clock reached 8 of 24 (13 of 96 single takes). The
registered rule did not credit the clock for the difference: Delta_J was 0.25, but the 2.5th percentile of its
bootstrap was exactly 0 and the rule needed it above 0. Scored with a second recogniser, a wav2vec2 CTC model,
only 2 of the 24 kept takes passed, so it did not confirm Whisper. Only 2 of the 96 single takes reached q >= 0.75.
Expect about one in three single takes to get half the notes and half the words; the best of 4 does better.
[`results/gpu_run3`](../../results/gpu_run3) holds Run B's files.

People who will take part in a blind listening test of Da Capo should not use the tool before they rate.

## What works and what does not

- One syllable per note. The tool warns about longer words and refuses a lyric whose word count differs from the
  note count. Rests take no word.
- Words without punctuation. A word must start with a letter a-z: ACE-Step's tokenizer joins a word that starts
  with a quote or a bracket (`'cause`, `(oh)`) to the word before it, so the tool refuses it. Other marks are
  sent as written, with a warning; B2's lyrics had none.
- A word's clock window runs from its note's onset to the next note's onset, the rule B2 used. After a rest,
  this means the window of the word before the rest stretches across it, so the clock keeps pulling that word
  during the rest. B2's melodies had no rests, so this was never tried, and the tool warns when it happens.
- Short phrases, up to about 20 seconds, work best. The tool warns past 20 s and stops at 60 s.
- Long words, melismas (one syllable over several notes) and fast passages are not handled. Notes shorter than
  0.25 s get a warning. The clock works in steps of 0.08 s, so a word whose window falls between two steps gets
  no clock at all; the tool names such words.
- English only, as in Run B.
- B2 sang in MIDI 57-76 (A3-E5) with a female voice. The male voice only changes the caption from "solo female
  vocal" to "solo male vocal"; Run B never tried it, and the tool warns when notes fall outside A2-E4 for it.
- The clip length follows the guide: a 4-beat lead-in, the melody, and about 3 s of the guide's tail.
- It needs a GPU of compute capability 8.0 or newer (L4, A100, H100); Run B ran on an L4. On older GPUs such as
  the T4 or V100, ACE-Step switches to float16 and eager attention, which Run B never used, so the tool stops
  before loading the model. In Colab, the L4 and A100 need compute units (Colab Pro or pay as you go); the free
  T4 does not work.

## Input

Melody as text: notes separated by spaces, each a name with octave (C4 is middle C, MIDI 60; sharps `#`, flats
`b`) or a MIDI number, then a colon and the length in beats (`1`, `0.5`, `3/2`, at least 1/24). A beat is a
quarter note in 4/4: 1 is a quarter note, 2 a half note, 0.5 an eighth note and 1.5 a dotted quarter. A note
without a length lasts one beat. `R` (or `rest`, or `-`) is a rest. Commas and bar lines `|` are ignored:

```
E4:1 G4:1 A4:2 | G4:1 E4:1 D4:2 | C4:2 R:1 D4:1
```

Melody as MIDI: a file with exactly one monophonic track. Positions are read in beats, so the file's own tempo does
not change them, and are rounded to 1/24 beat. A note lasts until the next onset unless a gap of half a beat or
more follows, which becomes a rest. The last note keeps its own length, and whole empty bars before the first note
are dropped. The tool warns when the file's tempo differs from the tempo it sings at.

The rest of the form:

| Setting | Default | What it does |
|---|---|---|
| lyric | | words separated by spaces, one per note |
| tempo | 100 | beats per minute, a whole number from 30 to 300 |
| voice | female | the caption's "solo female vocal" or "solo male vocal" |
| key | auto | the key ACE-Step is told, e.g. `D major` or `F# minor` (any letter case); `auto` chooses one from the notes |
| draws | 4 | takes with the clock, 1 to 8 |
| seed | 0 | the takes' seeds come from `dacapo.seeds.run_seeds("dacapo-sing-tool", str(seed), draw)` |
| compare | off | also make the same takes without the clock (`oohs_k3_none`), with the same seeds |

The key goes into ACE-Step's text prompt. B2 always gave the major key its melody was written in, and every note
lay in that key's scale. `auto` therefore ranks keys by the fewest beats of notes outside the key's scale (for a
minor key, natural minor with its raised 7th), then major before minor, then the Krumhansl-Kessler key profiles.
On B2's 24 test melodies and E4's 12 melodies it gives B2's key 32 times, and a key whose scale holds every note
all 36 times. The tool warns when the melody has notes outside the key it uses.

## Running it

In Colab: click Open in Colab at the top, choose an L4 or A100 runtime, fill in the form and choose Run all. The
notebook refuses a CPU or a T4, checks your melody and words, and only then installs the same pinned packages as
`notebook_runB.ipynb`. Drive is optional; without it the files download as a zip.

From the command line, check an input without any model:

```
python experiments/sing/sing.py check --melody "C4:1 E4:1 G4:2 A4:1 G4:1 E4:2 D4:2 C4:2" \
    --lyric "Hold my hand and walk with me now"
```

and make the takes on a GPU:

```
ACESTEP_REPO=/path/to/ACE-Step-1.5 ACESTEP_CHECKPOINTS_DIR=/path/to/checkpoints \
python experiments/sing/sing.py run --melody "C4:1 E4:1 G4:2 A4:1 G4:1 E4:2 D4:2 C4:2" \
    --lyric "Hold my hand and walk with me now" --draws 4 --compare --out /path/to/outputs
```

`run` checks the input, builds the item and renders the guide in its own process, then generates in
`$DACAPO_PY_ACE` and scores in `$DACAPO_PY_MAIN`, both this interpreter unless set (the agents' local setup keeps
ACE-Step and the measuring tools in two environments, as for Run B). The same settings always lead to the same
folder under `--out`, so running them again resumes a stopped run or shows the finished one. `--device cpu` is a
test: a take needs about a minute to generate once ACE-Step has loaded, and the first load can take several
minutes. The scorer then uses `openai/whisper-small.en` (the Whisper of Run B's tiny scale) instead of Run B's
Whisper large-v3, so the scores only show that the chain runs.

Status: on 2026-10-30 local CPU runs made one take of an 8-note melody, with and without the clock, and scored
them with Whisper small.en. They showed that the whole chain runs: the first run took 13.5 minutes, most of it
loading the models, and a later run with the models cached took about 3 minutes for both takes. The take with the
clock came out byte-identical in every run. Nobody has listened to these takes, and no GPU run has used the tool
yet.

Exit codes: 0 done, 1 the input is refused, 2 a setup problem (no GPU, no model, no scorer), 3 no take was made.

## What happens, step by step

Every step imports Run B's code and leaves it unchanged.

1. The item. `build_item` lays out the notes as `make_items_b2.build` laid out B2's: a 4-beat lead-in, the notes at
   the given tempo, each sounding 0.9 of its written length (0.95 for notes shorter than a beat, E4's rule that B2's
   code kept; B2's melodies had none), and a length of `ceil((4 + beats + 2) x 60 / tempo)` seconds. The lyric is
   `[Verse]` and the words on one line. Given as text, all 24 B2 test melodies and the 12 E4 melodies come out
   identical to their manifest entries (the tests check this).
2. The guide. `make_items_b2.render_one` renders the melody in GM Voice Oohs (program 53) with FluidSynth and
   FluidR3_GM at 48 kHz, gain 0.6, reverb and chorus off, exactly as B2's guides were rendered. The tool compares
   the soundfont's sha256 with B2's, records the result in `plan.json` and warns if they differ.
3. The takes. `plan_b2.make_run` gives each take B2's settings for S* (task `cover-nofsq`, cover strength 3/8, beta
   8 on the alignment heads {2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}, 8 steps, shift 3), and the tool sets
   only the seeds, the voice in the caption and the take's name. `dacapo.attn_bias.lyric_keys` finds each word's
   tokens with ACE-Step's own tokenizer, `plan_b2.spans` gives each word the window from its note's onset to the
   next onset, and `run_b2.generate_run` generates the take inside the clock's hook and checks that the hook ran
   40 times (5 layers x 8 steps). A take without the clock runs no hook.
4. The scores. `score_b2.B2Scorer` measures each take as B2's did (htdemucs stems, SwiftF0 on the vocal stem,
   Whisper word chunks, the CTC transcript), `score_b2.clip` turns the measurements into B2's values, and
   `score_b2.kept` keeps the take with the largest q (ties: higher WR, then higher NPA_v, then the earlier take).
   The kept take with the clock is copied to `kept.wav` and `kept.mp3`.

## The scores

| Column | Meaning |
|---|---|
| q | the vocal gate times the smaller of NPA_v and WR; q >= 0.5 was B2's "joint" bar, q >= 0.75 its strict bar |
| notes right | NPA_v: notes sung within 50 cents of the written pitch, in the written octave, on the vocal stem |
| words recognised | WR: lyric words Whisper heard, matched in order |
| voice found (gate) | E6's vocal gate: a vocal stem share of at least 0.2, at least 1 s of active voice, a usable peak |

On a GPU the scorer is Whisper large-v3 at Run B's pinned revision. The CTC model's transcript and its word
recall are saved too, as a second opinion. Whisper also writes "thank you" and a few other phrases on wordless
audio, and it may spell a homophone the other way, so the tool warns when the lyric contains such words.

## What a run writes

Each run folder (named after the first words and a hash of the settings) holds:

- `plan.json`: the settings, the item, the guide's sha256 and soundfont, the recipe, the seeds and their
  derivation, every take's settings, the Da Capo commit and the sha256 of every code file used
- `sing_oohs.wav`: the guide
- `take1_clock.wav`, `take1_clock.mp3`, ... (and `take1_noclock.*` with compare); `kept.wav` and `kept.mp3`
- `ledger.jsonl`, `scores.jsonl` and `gen_setup.json`: what generation and scoring recorded
- `sing_result.json`: settings, seeds, code, scorer, every take's scores and Whisper's transcript, and the kept take
- `logs/`: the full output of generation and scoring

## Tests

```
python -m pytest -q -p no:cacheprovider experiments/sing/tests
```

They cover the melody text and MIDI parsing, the word-note check and the warnings, the items, windows and run
settings against B2's own functions on B2's items, the seeds, the selector, the output JSON, resuming, and the
notebook. ACE-Step is not run in the tests.
