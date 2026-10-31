# Post-hoc checks of Run B

The agents made these checks of B2 Stage T after Run B, to back numbers that the write-up quotes. They are post-hoc and exploratory. None of them was registered, and none can change a registered verdict. [`experiments/checks/runb_secondary.py`](../../experiments/checks/runb_secondary.py) computes every value from one file, [`results/gpu_run3/b2_result.json`](../gpu_run3/b2_result.json), whose sha256 it records, and writes `runb_secondary.json`. Each value there sits next to a one-line definition. The script needs only the standard library, numpy and the repository's word matcher (`dacapo/words.py`), and its tests are in `experiments/checks/tests`.

```
python experiments/checks/runb_secondary.py
```

Terms follow `experiments/e8_clock_sing/decide_b2.py`. A clip's score is q = gate × min(NPA_v, WR). For each melody and arm the selector keeps the draw with the largest q, and the melody is joint if that q is at least 0.5. J counts the joint melodies out of 24. Seed-paired clips are the same melody and draw in two arms; B2 derives its seeds from the stage, melody and draw only, so every arm shares them. The arms are S\* (the clocked recipe), S\*_b0 (the same guide without the clock, also run as ctx_oohs_k3 and B\*), S\*_rand (the clock on random heads), S\*_shuf (the clock with rotated windows) and ctx_oohs_k4 (the guide for 4 steps, no clock). Values are given below in that arm order.

## Consistency with the stored summary

| Number | Value | Meaning |
|---|---|---|
| Kept draws, J | reproduced | The registered selector, applied to the clip rows, keeps the stored draw for every arm and melody, and gives the stored J of every arm |
| WR from the transcript | 480 of 480 | Splitting the stored Whisper transcript at white space and hyphens and matching it with `dacapo/words.py` gives the stored WR on every clip. The run itself matched Whisper's timed word chunks, which the result file does not keep |
| NPA_v × notes | whole numbers (within 6e-6) | NPA_v is a count of correct notes out of the line's notes, and each line has one note per word |

## P1: the bootstrap of ΔJ

| Number | Value | Meaning |
|---|---|---|
| ΔJ | 0.25 | J(S\*) minus the best unclocked comparator's J (S\*_b0, 8 of 24) |
| 2.5th percentile | 0.0 | The registered bootstrap: `default_rng(0).integers(0, 24, (10000, 24))`, the maximum over comparators taken again in every resample, numpy's linear percentile. Same as the stored value |
| Resamples with ΔJ ≤ 0 | 286 of 10,000 | In the same 10,000 resamples |
| Resamples with ΔJ < 0 | 104 of 10,000 | In the same 10,000 resamples |
| Seeds 0 to 19 | 2.5th percentile 0.0 for every seed; 2.78% to 3.50% of resamples ≤ 0 | The same bootstrap with `default_rng(s)` |
| Exact P(ΔJ ≤ 0) | 0.029987 | The ideal bootstrap: all 24^24 equally likely resamples, counted exactly by grouping the melodies into categories of J(S\*) − J(b) |
| Exact P(ΔJ < 0) | 0.013158 | Counted the same way |
| Ideal 2.5th percentile | exactly 0 | Because P(ΔJ < 0) is below 0.025 and P(ΔJ ≤ 0) is above it |

## Melody by melody: S\* against S\*_b0

| Number | Value | Meaning |
|---|---|---|
| Wins, losses, ties | 8, 2, 14 | Melodies joint for S\* only, for S\*_b0 only, and for both or neither |
| Sign test | p = 0.109 | Exact two-sided binomial test on the 10 untied melodies |

## The second recogniser (CTC)

| Number | Value | Meaning |
|---|---|---|
| J with CTC WR, same kept draws | 2, 3, 1, 2, 1 | Melodies whose kept draw, chosen by Whisper's q, has gate × min(NPA_v, CTC WR) ≥ 0.5 |
| Mean CTC WR | 0.179, 0.160, 0.121, 0.128, 0.045 | Over each arm's 96 clips |
| Kept draws, S\* minus S\*_b0 | −0.080; 9 up, 13 down, 2 equal | Mean over melodies of the CTC WR of S\*'s kept draw minus that of S\*_b0's kept draw, and the count of melodies by sign |
| All 96 seed-paired clips, S\* minus S\*_b0 | +0.019 | Mean difference in CTC WR |
| Correlation of Whisper WR and CTC WR | 0.53, 0.76, 0.76, 0.61, 0.79 | Pearson, over each arm's 96 clips |
| J with draws re-selected by CTC | 4, 3, 3, 2, 1 | The selector keeps the draw with the largest gate × min(NPA_v, CTC WR) |

## The capability margin (P2)

| Number | Value | Meaning |
|---|---|---|
| J(S\*), one note fewer | 12 | Every clip loses one correct note: NPA_v becomes (correct notes − 1) / (the line's notes), never below 0, and the selector picks again |
| J(S\*), two notes fewer | 10 | The same with two correct notes |
| J(S\*), one word fewer | 14 | Every clip loses one matched word: WR becomes (matched words − 1) / (the line's words) |
| J(S\*), two words fewer | 12 | The same with two matched words |
| Melodies lost with one note fewer | b2t12, b2t18 | S\*'s joint melodies that have no note to spare |

## WR at chance

| Number | Value | Meaning |
|---|---|---|
| WR chance | 0.145, 0.092, 0.086, 0.113, 0.026 | Each clip's transcript scored against each of the other 23 test lyrics with the registered matcher, averaged over the 23, then over the arm's 96 clips |
| Highest WR against another lyric | 0.417 (S\*, S\*_b0, S\*_rand), 0.385 (S\*_shuf, ctx_oohs_k4) | No clip reaches 0.5 against a lyric it was not singing |
| WR gain of S\* over S\*_b0 | 0.174, or 0.121 above chance | Mean WR difference, before and after subtracting each arm's WR chance |

## Words in any order

| Number | Value | Meaning |
|---|---|---|
| J, any order | 15, 8, 8, 10, 3 | q from the stored any-order WR (`wr_multiset`, a registered secondary measure), the selector picking by that q |
| S\* minus S\*_shuf | 0.078 [0.015, 0.141] | Mean kept any-order q, with the 95% interval of the registered item bootstrap (seed 0, 10,000 resamples) |
| S\* minus S\*_b0 | 0.139 [0.068, 0.210] | The same against the guide without the clock |

## Word onsets on seed-paired clips

| Number | Value | Meaning |
|---|---|---|
| Clips | 39 | Seed-paired clips with an onset error (at least 4 matched words) in both S\* and S\*_b0 |
| Median onset error | 0.40 s (S\*), 1.72 s (S\*_b0) | Median absolute distance of Whisper's word onsets from the written note onsets, over those 39 pairs |

## A longest-match matcher

| Number | Value | Meaning |
|---|---|---|
| Clips with more words | 12 of 480 | A longest in-order match finds more lyric words than the registered matcher: one more on 11 clips, two more on 1 |
| J with that matcher | 14, 8, 8, 7, 3 | Every arm's melody-by-melody J is unchanged, so ΔJ, its bootstrap and the verdict are unchanged |

## Hallucination strings

| Number | Value | Meaning |
|---|---|---|
| S\*'s joint transcripts with a registered string | 0 of 31 | None of the transcripts of S\*'s 31 joint clips contains, as whole words, one of the E4 hallucination strings registered in addendum 3 |

# Post-hoc checks of Run C

The agents made these checks of C1 after Run C, to back numbers that the write-up quotes, and some of them set C1 next to B2. They are post-hoc and exploratory. None of them was registered, and none can change a registered verdict. [`experiments/checks/runc_secondary.py`](../../experiments/checks/runc_secondary.py) reads two files, [`results/gpu_run4/c1_result.json`](../gpu_run4/c1_result.json) and [`results/gpu_run3/b2_result.json`](../gpu_run3/b2_result.json), records their sha256 and writes `runc_secondary.json`, where each value sits next to a one-line definition. It needs the standard library, numpy and `dacapo/words.py`, and its tests are in `experiments/checks/tests`.

```
python experiments/checks/runc_secondary.py
```

Terms follow `experiments/e9_clock_replicate/decide_c1.py`. A clip's score is q = gate × min(NPA_v, WR). The selector keeps each melody's draw with the largest q, and the melody is joint if that q is at least 0.5. J is the share of joint melodies, out of 96 in C1 and 24 in B2, and ΔJ is J(S\*) − J(S\*_b0). C1's registered bootstrap resamples the 96 melodies with `default_rng(0).integers(0, 96, (10000, 96))`. Where C1 and B2 are compared, B2's 24 melodies are resampled separately with `default_rng(1).integers(0, 24, (10000, 24))`, so the two runs vary independently. Intervals are 95% percentile intervals.

## Consistency with the stored summaries

| Number | Value | Meaning |
|---|---|---|
| Kept draws, J | reproduced | The registered selector, applied to the clip rows, keeps the stored draw and gives the stored J for every C1 arm and for B2's S\* and S\*_b0 |
| Bootstrap index | reproduced | The sha256 of the index matrix equals the one `c1_result.json` records |
| R1 | reproduced | ΔJ 0.3125, 2.5th percentile 0.197917 and interval 0.197917 to 0.427083, as stored |
| Pooled estimate | reproduced | ΔJ and the mean kept q difference over B2's 24 melodies and C1's 96, as stored |
| NPA_v × notes, WR × words | whole numbers (within 6e-6) | Over S\*'s 384 clips, so NPA_v and WR count notes and words out of the line's n. Each line has one note per word |

## Melody by melody: S\* against S\*_b0

| Number | Value | Meaning |
|---|---|---|
| Wins, losses, ties | 36, 6, 54 | Melodies joint for S\* only, for S\*_b0 only, and for both or neither |
| Sign test | p = 2.8 × 10⁻⁶ | Exact two-sided binomial test on the 42 untied melodies |

## J(S\*) and the capability bar (R2)

| Number | Value | Meaning |
|---|---|---|
| J(S\*) | 0.490 (47 of 96) | As registered |
| Interval | 0.39 to 0.59 | J(S\*) over the 10,000 registered resamples |
| Resamples at 0.5 or above | 46.1% | Resamples in which S\* would have reached the bar |
| One correct note more, or fewer | 54, or 35 | Every S\* clip gains or loses one correct note: NPA_v becomes (correct notes ± 1) / n, kept between 0 and 1, and the selector picks again. A clip that fails the gate stays at q = 0 |
| One matched word more, or fewer | 54, or 40 | The same with one matched word: WR becomes (matched words ± 1) / n |
| One note and one word more, or fewer | 63, or 27 | Both at once |
| Melodies lost with one note fewer | 12 | And 7 melodies gained with one note more |

## Run C next to B2

| Number | Value | Meaning |
|---|---|---|
| ΔJ | 0.3125 (C1), 0.25 (B2) | J(S\*) − J(S\*_b0) in each run |
| C1's ΔJ minus B2's | 0.0625 (−0.20 to 0.32) | With the independent resamples. 33.9% of them are at or below 0 |
| Mean kept q difference | 0.184 (C1), 0.131 (B2) | Mean over melodies of S\*'s kept q minus S\*_b0's |
| C1's minus B2's | 0.053 (−0.03 to 0.14) | The same resamples |
| J(S\*), B2 to C1 | 0.583 to 0.490, a change of −0.094 (−0.31 to 0.13) | |
| J(S\*_b0), B2 to C1 | 0.333 to 0.177, a change of −0.156 (−0.35 to 0.04) | |

Every run-to-run interval above contains 0. The two runs' gaps can't be told apart, and neither recipe's drop from B2 to C1 is clear on its own.

## Wordless output and the second recogniser

| Number | Value | Meaning |
|---|---|---|
| Wordless rate | 0.036, 0.414, 0.419, 0.089 | Stored rate per arm (S\*, S\*_b0, S\*_rand, S\*_shuf): the share of scored clips whose Whisper transcript the registered scorer calls wordless, meaning empty, at least half vowel sounds, or nothing but the registered hallucination strings and vowel sounds. The clip flags give the same rates. B2: 0.021 (S\*) and 0.281 (S\*_b0) |
| S\*'s joint kept draws also joint by CTC on the vocal | 9 of 47 (19%) | Of the melodies whose kept draw is joint by Whisper, those whose kept draw also has gate × min(NPA_v, CTC WR on the separated vocal) ≥ 0.5 |
| The same for the other arms | 8 of 17, 6 of 20, 2 of 9 | S\*_b0, S\*_rand, S\*_shuf. In no arm is a kept draw joint by CTC and not by Whisper |

## A single draw instead of the best of 4

| Number | Value | Meaning |
|---|---|---|
| J with one draw | 0.216 (S\*), 0.060 (S\*_b0) in C1; 0.323 and 0.135 in B2 | Share of all clips with q ≥ 0.5, which is best-of-n at n = 1 |
| ΔJ with one draw | 0.156 (0.10 to 0.21) in C1, 0.188 (0.07 to 0.30) in B2 | Intervals from each run's registered resamples, the draws of a melody kept together |
