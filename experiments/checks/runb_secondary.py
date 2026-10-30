"""Post-hoc secondary checks of Run B's B2 result. Exploratory and not registered: the agents wrote them after
Run B to check numbers that the write-up quotes. No check here can change a registered verdict.

Reads only results/gpu_run3/b2_result.json and writes results/checks/runb_secondary.json, where every value
carries a one-line definition. Needs the standard library, numpy and dacapo/words.py (pure Python: the
registered word normaliser and in-order matcher).

  python experiments/checks/runb_secondary.py [--src results/gpu_run3/b2_result.json] [--out results/checks/runb_secondary.json]

Terms, as in experiments/e8_clock_sing/decide_b2.py: per clip q = gate * min(NPA_v, WR); per item and arm the
selector keeps the draw with the largest q (ties: higher WR, then higher NPA_v, then the lower draw); an item is
joint if its kept q >= 0.5; J(arm) is the share of the 24 items that are joint. Values are rounded to 1e-6
before every comparison. Seed-paired clips are the same item and draw in two arms (B2 derives the seeds from
the stage, item and draw only, so every arm shares them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dacapo.words import lyric_words, match_words, norm_word  # noqa: E402

SRC = ROOT / "results" / "gpu_run3" / "b2_result.json"
OUT = ROOT / "results" / "checks" / "runb_secondary.json"
ARMS = ("S*", "S*_b0", "S*_rand", "S*_shuf", "ctx_oohs_k4")  # the distinct Stage T arms; S*_b0 is also ctx_oohs_k3 and B*
JOINT = 0.5
N_BOOT, BOOT_SEED = 10000, 0  # decide_b2.boot_index
SENS_SEEDS = range(20)
# The registered E4 hallucination strings (experiments/e8_clock_sing/score_b2.py HALLUCINATIONS, addendum 3).
HALLUCINATIONS = ("thank you for watching", "transcript emily beynon", "satsang with mooji", "thank you", "oh my god",
                  "bf watch tv", "bells chiming", "oh god", "oh yeah", "so", "um")


def rnd(x) -> float:
    return round(float(x), 6)


def val(definition: str, value) -> dict:
    return {"definition": definition, "value": value}


# rows

def load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw)["results"], hashlib.sha256(raw).hexdigest()


def arm_rows(res: dict) -> dict[str, dict[str, list[dict]]]:
    """{arm: {item: [clip rows sorted by draw]}}; a merged run belongs to every label it carries."""
    out: dict = {}
    for c in res["clips"]:
        for lab in c["labels"]:
            out.setdefault(lab, {}).setdefault(c["item"], []).append(c)
    for arm in out.values():
        for draws in arm.values():
            draws.sort(key=lambda c: c["draw"])
    return out


def kept(draws: list[dict]) -> dict:
    """score_b2.kept, the registered selector."""
    return max(draws, key=lambda c: (c["q"], c["wr"], c["npa_v"], -c["draw"]))


def n_words(c: dict) -> int:
    """Lyric words of the clip's line; B2 sings one note per one-syllable word, so this is also its note count."""
    return len(lyric_words(c["lyrics"]))


def hyp_words(text: str | None) -> list[str]:
    """The stored Whisper transcript as normalised words, split at white space and hyphens. With this split the
    registered matcher reproduces the stored WR of every clip (checked in 'consistency')."""
    return [w for w in (norm_word(x) for x in (text or "").replace("-", " ").split()) if w]


def recall(lyric: list[str], hyp: list[str]) -> float:
    return len(match_words(lyric, hyp)) / len(lyric)


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def item_j(arm: dict[str, list[dict]], items: list[str], score) -> np.ndarray:
    """Per item, 1.0 if the draw with the largest score(clip) has score >= 0.5 (the selector re-picks by score)."""
    return np.array([float(max(rnd(score(c)) for c in arm[i]) >= JOINT) for i in items])


def kept_mean(arm: dict[str, list[dict]], items: list[str], score) -> np.ndarray:
    return np.array([max(rnd(score(c)) for c in arm[i]) for i in items])


def boot_index(n_items: int, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_items, size=(N_BOOT, n_items))


def delta_boot(js: np.ndarray, comps: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Delta_J in every resample with the max over the comparators retaken (decide_b2.contrasts), rounded to 1e-6."""
    best = np.max(np.stack([c[idx].mean(1) for c in comps]), axis=0)
    return np.round(js[idx].mean(1) - best, 6)


def ideal_bootstrap(js: np.ndarray, comps: list[np.ndarray]) -> dict:
    """Exact P(Delta_J <= 0) and P(Delta_J < 0) over all n^n equally likely resamples of the n items. Items are
    grouped by their category (J of S* minus J of each distinct comparator); the count of resamples reaching each
    vector of summed differences is built up one draw at a time in exact integers. Delta_J <= 0 when S* does not
    beat the best comparator, i.e. the smallest summed difference is <= 0."""
    n = len(js)
    cats = Counter(tuple(int(js[i]) - int(c[i]) for c in comps) for i in range(n))
    ways = {tuple(0 for _ in comps): 1}
    for _ in range(n):
        nxt: Counter = Counter()
        for state, w in ways.items():
            for v, k in cats.items():
                nxt[tuple(s + x for s, x in zip(state, v))] += w * k
        ways = nxt
    total = n ** n
    le = sum(w for s, w in ways.items() if min(s) <= 0)
    lt = sum(w for s, w in ways.items() if min(s) < 0)
    return {"le": le / total, "lt": lt / total,
            "categories": {",".join(map(str, k)): v for k, v in sorted(cats.items(), reverse=True)}}


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial test of wins against losses at p = 1/2, ties left out."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def tokens(text: str | None) -> list[str]:
    """score_b2.tokens: the transcript's normalised words, as the registered wordless rule reads them."""
    return [w for w in (norm_word(x) for x in re.findall(r"[a-z0-9']+", (text or "").lower())) if w]


# the checks

def analyse(res: dict) -> dict:
    A = arm_rows(res)
    items = sorted(A["S*"])
    n_items = len(items)
    stored = res["arms"]
    comparators = list(res["contrasts"]["delta_j"]["comparators"])
    clips = res["clips"]
    out: dict = {}

    # consistency of the rows with the stored summary
    kept_ok = all(kept(A[a][i])["run_id"] == stored[a]["items"][i]["kept_run"] for a in stored for i in items)
    j_reg = {a: item_j(A[a], items, lambda c: c["q"]) for a in A}
    j_ok = all(rnd(j_reg[a].mean()) == stored[a]["J"] for a in stored)
    wr_ok = sum(rnd(recall(lyric_words(c["lyrics"]), hyp_words(c["whisper_text"]))) == c["wr"] for c in clips)
    npa_dev = max(abs(c["npa_v"] * n_words(c) - round(c["npa_v"] * n_words(c))) for c in clips)
    wr_dev = max(abs(c["wr"] * n_words(c) - round(c["wr"] * n_words(c))) for c in clips)
    out["consistency"] = {
        "kept_draws_reproduced": val("The registered selector applied to the clip rows keeps the stored kept run "
                                     "for every arm and item", kept_ok),
        "j_reproduced": val("J of every arm recomputed from the clip rows equals the stored J", j_ok),
        "wr_reproduced_from_transcript": val(f"Stage T clips (of {len(clips)}) whose stored WR is reproduced by "
                                             "the registered matcher from the stored transcript split at white "
                                             "space and hyphens", wr_ok),
        "npa_v_times_notes_max_deviation": val("Largest distance of NPA_v x (the line's note count) from a whole "
                                               "number over all clips; near 0 means NPA_v is a count of correct "
                                               "notes out of the line's notes", rnd(npa_dev)),
        "wr_times_words_max_deviation": val("Largest distance of WR x (the line's word count) from a whole number "
                                            "over all clips", rnd(wr_dev)),
    }

    # (a) P1 bootstrap robustness
    js = j_reg["S*"]
    comp_vecs = [j_reg[b] for b in comparators]
    distinct = []
    for v in comp_vecs:
        if not any(np.array_equal(v, u) for u in distinct):
            distinct.append(v)
    d0 = delta_boot(js, comp_vecs, boot_index(n_items))
    sens = [delta_boot(js, comp_vecs, boot_index(n_items, s)) for s in SENS_SEEDS]
    exact = ideal_bootstrap(js, distinct)
    p25 = rnd(np.percentile(d0, 2.5))
    out["p1_bootstrap"] = {
        "comparators": val("The unclocked guided arms over which P1 takes the max (S*_b0, ctx_oohs_k3 and B* are one "
                           "run, so two distinct J vectors)", comparators),
        "delta_j": val("J(S*) minus the largest comparator J", rnd(js.mean() - max(v.mean() for v in comp_vecs))),
        "p2_5": val("2.5th percentile (numpy linear) of the bootstrap Delta_J: 10,000 resamples of the 24 items, "
                    "default_rng(0).integers(0, 24, (10000, 24)), max over comparators retaken (decide_b2)", p25),
        "n_le_0": val("Resamples of those 10,000 with Delta_J <= 0", int((d0 <= 0).sum())),
        "n_lt_0": val("Resamples of those 10,000 with Delta_J < 0", int((d0 < 0).sum())),
        "seeds_0_19_p2_5": val("The same 2.5th percentile with default_rng(s) for s = 0 to 19",
                               [rnd(np.percentile(d, 2.5)) for d in sens]),
        "seeds_0_19_share_le_0": val("Share of the 10,000 resamples with Delta_J <= 0, for s = 0 to 19",
                                     [rnd((d <= 0).mean()) for d in sens]),
        "exact_p_le_0": val("Exact ideal-bootstrap P(Delta_J <= 0): all 24^24 equally likely resamples counted by "
                            "item category", rnd(exact["le"])),
        "exact_p_lt_0": val("Exact ideal-bootstrap P(Delta_J < 0), counted the same way", rnd(exact["lt"])),
        "exact_p2_5_is_0": val("The ideal bootstrap's 2.5th percentile is exactly 0, since P(Delta_J < 0) < 0.025 "
                               "<= P(Delta_J <= 0)", bool(exact["lt"] < 0.025 <= exact["le"])),
        "item_categories": val("Items per category, the category being J(S*) - J(b) for each distinct comparator b "
                               "(S*_b0, ctx_oohs_k4)", exact["categories"]),
    }

    # (b) sign test of S* against S*_b0 on per-item J
    diff = j_reg["S*"] - j_reg["S*_b0"]
    w, lo, t = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
    out["sign_test_s_star_vs_b0"] = {
        "wins": val("Items joint for S* and not for S*_b0", w),
        "losses": val("Items joint for S*_b0 and not for S*", lo),
        "ties": val("Items joint for both or for neither", t),
        "p_two_sided": val("Exact two-sided binomial sign test on the untied items (p = 1/2)", rnd(sign_test(w, lo))),
    }

    # (c) the CTC recogniser
    kept_rows = {a: {i: kept(A[a][i]) for i in items} for a in ARMS}
    ctc_same = {a: int(sum(rnd(kept_rows[a][i]["q_ctc"]) >= JOINT for i in items)) for a in ARMS}
    ctc_resel = {a: int(item_j(A[a], items, lambda c: c["q_ctc"]).sum()) for a in ARMS}
    kd = np.array([kept_rows["S*"][i]["wr_ctc"] - kept_rows["S*_b0"][i]["wr_ctc"] for i in items])
    paired = np.array([a["wr_ctc"] - b["wr_ctc"] for i in items for a, b in zip(A["S*"][i], A["S*_b0"][i])])
    assert all(a["draw"] == b["draw"] for i in items for a, b in zip(A["S*"][i], A["S*_b0"][i]))

    def corr(arm):
        rows = [c for i in items for c in A[arm][i]]
        return rnd(np.corrcoef([c["wr"] for c in rows], [c["wr_ctc"] for c in rows])[0, 1])

    out["ctc"] = {
        "j_ctc_same_kept_draws": val("Items (of 24) whose kept draw, chosen by Whisper's q, has gate * min(NPA_v, "
                                     "CTC WR) >= 0.5", ctc_same),
        "mean_wr_ctc": val("Mean CTC WR over the arm's 96 clips",
                           {a: rnd(np.mean([c["wr_ctc"] for i in items for c in A[a][i]])) for a in ARMS}),
        "kept_s_star_minus_b0_mean_wr_ctc": val("Mean over items of CTC WR of S*'s kept draw minus CTC WR of "
                                                "S*_b0's kept draw", rnd(kd.mean())),
        "kept_s_star_minus_b0_up_down_equal": val("Items where that difference is > 0, < 0 and = 0",
                                                  [int((kd > 0).sum()), int((kd < 0).sum()), int((kd == 0).sum())]),
        "paired_96_s_star_minus_b0_mean_wr_ctc": val("Mean over the 96 seed-paired clips of S*'s CTC WR minus "
                                                     "S*_b0's", rnd(paired.mean())),
        "corr_wr_whisper_ctc": val("Pearson correlation over the arm's 96 clips of Whisper WR and CTC WR",
                                   {a: corr(a) for a in ARMS}),
        "j_ctc_reselected": val("Items (of 24) joint when the selector keeps the draw with the largest q_ctc = "
                                "gate * min(NPA_v, CTC WR)", ctc_resel),
    }

    # (d) the P2 margin in notes and in words
    def j_minus(arm, notes=0, words=0):
        def score(c):
            n = n_words(c)
            npa = max(round(c["npa_v"] * n) - notes, 0) / n
            wr = max(round(c["wr"] * n) - words, 0) / n
            return (1.0 if c["gate"] else 0.0) * min(npa, wr)
        return item_j(A[arm], items, score)

    lost1 = [i for i, a, b in zip(items, j_reg["S*"], j_minus("S*", notes=1)) if a and not b]
    out["p2_margin"] = {
        "j_s_star_minus_1_note": val("Items (of 24) joint for S* when every clip has one correct note fewer: "
                                     "NPA_v = (correct notes - 1) / (the line's notes), floor 0, the selector "
                                     "re-picking", int(j_minus("S*", notes=1).sum())),
        "j_s_star_minus_2_notes": val("The same with two correct notes fewer", int(j_minus("S*", notes=2).sum())),
        "j_s_star_minus_1_word": val("Items (of 24) joint for S* when every clip has one matched word fewer: "
                                     "WR = (matched words - 1) / (the line's words), floor 0, the selector "
                                     "re-picking", int(j_minus("S*", words=1).sum())),
        "j_s_star_minus_2_words": val("The same with two matched words fewer", int(j_minus("S*", words=2).sum())),
        "items_lost_minus_1_note": val("S*'s joint items that are no longer joint with one correct note fewer",
                                       lost1),
    }

    # (e) WR at chance
    lyr = {i: lyric_words(A["S*"][i][0]["lyrics"]) for i in items}

    def chance(c):
        h = hyp_words(c["whisper_text"])
        v = [recall(lyr[o], h) for o in items if o != c["item"]]
        return float(np.mean(v)), max(v)

    ch = {a: [chance(c) for i in items for c in A[a][i]] for a in ARMS}
    ch_mean = {a: rnd(np.mean([x[0] for x in ch[a]])) for a in ARMS}
    wr_mean = {a: rnd(np.mean([c["wr"] for i in items for c in A[a][i]])) for a in ARMS}
    out["wr_chance"] = {
        "wr_chance": val("Mean over the arm's 96 clips of the clip's WR against each of the other 23 test lyrics "
                         "(stored transcript, registered matcher), averaged over the 23", ch_mean),
        "max_wr_against_other_lyric": val("Largest WR of any clip of the arm against any other test lyric",
                                          {a: rnd(max(x[1] for x in ch[a])) for a in ARMS}),
        "wr_above_chance": val("Mean WR minus WR chance", {a: rnd(wr_mean[a] - ch_mean[a]) for a in ARMS}),
        "s_star_minus_b0_wr_gain": val("Mean WR of S* minus that of S*_b0", rnd(wr_mean["S*"] - wr_mean["S*_b0"])),
        "s_star_minus_b0_wr_gain_above_chance": val("The same gain after subtracting each arm's WR chance",
                                                    rnd((wr_mean["S*"] - ch_mean["S*"])
                                                        - (wr_mean["S*_b0"] - ch_mean["S*_b0"]))),
    }

    # (f) words in any order, and onset error on seed-paired clips
    def q_any(c):
        return (1.0 if c["gate"] else 0.0) * min(c["npa_v"], c["wr_multiset"])

    idx = boot_index(n_items)
    kq = {a: kept_mean(A[a], items, q_any) for a in ARMS}

    def contrast(b):
        d = kq["S*"] - kq[b]
        bs = d[idx].mean(1)
        return {"mean": rnd(d.mean()), "ci95": [rnd(np.percentile(bs, 2.5)), rnd(np.percentile(bs, 97.5))]}

    pairs = [(a["onset_err_abs"], b["onset_err_abs"]) for i in items for a, b in zip(A["S*"][i], A["S*_b0"][i])
             if a["onset_err_abs"] is not None and b["onset_err_abs"] is not None]
    out["any_order"] = {
        "j_any_order": val("Items (of 24) joint with q = gate * min(NPA_v, WR in any order), the stored "
                           "wr_multiset, the selector re-picking by that q",
                           {a: int(item_j(A[a], items, q_any).sum()) for a in ARMS}),
        "mean_kept_q_any_order": val("Mean over items of the largest any-order q", {a: rnd(kq[a].mean()) for a in ARMS}),
        "contrast_s_star_minus_shuf": val("Mean over items of S*'s minus S*_shuf's kept any-order q, with the 95% "
                                          "interval of the registered item bootstrap (default_rng(0), 10,000 "
                                          "resamples, percentiles)", contrast("S*_shuf")),
        "contrast_s_star_minus_b0": val("The same for S* minus S*_b0", contrast("S*_b0")),
    }
    out["onset_seed_paired"] = {
        "n_clips": val("Seed-paired clips (same item and draw) with an onset error in both S* and S*_b0 (each "
                       "clip with at least 4 matched words)", len(pairs)),
        "median_s_star": val("Median absolute onset error (s) of S* over those clips",
                             rnd(np.median([p[0] for p in pairs]))),
        "median_b0": val("Median absolute onset error (s) of S*_b0 over those clips",
                         rnd(np.median([p[1] for p in pairs]))),
    }

    # (g) a longest in-order matcher instead of difflib's matching blocks
    lcs_wr, more = {}, {}
    for c in clips:
        lw, h = lyric_words(c["lyrics"]), hyp_words(c["whisper_text"])
        n_lcs = lcs_len(lw, h)
        lcs_wr[c["run_id"]] = n_lcs / len(lw)
        if n_lcs > len(match_words(lw, h)):
            more[c["run_id"]] = n_lcs - len(match_words(lw, h))
    j_lcs = {a: item_j(A[a], items, lambda c: (1.0 if c["gate"] else 0.0) * min(c["npa_v"], lcs_wr[c["run_id"]]))
             for a in A}
    out["longest_match"] = {
        "n_clips_more_words": val("Stage T clips (of 480) where the longest in-order match of lyric and transcript "
                                  "words finds more lyric words than the registered matcher", len(more)),
        "extra_words": val("Extra words found on those clips, as {count: clips}",
                           {str(k): v for k, v in sorted(Counter(more.values()).items())}),
        "clips": val("Those clips", sorted(more)),
        "j_with_longest_match": val("Items (of 24) joint with WR from the longest in-order match, the selector "
                                    "re-picking", {a: int(j_lcs[a].sum()) for a in ARMS}),
        "item_j_unchanged": val("Every arm's per-item J is the same as with the registered matcher, so Delta_J, "
                                "its bootstrap and the verdict do not change",
                                all(np.array_equal(j_lcs[a], j_reg[a]) for a in A)),
    }

    # (h) the registered hallucination strings in S*'s joint transcripts
    joint = [c for i in items for c in A["S*"][i] if c["q"] >= JOINT]
    hits = [c["run_id"] for c in joint
            if any(f" {h} " in " " + " ".join(tokens(c["whisper_text"])) + " " for h in HALLUCINATIONS)]
    out["hallucination_strings"] = {
        "n_s_star_joint_clips": val("S*'s joint clips (q >= 0.5) over all 96", len(joint)),
        "n_containing_a_registered_string": val("Of those, transcripts that contain one of the registered E4 "
                                                "hallucination strings as whole words", len(hits)),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    res, sha = load(Path(a.src))
    doc = {"about": "Post-hoc, exploratory checks of B2 Stage T made by the agents after Run B; not registered, and "
                    "they cannot change a registered verdict. Computed only from the file named in source.",
           "source": {"file": "results/gpu_run3/b2_result.json", "sha256": sha},
           "script": "experiments/checks/runb_secondary.py",
           **analyse(res)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
