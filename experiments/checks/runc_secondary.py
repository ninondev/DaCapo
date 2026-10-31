"""Post-hoc secondary checks of Run C's C1 result, read next to Run B's B2. Exploratory and not registered: the
agents wrote them after Run C to check numbers that the write-up quotes. No check here can change a registered
verdict.

Reads only results/gpu_run4/c1_result.json and results/gpu_run3/b2_result.json and writes
results/checks/runc_secondary.json, where every value carries a one-line definition. Needs the standard library,
numpy and dacapo/words.py (pure Python: the registered lyric word splitter).

  python experiments/checks/runc_secondary.py [--c1 results/gpu_run4/c1_result.json]
      [--b2 results/gpu_run3/b2_result.json] [--out results/checks/runc_secondary.json]

Terms, as in experiments/e9_clock_replicate/decide_c1.py (and decide_b2.py for B2): per clip q = gate * min(NPA_v,
WR); per item and arm the selector keeps the draw with the largest q (ties: higher WR, then higher NPA_v, then the
lower draw); an item is joint if its kept q (bo4) >= 0.5; J(arm) is the share of the items that are joint (96 in
C1, 24 in B2). Delta_J = J(S*) - J(S*_b0). Values are rounded to 1e-6 before every comparison; percentiles are
numpy's linear ones. C1's registered bootstrap index is default_rng(0).integers(0, 96, (10000, 96)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dacapo.words import lyric_words  # noqa: E402

C1_SRC = ROOT / "results" / "gpu_run4" / "c1_result.json"
B2_SRC = ROOT / "results" / "gpu_run3" / "b2_result.json"
OUT = ROOT / "results" / "checks" / "runc_secondary.json"
C1_ARMS = ("S*", "S*_b0", "S*_rand", "S*_shuf")
PAIR = ("S*", "S*_b0")
JOINT = 0.5
N_BOOT = 10000
C1_SEED = 0   # decide_c1: the registered C1 index
B2_SEED = 0   # decide_b2: the registered B2 index, used where B2 is resampled on its own
B2_SEED_JOINT = 1  # a separate stream for B2 where it is resampled next to C1, so the two indexes are independent


def rnd(x) -> float:
    return round(float(x), 6)


def sig(x) -> float:
    """Six significant digits, for p-values far below 1e-6."""
    return float(f"{float(x):.6g}")


def val(definition: str, value) -> dict:
    return {"definition": definition, "value": value}


# rows

def load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw)["results"], hashlib.sha256(raw).hexdigest()


def arm_rows(res: dict) -> dict[str, dict[str, list[dict]]]:
    """{arm: {item: [clip rows sorted by draw]}}; a merged run (B2's S*_b0 = ctx_oohs_k3 = B*) belongs to every
    label it carries."""
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
    """Lyric words of the clip's line. C1 and B2 sing one note per one-syllable word, so this is also the note
    count."""
    return len(lyric_words(c["lyrics"]))


def item_j(arm: dict[str, list[dict]], items: list[str], score) -> np.ndarray:
    """Per item, 1.0 if the draw with the largest score(clip) has score >= 0.5 (the selector re-picks by score)."""
    return np.array([float(max(rnd(score(c)) for c in arm[i]) >= JOINT) for i in items])


def kept_q(arm: dict[str, list[dict]], items: list[str]) -> np.ndarray:
    return np.array([kept(arm[i])["q"] for i in items])


def boot_index(n_items: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_items, size=(N_BOOT, n_items))


def ci(x: np.ndarray) -> list[float]:
    return [rnd(np.percentile(x, 2.5)), rnd(np.percentile(x, 97.5))]


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial test of wins against losses at p = 1/2, ties left out."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def shifted(notes: int = 0, words: int = 0):
    """q of a clip with `notes` more correct notes and `words` more matched words (negative: fewer), each count
    kept within 0 and the line's n; a clip that fails the gate stays at q = 0."""
    def score(c):
        n = n_words(c)
        npa = min(max(round(c["npa_v"] * n) + notes, 0), n) / n
        wr = min(max(round(c["wr"] * n) + words, 0), n) / n
        return (1.0 if c["gate"] else 0.0) * min(npa, wr)
    return score


def best_of_1(arm: dict[str, list[dict]], items: list[str]) -> np.ndarray:
    """Per item, the share of its draws with q >= 0.5: best of 1 averaged over the single draws (decide_b2.best_of_n
    at n = 1)."""
    return np.array([np.mean([rnd(c["q"]) >= JOINT for c in arm[i]]) for i in items])


# the checks

def analyse(c1: dict, b2: dict) -> dict:
    A, B = arm_rows(c1), arm_rows(b2)
    items, b2_items = sorted(A["S*"]), sorted(B["S*"])
    n, nb = len(items), len(b2_items)
    jc = {a: item_j(A[a], items, lambda c: c["q"]) for a in C1_ARMS}
    jb = {a: item_j(B[a], b2_items, lambda c: c["q"]) for a in PAIR}
    qc = {a: kept_q(A[a], items) for a in PAIR}
    qb = {a: kept_q(B[a], b2_items) for a in PAIR}
    idx = boot_index(n, C1_SEED)
    out: dict = {}

    # consistency of the rows with the stored summaries
    sa, sb = c1["arms"], b2["arms"]
    kept_ok = all(kept(A[a][i])["run_id"] == sa[a]["items"][i]["kept_run"] for a in C1_ARMS for i in items)
    kept_ok_b2 = all(kept(B[a][i])["run_id"] == sb[a]["items"][i]["kept_run"] for a in PAIR for i in b2_items)
    d_boot = np.round(jc["S*"][idx].mean(1) - jc["S*_b0"][idx].mean(1), 6)
    r1 = c1["rule"]["R1"]
    s_clips = [c for i in items for c in A["S*"][i]]
    npa_dev = max(abs(c["npa_v"] * n_words(c) - round(c["npa_v"] * n_words(c))) for c in s_clips)
    wr_dev = max(abs(c["wr"] * n_words(c) - round(c["wr"] * n_words(c))) for c in s_clips)
    pdj = np.concatenate([jb["S*"] - jb["S*_b0"], jc["S*"] - jc["S*_b0"]])
    pdq = np.concatenate([qb["S*"] - qb["S*_b0"], qc["S*"] - qc["S*_b0"]])
    pidx = boot_index(len(pdj), 0)
    pooled = {"delta_j": {"value": rnd(pdj.mean()), "ci95": ci(pdj[pidx].mean(1))},
              "mean_bo4_q": {"value": rnd(pdq.mean()), "ci95": ci(pdq[pidx].mean(1))}}
    stored_pool = c1["pooled_with_b2"]
    out["consistency"] = {
        "kept_draws_reproduced": val("The registered selector applied to C1's clip rows keeps the stored kept run for "
                                     "every arm and item, and applied to B2's rows does so for S* and S*_b0",
                                     kept_ok and kept_ok_b2),
        "j_reproduced": val("J of every C1 arm, and of B2's S* and S*_b0, recomputed from the clip rows equals the "
                            "stored J", all(rnd(jc[a].mean()) == sa[a]["J"] for a in C1_ARMS)
                            and all(rnd(jb[a].mean()) == sb[a]["J"] for a in PAIR)),
        "bootstrap_index_reproduced": val("sha256 of default_rng(0).integers(0, 96, (10000, 96)) as int64 bytes equals "
                                          "c1_result's bootstrap.index_sha256",
                                          hashlib.sha256(idx.tobytes()).hexdigest()
                                          == c1["bootstrap"]["index_sha256"]),
        "r1_reproduced": val("Delta_J, its 2.5th percentile and its 95% interval recomputed with that index equal "
                             "c1_result's rule.R1",
                             rnd(jc["S*"].mean() - jc["S*_b0"].mean()) == r1["delta_j"]
                             and rnd(np.percentile(d_boot, 2.5)) == r1["p2_5"] and ci(d_boot) == r1["ci95"]),
        "pooled_reproduced": val("The exploratory pooled Delta_J and mean bo4 q difference over B2's 24 items then "
                                 "C1's 96, with default_rng(0).integers(0, 120, (10000, 120)), equal c1_result's "
                                 "pooled_with_b2", pooled["delta_j"]["value"] == stored_pool["delta_j"]["value"]
                                 and pooled["delta_j"]["ci95"] == stored_pool["delta_j"]["ci95"]
                                 and pooled["mean_bo4_q"] == stored_pool["mean_bo4_q"]),
        "npa_v_times_notes_max_deviation": val("Largest distance of NPA_v x (the line's note count) from a whole "
                                               "number over S*'s 384 clips; near 0 means NPA_v is a count of correct "
                                               "notes out of the line's notes", rnd(npa_dev)),
        "wr_times_words_max_deviation": val("Largest distance of WR x (the line's word count) from a whole number "
                                            "over S*'s 384 clips", rnd(wr_dev)),
    }

    # (a) sign test of S* against S*_b0 on per-item J
    diff = jc["S*"] - jc["S*_b0"]
    w, lo, t = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
    out["sign_test_s_star_vs_b0"] = {
        "wins": val("C1 items joint for S* and not for S*_b0", w),
        "losses": val("C1 items joint for S*_b0 and not for S*", lo),
        "ties": val("C1 items joint for both or for neither", t),
        "p_two_sided": val("Exact two-sided binomial sign test on the untied items (p = 1/2), six significant "
                           "digits", sig(sign_test(w, lo))),
    }

    # (b) the item bootstrap of J(S*)
    jbs = np.round(jc["S*"][idx].mean(1), 6)
    out["j_s_star_bootstrap"] = {
        "j": val("J(S*) in C1, the share of the 96 items joint", rnd(jc["S*"].mean())),
        "ci95": val("2.5th and 97.5th percentiles of J(S*) over the registered C1 resamples, "
                    "default_rng(0).integers(0, 96, (10000, 96))", ci(jbs)),
        "share_ge_0_5": val("Share of those 10,000 resamples with J(S*) >= 0.5, the R2 bar", rnd((jbs >= JOINT).mean())),
    }

    # (c) the R2 margin in notes and in words
    cases = {"plus_1_note": (1, 0, "one correct note more"), "minus_1_note": (-1, 0, "one correct note fewer"),
             "plus_1_word": (0, 1, "one matched word more"), "minus_1_word": (0, -1, "one matched word fewer"),
             "plus_1_note_and_1_word": (1, 1, "one correct note and one matched word more"),
             "minus_1_note_and_1_word": (-1, -1, "one correct note and one matched word fewer")}
    sens = {k: item_j(A["S*"], items, shifted(dn, dw)) for k, (dn, dw, _) in cases.items()}
    shift_def = ("NPA_v = (correct notes + dn) / n and WR = (matched words + dw) / n, n the line's word count (= its "
                 "note count), each count kept within 0 and n, gate-failed clips at q = 0, the selector re-picking")
    out["r2_margin"] = {
        "n_items_joint": val("Items (of 96) joint for S* as registered", int(jc["S*"].sum())),
        **{f"n_items_joint_{k}": val(f"Items (of 96) joint for S* when every S* clip has {txt} (dn = {dn}, "
                                     f"dw = {dw}): {shift_def}", int(sens[k].sum()))
           for k, (dn, dw, txt) in cases.items()},
        "items_lost_minus_1_note": val("S*'s joint items that are no longer joint with one correct note fewer",
                                       int(((jc["S*"] == 1) & (sens["minus_1_note"] == 0)).sum())),
        "items_gained_plus_1_note": val("S*'s non-joint items that become joint with one correct note more",
                                        int(((jc["S*"] == 0) & (sens["plus_1_note"] == 1)).sum())),
    }

    # (d) C1 against B2: the gap and the mean kept q difference
    ib = boot_index(nb, B2_SEED_JOINT)
    dc, db = jc["S*"] - jc["S*_b0"], jb["S*"] - jb["S*_b0"]
    gc, gb = qc["S*"] - qc["S*_b0"], qb["S*"] - qb["S*_b0"]
    bd = np.round(dc[idx].mean(1) - db[ib].mean(1), 6)
    bq = np.round(gc[idx].mean(1) - gb[ib].mean(1), 6)
    boot_note = ("C1's 96 items resampled with default_rng(0).integers(0, 96, (10000, 96)) and B2's 24 items "
                 "independently with default_rng(1).integers(0, 24, (10000, 24)), resample r of each paired")
    out["c1_vs_b2"] = {
        "delta_j_c1": val("C1's J(S*) - J(S*_b0)", rnd(dc.mean())),
        "delta_j_b2": val("B2's J(S*) - J(S*_b0), on its 24 test items (S*_b0 alone; B2's registered P1 took the max "
                          "over comparators, which was S*_b0's J too)", rnd(db.mean())),
        "delta_j_c1_minus_b2": val("C1's Delta_J minus B2's", rnd(dc.mean() - db.mean())),
        "delta_j_c1_minus_b2_ci95": val(f"95% interval of that difference: {boot_note}", ci(bd)),
        "delta_j_c1_minus_b2_share_le_0": val("Share of those 10,000 paired resamples where the difference is <= 0",
                                              rnd((bd <= 0).mean())),
        "mean_kept_q_diff_c1": val("C1's mean over items of bo4 q(S*) - bo4 q(S*_b0)", rnd(gc.mean())),
        "mean_kept_q_diff_b2": val("B2's mean over items of bo4 q(S*) - bo4 q(S*_b0)", rnd(gb.mean())),
        "mean_kept_q_diff_c1_minus_b2": val("C1's mean kept q difference minus B2's", rnd(gc.mean() - gb.mean())),
        "mean_kept_q_diff_c1_minus_b2_ci95": val(f"95% interval of that difference: {boot_note}", ci(bq)),
    }

    # (e) the change in J from B2 to C1
    chg = {}
    for a, name in zip(PAIR, ("s_star", "s_b0")):
        b = np.round(jc[a][idx].mean(1) - jb[a][ib].mean(1), 6)
        chg[f"j_b2_{name}"] = val(f"B2's J({a}), share of its 24 test items joint", rnd(jb[a].mean()))
        chg[f"j_c1_{name}"] = val(f"C1's J({a}), share of its 96 items joint", rnd(jc[a].mean()))
        chg[f"change_{name}"] = val(f"C1's J({a}) minus B2's", rnd(jc[a].mean() - jb[a].mean()))
        chg[f"change_{name}_ci95"] = val(f"95% interval of that change, with the paired independent resamples of "
                                         f"c1_vs_b2", ci(b))
    out["b2_to_c1_change"] = chg

    # (f) wordless output, and how often the CTC recogniser on the stem agrees with Whisper on the kept draws
    wl_rows = {a: [c for i in items for c in A[a][i] if c["scored"]] for a in C1_ARMS}
    wl = {a: rnd(np.mean([bool(c["wordless"]) for c in wl_rows[a]])) for a in C1_ARMS}
    agree = {}
    for a in C1_ARMS:
        ks = [kept(A[a][i]) for i in items]
        wj = [c for c in ks if rnd(c["q"]) >= JOINT]
        both = [c for c in wj if rnd(c["q_ctc_stem"]) >= JOINT]
        cj = [c for c in ks if rnd(c["q_ctc_stem"]) >= JOINT]
        agree[a] = {"whisper_joint": len(wj), "also_ctc_stem_joint": len(both),
                    "share": rnd(len(both) / len(wj)) if wj else None,
                    "ctc_stem_joint": len(cj), "ctc_stem_joint_not_whisper": len(cj) - len(both)}
    out["wordless_and_ctc"] = {
        "wordless_rate": val("c1_result's wordless_rate per arm: share of the arm's scored clips whose Whisper "
                             "transcript the registered scorer calls wordless (score_b2.wordless: empty or at least "
                             "half vowel sounds, or only registered E4 hallucination strings and vowel sounds)",
                             {a: sa[a]["wordless_rate"] for a in C1_ARMS}),
        "wordless_rate_reproduced": val("The same share recomputed from the clip rows' wordless flags equals the "
                                        "stored rate for every arm", all(wl[a] == sa[a]["wordless_rate"]
                                                                         for a in C1_ARMS)),
        "wordless_rate_b2": val("b2_result's wordless_rate for S* and S*_b0, for comparison",
                                {a: sb[a]["wordless_rate"] for a in PAIR}),
        "kept_whisper_joint_also_ctc_stem_joint": val(
            "Per arm, over the 96 kept draws (kept by Whisper's q): whisper_joint = kept q >= 0.5; "
            "also_ctc_stem_joint = of those, q_ctc_stem = gate * min(NPA_v, CTC WR on the vocal stem) >= 0.5; "
            "share = also_ctc_stem_joint / whisper_joint; ctc_stem_joint = kept draws with q_ctc_stem >= 0.5 (J by CTC "
            "on the stem x 96); ctc_stem_joint_not_whisper = those not joint by Whisper", agree),
        "ctc_stem_j_reproduced": val("ctc_stem_joint / 96 equals the stored J_ctc_stem of every arm",
                                     all(rnd(agree[a]["ctc_stem_joint"] / n) == sa[a]["J_ctc_stem"]
                                         for a in C1_ARMS)),
    }

    # (g) a single draw instead of the best of 4
    b1c = {a: best_of_1(A[a], items) for a in PAIR}
    b1b = {a: best_of_1(B[a], b2_items) for a in PAIR}
    ib0 = boot_index(nb, B2_SEED)
    e1c, e1b = b1c["S*"] - b1c["S*_b0"], b1b["S*"] - b1b["S*_b0"]
    out["best_of_1"] = {
        "j1_c1": val("C1's best-of-1 J: share of all (item, draw) clips with q >= 0.5, per arm",
                     {a: rnd(b1c[a].mean()) for a in PAIR}),
        "j1_b2": val("B2's best-of-1 J, the same", {a: rnd(b1b[a].mean()) for a in PAIR}),
        "j1_reproduced": val("Those equal best_of_n['1']['j'] stored in c1_result and b2_result",
                             all(rnd(b1c[a].mean()) == sa[a]["best_of_n"]["1"]["j"]
                                 and rnd(b1b[a].mean()) == sb[a]["best_of_n"]["1"]["j"] for a in PAIR)),
        "delta_j1_c1": val("C1's best-of-1 J(S*) - J(S*_b0)", rnd(e1c.mean())),
        "delta_j1_c1_ci95": val("Its 95% interval over the registered C1 resamples, default_rng(0).integers(0, 96, "
                                "(10000, 96)), the draws of an item kept together", ci(e1c[idx].mean(1))),
        "delta_j1_b2": val("B2's best-of-1 J(S*) - J(S*_b0)", rnd(e1b.mean())),
        "delta_j1_b2_ci95": val("Its 95% interval over B2's registered resamples, default_rng(0).integers(0, 24, "
                                "(10000, 24))", ci(e1b[ib0].mean(1))),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", default=str(C1_SRC))
    ap.add_argument("--b2", default=str(B2_SRC))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    c1, c1_sha = load(Path(a.c1))
    b2, b2_sha = load(Path(a.b2))
    doc = {"about": "Post-hoc, exploratory checks of C1 (Run C) made by the agents after Run C, some read next to B2; "
                    "not registered, and they cannot change a registered verdict. Computed only from the files named "
                    "in sources.",
           "sources": {"c1": {"file": "results/gpu_run4/c1_result.json", "sha256": c1_sha},
                       "b2": {"file": "results/gpu_run3/b2_result.json", "sha256": b2_sha}},
           "script": "experiments/checks/runc_secondary.py",
           **analyse(c1, b2)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
