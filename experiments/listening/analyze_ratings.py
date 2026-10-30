"""Analysis of the two blind listening packs, fixed in registration/listening_plan.json before any rating is in.

  python experiments/listening/analyze_ratings.py check --template PACK/ratings.csv --ratings r1.csv [r2.csv ...]
  python experiments/listening/analyze_ratings.py b2    --key KEY --ratings r1.csv [...]
                                                        --scores results/gpu_run3/b2_result.json
                                                        [--exclude-file exclusions.json] --out out.json
  python experiments/listening/analyze_ratings.py runa  --key KEY --ratings r1.csv [...] --owner NAME|none
                                                        --e5 results/gpu_run2/e5v2_main.json
                                                        --e6 results/gpu_run2/e6_readout.json
                                                        [--exclude-file exclusions.json] --out out.json

'check' reads every sheet against the pack's own blank ratings.csv, with no key, and lists every problem that
would stop the analysis; it runs at closing, before any key is opened. A rater is named by the ratings file:
ratings_xiaoming.csv is rater 'xiaoming'. The exclusions file lists raters who heard a labelled demo clip, saw a
key or read the per-clip results before rating; it is committed before the key is opened. Both packs are
exploratory (addendum 2 and addendum 3); nothing here changes a registered verdict.

B2 pack (24 pairs, S* against S*_b0, the side of each pair drawn by the operating system's random source): an answer
A or B is a vote for the recipe on that side. Per question and pair, a_p is the A answers minus the B answers over
raters and s_p is +1 if S* is on side A, -1 if on side B. The vote margin for the clock is D = sum_p s_p a_p. The test
is a randomization test conditional on m, the number of pairs with S* on side A: the sides were independent fair
draws, so given m every one of the C(24, m) side patterns was equally likely, and under the sharp null that every
rater would give the same answers whichever recipe sat on which side, the null distribution runs over those
patterns. The statistic is the centred margin Dc = D - (2m - 24) mean(a), which equals 2m(24 - m)/24 times the mean
of a where S* is on A minus the mean of a where S* is on B, so a preference for clip A or clip B cannot move it.

Run A pack (60 clips: 5 per E5 condition and 20 E6 clips, yes / no / unsure questions): yes rates per condition and
per E6 set, an exact permutation test of 'the gap repeats an earlier phrase' between lit and ctl clips (with and
without the owner's sheet), agreement of the listeners' majority with the pitch tracker's per-draw choice, and, for
the E6 clips, agreement with the vocal gate and with Whisper's word recall.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

ALPHA = 0.05
B2_QUESTIONS = ("melody_right", "words_clear")
RUNA_QUESTIONS = ("vocals_present", "words_intelligible", "accompaniment", "gap_repeats_earlier_motif")
ARM, BASE = "S*", "S*_b0"


def norm(value: str) -> str:
    """Unicode NFKC (full-width letters become ordinary ones), lower case, no spaces, no final . 。 ! ！ ~ ～."""
    v = unicodedata.normalize("NFKC", value or "").casefold()
    v = re.sub(r"\s+", "", v)
    return re.sub(r"[.。!！~～]+$", "", v)


def _table(groups: dict) -> dict:
    return {norm(w): label for label, words in groups.items() for w in words}


# Answers after norm(). 'blank' means no answer, as an empty cell does. Anything else is 'invalid' (no answer, reported).
B2_ANSWERS = _table({
    "A": ["a", "a段", "第一个", "第一段", "前者"],
    "B": ["b", "b段", "第二个", "第二段", "后者"],
    "same": ["same", "一样", "都一样", "一样好", "差不多", "都差不多", "相同", "没区别", "没有区别", "都好", "都对",
             "都清楚", "both", "equal", "no difference"],
    "neither": ["neither", "都不", "都不行", "都不好", "两个都不行", "两个都不好", "两个都不", "都不对", "都不清楚",
                "都听不清", "none"],
    "blank": ["不确定", "不知道", "说不清", "?", "unsure", "not sure"],
})
YES_NO = _table({
    "yes": ["yes", "y", "是", "是的", "有", "有的", "能", "对", "可以", "√", "听得清", "能听清", "重复", "有重复",
            "有伴奏"],
    "no": ["no", "n", "否", "不是", "没有", "没", "无", "不", "不能", "不对", "不可以", "×", "听不清", "不重复",
           "没重复", "没有伴奏"],
    "unsure": ["unsure", "not sure", "?", "不确定", "不太确定", "不知道", "说不清", "不好说"],
})

# The units a listener has heard before the gap in each E5 condition, by the names of the per-draw choice. 'strict'
# counts exact repeats only; 'moved' also counts a heard unit moved by the condition's step.
EARLIER = {"lit": {"A", "B"}, "ctl": {"A", "B", "C"}, "rel+2": {"A", "B", "T+2A"}, "rel+5": {"A", "B", "T+5A"},
           "rel-3": {"A", "B", "T-3A"}, "seq": {"A", "T+2A", "T+4A"}, "ana": {"A", "T+5A", "B"}, "null": set()}
MOVED = {"rel+2": {"T+2B"}, "rel+5": {"T+5B"}, "rel-3": {"T-3B"}, "seq": {"T+6A"}, "ana": {"T+5B"}}
PRIMARY_AGREEMENT = ("lit", "ctl", "null")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rnd(x):
    return None if x is None else round(float(x), 6)


def sig(p):
    """A p value kept to 6 significant digits, so a very small exact p never prints as 0."""
    return None if p is None else float(f"{p:.6g}")


def decode(raw: bytes, path) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: not UTF-8, UTF-16 or GB18030; save the sheet as CSV")


def header(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", unicodedata.normalize("NFKC", name or "").strip().casefold())


def item_id(value: str, kind: str) -> str:
    """'pair_1', 'Pair 01', 'pair-01', '01' and full-width forms all become 'pair_01' (or 'clip_01')."""
    v = unicodedata.normalize("NFKC", value or "").strip().casefold()
    m = re.fullmatch(rf"(?:{kind})?[\s_\-]*0*(\d+)", v)
    return f"{kind}_{int(m[1]):02d}" if m else v


def read_table(path: Path, kind: str, questions: tuple) -> list[dict]:
    """Rows of a ratings sheet saved by a spreadsheet as CSV (comma, semicolon or tab; UTF-8, UTF-16 or GB18030;
    any line ending). Extra fields past the header are joined into notes. Raises if a needed column is missing."""
    text = decode(Path(path).read_bytes(), path).lstrip("\r\n")
    first = text.splitlines()[0] if text else ""
    delim = max((",", ";", "\t"), key=first.count)
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delim, restkey="_extra")
    fields = {header(f) for f in (reader.fieldnames or [])}
    missing = [c for c in (kind, *questions) if c not in fields]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {sorted(fields)}")
    rows = []
    for r in reader:
        extra = r.pop("_extra", None) or []
        row = {header(k): (v or "").strip() for k, v in r.items() if k is not None}
        if extra:
            row["notes"] = ",".join([row.get("notes", "")] + [str(x) for x in extra])
        if any(row.values()):
            rows.append(row)
    return rows


def answer(value: str, table: dict) -> str:
    v = norm(value)
    if not v:
        return "blank"
    return table.get(v, "invalid")


def rater_name(path: Path) -> str:
    stem = re.sub(r"\s*\(\d+\)$", "", unicodedata.normalize("NFKC", Path(path).stem).strip())
    return stem[len("ratings_"):] if stem.casefold().startswith("ratings_") and len(stem) > 8 else stem


def read_exclusions(path: Path | None) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {unicodedata.normalize("NFKC", d["name"]).casefold(): d for d in data}


def read_sheet(p: Path, kind: str, questions: tuple, table: dict, ids) -> dict:
    """One rater's sheet. Raises on an id that is not in the pack, on a row with answers but no id, and on an id
    given twice with different answers."""
    rows = {}
    for row in read_table(p, kind, questions):
        i = item_id(row.get(kind, ""), kind)
        if not i:
            if any(row.get(q) for q in questions):
                raise ValueError(f"{p}: a row has answers but no {kind} name")
            continue
        if ids is not None and i not in ids:
            raise ValueError(f"{p}: {kind} {row.get(kind)!r} is not in the pack")
        if i in rows and any(answer(rows[i].get(q, ""), table) != answer(row.get(q, ""), table) for q in questions):
            raise ValueError(f"{p}: {i} appears twice with different answers")
        if i not in rows or any(row.get(q) for q in questions):
            rows[i] = row
    answers = {i: {q: answer(r.get(q, ""), table) for q in questions} for i, r in rows.items()}
    invalid = {}
    for r in rows.values():
        for q in questions:
            if answer(r.get(q, ""), table) == "invalid":
                v = norm(r.get(q, ""))
                invalid[v] = invalid.get(v, 0) + 1
    return {"path": str(p), "sha256": sha256(p), "answers": answers, "n_rows": len(rows),
            "n_answered": sum(a not in ("blank", "invalid") for v in answers.values() for a in v.values()),
            "invalid": invalid, "missing_ids": sorted(set(ids) - set(rows)) if ids is not None else []}


def load_raters(paths, exclude=(), kind="pair", questions=B2_QUESTIONS, table=B2_ANSWERS, ids=None,
                identical_ok=()) -> tuple[dict, dict, list]:
    """One sheet per rater. Returns (raters, excluded, empty). Raises on two sheets for one rater (names compared
    after NFKC and without case), on byte-identical sheets under two names (unless both names are in identical_ok,
    after the owner confirmed two people), on an excluded name with no sheet, and on any sheet error of
    read_sheet. A sheet with no readable answer is left out and listed in empty."""
    excl = {unicodedata.normalize("NFKC", e).casefold() for e in exclude}
    same_ok = {unicodedata.normalize("NFKC", e).casefold() for e in identical_ok}
    raters, dropped, empty, seen_names, seen_hash = {}, {}, [], set(), {}
    for p in paths:
        name = rater_name(p)
        key = name.casefold()
        if key in seen_names:
            raise ValueError(f"two ratings files for rater {name!r}")
        seen_names.add(key)
        h = sha256(p)
        if h in seen_hash and not (key in same_ok and seen_hash[h][1] in same_ok):
            raise ValueError(f"{p} is byte-identical to {seen_hash[h][0]}")
        seen_hash.setdefault(h, (str(p), key))
        if key in excl:
            dropped[name] = {"file": Path(p).name, "sha256": h}
            continue
        r = read_sheet(p, kind, questions, table, ids)
        if r["n_answered"] == 0:
            empty.append({"name": name, "file": Path(p).name, "sha256": h, "unreadable": r["invalid"]})
            continue
        raters[name] = r
    unmatched = excl - {n.casefold() for n in dropped}
    if unmatched:
        raise ValueError(f"excluded rater(s) with no ratings file: {sorted(unmatched)}")
    return raters, dropped, empty


def check_sheets(template: Path, paths) -> int:
    """Check every sheet against the pack's own blank ratings.csv, without any key. Prints one line per sheet and
    every problem; returns 1 if any sheet would stop the analysis."""
    first = header(decode(Path(template).read_bytes(), template).lstrip("\r\n").splitlines()[0].split(",")[0])
    kind, questions, table = (("pair", B2_QUESTIONS, B2_ANSWERS) if first == "pair" else
                              ("clip", RUNA_QUESTIONS, YES_NO))
    ids = {item_id(r[kind], kind) for r in read_table(template, kind, ())}
    bad = 0
    for p in paths:
        try:
            r = read_sheet(p, kind, questions, table, ids)
            print(f"{rater_name(p)}: {r['n_answered']} readable answers in {r['n_rows']} rows, "
                  f"{len(r['missing_ids'])} rows missing, unreadable {r['invalid'] or 'none'}")
            if r["n_answered"] == 0:
                print("  no readable answer: this sheet would be left out")
        except ValueError as e:
            bad = 1
            print(f"STOP {e}")
    try:
        load_raters(paths, (), kind, questions, table, ids)
    except ValueError as e:
        bad = 1
        print(f"STOP {e}")
    return bad


def signflip_conditional(a, s) -> dict:
    """Exact conditional randomization test. a: per pair, A answers minus B answers (integers); s: per pair, +1 if S*
    is on side A, else -1. m is the number of pairs with S* on A. Null: all side patterns with m entries +1, equally
    likely. D = sum s_p a_p is the vote margin for the clock; its average over those patterns is (2m - n) mean(a),
    so the test uses the centred margin Dc = D - (2m - n) mean(a) = 2m(n - m)/n x (mean of a where S* is on A minus
    mean of a where S* is on B), which a constant preference for clip A or clip B cannot move.
    Returns D, Dc, m, the two-sided p = P(|Dc_null| >= |Dc|), and the unconditional two-sided p of D over all 2^n
    sign patterns, for reference only."""
    a = [int(x) for x in a]
    s = [1 if x > 0 else -1 for x in s]
    n, m = len(a), sum(x > 0 for x in s)
    d = sum(x * y for x, y in zip(a, s))
    tot = sum(a)
    t_obs = sum(x for x, y in zip(a, s) if y > 0)
    z_obs = n * t_obs - m * tot                     # = n (Dc) / 2, an integer
    ways = [dict() for _ in range(m + 1)]          # ways[k][t]: patterns with k pairs on A whose a-sum on A is t
    ways[0][0] = 1
    for x in a:
        for k in range(m, 0, -1):
            for t, w in ways[k - 1].items():
                ways[k][t + x] = ways[k].get(t + x, 0) + w
    total = math.comb(n, m)
    two = sum(w for t, w in ways[m].items() if abs(n * t - m * tot) >= abs(z_obs)) / total
    off = sum(abs(x) for x in a)
    dist = np.zeros(2 * off + 1)
    dist[off] = 1.0
    for x in a:
        if x:
            k = abs(x)
            new = np.zeros_like(dist)
            new[k:] += 0.5 * dist[:-k]
            new[:-k] += 0.5 * dist[k:]
            dist = new
    vals = np.arange(-off, off + 1)
    uncond = float(dist[np.abs(vals) >= abs(d)].sum())
    return {"D": d, "D_centred": 2 * z_obs / n if n else 0.0, "z": z_obs, "m": m,
            "p_two_sided": min(1.0, two), "p_unconditional": min(1.0, uncond)}


def b2_label(d: int, p: float) -> str:
    if p < ALPHA and d > 0:
        return "listeners favour the clock"
    if p < ALPHA and d < 0:
        return "listeners favour no clock"
    return "no clear preference"


def analyse_b2(key: dict, raters: dict, scores: dict) -> dict:
    pairs = key["pairs"]
    names = sorted(pairs)
    clips = {c["run_id"]: c for c in scores.get("results", {}).get("clips", [])}
    for p in names:
        if sorted([pairs[p]["A"]["arm"], pairs[p]["B"]["arm"]]) != sorted([ARM, BASE]):
            raise ValueError(f"{p}: the key does not pair S* with S*_b0")
        for side in ("A", "B"):
            if pairs[p][side]["run_id"] not in clips:
                raise ValueError(f"{p}: run {pairs[p][side]['run_id']} is not in the scores file")
    s = [1 if pairs[p]["A"]["arm"] == ARM else -1 for p in names]
    machine = {"words_clear": (("whisper_wr", "wr"), ("ctc_wr", "wr_ctc")), "melody_right": (("npa_v", "npa_v"),)}
    out = {"n_pairs": len(names), "raters": sorted(raters), "questions": {}, "per_pair": {p: {"item": pairs[p]["item"]} for p in names},
           "raters_detail": {n: {"n_rows": r["n_rows"], "n_answered": r["n_answered"], "missing_ids": r["missing_ids"],
                                 "invalid_answers": r["invalid"]} for n, r in raters.items()}}
    for q in B2_QUESTIONS:
        counts = {"S*": 0, "S*_b0": 0, "same": 0, "neither": 0, "blank": 0, "invalid": 0, "A": 0, "B": 0}
        per_rater, a_list = {}, []
        agree = {f: [0, 0] for f, _ in machine[q]}
        for p, sp in zip(names, s):
            side = {"A": pairs[p]["A"], "B": pairs[p]["B"]}
            ap = 0
            for rname in sorted(raters):
                v = raters[rname]["answers"].get(p, {}).get(q, "blank")
                pr = per_rater.setdefault(rname, {"S*": 0, "S*_b0": 0, "other": 0})
                counts[v] += 1
                if v in ("A", "B"):
                    ap += 1 if v == "A" else -1
                    arm = side[v]["arm"]
                    counts[arm] += 1
                    pr[arm] += 1
                    other = "B" if v == "A" else "A"
                    for field, metric in machine[q]:
                        x, y = clips[side[v]["run_id"]].get(metric), clips[side[other]["run_id"]].get(metric)
                        if x is None or y is None or round(x, 6) == round(y, 6):
                            continue
                        agree[field][0] += int(x > y)
                        agree[field][1] += 1
                else:
                    pr["other"] += 1
            a_list.append(ap)
            out["per_pair"][p][q] = sp * ap
        test = signflip_conditional(a_list, s)
        decisive = counts["S*"] + counts["S*_b0"]
        side_votes = {1: [0, 0], -1: [0, 0]}         # per side of S*: [votes for S*, A and B answers]
        for p, sp in zip(names, s):
            for rname in raters:
                v = raters[rname]["answers"].get(p, {}).get(q, "blank")
                if v in ("A", "B"):
                    side_votes[sp][1] += 1
                    side_votes[sp][0] += int((v == "A") == (sp == 1))
        shares = [side_votes[k][0] / side_votes[k][1] for k in (1, -1) if side_votes[k][1]]
        balanced = rnd(sum(shares) / 2) if len(shares) == 2 else None
        margins = [x * y for x, y in zip(a_list, s)]
        out["questions"][q] = {
            "counts": {k: counts[k] for k in ("S*", "S*_b0", "same", "neither", "blank", "invalid")},
            "share_for_clock_among_A_and_B_answers": rnd(counts["S*"] / decisive) if decisive else None,
            "balanced_share_for_clock": balanced,
            "share_of_A_among_A_and_B_answers": rnd(counts["A"] / decisive) if decisive else None,
            "pairs_net": {"for_clock": sum(x > 0 for x in margins), "for_no_clock": sum(x < 0 for x in margins),
                          "tied_or_empty": sum(x == 0 for x in margins)},
            "test": {"D": test["D"], "D_centred": rnd(test["D_centred"]), "m": test["m"],
                     "p_two_sided": sig(test["p_two_sided"]), "p_unconditional_reference": sig(test["p_unconditional"])},
            "label": b2_label(test["z"], test["p_two_sided"]),
            "per_rater": per_rater,
            "agreement_with_machine": {k: {"agree": v[0], "n": v[1], "share": rnd(v[0] / v[1]) if v[1] else None}
                                       for k, v in agree.items()},
        }
    return out


def yes_share(answers: list[str]):
    y, n = answers.count("yes"), answers.count("no")
    return rnd(y / (y + n)) if y + n else None


def majority(answers: list[str]):
    y, n = answers.count("yes"), answers.count("no")
    return None if y == n else (y > n)


def perm_test_two_groups(values_a: list[float], values_b: list[float]) -> dict:
    """Exact permutation test of mean(a) - mean(b) over all relabellings of the pooled clips (two-sided)."""
    pooled = list(values_a) + list(values_b)
    na, n = len(values_a), len(pooled)
    if na == 0 or n == na:
        return {"diff": None, "p_two_sided": None, "n_perm": 0, "n_lit": na, "n_ctl": n - na, "label": "not testable"}
    obs = np.mean(values_a) - np.mean(values_b)
    hits = total = 0
    for idx in itertools.combinations(range(n), na):
        s = set(idx)
        a = [pooled[i] for i in idx]
        b = [pooled[i] for i in range(n) if i not in s]
        total += 1
        hits += abs(np.mean(a) - np.mean(b)) >= abs(obs) - 1e-12
    p = hits / total
    label = ("listeners hear more repeats after lit" if p < ALPHA and obs > 0 else
             "listeners hear fewer repeats after lit" if p < ALPHA and obs < 0 else "no clear difference")
    return {"diff": rnd(obs), "p_two_sided": sig(p), "n_perm": total, "n_lit": na, "n_ctl": n - na, "label": label}


def cohen_kappa(x: list[bool], y: list[bool]):
    n = len(x)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return rnd((po - pe) / (1 - pe)) if pe < 1 else None


def analyse_runa(key: dict, raters: dict, e5: dict, e6: dict, owner: str | None = None) -> dict:
    clips = key["clips"]
    names = sorted(clips)
    rows5 = {(r["id"], r["cond"], r["seed"]): r for r in e5.get("results", {}).get("rows", [])}
    rows6 = {r["id"]: r for r in e6.get("results", {}).get("rows", [])}
    attempt = (e5.get("config") or {}).get("attempt", 1)
    want_dir = "outputs_v2_main" + ("" if attempt in (1, None) else f"_a{attempt}")
    for c in names:
        k = clips[c]
        if k["experiment"] == "e5":
            if (k["id"], k["cond"], k["seed"]) not in rows5:
                raise ValueError(f"{c}: E5 run {k['id']}/{k['cond']}/{k['seed']} is not in the e5 file")
            if k.get("phase_dir") != want_dir:
                raise ValueError(f"{c}: key names {k.get('phase_dir')}, the e5 file is {want_dir}")
        elif k["id"] not in rows6:
            raise ValueError(f"{c}: E6 clip {k['id']} is not in the e6 file")

    def answers(c, q, who):
        return [raters[r]["answers"].get(c, {}).get(q, "blank") for r in who]

    everyone = sorted(raters)
    groups = {}
    for c in names:
        k = clips[c]
        g = f"e5:{k['cond']}" if k["experiment"] == "e5" else f"e6:{k.get('set', '')}"
        groups.setdefault(g, []).append(c)
    groups["e6:all"] = [c for c in names if clips[c]["experiment"] == "e6"]
    out = {"n_clips": len(names), "raters": everyone, "owner": owner, "yes_rates": {}, "tests": {}, "agreement": {},
           "raters_detail": {n: {"n_rows": r["n_rows"], "n_answered": r["n_answered"], "missing_ids": r["missing_ids"],
                                 "invalid_answers": r["invalid"]} for n, r in raters.items()}}
    for g, cs in sorted(groups.items()):
        qs = RUNA_QUESTIONS if g.startswith("e5:") else RUNA_QUESTIONS[:3]
        out["yes_rates"][g] = {q: yes_share(sum((answers(c, q, everyone) for c in cs), [])) for q in qs}
        out["yes_rates"][g]["n_clips"] = len(cs)

    def lit_ctl(who):
        share = {c: yes_share(answers(c, "gap_repeats_earlier_motif", who)) for c in names}
        lit = [share[c] for c in groups.get("e5:lit", []) if share[c] is not None]
        ctl = [share[c] for c in groups.get("e5:ctl", []) if share[c] is not None]
        return perm_test_two_groups(lit, ctl)

    out["tests"]["repeat_lit_minus_ctl"] = lit_ctl(everyone)
    if owner is not None:
        others = [r for r in everyone if r.casefold() != owner.casefold()]
        if len(others) == len(everyone):
            raise ValueError(f"owner {owner!r} has no ratings file")
        out["tests"]["repeat_lit_minus_ctl_without_owner"] = lit_ctl(others) if others else None

    for mapping in ("strict", "moved"):
        res = {"all": {"human": [], "tracker": []}, "primary": {"human": [], "tracker": []}}
        table = {}
        for c in names:
            k = clips[c]
            if k["experiment"] != "e5":
                continue
            row = rows5[(k["id"], k["cond"], k["seed"])]
            if row.get("choice") == "tie":
                continue
            m = majority(answers(c, "gap_repeats_earlier_motif", everyone))
            if m is None:
                continue
            heard = EARLIER[k["cond"]] | (MOVED.get(k["cond"], set()) if mapping == "moved" else set())
            t = row.get("choice") in heard
            for part in ("all",) + (("primary",) if k["cond"] in PRIMARY_AGREEMENT else ()):
                res[part]["human"].append(m)
                res[part]["tracker"].append(t)
            cell = table.setdefault(k["cond"], {"both_yes": 0, "both_no": 0, "human_only": 0, "tracker_only": 0})
            cell["both_yes" if m and t else "both_no" if not m and not t else "human_only" if m else "tracker_only"] += 1
        out["agreement"][f"repeat_vs_tracker_{mapping}"] = {
            part: {"n": len(v["human"]), "agree": sum(a == b for a, b in zip(v["human"], v["tracker"])),
                   "kappa": cohen_kappa(v["human"], v["tracker"])} for part, v in res.items()}
        out["agreement"][f"repeat_vs_tracker_{mapping}"]["by_condition"] = table

    for q, fn in (("vocals_present", lambda r: bool(r.get("vocal"))),
                  ("words_intelligible", lambda r: (r.get("n_matched") or 0) / max(1, r.get("n_words") or 0) >= 0.5)):
        h, m_ = [], []
        for c in groups["e6:all"]:
            mj = majority(answers(c, q, everyone))
            if mj is None:
                continue
            h.append(mj)
            m_.append(fn(rows6[clips[c]["id"]]))
        out["agreement"][f"e6_{q}"] = {"n": len(h), "agree": sum(a == b for a, b in zip(h, m_)), "kappa": cohen_kappa(h, m_)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pack", choices=("check", "b2", "runa"))
    ap.add_argument("--key", type=Path, help="the pack's key (b2, runa)")
    ap.add_argument("--template", type=Path, help="the pack's own blank ratings.csv (check)")
    ap.add_argument("--ratings", required=True, nargs="+", type=Path)
    ap.add_argument("--exclude-file", type=Path)
    ap.add_argument("--identical-ok", nargs="*", default=[],
                    help="raters whose byte-identical sheets the owner confirmed come from different people")
    ap.add_argument("--owner", help="runa: the owner's rater name, or 'none' if the owner did not rate")
    ap.add_argument("--scores", type=Path, help="b2_result.json (b2)")
    ap.add_argument("--e5", type=Path, help="e5v2_main.json (runa)")
    ap.add_argument("--e6", type=Path, help="e6_readout.json (runa)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.pack == "check":
        if not a.template:
            ap.error("check needs --template (the pack's blank ratings.csv)")
        return check_sheets(a.template, a.ratings)
    need = {"b2": [("--key", a.key), ("--scores", a.scores), ("--out", a.out)],
            "runa": [("--key", a.key), ("--e5", a.e5), ("--e6", a.e6), ("--owner", a.owner), ("--out", a.out)]}[a.pack]
    missing = [n for n, v in need if not v]
    if missing:
        ap.error(f"{a.pack} needs {' '.join(missing)}")
    unused = {"b2": [("--owner", a.owner), ("--e5", a.e5), ("--e6", a.e6)], "runa": [("--scores", a.scores)]}[a.pack]
    for n, v in unused:
        if v:
            print(f"note: {n} is not used for {a.pack}", file=sys.stderr)
    excl = read_exclusions(a.exclude_file)
    key = json.loads(a.key.read_text(encoding="utf-8"))
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    if a.pack == "b2":
        raters, dropped, empty = load_raters(a.ratings, excl, "pair", B2_QUESTIONS, B2_ANSWERS, set(key["pairs"]),
                                             a.identical_ok)
    else:
        raters, dropped, empty = load_raters(a.ratings, excl, "clip", RUNA_QUESTIONS, YES_NO, set(key["clips"]),
                                             a.identical_ok)
    if not raters:
        raise ValueError("no rater left after exclusions")
    if a.pack == "b2":
        res = analyse_b2(key, raters, load(a.scores))
    else:
        owner = None if a.owner.casefold() == "none" else a.owner
        if owner is not None and owner.casefold() not in {n.casefold() for n in raters}:
            owner = None
            print("note: the owner's sheet is absent or excluded; the lit-ctl test runs once", file=sys.stderr)
        res = analyse_runa(key, raters, load(a.e5), load(a.e6), owner)
        res["owner_sheet"] = owner or "none"
    here = Path(__file__).resolve()
    res["left_out_no_readable_answer"] = empty
    res["provenance"] = {"code_sha256": sha256(here), "key_sha256": sha256(a.key),
                         "excluded": {n: {**excl[n.casefold()], **v} for n, v in sorted(dropped.items())},
                         "exclusions_file_sha256": sha256(a.exclude_file) if a.exclude_file else None,
                         "ratings": {n: {"file": Path(r["path"]).name, "sha256": r["sha256"]} for n, r in raters.items()},
                         "plan": "experiments/listening/registration/listening_plan.json"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for n, r in raters.items():
        print(f"rater {n}: {r['n_answered']} answers in {r['n_rows']} rows, {len(r['missing_ids'])} rows missing, "
              f"{sum(r['invalid'].values())} unreadable answers {sorted(r['invalid'])}")
    if a.pack == "b2":
        for q, v in res["questions"].items():
            t = v["test"]
            print(f"{q}: clock {v['counts']['S*']}, no clock {v['counts']['S*_b0']}, D {t['D']}, centred "
                  f"{t['D_centred']}, m {t['m']}, p {t['p_two_sided']} (all coin tosses: "
                  f"{t['p_unconditional_reference']}) -> {v['label']}")
    else:
        print(json.dumps(res["tests"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
