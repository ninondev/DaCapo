"""C1 items for Run C (addendum 4): the 96 test lines and their melodies, the Voice Oohs guides, and the check of
ACE-Step's lyric layout.

  python make_items_c1.py check               the line rules that need no tokenizer
  python make_items_c1.py build               manifest_c1.json from lines_c1.txt (generator seed 33)
  python make_items_c1.py render [--out DIR]  the Voice Oohs guides of c1t00-c1t95, item00 (A0) and item01 (tiny)
                                              -> guides_c1.json with their sha256
  python make_items_c1.py render --check      render again and exit 1 if any sha256 differs from guides_c1.json
  python make_items_c1.py tok-check [--ckpt DIR]   the lyric keys of the 98 items -> tok_check_c1.json

The melodies follow make_items_b2's procedure exactly: E4's rhythm() and pitches() (experiments/e4_sing/
make_items.py, imported, never run), one note per word, keys C, D, F and G in turn, 100 BPM, a 4-beat lead-in,
16 beats, MIDI 57-76, each note sounding for 0.9 of its length; only the generator seed is new (33). The guides
are rendered as B2 renders them (make_items_b2.render_one: dacapo.render at 48 kHz, FluidSynth, FluidR3_GM,
gain 0.6, reverb and chorus off, program 53). The lines were written by the agents under addendum 3's line
rules (make_items_b2.line_problems, imported unchanged) and three more: no line from E4, E6, B2 or E7, no two
Run C lines sharing more than 6 words, and no Run C line sharing more than 6 words with a line of E4, E6, B2 or E7
(counted with repeats). build refuses a line that breaks a rule.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "experiments" / "e8_clock_sing"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import make_items_b2 as M  # noqa: E402
import plan_b2 as P2  # noqa: E402
import plan_c1 as P  # noqa: E402

from dacapo.words import group_words, lyric_words  # noqa: E402

SEED = 33
N_LINES = 96
MAX_SHARED = 6
AUTHOR = ("The 96 lines were written by the agents under the line rules of addendum 3 (section items) and the rules "
          "added for Run C (addendum 4), before any Run C generation.")
E7_MELODIES = ROOT / "experiments" / "e7_melody_adapter" / "melodies_suno.json"
manifest_text = M.manifest_text
guide_mismatches = M.guide_mismatches   # run_c1.setup_check compares the guides it renders


# line rules

def b2_lines() -> list[str]:
    return [x for x in P2.LINES.read_text().splitlines() if x.strip()]


def e7_lines() -> list[str]:
    items = json.loads(E7_MELODIES.read_text())["items"]
    return [x.strip() for it in items for x in it["lyrics"].splitlines() if x.strip()]


def earlier_lines() -> set[str]:
    """E4's and E6's lines (make_items_b2.earlier_lines), B2's 24 and the lines of E7's 12 lyrics, lower case."""
    return M.earlier_lines() | {x.lower() for x in b2_lines()} | {x.lower() for x in e7_lines()}


def line_problems(line: str, old: set[str] | None = None) -> list[str]:
    """Addendum 3's rules that need no tokenizer (make_items_b2.line_problems) and no earlier line."""
    old = earlier_lines() if old is None else old
    out = M.line_problems(line, set())
    if line.strip().lower() in old:
        out.append("a line from E4, E6, B2 or E7")
    return out


def shared_words(a: str, b: str) -> int:
    """Words two lines share, counted with repeats, case-insensitive."""
    return sum((Counter(a.lower().split()) & Counter(b.lower().split())).values())


def overlap_problems(lines: list[str]) -> list[str]:
    return [f"lines {i} and {j} share {n} words" for (i, a), (j, b) in itertools.combinations(enumerate(lines), 2)
            if (n := shared_words(a, b)) > MAX_SHARED]


def earlier_overlap_problems(lines: list[str], old: set[str] | None = None) -> list[str]:
    """A Run C line that shares more than MAX_SHARED words (with repeats) with a line of E4, E6, B2 or E7."""
    old = sorted(earlier_lines() if old is None else old)
    out = []
    for i, a in enumerate(lines):
        n, b = max(((shared_words(a, b), b) for b in old), default=(0, ""))
        if n > MAX_SHARED:
            out.append(f"line {i} shares {n} words with the earlier line '{b}'")
    return out


def read_lines() -> list[str]:
    return [x for x in P.LINES.read_text().splitlines() if x.strip()]


def rule_problems(lines: list[str]) -> dict[str, list[str]]:
    """Every rule that needs no tokenizer, over the whole file. Empty when every rule holds."""
    old = earlier_lines()
    bad = {ln: p for ln in lines if (p := line_problems(ln, old))}
    if len(lines) != N_LINES or len(set(lines)) != N_LINES:
        bad["(file)"] = [f"{len(lines)} lines, {len(set(lines))} distinct; {N_LINES} needed"]
    if (ov := overlap_problems(lines)):
        bad["(overlap)"] = ov
    if (ov := earlier_overlap_problems(lines, old)):
        bad["(overlap with earlier)"] = ov
    return bad


# manifest

def build(lines: list[str]) -> dict:
    """The 96 test items, exactly as make_items_b2.build builds B2's (E4's rhythm, pitches and note lengths), from
    numpy default_rng(33)."""
    m = M.e4()
    from dacapo.symbolic import Note
    rng = np.random.default_rng(SEED)
    items = []
    for i, line in enumerate(lines):
        words = line.split()
        key = list(m.KEYS)[i % len(m.KEYS)]
        durs = m.rhythm(len(words), rng)
        ps = m.pitches(len(words), m.KEYS[key], rng)
        notes, t = [], m.LEAD_IN
        for p, d in zip(ps, durs):
            notes.append(Note(p, t, d * 0.9 if d >= 1 else d * 0.95))
            t += d
        iid = f"c1t{i:02d}"
        items.append({"id": iid, "split": "test", "key": key, "words": words, "lyrics": "[Verse]\n" + line,
                      "melody": {"bpm": m.BPM, "notes": [asdict(n) for n in notes]},
                      "guides": {P.GUIDE: f"guides/{iid}_{P.GUIDE}.wav"},
                      "duration": float(np.ceil((m.LEAD_IN + 16 + 2) * 60 / m.BPM))})
    return {"seed": SEED, "bpm": m.BPM, "lines": "experiments/e9_clock_replicate/lines_c1.txt",
            "lines_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), "author": AUTHOR,
            "generator": "experiments/e4_sing/make_items.py rhythm() and pitches(), numpy default_rng(33), keys C, D, F, G "
                         "in turn", "items": items}


def melody_key(it: dict) -> tuple:
    return tuple((n["pitch"], n["start"], n["dur"]) for n in it["melody"]["notes"])


def melody_problems(man: dict) -> list[str]:
    """A Run C melody that repeats another Run C melody, a B2 melody or an E4 melody."""
    old = {melody_key(it): it["id"] for it in P2.dev_items() + P2.test_items()}
    seen: dict = {}
    out = []
    for it in man["items"]:
        k = melody_key(it)
        if k in old:
            out.append(f"{it['id']} repeats {old[k]}")
        if k in seen:
            out.append(f"{it['id']} repeats {seen[k]}")
        seen[k] = it["id"]
    return out


def cmd_check(args) -> int:
    bad = rule_problems(read_lines())
    print("line rules:", "all 96 lines pass" if not bad else json.dumps(bad, indent=1))
    return 1 if bad else 0


def cmd_build(args) -> int:
    lines = read_lines()
    bad = rule_problems(lines)
    man = build(lines)
    if (mp := melody_problems(man)):
        bad["(melodies)"] = mp
    if bad:
        print("rules fail, nothing written:", json.dumps(bad, indent=1))
        return 1
    P.C1_MANIFEST.write_text(manifest_text(man))
    print(f"wrote {P.C1_MANIFEST.relative_to(ROOT)}: {len(man['items'])} items")
    return 0


# guides

def render_all(out: Path, workers: int = 4) -> dict:
    """The Voice Oohs guide of every Run C item, of item00 (A0) and of item01 (tiny), as B2 renders them."""
    out.mkdir(parents=True, exist_ok=True)
    its = P.extra_items() + P.test_items()
    with ThreadPoolExecutor(workers) as ex:
        done = list(ex.map(lambda it: M.render_one(it, P.GUIDE, out), its))
    sha = {iid: {g: h} for iid, g, h in done}
    return {"renderer": "dacapo.render.render: FluidSynth, gain 0.6, reverb and chorus off, 48 kHz",
            "programs": {P.GUIDE: P2.GUIDES[P.GUIDE]}, **M.soundfont(), "sha256": {it["id"]: sha[it["id"]] for it in its}}


def b2_guide_problems(got: dict) -> list[str]:
    """item00's and item01's guides equal B2's (guides_b2.json), which rendered them the same way."""
    b2 = P2.read_json(P2.GUIDES_JSON)
    out = [f"{i} differs from guides_b2.json" for i in P.EXTRA_ITEMS if got["sha256"][i][P.GUIDE] != b2["sha256"][i][P.GUIDE]]
    if got["soundfont_sha256"] != b2["soundfont_sha256"]:
        out.append("soundfont differs from guides_b2.json")
    return out


def cmd_render(args) -> int:
    got = render_all(Path(args.out))
    if args.check:
        bad = M.guide_mismatches(got, P.read_json(P.GUIDES_JSON))
        n = sum(len(v) for v in got["sha256"].values())
        print("guides:", f"all {n} match guides_c1.json" if not bad else f"MISMATCH {bad}")
        return 1 if bad else 0
    bad = b2_guide_problems(got)
    if bad:
        print("guides not written:", bad)
        return 1
    P.GUIDES_JSON.write_text(json.dumps(got, indent=1) + "\n")
    print(f"wrote {P.GUIDES_JSON.relative_to(ROOT)}: {sum(len(v) for v in got['sha256'].values())} guides, "
          f"soundfont {got['soundfont_sha256'][:8]}, {got['fluidsynth']}")
    return 0


# tokenizer check

def tok_check(ckpt: Path) -> dict:
    """ACE-Step's lyric layout for every item, replicated with its tokenizer exactly as make_items_b2.tok_check
    does for B2 (the generation stage checks it against ACE-Step's own path)."""
    from transformers import AutoTokenizer
    tdir = Path(ckpt) / "Qwen3-Embedding-0.6B"
    tok = AutoTokenizer.from_pretrained(str(tdir))
    hdr = tok.encode(M.HEADER, add_special_tokens=False)
    out = {}
    for it in P.extra_items() + P.test_items():
        ids = tok(f"{M.HEADER}{it['lyrics']}<|endoftext|>", padding="longest", truncation=True, max_length=2048).input_ids
        s, e = len(hdr), ids.index(M.EOT)
        words = group_words(M.decode_incrementally(tok, ids[s:e]))
        keys = [[s + t for t in w.tokens] for w in words]
        problems = []
        if ids[:s] != hdr or s != M.HEADER_TOKENS:
            problems.append(f"header: s = {s}")
        if [w.norm for w in words] != lyric_words(it["lyrics"]):
            problems.append("token words differ from the lyric words")
        if len(words) != len(it["melody"]["notes"]):
            problems.append(f"{len(words)} words for {len(it['melody']['notes'])} notes")
        if any(not 1 <= len(w.tokens) <= 2 for w in words):
            problems.append(f"words of 3+ tokens: {[w.text for w in words if len(w.tokens) > 2]}")
        out[it["id"]] = {"start": s, "end": e, "n_lyr": len(ids), "words": [w.norm for w in words], "keys": keys,
                         "multi_token_words": [w.text for w in words if len(w.tokens) > 1], "problems": problems}
    return {"tokenizer": "Qwen3-Embedding-0.6B of the registered ACE-Step checkpoint",
            "tokenizer_json_sha256": M.sha256(tdir / "tokenizer.json"), "header": M.HEADER, "items": out,
            "passed": all(not v["problems"] for v in out.values())}


def b2_tok_problems(res: dict) -> list[str]:
    """item00's and item01's layouts equal B2's tok_check_b2.json."""
    b2 = P2.read_json(P2.TOK_JSON)
    out = [f"{i} differs from tok_check_b2.json" for i in P.EXTRA_ITEMS if res["items"][i] != b2["items"][i]]
    if res["tokenizer_json_sha256"] != b2["tokenizer_json_sha256"]:
        out.append("tokenizer.json differs from tok_check_b2.json")
    return out


def cmd_tok(args) -> int:
    res = tok_check(Path(args.ckpt) if args.ckpt else M.default_ckpt())
    bad = {k: v["problems"] for k, v in res["items"].items() if v["problems"]}
    b2 = b2_tok_problems(res)
    if b2:
        print("tokenizer check not written:", b2)
        return 1
    P.TOK_JSON.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {P.TOK_JSON.relative_to(ROOT)}:", f"all {len(res['items'])} items pass" if res["passed"] else f"FAILED {bad}")
    return 0 if res["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "build", "render", "tok-check"])
    ap.add_argument("--out", default=str(P.GUIDES_DIR))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    return {"check": cmd_check, "build": cmd_build, "render": cmd_render, "tok-check": cmd_tok}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
