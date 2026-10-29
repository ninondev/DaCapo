"""B2 items for Run B (addendum 3): the 24 test lines and their melodies, the guides of all 36 B2 items,
and the check of ACE-Step's lyric layout.

  python make_items_b2.py build               manifest_b2.json from lines_b2.txt (generator seed 22)
  python make_items_b2.py render [--out DIR]  72 guides (item00-item11 and b2t00-b2t23, Voice Oohs and
                                              clarinet) -> guides_b2.json with their sha256
  python make_items_b2.py render --check      render again and exit 1 if any sha256 differs from guides_b2.json
  python make_items_b2.py tok-check [--ckpt DIR]   the lyric keys of all 36 items -> tok_check_b2.json

The test melodies use E4's rhythm() and pitches() unchanged (experiments/e4_sing/make_items.py, imported,
never run): one note per word, keys C, D, F and G in turn, 100 BPM, a 4-beat lead-in, 16 beats, MIDI
57-76, each note sounding for 0.9 of its length. The guides are rendered as E4 renders them (dacapo.render
at 48 kHz: FluidSynth, FluidR3_GM, gain 0.6, reverb and chorus off). The lines were written by the agents
under the line rules below; build refuses a line that breaks one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import plan_b2 as P  # noqa: E402

from dacapo.words import group_words, lyric_words  # noqa: E402

SEED = 22
N_WORDS = (11, 13)
HEADER = "# Languages\nen\n\n# Lyric\n"          # ACE-Step's lyric prompt (prompt_utils.py:27-29)
HEADER_TOKENS = 9
EOT = 151643                                     # <|endoftext|>
HOMOPHONES = frozenset("sea see night knight one won sun son hear here road rode whole hole by buy new knew right "
                       "write red read tale tail blue blew week weak made maid meet meat".split())
HALLUCINATION_WORDS = frozenset(("thank", "you", "watch", "subscribe"))
AUTHOR = ("The 24 lines were written by the agents under the line rules of addendum 3 (section items), before any "
          "B2 generation.")


def load_module(name: str, path: Path):
    """Import a script by path under its own name (experiments/e4_sing and experiments/e5_motif both hold
    a make_items.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def e4():
    return load_module("e4_make_items", ROOT / "experiments" / "e4_sing" / "make_items.py")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# line rules

def syllables(word: str) -> int:
    """Vowel groups, less a silent final e, es or ed (a leading y is a consonant). Exact enough for the plain
    one-syllable words the rules allow; every word of E4's twelve lines counts 1."""
    w = word.lower()
    w = w[1:] if w.startswith("y") else w
    n = len(re.findall(r"[aeiouy]+", w))
    if n > 1 and w.endswith("e") and not (w.endswith("le") and len(w) > 2 and w[-3] not in "aeiouy"):
        n -= 1
    elif n > 1 and w.endswith("es") and not w.endswith(("ses", "xes", "zes", "ches", "shes", "ces", "ges")):
        n -= 1
    elif n > 1 and w.endswith("ed") and not w.endswith(("ted", "ded")):
        n -= 1
    return n


def earlier_lines() -> set[str]:
    sys.path.insert(0, str(ROOT / "experiments" / "e6_clock"))
    import readout
    lines = {x.lower() for x in e4().LINES}
    for verse in readout.VERSES.values():
        lines |= {x.strip().lower() for x in verse.splitlines() if x.strip() and not x.startswith("[")}
    return lines


def line_problems(line: str, old: set[str] | None = None) -> list[str]:
    """The rules that need no tokenizer (tok_check adds the rest)."""
    old = earlier_lines() if old is None else old
    words = line.split()
    out = []
    if not N_WORDS[0] <= len(words) <= N_WORDS[1]:
        out.append(f"{len(words)} words")
    for j, w in enumerate(words):
        if not re.fullmatch(r"[A-Za-z]+", w):
            out.append(f"{w!r}: not letters only")
            continue
        want = w if w == "I" else (w[0].upper() + w[1:].lower() if j == 0 else w.lower())
        if w != want:
            out.append(f"{w!r}: capitalised unlike E4 (first word and I only)")
        if syllables(w) != 1:
            out.append(f"{w!r}: {syllables(w)} syllables")
        if w.lower() in HOMOPHONES:
            out.append(f"{w!r}: registered homophone")
        if w.lower() in HALLUCINATION_WORDS:
            out.append(f"{w!r}: a word Whisper produces on wordless audio")
    if line.strip().lower() in old:
        out.append("an E4 or E6 line")
    return out


def read_lines() -> list[str]:
    return [x for x in P.LINES.read_text().splitlines() if x.strip()]


# manifest

def build(lines: list[str]) -> dict:
    """The 24 test items, exactly as E4's main() builds its items (the same rhythm, pitches and note
    lengths), from numpy default_rng(22)."""
    m = e4()
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
        iid = f"b2t{i:02d}"
        items.append({"id": iid, "split": "test", "key": key, "words": words, "lyrics": "[Verse]\n" + line,
                      "melody": {"bpm": m.BPM, "notes": [asdict(n) for n in notes]},
                      "guides": {g: f"guides/{iid}_{g}.wav" for g in P.GUIDES},
                      "duration": float(np.ceil((m.LEAD_IN + 16 + 2) * 60 / m.BPM))})
    return {"seed": SEED, "bpm": m.BPM, "lines": "experiments/e8_clock_sing/lines_b2.txt",
            "lines_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), "author": AUTHOR,
            "generator": "experiments/e4_sing/make_items.py rhythm() and pitches(), numpy default_rng(22), keys C, D, F, G "
                         "in turn", "items": items}


def manifest_text(man: dict) -> str:
    return json.dumps(man, indent=1) + "\n"


def cmd_build(args) -> int:
    lines = read_lines()
    old = earlier_lines()
    bad = {ln: p for ln in lines if (p := line_problems(ln, old))}
    if len(lines) != 24 or len(set(lines)) != 24:
        bad["(file)"] = [f"{len(lines)} lines, {len(set(lines))} distinct; 24 needed"]
    if bad:
        print("line rules fail:", json.dumps(bad, indent=1))
        return 1
    P.B2_MANIFEST.write_text(manifest_text(build(lines)))
    print(f"wrote {P.B2_MANIFEST.relative_to(ROOT)}: 24 items")
    return 0


# guides

def render_one(it: dict, guide: str, out: Path) -> tuple[str, str, str]:
    """As experiments/e6_clock/runb_cpu_test.py renders item00's guide: dacapo.render at 48 kHz."""
    from dacapo.render import render
    from dacapo.symbolic import Melody, Note
    path = P.guide_path(out, it["id"], guide)
    bpm = float(it["melody"]["bpm"])
    m = Melody(tuple(Note(**n) for n in it["melody"]["notes"]), bpm)
    render([m.to_instrument(P.GUIDES[guide])], bpm, path, sr=48000)
    return it["id"], guide, sha256(path)


def soundfont() -> dict:
    from dacapo.render import default_sf2
    sf2 = default_sf2()
    try:
        fs = subprocess.run(["fluidsynth", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        fs = None
    return {"fluidsynth": fs, "soundfont": sf2.name, "soundfont_sha256": sha256(sf2)}


def render_all(out: Path, workers: int = 4) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    its = P.dev_items() + P.test_items()
    with ThreadPoolExecutor(workers) as ex:
        done = list(ex.map(lambda a: render_one(*a, out), [(it, g) for it in its for g in P.GUIDES]))
    sha: dict = {}
    for iid, g, h in done:
        sha.setdefault(iid, {})[g] = h
    return {"renderer": "dacapo.render.render: FluidSynth, gain 0.6, reverb and chorus off, 48 kHz",
            "programs": P.GUIDES, **soundfont(), "sha256": {it["id"]: sha[it["id"]] for it in its}}


def guide_mismatches(got: dict, want: dict) -> list[str]:
    out = [f"{i} {g}" for i in want["sha256"] for g in want["sha256"][i]
           if got["sha256"].get(i, {}).get(g) != want["sha256"][i][g]]
    if got["soundfont_sha256"] != want["soundfont_sha256"]:
        out.append(f"soundfont {got['soundfont_sha256']}")
    return out


def cmd_render(args) -> int:
    got = render_all(Path(args.out))
    if args.check:
        bad = guide_mismatches(got, P.read_json(P.GUIDES_JSON))
        print("guides:", "all 72 match guides_b2.json" if not bad else f"MISMATCH {bad}")
        return 1 if bad else 0
    P.GUIDES_JSON.write_text(json.dumps(got, indent=1) + "\n")
    print(f"wrote {P.GUIDES_JSON.relative_to(ROOT)}: {sum(len(v) for v in got['sha256'].values())} guides, "
          f"soundfont {got['soundfont_sha256'][:8]}, {got['fluidsynth']}")
    return 0


# tokenizer check

def decode_incrementally(tok, ids: list[int]) -> list[str]:
    """MusicStampsAligner._decode_tokens_incrementally (acestep/core/scoring/dit_alignment.py:199)."""
    out, prev = [], b""
    for i in range(len(ids)):
        cur = tok.decode(ids[:i + 1], skip_special_tokens=False).encode("utf-8", errors="surrogatepass")
        new = cur[len(prev):] if len(cur) >= len(prev) else b""
        out.append(new.decode("utf-8", errors="replace"))
        prev = cur
    return out


def tok_check(ckpt: Path) -> dict:
    """ACE-Step's lyric layout for every item, replicated with its tokenizer (the generation stage checks it
    against ACE-Step's own path): header length s, n_lyr, and the absolute keys of each word."""
    from transformers import AutoTokenizer
    tdir = Path(ckpt) / "Qwen3-Embedding-0.6B"
    tok = AutoTokenizer.from_pretrained(str(tdir))
    hdr = tok.encode(HEADER, add_special_tokens=False)
    out = {}
    for it in P.dev_items() + P.test_items():
        ids = tok(f"{HEADER}{it['lyrics']}<|endoftext|>", padding="longest", truncation=True, max_length=2048).input_ids
        s, e = len(hdr), ids.index(EOT)
        words = group_words(decode_incrementally(tok, ids[s:e]))
        keys = [[s + t for t in w.tokens] for w in words]
        problems = []
        if ids[:s] != hdr or s != HEADER_TOKENS:
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
            "tokenizer_json_sha256": sha256(tdir / "tokenizer.json"), "header": HEADER, "items": out,
            "passed": all(not v["problems"] for v in out.values())}


def default_ckpt() -> Path:
    c = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
    return Path(c) if c else Path(os.environ.get("ACESTEP_REPO", ".")) / "checkpoints"


def cmd_tok(args) -> int:
    res = tok_check(Path(args.ckpt) if args.ckpt else default_ckpt())
    P.TOK_JSON.write_text(json.dumps(res, indent=1) + "\n")
    bad = {k: v["problems"] for k, v in res["items"].items() if v["problems"]}
    print(f"wrote {P.TOK_JSON.relative_to(ROOT)}:", "all 36 items pass" if res["passed"] else f"FAILED {bad}")
    return 0 if res["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "render", "tok-check"])
    ap.add_argument("--out", default=str(P.GUIDES_DIR))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    return {"build": cmd_build, "render": cmd_render, "tok-check": cmd_tok}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
