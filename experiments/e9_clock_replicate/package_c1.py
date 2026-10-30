"""The C1 package (addendum 4, section package): the result files, ledgers and logs, the demo by rule, and a blind
A/B pack of 24 pairs. B2's packaging (experiments/e8_clock_sing/package_b2.py) is imported and never edited: the
audio conversion, the listener README and the reuse of an earlier key's sides come from it.

The demo and the pack come from c1_result.json, that is, from the draws the selector keeps: for each test item,
S*'s kept draw next to S*_b0's kept draw. Nothing is picked by ear. The pack's 24 items and their order are the
first 24 of numpy default_rng(20261031).permutation over the demo items in item order (all 96 at the full
scale); which clip of a pair is A comes from the operating system's random source (secrets.randbelow(2)) and is
written only to the key, so neither the code nor the seed tells a listener which side is S*. Packaging again
keeps the sides of an earlier key for the same pairs. The pack and its key go outside the package. The pack is
exploratory.
"""
from __future__ import annotations

import csv
import json
import secrets
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "experiments" / "e8_clock_sing"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import package_b2 as PB  # noqa: E402

BLIND_SEED = 20261031
N_PAIRS = 24
DEMO_KBPS, PACK_KBPS = PB.DEMO_KBPS, PB.PACK_KBPS
README = PB.README


def pack_pairs(demo: list[dict], n: int = N_PAIRS, seed: int = BLIND_SEED) -> list[dict]:
    """The pack's pairs in pack order: the first n of default_rng(seed).permutation over the demo pairs in item
    order."""
    pairs = sorted((d for d in demo if d.get("comparator_run")), key=lambda d: d["item"])
    order = np.random.default_rng(seed).permutation(len(pairs))[:n]
    return [pairs[int(j)] for j in order]


def copy_results(paths, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    got = []
    files = [paths.res / "c1_result.json", paths.res / "c1_summary.json", paths.a0_report, paths.a0 / "a0_gen.json",
             paths.a0 / "a0_scores.json", paths.setup / "setup_check.json", paths.setup / "setup_model.json"]
    for f in files:
        if f.exists():
            shutil.copy(f, out / f.name)
            got.append(f.name)
    led = out / "ledgers"
    led.mkdir(exist_ok=True)
    for f, name in ((paths.ledger(), "test_generation.jsonl"), (paths.scores(), "test_scores.jsonl")):
        if f.exists():
            shutil.copy(f, led / name)
            got.append(f"ledgers/{name}")
    return got


def blind_pack(paths, demo: list[dict], items: dict, pack_zip: Path, key_path: Path) -> dict:
    """The 24 pairs of pack_pairs, A and B drawn per pair with secrets (or kept from an earlier key of the same
    pairs), each with the item's Voice Oohs guide as the melody reference."""
    pairs = pack_pairs(demo)
    pack = pack_zip.with_suffix("")
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    before = PB.earlier_sides(key_path)
    key, rows = {}, []
    for k, d in enumerate(pairs):
        name = f"pair_{k + 1:02d}"
        clips = [("S*", d["s_star_run"]), (d["comparator"], d["comparator_run"])]
        on_a = before.get((d["item"], frozenset((d["s_star_run"], d["comparator_run"]))))
        flip = on_a == d["comparator_run"] if on_a else bool(secrets.randbelow(2))
        if flip:
            clips.reverse()
        files = {}
        for side, (arm, rid) in zip("AB", clips):
            files[side] = PB.audio(paths.test / f"{rid}.wav", pack / f"{name}_{side}", PACK_KBPS)
        files["melody"] = PB.audio(paths.guides / f"{d['item']}_oohs.wav", pack / f"{name}_melody", PACK_KBPS)
        key[name] = {"item": d["item"], "A": {"arm": clips[0][0], "run_id": clips[0][1]},
                     "B": {"arm": clips[1][0], "run_id": clips[1][1]}, "files": files}
        rows.append([name, items[d["item"]]["lyrics"].split("\n", 1)[-1], "", "", ""])
    with open(pack / "ratings.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "lyric", "melody_right", "words_clear", "notes"])
        w.writerows(rows)
    (pack / "README.txt").write_text(README)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({"order_seed": BLIND_SEED, "n_pairs": len(key),
                                    "items": "the first 24 of default_rng(20261031).permutation over the demo items",
                                    "sides": "secrets.randbelow(2) per pair", "pairs": key}, indent=1) + "\n")
    zipped = shutil.make_archive(str(pack), "zip", pack)
    shutil.rmtree(pack)
    return {"n_pairs": len(key), "items": [d["item"] for d in pairs], "zip": str(zipped), "key": str(key_path)}


def package(paths, pkg: Path, key_path: Path, pack_zip: Path, items: dict) -> dict:
    """Fill pkg/c1 with the results and the demo; the blind pack goes to pack_zip and its key to key_path, both
    outside pkg."""
    for p, what in ((key_path, "key"), (pack_zip, "pack")):
        if p.resolve().is_relative_to(pkg.resolve()):
            raise ValueError(f"the blind-pack {what} must not go inside the package")
    out = pkg / "c1"
    if out.exists():
        shutil.rmtree(out)
    files = copy_results(paths, out)
    resf = paths.res / "c1_result.json"
    res = json.loads(resf.read_text())["results"] if resf.exists() else {}
    demo = res.get("demo") or []
    shown = []
    for d in demo:
        row = dict(d)
        row["s_star_file"] = PB.audio(paths.test / f"{d['s_star_run']}.wav", out / "demo" / f"{d['item']}__S_star__{d['s_star_run']}")
        if d.get("comparator_run"):
            row["comparator_file"] = PB.audio(paths.test / f"{d['comparator_run']}.wav",
                                              out / "demo" / f"{d['item']}__S_b0__{d['comparator_run']}")
        shown.append(row)
    if shown:
        (out / "demo").mkdir(parents=True, exist_ok=True)
        (out / "demo" / "demo.json").write_text(json.dumps({"rule": "per item, the S* draw the selector keeps and S*_b0's "
                                                            "kept draw", "verdict": res.get("verdict"), "clips": shown},
                                                           indent=1) + "\n")
    blind = blind_pack(paths, demo, items, pack_zip, key_path) if demo else {"n_pairs": 0}
    return {"state": "ok", "files": files, "demo_clips": sum(bool(r.get("s_star_file")) + bool(r.get("comparator_file"))
                                                             for r in shown), "blind_pack": blind,
            "verdict": res.get("verdict")}
