"""The B2 package (addendum 3, section reporting): the result files and ledgers, the demo by rule, and a blind
A/B pack of the same pairs. The pack and its key are written outside the package, so the package (with the
demo named by arm) and the pack can go to different people.

The demo and the pack come from b2_result.json, that is, from the draws the selector keeps: for each test
item, S*'s kept draw next to the primary comparator's kept draw. Nothing is picked by ear. Clips become
MP3 through ffmpeg when it is installed (96 kbps in the demo, 128 kbps in the pack, so the two never share a
file), and are copied as WAV otherwise. The pair order comes from numpy default_rng(20261029); which clip
of a pair is A comes from the operating system's random source and is written only to the key, so neither
the code nor the seed tells a listener which side is S*. Packaging again keeps the sides of an earlier key
for the same pairs, so a pack already handed out still matches its key.
"""
from __future__ import annotations

import csv
import json
import secrets
import shutil
import subprocess
from pathlib import Path

import numpy as np

BLIND_SEED = 20261029
DEMO_KBPS, PACK_KBPS = 96, 128
README = """Each pair is one melody with one lyric. pair_NN_melody plays the melody as a wordless guide; the lyric is
in ratings.csv. Listen to A and B in any order, as often as you like, and fill in ratings.csv:
  melody_right   which clip sings the melody more correctly: A, B, same or neither
  words_clear    in which clip can you make out more of the lyric: A, B, same or neither
Leave a cell empty if you are unsure. Please do this before you hear the published demo clips. The key is kept
elsewhere; do not look for it.
"""


def audio(src: Path, dst_stem: Path, kbps: int = DEMO_KBPS) -> str | None:
    """MP3 (ffmpeg) or, without ffmpeg, a WAV copy. Returns the file name, or None if src is missing."""
    if not src.exists():
        return None
    dst_stem.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        dst = dst_stem.with_suffix(".mp3")
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-b:a", f"{kbps}k", str(dst)], check=False)
        if p.returncode == 0:
            return dst.name
    dst = dst_stem.with_suffix(".wav")
    shutil.copy(src, dst)
    return dst.name


def copy_results(paths, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    got = []
    files = [paths.selection, paths.res / "b2_result.json", paths.res / "b2_summary.json", paths.a0_report,
             paths.a0 / "a0_gen.json", paths.a0 / "a0_scores.json", paths.setup / "setup_check.json",
             paths.setup / "setup_model.json"]
    for f in files:
        if f.exists():
            shutil.copy(f, out / f.name)
            got.append(f.name)
    led = out / "ledgers"
    led.mkdir(exist_ok=True)
    for stage in ("dev", "test"):
        for f, name in ((paths.ledger(stage), f"{stage}_generation.jsonl"), (paths.scores(stage), f"{stage}_scores.jsonl")):
            if f.exists():
                shutil.copy(f, led / name)
                got.append(f"ledgers/{name}")
    return got


def earlier_sides(key_path: Path) -> dict[tuple, str]:
    """(item, both run ids) -> the run an earlier key put on A, so packaging again keeps the pack that listeners
    may already have."""
    try:
        pairs = json.loads(key_path.read_text())["pairs"]
        return {(p["item"], frozenset((p["A"]["run_id"], p["B"]["run_id"]))): p["A"]["run_id"] for p in pairs.values()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def blind_pack(paths, demo: list[dict], items: dict, pack_zip: Path, key_path: Path) -> dict:
    """The demo pairs in the order of numpy default_rng(20261029), A and B drawn per pair with secrets (or kept
    from an earlier key of the same pairs)."""
    pairs = [d for d in demo if d.get("comparator_run")]
    pack = pack_zip.with_suffix("")
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    order = np.random.default_rng(BLIND_SEED).permutation(len(pairs))
    before = earlier_sides(key_path)
    key, rows = {}, []
    for k, j in enumerate(order):
        d = pairs[int(j)]
        name = f"pair_{k + 1:02d}"
        clips = [("S*", d["s_star_run"]), (d["comparator"], d["comparator_run"])]
        on_a = before.get((d["item"], frozenset((d["s_star_run"], d["comparator_run"]))))
        flip = on_a == d["comparator_run"] if on_a else bool(secrets.randbelow(2))
        if flip:
            clips.reverse()
        files = {}
        for side, (arm, rid) in zip("AB", clips):
            files[side] = audio(paths.test / f"{rid}.wav", pack / f"{name}_{side}", PACK_KBPS)
        files["melody"] = audio(paths.guides / f"{d['item']}_oohs.wav", pack / f"{name}_melody", PACK_KBPS)
        key[name] = {"item": d["item"], "A": {"arm": clips[0][0], "run_id": clips[0][1]},
                     "B": {"arm": clips[1][0], "run_id": clips[1][1]}, "files": files}
        rows.append([name, items[d["item"]]["lyrics"].split("\n", 1)[-1], "", "", ""])
    with open(pack / "ratings.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "lyric", "melody_right", "words_clear", "notes"])
        w.writerows(rows)
    (pack / "README.txt").write_text(README)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({"order_seed": BLIND_SEED, "sides": "secrets.randbelow(2) per pair", "pairs": key},
                                   indent=1) + "\n")
    zipped = shutil.make_archive(str(pack), "zip", pack)
    shutil.rmtree(pack)
    return {"n_pairs": len(key), "zip": str(zipped), "key": str(key_path)}


def package(paths, pkg: Path, key_path: Path, pack_zip: Path, items: dict) -> dict:
    """Fill pkg/b2 with the results and the demo; the blind pack goes to pack_zip and its key to key_path, both
    outside pkg."""
    for p, what in ((key_path, "key"), (pack_zip, "pack")):
        if p.resolve().is_relative_to(pkg.resolve()):
            raise ValueError(f"the blind-pack {what} must not go inside the package")
    out = pkg / "b2"
    if out.exists():
        shutil.rmtree(out)
    files = copy_results(paths, out)
    res = json.loads((paths.res / "b2_result.json").read_text())["results"] if (paths.res / "b2_result.json").exists() else {}
    demo = res.get("demo") or []
    shown = []
    for d in demo:
        row = dict(d)
        row["s_star_file"] = audio(paths.test / f"{d['s_star_run']}.wav", out / "demo" / f"{d['item']}__S_star__{d['s_star_run']}")
        if d.get("comparator_run"):
            row["comparator_file"] = audio(paths.test / f"{d['comparator_run']}.wav",
                                           out / "demo" / f"{d['item']}__comparator__{d['comparator_run']}")
        shown.append(row)
    if shown:
        (out / "demo").mkdir(parents=True, exist_ok=True)
        (out / "demo" / "demo.json").write_text(json.dumps({"rule": "per item, the S* draw the selector keeps and the "
                                                            "primary comparator's kept draw", "verdict": res.get("verdict"),
                                                            "reel": res.get("reel"), "clips": shown}, indent=1) + "\n")
    blind = blind_pack(paths, demo, items, pack_zip, key_path) if demo else {"n_pairs": 0}
    return {"state": "ok", "files": files, "demo_clips": sum(bool(r.get("s_star_file")) + bool(r.get("comparator_file"))
                                                             for r in shown), "blind_pack": blind,
            "verdict": res.get("verdict")}
