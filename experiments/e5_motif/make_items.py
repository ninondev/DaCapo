"""E5 stimuli: in-context motif tests for a song DiT, with exact symbolic answers.

Source layout (120 BPM, 1 unit = 2 bars = 4 s = 100 latent frames):
    [U1][U2][U3][  gap 12-16 s, silent  ][D (1 bar)]   -> 18 s, repainting_start/end = 12/16
Conditions (U1 U2 U3):
    lit      A  B  A          rule answer: B            (A B A -> B)
    ctl      A  B  C          lag control: no rule; B would be pure 8-s lag copying
    rel+2/+5/-3  A  B  T_k(A)  rule answer: T_k(B)      (transposition-relational induction)
    seq      A  T2(A)  T4(A)  rule answer: T6(A)        (sequence extrapolation)
    ana      A  T5(A)  B      rule answer: T5(B)        (analogy A : A' :: B : ?)
    null     -  -  -          no context (chance for every hypothesis)
Every hypothesis is an exact symbolic object (rhythm + pitches), also rendered for calibration.
B~ is a derangement of B's pitches (same rhythm and pitch multiset, >= 75% positions differ).
v1 (manifest.json, generator seed 101): melodies 0-1 are pilot items, 2-13 confirmatory.
v2 (manifest_v2.json, generator seed 303): 36 fresh confirmatory melodies v2c00-v2c35. Their A, B, C and B2
units keep every note inside the scored gap window (unit_in_window). The first v2 draw (seed 202, no window
constraint) failed K0 and is kept in k0_first_run/.

  python make_items.py                       v1: render sources and renders, write manifest.json (as registered)
  python make_items.py build-v2              once, locally: write manifest_v2.json with render sha256
  python make_items.py build-p0b             once, locally: write manifest_p0b.json (v1 items, both lengths)
  python make_items.py render --manifest M [--ids a,b] [--hyp-programs 73,54,71] [--keep-hyps]
                                             render from a committed manifest and check every sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from dacapo.render import render
from dacapo.symbolic import Melody, Note, random_melody

HERE = Path(__file__).parent
BPM, UNIT, GAP_START, GAP_END = 120.0, 8.0, 12.0, 16.0  # UNIT in beats (2 bars); gap in seconds
FLUTE = 73
CONDITIONS = ["lit", "ctl", "rel+2", "rel+5", "rel-3", "seq", "ana", "null"]
T18_SAMPLES, T18_FADE = 864000, 2400
V2 = {"n": 36, "seed": 303, "prefix": "v2c", "manifest": "manifest_v2.json", "sources": "sources_v2",
      "sources_t18": "sources_v2_t18", "hyp_dir": "hyp_renders_v2", "programs": (73, 54, 71)}


def unit(rng: np.random.Generator, tonic: int) -> Melody:
    while True:
        m = random_melody(rng, bars=2, bpm=BPM, tonic=tonic)
        if 6 <= len(m.notes) <= 12 and len(set(m.pitches())) >= 4:
            return m


# v2 units: with a unit placed at 12.0 s, a note is scored only if its onset is >= 12.4 s and it ends by
# 15.85 s (score_analogy.notes_in_gap). Keeping notes with onset >= beat 1.0 and end <= beat 7.5 puts every
# note at 12.5-15.75 s, so a wrong-note draw can only change a scored note.
WINDOW_FIRST_BEAT, WINDOW_LAST_END = 1.0, 7.5


def unit_in_window(rng: np.random.Generator, tonic: int) -> Melody:
    while True:
        m = random_melody(rng, bars=2, bpm=BPM, tonic=tonic)
        m = Melody(tuple(n for n in m.notes if n.start >= WINDOW_FIRST_BEAT and n.start + n.dur <= WINDOW_LAST_END), BPM)
        if 6 <= len(m.notes) <= 12 and len(set(m.pitches())) >= 4:
            return m


def derange(m: Melody, rng: np.random.Generator) -> Melody:
    p = m.pitches()
    for _ in range(2000):
        q = list(rng.permutation(p))
        if np.mean([a != b for a, b in zip(p, q)]) >= 0.75:
            return Melody(tuple(replace(n, pitch=int(x)) for n, x in zip(m.notes, q)), m.bpm)
    raise RuntimeError("no derangement")


def differ(a: Melody, b: Melody) -> bool:
    """Pitch sequences differ at >= 75% of aligned positions (or rhythms differ)."""
    pa, pb = a.pitches(), b.pitches()
    if len(pa) != len(pb) or [n.start for n in a.notes] != [n.start for n in b.notes]:
        return True
    return np.mean([x != y for x, y in zip(pa, pb)]) >= 0.75


def at(m: Melody, beats: float) -> Melody:
    return m.shift(beats)


def build(rng: np.random.Generator, idx: int, unit_fn=unit) -> dict:
    """unit_fn draws A, B, C and B2; D (the bar after the gap) always comes from unit()."""
    tonic = int(rng.choice([67, 69, 72]))
    while True:
        A, B, C, B2 = (unit_fn(rng, tonic) for _ in range(4))
        D = Melody(tuple(n for n in unit(rng, tonic).notes if n.start < 4.0), BPM)  # 1 bar
        Bt = derange(B, rng)
        hyps = {"B": B, "B~": Bt, "A": A, "C": C, "D": D}
        for k in (2, 5, -3, 4, 6, 7, 1, 3, -2, -4):
            hyps[f"T{k:+d}B"], hyps[f"T{k:+d}B~"], hyps[f"T{k:+d}A"] = B.transpose(k), Bt.transpose(k), A.transpose(k)
        core = [B, Bt, A, C, D, B.transpose(2), B.transpose(5), B.transpose(-3), A.transpose(6), A.transpose(4)]
        if all(differ(x, y) for i, x in enumerate(core) for y in core[i + 1:]):
            break
    contexts = {"lit": [A, B, A], "ctl": [A, B, C], "rel+2": [A, B, A.transpose(2)], "rel+5": [A, B, A.transpose(5)],
                "rel-3": [A, B, A.transpose(-3)], "seq": [A, A.transpose(2), A.transpose(4)],
                "ana": [A, A.transpose(5), B], "null": []}
    return {"A": A, "B": B, "C": C, "D": D, "B2": B2, "hyps": hyps, "contexts": contexts, "tonic": tonic}


def source(ctx: list[Melody], D: Melody) -> Melody:
    notes = []
    for i, u in enumerate(ctx):
        notes += list(at(u, i * UNIT).notes)
    notes += list(at(D, 4 * UNIT).notes)
    return Melody(tuple(notes), BPM)


def hyp_full(m: Melody, D: Melody) -> Melody:
    """A hypothesis in the gap of the 'null' layout, followed by D: what the ruler gate scores."""
    return Melody(tuple(at(m, 3 * UNIT).notes) + tuple(at(D, 4 * UNIT).notes), BPM)


def hyp_file(mid: str, h: str) -> str:
    return f"{mid}_{h.replace('~', 'x')}.wav"


def main(n: int = 14, seed: int = 101):
    """v1 items as registered: same RNG order, files and manifest bytes. The committed manifest.json is
    never replaced by different content."""
    rng = np.random.default_rng(seed)
    (HERE / "sources").mkdir(exist_ok=True)
    (HERE / "hyp_renders").mkdir(exist_ok=True)
    items = []
    for i in range(n):
        b = build(rng, i)
        mid = f"mel{i:02d}"
        conds = {}
        for c in CONDITIONS:
            src = source(b["contexts"][c], b["D"])
            wav = HERE / "sources" / f"{mid}_{c}.wav"
            render([src.to_instrument(FLUTE)], BPM, wav, sr=48000)
            conds[c] = f"sources/{wav.name}"
        # Calibration renders: each hypothesis sits in the gap of the 'null' layout.
        hyp_wavs = {}
        for h, m in b["hyps"].items():
            wav = HERE / "hyp_renders" / hyp_file(mid, h)
            render([hyp_full(m, b["D"]).to_instrument(FLUTE)], BPM, wav, sr=48000)
            hyp_wavs[h] = f"hyp_renders/{wav.name}"
        items.append({"id": mid, "split": "pilot" if i < 2 else "confirm", "tonic": b["tonic"], "sources": conds,
                      "hyp_renders": hyp_wavs,
                      "hyps": {h: [asdict(x) for x in m.notes] for h, m in b["hyps"].items()},
                      "B2": [asdict(x) for x in b["B2"].notes]})
    meta = {"seed": seed, "bpm": BPM, "unit_beats": UNIT, "gap_s": [GAP_START, GAP_END], "duration_s": 18.0,
            "lead_program": FLUTE, "conditions": CONDITIONS, "items": items}
    text = json.dumps(meta, indent=1)
    p = HERE / "manifest.json"
    if p.exists() and p.read_text() != text:
        raise RuntimeError("manifest.json exists with other content; it is the registered v1 manifest")
    p.write_text(text)
    print(n, "melodies x", len(CONDITIONS), "conditions")


# v2: committed manifests, rendering from them, and the 18 s length arm.

def sha256(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audio_sha256(p: str | Path) -> str:
    from dacapo.motif_integrity import audio_sha256 as f
    return f(p)


def melody(notes: list[dict]) -> Melody:
    return Melody(tuple(Note(**n) for n in notes), BPM)


def contexts_from_hyps(hyps: dict) -> tuple[dict, Melody]:
    """The eight contexts exactly as build() makes them, rebuilt from a manifest's note lists."""
    m = {h: melody(hyps[h]) for h in ("A", "B", "C", "D", "T+2A", "T+4A", "T+5A", "T-3A")}
    A, B = m["A"], m["B"]
    ctx = {"lit": [A, B, A], "ctl": [A, B, m["C"]], "rel+2": [A, B, m["T+2A"]], "rel+5": [A, B, m["T+5A"]],
           "rel-3": [A, B, m["T-3A"]], "seq": [A, m["T+2A"], m["T+4A"]], "ana": [A, m["T+5A"], B], "null": []}
    return ctx, m["D"]


def trim_t18(src: Path, dst: Path) -> dict:
    """Cut a rendered source to 864000 samples (18.000 s, 450 latent frames) with a 50 ms half-cosine
    fade-out, written as FLOAT WAV. The cut ends inside D's last note or its release; D is unchanged."""
    import soundfile as sf

    x, sr = sf.read(str(src), always_2d=True, dtype="float32")
    if sr != 48000 or len(x) <= T18_SAMPLES:
        raise ValueError(f"{src}: expected a 48 kHz source longer than {T18_SAMPLES} samples")
    y = x[:T18_SAMPLES].copy()
    y[-T18_FADE:] *= (0.5 * (1 + np.cos(np.linspace(0, np.pi, T18_FADE))))[:, None]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, subtype="FLOAT")
    return {"samples_in": len(x), "peak_in_fade_region": float(np.abs(x[T18_SAMPLES - T18_FADE:T18_SAMPLES]).max()),
            "peak_removed": float(np.abs(x[T18_SAMPLES:]).max())}


def disjoint_from_v1(items: list[dict], v1: dict) -> None:
    """No v2 unit (A, B, C, D, B2) has the (start, dur, pitch) sequence of any v1 unit."""
    key = lambda notes: tuple((n["start"], n["dur"], n["pitch"]) for n in notes)
    old = {key(it["hyps"][u]) for it in v1["items"] for u in ("A", "B", "C", "D")} | {key(it["B2"]) for it in v1["items"]}
    for it in items:
        for u in ("A", "B", "C", "D"):
            if key(it["hyps"][u]) in old:
                raise RuntimeError(f"{it['id']} unit {u} repeats a v1 unit")
        if key(it["B2"]) in old:
            raise RuntimeError(f"{it['id']} unit B2 repeats a v1 unit")


def v2_items(n: int = V2["n"], seed: int = V2["seed"], prefix: str = V2["prefix"]) -> list[dict]:
    """Symbolic v2 items: build() in the same RNG order as v1, with other seed and ids and window-bound units."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        b = build(rng, i, unit_fn=unit_in_window)
        mid = f"{prefix}{i:02d}"
        items.append({"id": mid, "split": "confirm", "tonic": b["tonic"],
                      "sources": {"rend": {c: f"{V2['sources']}/{mid}_{c}.wav" for c in CONDITIONS},
                                  "t18": {c: f"{V2['sources_t18']}/{mid}_{c}.wav" for c in CONDITIONS}},
                      **{hyp_key(p): {h: f"{hyp_dir(p)}/{hyp_file(mid, h)}" for h in b["hyps"]} for p in V2["programs"]},
                      "hyps": {h: [asdict(x) for x in m.notes] for h, m in b["hyps"].items()},
                      "B2": [asdict(x) for x in b["B2"].notes]})
    return items


def hyp_key(program: int) -> str:
    return "hyp_renders" if program == FLUTE else f"hyp_renders_p{program}"


def hyp_dir(program: int) -> str:
    return V2["hyp_dir"] if program == FLUTE else f"{V2['hyp_dir']}_p{program}"


def render_sources(it: dict, root: Path, arms=("rend", "t18")) -> dict:
    """Render one item's eight rend sources and cut the t18 arm. Returns {arm: {cond: sha256}}, the
    rend length and what the cut removed."""
    ctx, D = contexts_from_hyps(it["hyps"])
    shas, lengths, cut = {"rend": {}, "t18": {}}, set(), {}
    import soundfile as sf
    for c in CONDITIONS:
        rend = root / it["sources"]["rend"][c]
        rend.parent.mkdir(parents=True, exist_ok=True)
        render([source(ctx[c], D).to_instrument(FLUTE)], BPM, rend, sr=48000)
        shas["rend"][c] = audio_sha256(rend)
        lengths.add(sf.info(str(rend)).frames)
        if "t18" in arms:
            cut[c] = trim_t18(rend, root / it["sources"]["t18"][c])
            shas["t18"][c] = audio_sha256(root / it["sources"]["t18"][c])
    if len(lengths) != 1:
        raise RuntimeError(f"{it['id']}: conditions differ in length {sorted(lengths)}")
    if "rend" not in arms:
        for c in CONDITIONS:
            (root / it["sources"]["rend"][c]).unlink()
        shas.pop("rend")
    return {"sha256": shas, "rend_samples": lengths.pop(), "cut": cut}


def render_hyps(it: dict, root: Path, program: int) -> dict:
    D = melody(it["hyps"]["D"])
    out = {}
    for h, rel in it[hyp_key(program)].items():
        wav = root / rel
        wav.parent.mkdir(parents=True, exist_ok=True)
        render([hyp_full(melody(it["hyps"][h]), D).to_instrument(program)], BPM, wav, sr=48000)
        out[h] = audio_sha256(wav)
    return out


def _cut_summary(it: dict, cut: dict) -> dict:
    d_end = max(n["start"] + n["dur"] for n in it["hyps"]["D"])
    return {"D_end_beat": d_end, "max_peak_in_fade_region": max(v["peak_in_fade_region"] for v in cut.values()),
            "max_peak_removed": max(v["peak_removed"] for v in cut.values())}


def _build_v2_item(args) -> dict:
    it, tmp = args
    import shutil
    import soundfile as sf

    r = render_sources(it, tmp)
    it["source_samples"] = {"rend": r["rend_samples"], "t18": T18_SAMPLES}
    it["source_sha256"] = r["sha256"]
    it["t18_cut"] = _cut_summary(it, r["cut"])
    it["hyp_render_sha256"] = {str(p): render_hyps(it, tmp, p) for p in V2["programs"]}
    assert all(sf.info(str(tmp / p)).frames == T18_SAMPLES for p in it["sources"]["t18"].values())
    shutil.rmtree(tmp)
    print(it["id"], it["source_samples"]["rend"], flush=True)
    return it


def build_v2(root: Path = HERE, workers: int = 3) -> Path:
    """Write manifest_v2.json once: symbolic items plus the sha256 of every source and hypothesis render,
    rendered here and deleted again item by item (the renders are re-made from the manifest)."""
    from multiprocessing import Pool

    out = root / V2["manifest"]
    if out.exists():
        raise RuntimeError(f"{out} exists; the v2 manifest is written once")
    v1 = json.loads((HERE / "manifest.json").read_text())
    items = v2_items()
    disjoint_from_v1(items, v1)
    with Pool(workers) as pool:
        items = pool.map(_build_v2_item, [(it, root / f"_build_v2_{it['id']}") for it in items], chunksize=1)
    meta = {"seed": V2["seed"], "bpm": BPM, "unit_beats": UNIT, "gap_s": [GAP_START, GAP_END], "duration_s": 18.0,
            "lead_program": FLUTE, "programs": {"flute": 73, "synth_voice": 54, "clarinet": 71},
            "conditions": CONDITIONS, "length_arms": _length_arms(), "items": items}
    out.write_text(json.dumps(meta, indent=1))
    return out


def _length_arms() -> dict:
    return {"rend": "as rendered by FluidSynth", "t18": {"samples": T18_SAMPLES, "fade_samples": T18_FADE,
                                                        "fade": "half-cosine", "subtype": "FLOAT"}}


def build_p0b(root: Path = HERE) -> Path:
    """Write manifest_p0b.json once: the v1 items mel00-mel13 (symbolic content unchanged) with both
    source lengths and the sha256 of every source, rendered here and deleted again."""
    import shutil

    out = root / "manifest_p0b.json"
    if out.exists():
        raise RuntimeError(f"{out} exists; the Phase 0b manifest is written once")
    man = json.loads((HERE / "manifest.json").read_text())
    tmp = root / "_build_p0b"
    for it in man["items"]:
        it["sources"] = {"rend": dict(it["sources"]), "t18": {c: f"sources_t18/{Path(p).name}" for c, p in it["sources"].items()}}
        r = render_sources(it, tmp)
        it["source_samples"] = {"rend": r["rend_samples"], "t18": T18_SAMPLES}
        it["source_sha256"] = r["sha256"]
        it["t18_cut"] = _cut_summary(it, r["cut"])
        shutil.rmtree(tmp)
        print(it["id"], it["source_samples"]["rend"], flush=True)
    man["derived_from"] = {"manifest.json": sha256(HERE / "manifest.json")}
    man["length_arms"] = _length_arms()
    out.write_text(json.dumps(man, indent=1))
    return out


def render_from_manifest(manifest: Path, root: Path, ids: list[str] | None = None, hyp_programs=(),
                         keep_hyps: bool = True, expect_sha256: str | None = None) -> dict:
    """Render the sources (both lengths) and the requested hypothesis renders of a committed manifest and
    compare every file with the sha256 recorded in it. Existing files with the recorded sha256 are kept.
    Never writes a manifest. Returns {"ok", "mismatch": [...], "n_files"}."""
    if expect_sha256 is not None and sha256(manifest) != expect_sha256:
        raise RuntimeError(f"{manifest} differs from the registered sha256")
    man = json.loads(Path(manifest).read_text())
    bad, n = [], 0
    for it in man["items"]:
        if ids and it["id"] not in ids:
            continue
        rec = it["source_sha256"]
        have = all((root / it["sources"][a][c]).exists() and audio_sha256(root / it["sources"][a][c]) == rec[a][c]
                   for a in rec for c in rec[a])
        if not have:
            render_sources(it, root)
        for a in rec:
            for c, want in rec[a].items():
                n += 1
                if audio_sha256(root / it["sources"][a][c]) != want:
                    bad.append(it["sources"][a][c])
        for p in hyp_programs:
            got = render_hyps(it, root, p)
            for h, want in it["hyp_render_sha256"][str(p)].items():
                n += 1
                if got[h] != want:
                    bad.append(it[hyp_key(p)][h])
            if not keep_hyps:
                for rel in it[hyp_key(p)].values():
                    (root / rel).unlink(missing_ok=True)
    return {"ok": not bad, "mismatch": bad, "n_files": n}


def cli(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("build-v2")
    sub.add_parser("build-p0b")
    r = sub.add_parser("render")
    r.add_argument("--manifest", required=True)
    r.add_argument("--ids", default="")
    r.add_argument("--hyp-programs", default="")
    r.add_argument("--no-keep-hyps", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd is None:
        main()
        return 0
    if a.cmd == "build-v2":
        print(build_v2())
        return 0
    if a.cmd == "build-p0b":
        print(build_p0b())
        return 0
    root = Path(os.environ.get("E5_DIR", HERE))           # the folder the rest of E5 v2 reads (v2_config.e5_dir)
    res = render_from_manifest(root / a.manifest, root, [x for x in a.ids.split(",") if x],
                               [int(p) for p in a.hyp_programs.split(",") if p], keep_hyps=not a.no_keep_hyps)
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(cli())
