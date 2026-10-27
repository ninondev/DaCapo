"""Build the fixed Stage S melodies and their upload files.

Source records describe registration-time inputs. They are not Suno results.
Audio and MIDI are created only in an ignored or external output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import zipfile

import numpy as np

from dacapo.provenance import config_hash, environment, write_result
from dacapo.render import default_sf2, render_melody
from dacapo.symbolic import Melody, Note

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "melodies_suno.json"
SEED = 20261027
REGISTERED_UTC = "2026-10-27T13:58:24Z"
SAMPLE_RATE = 48000
PROGRAMS = {"piano": 0, "oohs": 53}
KEYS = (("C major", 60), ("D major", 62), ("F major", 65), ("G major", 67))
ROUTES = ("R1", "R2", "R3")
SEED_TAGS = ("e7-suno", "e7-smoke", "e7-data", "e7-dev", "e7-test", "e7-listen")
STYLE = "pop, full band, female vocal"
TABLE_FIELDS = ("item_id", "route", "take", "filename", "bpm", "key", "style",
                "actual_bpm", "actual_key", "actual_style", "route_available",
                "actual_option", "model_version", "date_utc", "credits_used", "notes")
ROUTE_AVAILABLE_VALUES = ("yes", "no", "unknown")

# Every line has eight one-syllable words. Four lines match 32 notes.
LYRICS = (
    ("I watch the sun rise past the hill", "The pale light warms the roof at dawn",
     "My small boat rocks on waves of gold", "We sail back home and sing once more"),
    ("The rain falls soft on leaves of green", "I hear each drop tap on the glass",
     "Your coat hangs near the old front door", "Come sit with me while clouds drift past"),
    ("We walk where tall reeds touch the stream", "The wind bends each thin stem in turn",
     "A white bird lifts its wings and flies", "I hold this day and let it burn"),
    ("The train rolls out past fields of wheat", "Its low hum keeps a slow warm beat",
     "My hand rests on the cold clear pane", "Some day this road leads home once more"),
    ("I plant this seed in soft dark earth", "The spring rain wakes its roots at night",
     "A small green shoot breaks through the ground", "We lift its leaves up to the light"),
    ("The moon hangs low right past the bay", "We count the lights that line the shore",
     "Your voice floats through the cool salt air", "I turn and hear that tune once more"),
    ("The snow lies deep on roof and road", "My boots leave marks near to the gate",
     "A red scarf moves through all the white", "Your lamp is on and I can wait"),
    ("We pass the shops and cross the square", "The bells call out the time at noon",
     "You tap your heel and take my hand", "We hum this song then change the tune"),
    ("A kite pulls hard on its thin string", "The blue sky fills its broad red sail",
     "I run with you right through the field", "We laugh and chase its long white tail"),
    ("The stars shine through the bare black trees", "Their light falls on the path like rain",
     "I hear your steps right by my own", "We find the way back home once more"),
    ("The bread is warm the tea is sweet", "We share the last red plum at dawn",
     "A small wren sings near to the door", "Its song stays close when you are gone"),
    ("The tide draws lines right through the sand", "A shell lies where the waves have been",
     "We leave smooth stones next to the path", "And trust the sea to sing once more"),
)


def json_bytes(value: Any) -> bytes:
    """Canonical source-file bytes. These bytes are the hash input."""
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lyrics_hash(manifest: dict) -> str:
    """Hash the canonical item-id to exact lyric-text mapping."""
    return hashlib.sha256(json_bytes({it["id"]: it["lyrics"] for it in manifest["items"]})).hexdigest()


def run_seeds(tag: str, item_id: str, draw: int) -> tuple[int, int]:
    """Local copy of the registered E5 v2 seed formula until it is available."""
    if tag not in SEED_TAGS or not item_id or not isinstance(draw, int) or draw < 0:
        raise ValueError("Use a registered e7 tag, a nonempty item id, and a nonnegative draw.")
    d = hashlib.sha256(f"{tag}|{item_id}|{draw}".encode()).digest()
    dit_seed = int.from_bytes(d[0:4], "little") & 0x7FFFFFFF
    rng_seed = int.from_bytes(d[4:8], "little") & 0x7FFFFFFF
    if rng_seed == dit_seed:
        rng_seed ^= 1
    return dit_seed, rng_seed


def _phrase(rng: np.random.Generator, center: int, n_tones: int) -> list[int]:
    out = [center]
    for _ in range(6):
        step = int(rng.choice((-2, -1, 0, 1, 2), p=(.05, .38, .14, .38, .05)))
        out.append(int(np.clip(out[-1] + step, max(0, center - 2), min(n_tones - 1, center + 2))))
    out.append(center)
    return out


def build_manifest(seed: int = SEED) -> dict:
    """Create twelve original ABAC melodies. A is repeated without change."""
    rng = np.random.default_rng(seed)
    scale = {0, 2, 4, 5, 7, 9, 11}
    rhythms = ((1.,) * 8, (.5, 1.5, 1., 1., .5, 1.5, 1., 1.),
               (1.5, .5, 1., 1., 1., .5, 1.5, 1.))
    items = []
    for i, lines in enumerate(LYRICS):
        key, tonic = KEYS[i % len(KEYS)]
        tones = [p for p in range(57, 77) if (p - tonic) % 12 in scale]
        center = int(np.argmin([abs(p - tonic - 4) for p in tones]))
        a = _phrase(rng, center, len(tones))
        shift = int(rng.choice((-1, 1)))
        b = [int(np.clip(p + shift, 0, len(tones) - 1)) for p in a]
        c = _phrase(rng, center, len(tones))
        target = tones.index(tonic)
        c[-2] = int(np.clip(target + 1, 0, len(tones) - 1))
        c[-1] = target
        rhythm = rhythms[int(rng.integers(len(rhythms)))]
        pitches = [tones[p] for phrase in (a, b, a, c) for p in phrase]
        notes, start = [], 0.0
        for pitch, duration in zip(pitches, rhythm * 4):
            notes.append({"pitch": pitch, "start": start, "dur": duration, "vel": 90})
            start += duration
        lyrics = "\n".join(lines)
        items.append({"id": f"mel{i:02d}", "key": key, "lyrics": lyrics,
                      "words": lyrics.split(), "phrase_form": "ABAC",
                      "melody": {"bpm": int(rng.integers(90, 111)), "notes": notes}})
    manifest = {"schema_version": 1, "seed": seed, "bars": 8, "meter": [4, 4], "items": items}
    validate_manifest(manifest)
    return manifest


def melody(item: dict) -> Melody:
    return Melody(tuple(Note(**n) for n in item["melody"]["notes"]), item["melody"]["bpm"])


def notes_seconds(item: dict) -> list[tuple[float, float, int]]:
    m = melody(item)
    return [(n.start * m.spb, (n.start + n.dur) * m.spb, n.pitch) for n in m.notes]


def validate_manifest(manifest: dict) -> None:
    if manifest["bars"] != 8 or manifest["meter"] != [4, 4] or len(manifest["items"]) != 12:
        raise ValueError("Stage S needs twelve eight-bar melodies in 4/4.")
    ids, contours = set(), set()
    for item in manifest["items"]:
        m = melody(item)
        words = item["lyrics"].split()
        if not re.fullmatch(r"mel\d{2}", item["id"]) or item["id"] in ids:
            raise ValueError("Melody ids must be unique melNN names.")
        ids.add(item["id"])
        if not 90 <= m.bpm <= 110 or not 24 <= len(m.notes) <= 40:
            raise ValueError("Tempo or note count is outside the registered range.")
        if len(words) != len(m.notes) or words != item["words"]:
            raise ValueError("Each note must have exactly one written word/syllable.")
        tonic = dict(KEYS)[item["key"]]
        if any(not 57 <= n.pitch <= 76 or (n.pitch - tonic) % 12 not in {0, 2, 4, 5, 7, 9, 11} for n in m.notes):
            raise ValueError("A note is out of range or out of key.")
        if m.notes[0].start != 0 or abs(m.length_beats - 32) > 1e-9:
            raise ValueError("The melody must cover exactly 32 beats.")
        if any(n.dur <= 0 or abs(n.start + n.dur - nxt.start) > 1e-9 for n, nxt in zip(m.notes, m.notes[1:])):
            raise ValueError("Notes must be positive, ordered, and contiguous.")
        if sum(abs(x) <= 2 for x in m.intervals()) / len(m.intervals()) < .7:
            raise ValueError("At least 70 percent of note transitions must be steps or repeats.")
        contour = tuple((n.pitch - m.notes[0].pitch, n.dur) for n in m.notes)
        if contour in contours:
            raise ValueError("Two melodies are identical after global transposition.")
        contours.add(contour)


def load_manifest(path: str | Path = MANIFEST) -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest


def _record(config: dict, results: dict, snapshot: dict, registered_utc: str) -> dict:
    return {"config": config, "config_hash": config_hash(config), "environment": snapshot,
            "environment_scope": "registration-time snapshot; not a measurement of this machine",
            "wall_seconds": 0.0, "finished_utc": registered_utc, "results": results}


def build_pack_record(manifest: dict, soundfont: str | Path, snapshot: dict,
                      registered_utc: str = REGISTERED_UTC) -> dict:
    cfg = {"experiment": "e7_suno_pack_registration", "seed": manifest["seed"],
           "generator_sha256": sha256_file(__file__), "asr_model": None,
           "asr_revision": None, "f0_tool": None, "separator": None}
    results = {"melodies_sha256": hashlib.sha256(json_bytes(manifest)).hexdigest(),
               "lyrics_sha256": lyrics_hash(manifest),
               "lyrics_by_item_sha256": {it["id"]: hashlib.sha256(it["lyrics"].encode()).hexdigest() for it in manifest["items"]},
               "lyrics_provenance": "Lyrics written by the agents for this experiment.",
               "render": {"programs": PROGRAMS, "sample_rate": SAMPLE_RATE, "reverb": False,
                          "soundfont": "FluidR3_GM.sf2", "soundfont_sha256": sha256_file(soundfont)}}
    return _record(cfg, results, snapshot, registered_utc)


def build_license_record(snapshot: dict, registered_utc: str = REGISTERED_UTC) -> dict:
    """List only Stage S resources. Later datasets require a new review first."""
    resources = [
        {"name": "FluidR3_GM", "kind": "soundfont", "license": "MIT",
         "url": "https://sources.debian.org/src/fluid-soundfont/3.1-5.3/debian/copyright/",
         "verification": "Installed Debian copyright file states MIT for the soundfont.",
         "use": "Render original test melodies; no soundfont is committed."},
        {"name": "FluidSynth", "kind": "renderer", "license": "LGPL-2.1-or-later",
         "url": "https://github.com/FluidSynth/fluidsynth/blob/master/LICENSE",
         "verification": "Installed Debian copyright file.", "use": "Local synthesis."},
        {"name": "SwiftF0", "kind": "pitch model", "version": "0.3.0", "license": "MIT",
         "url": "https://github.com/lars76/swift-f0", "verification": "Installed package metadata.", "use": "Measurement."},
        {"name": "FCPE", "kind": "pitch model", "version": "0.0.4", "license": "MIT",
         "url": "https://github.com/CNChTu/FCPE", "verification": "Installed torchfcpe metadata.", "use": "Measurement."},
        {"name": "torchcrepe", "kind": "pitch model", "version": "0.0.24", "license": "MIT",
         "url": "https://github.com/maxrmorrison/torchcrepe", "verification": "Installed package metadata.", "use": "Measurement."},
        {"name": "Basic Pitch", "kind": "transcription model", "version": "0.4.0", "license": "Apache-2.0",
         "url": "https://github.com/spotify/basic-pitch/blob/main/LICENSE",
         "verification": "Installed distribution LICENSE file.", "use": "Unvalidated diagnostic note-F1 only."},
        {"name": "ACE-Step 1.5", "kind": "song model", "license": "MIT",
         "url": "https://huggingface.co/ACE-Step/Ace-Step1.5",
         "revision": "19671f406d603126926c1b7e2adc169acbcade22",
         "verification": "Declared in the task brief; verify model card in the owner-run setup before use.",
         "use": "Untrained baseline generation in owner-run GPU stage."},
        {"name": "Whisper large-v3", "kind": "ASR model", "license": "Apache-2.0",
         "url": "https://huggingface.co/openai/whisper-large-v3",
         "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
         "verification": "Model-card declaration must be checked by owner-run setup before use.",
         "use": "Measurement only; no weights were downloaded in Cloud."},
        {"name": "htdemucs", "kind": "separator model", "package_version": "demucs==4.1.0", "license": "MIT",
         "url": "https://github.com/facebookresearch/demucs/blob/main/LICENSE",
         "verification": "Upstream license must be checked by owner-run setup before use.",
         "use": "Measurement only; no weights were downloaded in Cloud."},
        {"name": "Suno user outputs", "kind": "external service audio", "license": "Terms-dependent; not cleared for publication",
         "url": "https://suno.com/terms", "verification": "Owner must check the current terms before generation and publication.",
         "use": "Private measurement only; never training; no audio or scores are committed."},
    ]
    cfg = {"experiment": "e7_stage_s_license_registration", "generator_sha256": sha256_file(__file__),
           "asr_model": None, "asr_revision": None, "f0_tool": None, "separator": None}
    return _record(cfg, {"resources": resources, "external_datasets_used": [],
                         "policy": "Research-only licenses permit measurement, never training or repository inclusion."}, snapshot, registered_utc)


def write_source_records(*, refresh_snapshot: bool = False, registered_utc: str | None = None) -> None:
    """Regenerate sources; refresh provenance only for an explicit new registration."""
    manifest = build_manifest()
    record_path = HERE / "records" / "suno_pack.json"
    old = json.loads(record_path.read_text()) if record_path.exists() else None
    snapshot = old["environment"] if old and not refresh_snapshot else environment()
    registered = registered_utc or (old["finished_utc"] if old and not refresh_snapshot
                                    else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if snapshot.get("git_commit") in (None, "unknown"):
        raise RuntimeError("Run source generation from the repository root.")
    MANIFEST.write_bytes(json_bytes(manifest))
    record_path.parent.mkdir(exist_ok=True)
    record_path.write_bytes(json_bytes(build_pack_record(manifest, default_sf2(), snapshot, registered)))
    (HERE / "records" / "licenses.json").write_bytes(json_bytes(build_license_record(snapshot, registered)))


def _output_directory(path: str | Path) -> Path:
    out = Path(path).resolve()
    if out.is_relative_to(REPO) and not out.is_relative_to(HERE / "outputs"):
        raise ValueError("Outputs inside the checkout must be under e7_melody_adapter/outputs.")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_owner_table(path: Path, manifest: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        for item in manifest["items"]:
            for route in ROUTES:
                for take in ("a", "b"):
                    writer.writerow({"item_id": item["id"], "route": route, "take": take,
                                     "filename": f"S_{item['id']}_{route}_{take}.mp3",
                                     "bpm": item["melody"]["bpm"], "key": item["key"],
                                     "style": STYLE, "route_available": "unknown"})


def build_upload_pack(output_dir: str | Path, manifest_path: str | Path = MANIFEST,
                      *, require_clean: bool = True) -> dict:
    """Render the 48 upload assets and owner table, then write hashes and a zip.

    ``require_clean=False`` is only for local CPU validation. The public CLI
    never disables the clean-clone check.
    """
    if Path.cwd().resolve() != REPO:
        raise RuntimeError("Change directory to the DaCapo clone root before packaging.")
    snapshot = environment()
    if snapshot["git_commit"] == "unknown" or (require_clean and snapshot["git_dirty"] is not False):
        raise RuntimeError("Upload packaging requires a known commit and a clean clone.")
    manifest = load_manifest(manifest_path)
    reference = json.loads((HERE / "records" / "suno_pack.json").read_text())
    actual = build_pack_record(manifest, default_sf2(), reference["environment"], reference["finished_utc"])
    if reference != actual:
        raise RuntimeError("Source hashes or rendering inputs differ from the registered upload pack.")
    out = _output_directory(output_dir)
    started = time.time()
    assets = out / "upload"
    assets.mkdir()
    for item in manifest["items"]:
        m = melody(item)
        for name, program in PROGRAMS.items():
            render_melody(m, assets / f"{item['id']}_{name}.wav", program=program, sr=SAMPLE_RATE)
        m.to_midi(assets / f"{item['id']}.mid")
        (assets / f"{item['id']}_lyrics.txt").write_text(item["lyrics"] + "\n")
    _write_owner_table(assets / "suno_recording_table.csv", manifest)
    hashes = {f.relative_to(assets).as_posix(): sha256_file(f) for f in sorted(assets.iterdir())}
    cfg = {"experiment": "e7_suno_upload", "generator_sha256": sha256_file(__file__),
           "manifest_sha256": actual["results"]["melodies_sha256"], "lyrics_sha256": lyrics_hash(manifest),
           "render": actual["results"]["render"], "asr_model": None, "asr_revision": None,
           "f0_tool": None, "separator": None, "clean_clone_required": require_clean}
    hashes_path = write_result(out / "suno_upload_hashes.json", cfg, {"files": hashes}, started)
    payload = json.loads(hashes_path.read_text())
    if payload["environment"]["git_commit"] == "unknown" or (require_clean and payload["environment"]["git_dirty"] is not False):
        raise RuntimeError("Repository provenance changed during packaging.")
    archive_path = out / "e7_suno_upload.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (assets / name).read_bytes())
    return {"zip": str(archive_path), "hashes": str(hashes_path), "asset_count": len(hashes),
            "source_commit": payload["environment"]["git_commit"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    records = sub.add_parser("records", help="Regenerate deterministic source records.")
    records.add_argument("--refresh-snapshot", action="store_true",
                         help="Record the current environment and UTC time for a new registration.")
    render = sub.add_parser("render", help="Render the upload pack from a clean clone.")
    render.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.stage == "records":
        write_source_records(refresh_snapshot=args.refresh_snapshot)
    else:
        print(json.dumps(build_upload_pack(args.output), indent=2))


if __name__ == "__main__":
    main()
