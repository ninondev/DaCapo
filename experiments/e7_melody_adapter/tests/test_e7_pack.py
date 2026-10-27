"""Check the registered source package and its real rendered upload assets."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from dacapo.render import default_sf2
from experiments.e7_melody_adapter import pack, stage_s_lib as lib


def test_source_files_regenerate_byte_for_byte():
    """Retain the recorded environment; a new run must not rewrite history."""
    manifest = pack.build_manifest()
    assert pack.json_bytes(manifest) == pack.MANIFEST.read_bytes()
    record_path = pack.HERE / "records/suno_pack.json"
    record = json.loads(record_path.read_text())
    rebuilt = pack.build_pack_record(
        manifest, default_sf2(), record["environment"], record["finished_utc"],
    )
    assert pack.json_bytes(rebuilt) == record_path.read_bytes()
    license_path = pack.HERE / "records/licenses.json"
    licenses = json.loads(license_path.read_text())
    rebuilt_licenses = pack.build_license_record(
        licenses["environment"], licenses["finished_utc"],
    )
    assert pack.json_bytes(rebuilt_licenses) == license_path.read_bytes()
    assert record["results"]["melodies_sha256"] == hashlib.sha256(pack.MANIFEST.read_bytes()).hexdigest()
    assert record["environment"]["git_commit"] != "unknown"
    assert record["environment"]["git_dirty"] is True
    assert not licenses["results"]["external_datasets_used"]


def test_melodies_have_fixed_form_and_one_syllable_per_note():
    manifest = pack.load_manifest()
    assert len(manifest["items"]) == 12
    assert {item["key"] for item in manifest["items"]} == {"C major", "D major", "F major", "G major"}
    for item in manifest["items"]:
        melody = pack.melody(item)
        assert len(melody.notes) == len(item["lyrics"].split()) == 32
        assert len(item["lyrics"].splitlines()) == 4
        assert all(len(line.split()) == 8 for line in item["lyrics"].splitlines())
        assert melody.length_beats == 32
        assert 90 <= melody.bpm <= 110
        assert all(57 <= note.pitch <= 76 for note in melody.notes)
        # A repeats in bars five and six, with the same rhythm and pitches.
        assert [(note.pitch, note.dur) for note in melody.notes[:8]] == [
            (note.pitch, note.dur) for note in melody.notes[16:24]
        ]
        assert sum(abs(interval) <= 2 for interval in melody.intervals()) >= 0.7 * 31


def test_source_lyrics_avoid_numbers_and_registered_spelling_variants():
    ambiguous = {"zero", "one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten", "eleven", "twelve", "gray", "grey", "till", "til"}
    manifest = pack.build_manifest()
    for item in manifest["items"]:
        words = item["lyrics"].lower().split()
        assert len(words) == len(item["melody"]["notes"]) == 32
        assert not ambiguous.intersection(words)
        assert not any(character.isdigit() for character in item["lyrics"])


# Closed compounds that a transcriber usually writes as one word.
JOINED_FORMS = {
    "sunrise", "sunset", "sunlight", "sunshine", "someday", "sometime", "sometimes", "someone",
    "somewhere", "anymore", "anyone", "everyday", "everyone", "maybe", "into", "onto", "upon",
    "within", "without", "outside", "inside", "today", "tonight", "cannot", "moonlight",
    "starlight", "daylight", "footsteps", "rainfall", "rainfalls", "raindrop", "snowfall", "seashell",
    "blackbird", "nearby", "goodbye", "alright", "homeward", "sailboat", "farewell", "overhead",
}


def test_source_lyrics_avoid_split_compounds_that_asr_would_join():
    assert lib.word_recall("I watch the sun rise", "I watch the sunrise") < 1.
    manifest = pack.build_manifest()
    for item in manifest["items"]:
        words = [word.lower() for word in item["lyrics"].split()]
        assert len(words) == len(item["melody"]["notes"]) == 32
        joined = {a + b for a, b in zip(words, words[1:])}
        assert not JOINED_FORMS.intersection(joined), item["id"]
        assert not JOINED_FORMS.intersection(words), item["id"]


def test_recording_table_preserves_reference_settings_and_actual_fields(tmp_path):
    manifest = pack.build_manifest()
    target = tmp_path / "recording.csv"
    pack._write_owner_table(target, manifest)
    with target.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == pack.TABLE_FIELDS
        rows = list(reader)
    lookup = {item["id"]: item for item in manifest["items"]}
    assert len(rows) == 72
    for row in rows:
        item = lookup[row["item_id"]]
        assert float(row["bpm"]) == item["melody"]["bpm"]
        assert row["key"] == item["key"]
        assert row["style"] == pack.STYLE
        assert row["route_available"] == "unknown"
        assert row["actual_bpm"] == row["actual_key"] == row["actual_style"] == ""
    assert pack.ROUTE_AVAILABLE_VALUES == ("yes", "no", "unknown")


@pytest.mark.parametrize("fault", ["duplicate_id", "out_of_range", "missing_word", "wrong_meter"])
def test_manifest_validation_rejects_invalid_inputs(fault):
    manifest = copy.deepcopy(pack.load_manifest())
    if fault == "duplicate_id":
        manifest["items"][1]["id"] = manifest["items"][0]["id"]
    elif fault == "out_of_range":
        manifest["items"][0]["melody"]["notes"][0]["pitch"] = 80
    elif fault == "missing_word":
        manifest["items"][0]["lyrics"] = "one word"
    else:
        manifest["meter"] = [3, 4]
    with pytest.raises(ValueError):
        pack.validate_manifest(manifest)


def test_seed_domains_are_deterministic_and_disjoint():
    seen = set()
    for tag in pack.SEED_TAGS:
        for item in pack.load_manifest()["items"]:
            for draw in range(4):
                raw = hashlib.sha256(f"{tag}|{item['id']}|{draw}".encode()).digest()
                dit = int.from_bytes(raw[:4], "little") & 0x7FFFFFFF
                rng = int.from_bytes(raw[4:8], "little") & 0x7FFFFFFF
                if dit == rng:
                    rng ^= 1
                actual = pack.run_seeds(tag, item["id"], draw)
                assert actual == (dit, rng)
                assert dit != rng
                assert not seen.intersection(actual)
                seen.update(actual)
                # This becomes active automatically when E5 v2 is present.
                if importlib.util.find_spec("dacapo.seeds") is not None:
                    from dacapo.seeds import run_seeds
                    assert run_seeds(tag, item["id"], draw) == actual


def test_real_upload_package_contains_every_asset_and_hash(tmp_path, monkeypatch):
    """This is a CPU render test. Missing synthesis tools are failures."""
    assert shutil.which("fluidsynth"), "FluidSynth is required, not an optional test dependency."
    assert default_sf2().is_file()
    monkeypatch.chdir(pack.REPO)
    result = pack.build_upload_pack(tmp_path / "upload_result", require_clean=False)
    manifest = pack.load_manifest()
    expected = {"suno_recording_table.csv"}
    for item in manifest["items"]:
        expected.update({f"{item['id']}.mid", f"{item['id']}_lyrics.txt",
                         f"{item['id']}_piano.wav", f"{item['id']}_oohs.wav"})
    assert result["asset_count"] == 49
    record = json.loads(Path(result["hashes"]).read_text())
    assert set(record["results"]["files"]) == expected
    assert record["environment"]["git_commit"] != "unknown"
    assets = Path(result["zip"]).parent / "upload"
    with zipfile.ZipFile(result["zip"]) as archive:
        assert set(archive.namelist()) == expected
        for name, digest in record["results"]["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert archive.read(name) == (assets / name).read_bytes()
    for item in manifest["items"]:
        reference = pack.melody(item)
        midi = pretty_midi.PrettyMIDI(str(assets / f"{item['id']}.mid"))
        notes = midi.instruments[0].notes
        assert [note.pitch for note in notes] == reference.pitches()
        assert len(notes) == 32
        assert midi.get_tempo_changes()[1][0] == pytest.approx(reference.bpm, abs=0.001)
        assert (assets / f"{item['id']}_lyrics.txt").read_text() == item["lyrics"] + "\n"
        for instrument in ("piano", "oohs"):
            audio, sample_rate = sf.read(assets / f"{item['id']}_{instrument}.wav")
            assert sample_rate == 48000
            assert len(audio) / sample_rate >= reference.length_beats * reference.spb - 0.1
            assert np.isfinite(audio).all()
            assert np.max(np.abs(audio)) > 0.01
            assert np.max(np.abs(audio)) < 1.0
    with (assets / "suno_recording_table.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 72
    assert {row["filename"] for row in rows} == {
        f"S_{item['id']}_{route}_{take}.mp3"
        for item in manifest["items"] for route in ("R1", "R2", "R3") for take in ("a", "b")
    }
    assert {"actual_option", "model_version", "date_utc", "credits_used", "route_available",
            "bpm", "key", "style", "actual_bpm", "actual_key", "actual_style"} <= rows[0].keys()
    assert all(row["actual_option"] == "" for row in rows)


def test_pack_refuses_a_protected_output_location():
    with pytest.raises(ValueError, match="outputs"):
        pack._output_directory(pack.REPO / "results/e7_forbidden")
