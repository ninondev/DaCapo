"""E5 v2 settings: run lists, seeds, tie order, file names and expected lengths (main environment)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5_motif"
sys.path.insert(0, str(E5))

import v2_config as C  # noqa: E402
from dacapo.seeds import run_seeds  # noqa: E402

P0B = json.loads((E5 / "manifest_p0b.json").read_text())
MAIN = json.loads((E5 / "manifest_v2.json").read_text())


def test_sizes_and_order():
    p = C.jobs("p0b", P0B)
    assert len(p) == 1344 and len({(j["id"], j["cond"], j["arm"], j["seed"]) for j in p}) == 1344
    assert [j["id"] for j in p][:96] == ["mel00"] * 96                   # a melody finishes before the next starts
    m = C.jobs("main", MAIN, cell="rep-rend-flute")
    assert len(m) == 1152 and len({(j["id"], j["cond"], j["seed"]) for j in m}) == 1152
    assert sum(j["cond"] != "null" for j in m) == 1008
    assert len({(j["id"], j["cond"]) for j in m if j["cond"] != "null"}) == 252
    assert all(j["arm"] == "base" and j["split"] == "confirm" for j in m)


def test_cells_excluded_by_k0_are_not_planned():
    p = C.jobs("p0b", P0B, excluded_cells=tuple(c for c in C.CELLS if c.endswith("clarinet")))
    assert len(p) == 1344 * 8 // 12 and not any(j["arm"].endswith("clarinet") for j in p)


def test_pairing():
    p = C.jobs("p0b", P0B)
    by = {}
    for j in p:
        by.setdefault((j["id"], j["seed"]), set()).add((j["dit_seed"], j["rng_seed"]))
    assert all(len(v) == 1 for v in by.values())                         # every cell and condition shares the pair
    assert len({next(iter(v)) for v in by.values()}) == len(by)          # pairs differ across (item, draw)


def test_seed_table_is_the_committed_one():
    t = C.seed_table()
    committed = json.loads((E5 / "registration" / "e5v2_seeds.json").read_text())
    assert committed == json.loads(json.dumps(t))
    flat = [s for ph in ("a0", "p0b", "main") for v in t[ph].values() for pair in v for s in pair]
    assert len(flat) == len(set(flat)) == 416 and not set(flat) & {1001, 2002, 20261027}
    assert run_seeds("e5v2-main", "v2c00", 0) == (738209811, 1519084974)
    assert run_seeds("e5v2-p0b", "mel00", 0) == (1980629844, 1873831935)
    assert all(a != b for v in t["main"].values() for a, b in v)


def test_tie_order_and_captions():
    assert C.CELLS[:3] == ("rep-rend-flute", "rep-rend-synth_voice", "rep-rend-clarinet")
    assert C.CELLS[3] == "rep-t18-flute" and C.CELLS[6] == "t2m-rend-flute" and len(C.CELLS) == 12
    assert C.CAPTIONS["synth_voice"] == "solo synth voice melody, monophonic, no accompaniment"
    assert C.PROGRAMS == {"flute": 73, "synth_voice": 54, "clarinet": 71}
    assert not any("__" in c for c in C.CELLS)


def test_names_split_into_four_fields():
    from dacapo.motif_integrity import audio_name
    for j in C.jobs("p0b", P0B)[:24] + C.jobs("main", MAIN, cell="t2m-t18-clarinet")[:24]:
        mid, cond, arm, seed = Path(audio_name(j)).stem.split("__")
        assert (mid, cond, arm, seed) == (j["id"], j["cond"], j["arm"], j["seed"])


def test_expected_frames_and_meta():
    assert [C.expected_frames(n) for n in (864000, 980800, 992768, 1004800, 1016768)] == [450, 510, 517, 523, 529]
    assert [C.expected_duration_meta(n) for n in (864000, 980800, 1016768)] == [18, 20, 21]


def test_tiny_scale_runs_one_or_two_generations_per_stage():
    t = C.SCALES["tiny"]
    assert len(C.jobs("p0b", P0B, scale="tiny")) == 2
    assert len(C.jobs("main", P0B, scale="tiny", cell=t["p0b_cells"][0])) == 2
    assert C.min_pass_per_cell(1) == 1 and C.min_pass_per_cell(4) == 2


def test_tiny_scale_never_touches_confirmatory_melodies_or_registered_seeds():
    t = C.SCALES["tiny"]
    assert t["main_manifest"] == t["p0b_manifest"] == "manifest_p0b.json" and t["k0_manifest"] == "manifest_p0b.json"
    assert not set(t["main_ids"]) & set(C.MAIN_IDS) and not set(t["main_ids"]) & set(t["p0b_ids"])
    tiny = {s for ph in ("a0", "p0b", "main") for mid in ("mel00", "mel01") for d in C.DRAWS
            for s in run_seeds(t["tags"][ph], mid, d)}
    reg = {s for ph in ("a0", "p0b", "main") for v in C.seed_table()[ph].values() for pair in v for s in pair}
    assert not tiny & reg and not tiny & C.RESERVED


def test_generation_settings():
    assert C.REPAINT["dcw_enabled"] is False and C.REPAINT["repaint_mode"] == "balanced"
    assert C.REPAINT["chunk_mask_mode"] == "explicit" and C.REPAINT["repaint_strength"] == 0.5
    assert C.INSTRUCTIONS["rep"].startswith("Repaint") and C.INSTRUCTIONS["t2m"].startswith("Fill")
    assert (C.MAX_STARTS, C.MAX_RESUME_PASSES, C.K1_FRAC, C.Q1_FRAC, C.Q2_FRAC) == (3, 2, 0.75, 0.75, 0.75)


def test_notebook_installs_the_pinned_measurement_software():
    sys.path.insert(0, str(ROOT / "experiments"))
    import build_notebook as B
    pinned = json.loads((E5 / "registration" / "addendum_2.json").read_text())["measurement"]["pinned"]
    assert dict(x.split("==") for x in B.PINS + [B.BASIC_PITCH]) == pinned


def test_register_states_the_k0_outcome_from_the_files():
    import register
    add = json.loads((E5 / "registration" / "addendum_2.json").read_text())
    assert add["k0_outcome"] == register.k0_outcome(register.k0_summary())
    assert "flute (GM 73) failed" in add["k0_outcome"] and add["k0_first_run"].startswith("K0 ran on this machine")
