"""The v2 statistics reproduce every number the registered v1 scorer stored (population ALL, first-listed
ties) and the numbers in registration/v1_outcome.json."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from dacapo import motif_features as F
from dacapo import motif_stats as S

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results" / "gpu_run1"
V1_SHA = {"e5_main.json": "63609991cfafde56ae43ae7783e814fb7a2b96df4062badf42bbb559be61f366",
          "e5_phase0.json": "e976c5fb31b512e3e7ed345aaf2700800f9325a2cb5fef3d15b39925a99e3383",
          "e5_ruler_gate.json": "26b21b2b71be808665ddbe53716e0fb62796b179bc931b1eac20378807779a6f"}
NINE = ("CI_lit", "CI_ctl", "ANA", "RI+2", "RI+5", "RI-3", "LI+2", "LI+5", "LI-3")
r4 = lambda x: round(float(x), 4)


def _load(name):
    p = V1 / name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == V1_SHA[name], f"{name} changed"
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def v1():
    d = _load("e5_main.json")
    return d["results"]["rows"], d["results"]["summary"]


def test_v1_nine_stored_indices_exact(v1):
    rows, summ = v1
    for name in NINE:
        ix = S.index(rows, *S.INDEX_SPEC[name], S.pop_all)
        st = summ["base"][name]
        assert (r4(ix["mean"]), [r4(ix["lo"]), r4(ix["hi"])], ix["n_melodies"]) == \
               (st["mean"], st["ci95"], st["n_melodies"]), name


def test_pack_equals_v1_loop_bootstrap(v1):
    rows, _ = v1
    for name in NINE:
        pm = S.per_melody(rows, S.INDEX_SPEC[name][0], S.contrast(*S.INDEX_SPEC[name][1:]), S.pop_all)
        p = S.pack(pm)
        assert (p["mean"], p["lo"], p["hi"]) == pytest.approx(S.boot_mean_ci(pm.values()), abs=1e-12)


def test_v1_h2_difference_matches_outcome_file(v1):
    rows, _ = v1
    d = S.diff_index(rows, S.INDEX_SPEC["CI_lit"], S.INDEX_SPEC["CI_ctl"], S.pop_all)
    assert (r4(d["mean"]), r4(d["lo"]), r4(d["hi"])) == (0.1147, 0.0695, 0.1667)     # v1_outcome: 0.115 [0.070, 0.167]
    assert d["n_common"] == 12 and r4(d["paired_common_mean"]) == 0.1147
    p = S.paired_diff(rows, S.INDEX_SPEC["CI_lit"], S.INDEX_SPEC["CI_ctl"], S.pop_all)
    assert (r4(p["mean"]), r4(p["lo"]), r4(p["hi"])) == (0.1147, 0.0695, 0.1667)     # same melodies: same numbers


def test_v1_k3_registered_form_and_octave(v1):
    rows, _ = v1
    lit, ctl = (S.index(rows, *S.INDEX_SPEC[n], S.pop_all) for n in ("CI_lit", "CI_ctl"))
    assert not ctl["mean"] >= lit["mean"] - 0.05                                      # K3: 0.0445 < 0.1093
    o = S.index(rows, "lit", "B", "B~", S.pop_all, key="acc_oct")
    assert (r4(o["mean"]), r4(o["lo"]), r4(o["hi"])) == (0.1457, 0.0853, 0.2115)


def test_v1_choices_both_tie_rules(v1):
    rows, summ = v1
    for r in rows:
        assert S.classify(r["acc"], S.CLASS_SETS[r["cond"]], summ["tau"], tie="first") == r["choice"]
        assert S.classify(r["acc"], S.CLASS_SETS[r["cond"]], summ["tau"], tie="strict") == r["choice"]


def test_v1_summary_blocks(v1):
    rows, summ = v1
    for c, hyps in S.CLASS_SETS.items():
        sel = [r for r in rows if r["cond"] == c]
        assert {h: sum(r["choice"] == h for r in sel) for h in hyps + ["none"]} == summ["base"][f"choices_{c}"]
        q = {k: round(float(np.mean([r[k] for r in sel])), 3) for k in ("gap_voiced", "gap_polyphony", "gap_rms_db_vs_context")}
        assert q == summ["base"][f"quality_{c}"]


def test_v1_gate_counts_and_failure_types(v1):
    rows, _ = v1
    nn = [r for r in rows if r["cond"] != "null"]
    count = lambda sel: (sum(map(S.gate_v1, sel)), len(sel))
    assert count(nn) == (59, 120)
    assert count([r for r in nn if r["seed"] == "s1"]) == (59, 84)
    assert count([r for r in nn if r["seed"] == "s2"]) == (0, 36)
    assert count([r for r in rows if r["cond"] == "null"]) == (0, 12)
    fail = [r for r in nn if not S.gate_v1(r)]
    quiet = [r for r in fail if r["gap_rms_db_vs_context"] < -6]
    poly_only = [r for r in fail if r["gap_voiced"] >= 0.6 and abs(r["gap_rms_db_vs_context"]) <= 6]
    assert (len(fail), len(quiet), len(poly_only)) == (61, 52, 8)
    assert sum(r["gap_rms_db_vs_context"] < -15 for r in quiet) == 50
    null = [r for r in rows if r["cond"] == "null"]
    assert all(r["gap_voiced"] == 0.0 and r["choice"] == "none" for r in null)


def test_registered_gate_in_motif_features_is_the_v1_rule(v1):
    """The governing v2 gate (GATE_REGISTERED) gives the v1 gate_choice verdict on every stored v1 row."""
    rows, _ = v1
    for r in rows:
        q = {**r, "gap_rms_db_vs_context_v1": r["gap_rms_db_vs_context"], "gap_polyphony_octfold": r["gap_polyphony"]}
        assert F.gate_flags(q)["pass"] == S.gate_v1(r)


def test_v1_null_coincidence_not_computable(v1):
    rows, summ = v1
    out = S.item_rates(rows, "null", "seq", "T+6A", S.pop_all, summ["tau"])
    assert "error" in out                                                             # T+6A never scored on null gaps


def test_v1_pilot_through_pilot_decision():
    rows = _load("e5_phase0.json")["results"]["rows"]
    v2rows = [{**r, "status": "ok", "gate": {"pass": S.gate_v1(r)}} for r in rows]
    planned = [S.draw_key(r) for r in v2rows]
    d = S.pilot_decision(v2rows, planned, ["cap1", "cap2", "cap3"])
    assert {a: v["n_pass"] for a, v in d["per_arm"].items()} == {"cap1": 4, "cap2": 2, "cap3": 3}
    assert d["decision"] == "selected" and d["arm"] == "cap1"


def test_v1_ruler_gate_ties():
    rows = _load("e5_ruler_gate.json")["results"]["rows"]
    first = [S.classify(r["accs"], list(r["accs"]), 0.7, tie="first") == r["true"] for r in rows]
    strict = [S.classify(r["accs"], list(r["accs"]), 0.7, tie="strict") == r["true"] for r in rows]
    assert (sum(first), sum(strict), len(rows)) == (658, 655, 658)
    bad = sorted((r["id"], r["cond"], r["true"]) for r, ok in zip(rows, strict) if not ok)
    assert bad == [("mel09", "rel+2", "T+2A"), ("mel09", "rel+2", "T+2B"), ("mel09", "seq", "T+2A")]
    assert sum(strict) / len(rows) >= 0.95


def test_v1_manifest_has_scorable_references_and_validator_catches_edge_notes():
    sys.path.insert(0, str(ROOT / "experiments" / "e5_motif"))
    import score_analogy as A
    man = json.loads((ROOT / "experiments" / "e5_motif" / "manifest.json").read_text())
    assert A.validate_manifest(man) == []
    bad = copy.deepcopy(man)
    bad["items"][0]["hyps"]["B"] = [{"start": 7.5, "dur": 0.5, "pitch": 72, "vel": 90}]   # one note at 15.75-16.0 s
    assert A.validate_manifest(bad) == [f"{bad['items'][0]['id']}:B"]


def test_regress_v1_command():
    sys.path.insert(0, str(ROOT / "experiments" / "e5_motif"))
    import score_analogy as A
    assert A.regress_v1() == 0
