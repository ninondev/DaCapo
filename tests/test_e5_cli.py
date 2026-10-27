"""End-to-end tests of the scorer stages (design.json -> quality -> score) through subprocess, with the real
trackers on stereo fake outputs: sources rendered with FluidSynth, a hypothesis render pasted into the gap,
and the whole clip peak-normalised to -1 dBFS over both channels as ACE-Step does."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5_motif"
sys.path.insert(0, str(E5))

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not installed")]
CONDS = ["lit", "ctl", "rel+2", "rel+5", "rel-3", "seq", "ana", "null"]
EXT = ".flac"
FILL = {"lit": "B", "ctl": "B", "rel+2": "T+2B", "rel+5": "T+5B", "rel-3": "T-3B", "seq": "T+6A", "ana": "T+5B", "null": None}
UTC_ADD, UTC_RUN = "2026-10-27T12:00:00Z", "2026-11-01T10:00:00Z"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Sources and flute hypothesis renders of mel02, mel04 and mel05, rendered once."""
    import make_items
    root = tmp_path_factory.mktemp("rendered")
    man = json.loads((E5 / "manifest_p0b.json").read_text())
    items = [it for it in man["items"] if it["id"] in ("mel02", "mel04", "mel05")]
    for it in items:
        make_items.render_sources(it, root, arms=("rend",))
        make_items.render_hyps(it, root, 73)
    return root, {**man, "items": items}


def run(w, *args) -> int:
    env = {**os.environ, "E5_DIR": str(w["e5"]), "E5_RUN": str(w["run"]), "DACAPO_ROOT": str(ROOT)}
    p = subprocess.run([sys.executable, str(E5 / "score_analogy.py"), *args], env=env, capture_output=True, text=True)
    (w["run"] / "cli.log").open("a").write(f"$ {' '.join(args)} -> {p.returncode}\n{p.stdout[-4000:]}\n{p.stderr[-3000:]}\n")
    return p.returncode


def fake(src: Path, render: Path | None, out: Path, gain_db: float = 0.0) -> None:
    s, sr = sf.read(str(src), always_2d=True)
    s = s.copy()
    if render is not None:
        h, _ = sf.read(str(render), always_2d=True)
        s[12 * sr:16 * sr] = h[12 * sr:16 * sr] * 10 ** (gain_db / 20)
    pk = np.abs(s).max()
    if pk >= 1e-6:
        s = s / pk * 10 ** (-1 / 20)                     # ACE-Step: max over all channels
    sf.write(str(out), s, sr, subtype="PCM_16")


def measurement_section() -> dict:
    """The pinned measurement software of addendum 2, as installed here."""
    import score_analogy as SA
    pinned = json.loads((E5 / "registration" / "addendum_2.json").read_text())["measurement"]["pinned"]
    prov = SA.tracker_provenance(tuple(pinned))
    return {"pinned": prov["versions"], "onnx_sha256": prov["model_sha256"]}


def build(tmp: Path, rendered, pilot_fill_a=("B", 0.0), pilot_fill_b=("B", -30.0), main_gain=0.0, draws=2) -> dict:
    from dacapo.motif_integrity import audio_sha256
    import register
    src_root, man = rendered
    e5, rn = tmp / "e5", tmp / "run"
    shutil.copytree(src_root, e5, copy_function=os.link)                # hard links: the renders are only read
    (e5 / "manifest_p0b.json").write_text(json.dumps(man))
    reg = e5 / "registration" / "addendum_2.json"
    reg.parent.mkdir(parents=True)
    reg.write_text(json.dumps({"addendum_utc": UTC_ADD, "code_sha256": {f: sha(ROOT / f) for f in register.E5_CODE},
                               "measurement": measurement_section()}))
    (e5 / "results").mkdir()
    (e5 / "results" / "e5_ruler_gate_p73.json").write_text(json.dumps({"results": {"summary": {
        "passed": True, "tau": 0.7, "program": 73, "manifest_sha256": sha(e5 / "manifest_p0b.json")}}}))
    (rn / "results").mkdir(parents=True)
    items = {it["id"]: it for it in man["items"]}

    def phase(name, ids, arms, conds, base, fill_fn, split):
        od = rn / f"outputs_v2_{name}"
        od.mkdir()
        jobs, ev = [], [{"event": "pass", "pass": 1, "utc": UTC_RUN}]
        for i, mid in enumerate(ids):
            for d in range(draws):
                for a in arms:
                    for c in conds:
                        j = {"id": mid, "split": split, "cond": c, "arm": a, "seed": f"d{d}", "dit_seed": base + 10 * i + d,
                             "rng_seed": base + 5000 + 10 * i + d, "src": items[mid]["sources"]["rend"][c]}
                        jobs.append(j)
                        wav = od / f"{mid}__{c}__{a}__d{d}{EXT}"
                        hyp, gain = fill_fn(c, a, d)
                        fake(e5 / j["src"], e5 / items[mid]["hyp_renders"][hyp] if hyp else None, wav, gain)
                        k = {f: j[f] for f in ("id", "cond", "arm", "seed", "dit_seed", "rng_seed")}
                        ev += [{"event": "start", **k, "pass": 1, "utc": UTC_RUN},
                               {"event": "done", **k, "pass": 1, "utc": UTC_RUN, "wav_sha256": sha(wav)}]
        design = {"phase": name, "scale": "full", "attempt": 1, "manifest": "manifest_p0b.json",
                  "manifest_sha256": sha(e5 / "manifest_p0b.json"),
                  "sources_sha256": {s: audio_sha256(e5 / s) for s in sorted({j["src"] for j in jobs})},
                  "addendum_sha256": sha(reg), "ci_method": "melody", "k1_frac": 0.75, "q1_frac": 0.75, "q2_frac": 0.75,
                  "min_pass_per_cell": 1, "arm_order": list(arms), "excluded_arms": [],
                  "ruler_program": 73 if name == "main" else None, "pilot_dir": "outputs_v2_p0b", "jobs": jobs}
        (od / "design.json").write_text(json.dumps(design))
        (od / "ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))

    phase("p0b", ["mel02"], ["armA", "armB"], ["lit", "ctl"], 90000,
          lambda c, a, d: pilot_fill_a if a == "armA" else pilot_fill_b, "pilot")
    phase("main", ["mel04", "mel05"], ["base"], CONDS, 1000,
          lambda c, a, d: (FILL[c] if d == 0 or FILL[c] is None else ("B~" if c in ("lit", "ctl") else FILL[c]), main_gain),
          "confirm")
    return {"e5": e5, "run": rn}


def qfile(w, phase):
    return json.loads((w["run"] / "results" / f"e5v2_{phase}_quality.json").read_text())


def ledger_drop_done(w, phase, keys):
    od = w["run"] / f"outputs_v2_{phase}"
    ev = [json.loads(l) for l in (od / "ledger.jsonl").read_text().splitlines()]
    ev = [e for e in ev if not (e["event"] == "done" and (e["id"], e["cond"], e["seed"]) in keys)]
    (od / "ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    for i, c, d in keys:
        (od / f"{i}__{c}__base__{d}{EXT}").unlink()


@pytest.fixture(scope="module")
def good(tmp_path_factory, rendered):
    w = build(tmp_path_factory.mktemp("good"), rendered)
    codes = {"p0b": run(w, "quality", "p0b", "--workers", "2"), "main_q": run(w, "quality", "main", "--workers", "2")}
    return w, codes


def test_pilot_selects_the_passing_arm(good):
    w, codes = good
    assert codes["p0b"] == 0, (w["run"] / "cli.log").read_text()
    d = qfile(w, "p0b")["results"]["decision"]
    assert d["decision"] == "selected" and d["arm"] == "armA"
    assert d["per_arm"]["armA"]["n_pass"] == 4 and d["per_arm"]["armB"]["n_pass"] == 0
    sel = json.loads((w["run"] / "results" / "e5v2_phase0b.json").read_text())
    assert sel["decision"] == "selected" and sel["cell"] == "armA"


def test_main_quality_rows_and_integrity(good):
    w, codes = good
    assert codes["main_q"] == 0, (w["run"] / "cli.log").read_text()
    q = qfile(w, "main")
    res = q["results"]
    assert res["integrity"]["ok"], res["integrity"]
    rows = res["rows"]
    assert len(rows) == 32 and all(r["status"] == "ok" for r in rows)
    assert all(r["wav_info"]["subtype"] == "PCM_16" and r["wav_info"]["format"] == "FLAC" and r["wav_info"]["channels"] == 2
               for r in rows)
    assert all(abs(r["quality"]["peak_dbfs"] + 1.0) < 0.01 for r in rows)
    null = [r for r in rows if r["cond"] == "null"]
    assert all(not r["gate"]["pass"] for r in null) and len(res["null_gap_rms_dbfs"]) == 4
    assert all(r["gate"]["pass"] for r in rows if r["cond"] != "null"), \
        [(r["cond"], r["quality"]) for r in rows if not r["gate"]["pass"] and r["cond"] != "null"]
    assert res["decision"]["decision"] == "pass" and res["decision"]["n_nonnull"] == 28
    assert set(q["config"]["trackers"]["model_sha256"]) == {"swift_f0/model.onnx", "basic_pitch/nmp.onnx"}
    assert q["config"]["gate"]["name"] == "registered"


def test_quality_is_frozen_once_written(good):
    w, _ = good
    before = sha(w["run"] / "results" / "e5v2_main_quality.json")
    assert run(w, "quality", "main") == 0
    assert sha(w["run"] / "results" / "e5v2_main_quality.json") == before


def test_a_frozen_phase_writes_its_missing_selection_again(good):
    """A reset between the Phase 0b freeze and the selection file: reading the frozen file writes it."""
    w, _ = good
    sel = w["run"] / "results" / "e5v2_phase0b.json"
    keep = json.loads(sel.read_text())
    sel.unlink()
    assert run(w, "quality", "p0b") == 0
    again = json.loads(sel.read_text())
    assert (again["decision"], again["cell"], again["quality_file_sha256"]) == (keep["decision"], keep["cell"], keep["quality_file_sha256"])


def test_score_then_terminal_blocks_rescoring(good):
    w, _ = good
    assert run(w, "score", "main", "--workers", "2", "--n-boot", "500") == 2   # registered estimator: 10,000 resamples
    assert not (w["run"] / "results" / "e5v2_main.json").exists()
    assert run(w, "score", "main", "--workers", "2") == 0, (w["run"] / "cli.log").read_text()
    out = json.loads((w["run"] / "results" / "e5v2_main.json").read_text())
    rows = out["results"]["rows"]
    lit0 = next(r for r in rows if r["cond"] == "lit" and r["seed"] == "d0")
    assert len(lit0["acc"]) == 35 and set(lit0["of1"]) == set(lit0["acc"]) | {"B2"}
    assert lit0["choice"] == "B" and next(r for r in rows if r["cond"] == "lit" and r["seed"] == "d1")["choice"] == "B~"
    ctl0 = next(r for r in rows if r["cond"] == "ctl" and r["seed"] == "d0")
    assert ctl0["acc"]["T+2B"] is not None                               # union scoring: RI(X-ctl) computable
    summ = out["results"]["summary"]
    assert summ["paper_level"] == "not_evaluable" and summ["primary_population"] == "ALL"   # 2 melodies, floor 12
    assert out["config"]["n_boot"] == 10000 and out["config"]["data_sha256"] == {}
    term = json.loads((w["run"] / "results" / "e5_terminal.json").read_text())
    assert term["stage"] == "score" and term["strike"] is None and term["attempt"] == 1
    assert run(w, "score", "main") == 3
    assert run(w, "quality", "main") == 3


def test_parallel_and_serial_quality_rows_identical(good, tmp_path):
    w, _ = good
    ref = qfile(w, "p0b")["results"]["rows"]
    res = w["run"] / "results"
    keep = {p.name: p.read_bytes() for p in res.glob("*.json")}
    for p in res.glob("*.json"):
        p.unlink()
    try:
        assert run(w, "quality", "p0b", "--workers", "1") == 0
        assert qfile(w, "p0b")["results"]["rows"] == ref
    finally:
        for p in res.glob("*.json"):
            p.unlink()
        for n, b in keep.items():
            (res / n).write_bytes(b)


def test_missing_draws_wait_for_the_resume_passes_then_count_as_failures(tmp_path, rendered):
    w = build(tmp_path, rendered)
    gone = {("mel04", "rel+5", "d1"), ("mel05", "seq", "d1")}
    ledger_drop_done(w, "main", gone)
    assert run(w, "quality", "p0b") == 0
    assert run(w, "quality", "main") == 4                                # generation may still run them
    inc = json.loads((w["run"] / "results" / "e5v2_main_incomplete.json").read_text())
    assert inc["resumable_missing"] == 2 and not (w["run"] / "results" / "e5v2_main_quality.json").exists()
    with (w["run"] / "outputs_v2_main" / "ledger.jsonl").open("a") as fh:   # two resume passes that could not finish
        for p in (2, 3):
            fh.write(json.dumps({"event": "resume", "pass": p, "utc": UTC_RUN}) + "\n")
    assert run(w, "quality", "main") == 0
    d = qfile(w, "main")["results"]["decision"]
    assert d["missing_as_failure"] and d["n_missing"] == 2 and d["n_pass"] == 26 and d["decision"] == "pass"


def test_lost_audio_before_the_freeze_sends_generation_back(tmp_path, rendered):
    w = build(tmp_path, rendered)
    (w["run"] / "outputs_v2_main" / f"mel04__lit__base__d0{EXT}").unlink()
    assert run(w, "quality", "p0b") == 0
    assert run(w, "quality", "main") == 4
    assert json.loads((w["run"] / "results" / "e5v2_main_incomplete.json").read_text())["lost"] == 1


def test_wav_changed_after_quality_voids_the_attempt(tmp_path, rendered):
    w = build(tmp_path, rendered)
    assert run(w, "quality", "p0b") == 0 and run(w, "quality", "main") == 0
    p = w["run"] / "outputs_v2_main" / f"mel04__seq__base__d0{EXT}"
    b = bytearray(p.read_bytes())
    b[-10] ^= 1
    p.write_bytes(bytes(b))
    assert run(w, "score", "main", "--n-boot", "200") == 5
    assert json.loads((w["run"] / "results" / "e5v2_main.json").read_text())["paper_level"] == "invalid_run"
    void = json.loads((w["run"] / "results" / "e5v2_void.json").read_text())
    assert void["attempt"] == 1 and void["phase"] == "main" and "wav_unchanged_since_quality" in void["failed_checks"]
    assert not (w["run"] / "results" / "e5_terminal.json").exists()      # attempt 1: no strike yet


def test_infill_failure_is_terminal_and_diagnostic_is_separate(tmp_path, rendered):
    w = build(tmp_path, rendered, main_gain=-40.0)                       # every gap far too quiet
    assert run(w, "quality", "p0b") == 0
    assert run(w, "quality", "main") == 3
    term = json.loads((w["run"] / "results" / "e5_terminal.json").read_text())
    assert (term["paper_level"], term["stage"], term["strike"]) == ("infill_failure_v2_gate", "main_quality", 2)
    assert run(w, "score", "main") == 3
    assert run(w, "score", "main", "--diagnostic", "--n-boot", "200") == 0
    diag = json.loads((w["run"] / "results" / "e5v2_main_diagnostic.json").read_text())
    assert diag["config"]["label"].startswith("EXPLORATORY")


def test_k1_terminal(tmp_path, rendered):
    w = build(tmp_path, rendered, pilot_fill_a=("B", -40.0), pilot_fill_b=("B", -40.0))
    assert run(w, "quality", "p0b") == 3
    term = json.loads((w["run"] / "results" / "e5_terminal.json").read_text())
    assert (term["paper_level"], term["stage"], term["strike"]) == ("infill_failure_v2_K1", "K1_phase0b", 2)
    assert json.loads((w["run"] / "results" / "e5v2_phase0b.json").read_text())["decision"] == "k1_triggered"
    assert run(w, "quality", "main") == 3                                # the main grid is never scored after K1


def test_unregistered_code_voids_the_attempt(tmp_path, rendered):
    w = build(tmp_path, rendered)
    reg = w["e5"] / "registration" / "addendum_2.json"
    a = json.loads(reg.read_text())
    a["code_sha256"]["dacapo/motif_stats.py"] = "0" * 64
    reg.write_text(json.dumps(a))
    for ph in ("p0b", "main"):                                           # design.json names this addendum
        dp = w["run"] / f"outputs_v2_{ph}" / "design.json"
        d = json.loads(dp.read_text())
        d["addendum_sha256"] = sha(reg)
        dp.write_text(json.dumps(d))
    assert run(w, "quality", "p0b") == 5
    res = qfile(w, "p0b")["results"]
    assert [c["name"] for c in res["integrity"]["checks"] if not c["ok"]] == ["registered_code"]
    assert res["decision"] is None and "gate" not in res["rows"][0]           # no tracker ran: no decision to see
    assert not list((w["run"] / "outputs_v2_p0b" / "features").glob("*.npz"))
    assert not (w["run"] / "results" / "e5v2_phase0b.json").exists()
    assert run(w, "quality", "p0b", "--void-report") == 2                # only after the terminal outcome
    (w["run"] / "results" / "e5_terminal.json").write_text(json.dumps({"paper_level": "x", "stage": "test"}))
    assert run(w, "quality", "p0b", "--void-report", "--workers", "2") == 0
    rep = json.loads((w["run"] / "results" / "e5v2_p0b_quality_void_report.json").read_text())
    assert rep["config"]["label"].startswith("EXPLORATORY") and rep["results"]["decision"]["decision"] == "selected"


def test_an_edited_manifest_or_data_file_voids_the_attempt(tmp_path, rendered):
    """Check 10 covers the registered data: a manifest whose sha256 differs from data_sha256 fails it."""
    w = build(tmp_path, rendered)
    reg = w["e5"] / "registration" / "addendum_2.json"
    a = json.loads(reg.read_text())
    a["data_sha256"] = {"experiments/e5_motif/manifest_p0b.json": "0" * 64}
    reg.write_text(json.dumps(a))
    for ph in ("p0b", "main"):
        dp = w["run"] / f"outputs_v2_{ph}" / "design.json"
        d = json.loads(dp.read_text())
        d["addendum_sha256"] = sha(reg)
        dp.write_text(json.dumps(d))
    assert run(w, "quality", "p0b") == 5
    chk = {c["name"]: c for c in qfile(w, "p0b")["results"]["integrity"]["checks"]}
    assert not chk["registered_code"]["ok"] and "experiments/e5_motif/manifest_p0b.json" in chk["registered_code"]["detail"]["changed_files"]


def test_measurement_software_that_differs_is_refused(tmp_path, rendered):
    w = build(tmp_path, rendered)
    reg = w["e5"] / "registration" / "addendum_2.json"
    a = json.loads(reg.read_text())
    a["measurement"]["pinned"]["librosa"] = "0.0.1"
    reg.write_text(json.dumps(a))
    assert run(w, "measurement") == 2
    assert run(w, "quality", "p0b") == 2 and not (w["run"] / "results" / "e5v2_p0b_quality.json").exists()


def test_ruler_gate_failure_exits_3_without_a_terminal(tmp_path, rendered):
    w = build(tmp_path, rendered)
    man = json.loads((w["e5"] / "manifest_p0b.json").read_text())
    (w["e5"] / "manifest_k0.json").write_text(json.dumps({**man, "items": [it for it in man["items"] if it["id"] == "mel04"]}))
    b = w["e5"] / "hyp_renders" / "mel04_B.wav"
    x, sr = sf.read(str(b), always_2d=True)
    b.unlink()                                                           # a new file, not the shared hard link
    sf.write(str(b), np.zeros_like(x), sr, subtype="PCM_16")             # B render destroyed: K0 must fail
    out = w["run"] / "k0.json"
    assert run(w, "gate", "--program", "73", "--manifest", "manifest_k0.json", "--out", str(out), "--keep-renders") == 3
    g = json.loads(out.read_text())["results"]["summary"]
    assert g["passed"] is False and g["manifest_sha256"] == sha(w["e5"] / "manifest_k0.json") and "n_tie_rows" in g
    assert g["tau"] == 0.7 and len(g["wrong_note_detail"]) == 2
    assert not (w["run"] / "results" / "e5_terminal.json").exists()
