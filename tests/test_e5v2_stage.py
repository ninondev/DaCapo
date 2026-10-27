"""Stage wiring of Run A for E5 v2: exit codes to statuses, resume passes, voids, terminal outcomes and the
tiny-scale overrides, with the commands replaced by scripted fakes; and the per-scale folders of run_all."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5_motif"
sys.path.insert(0, str(E5))
sys.path.insert(0, str(ROOT / "experiments"))

import stage as S  # noqa: E402
import v2_config as C  # noqa: E402


def _k0_file(e5: Path, program: int, passed: bool) -> None:
    (e5 / "results").mkdir(exist_ok=True)
    (e5 / "results" / f"e5_ruler_gate_p{program}.json").write_text(json.dumps({"results": {"summary": {"passed": passed}}}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    e5, run = tmp_path / "e5", tmp_path / "run"
    (e5 / "registration").mkdir(parents=True)
    (run / "results").mkdir(parents=True)
    f = "dacapo/seeds.py"
    add = {"addendum_utc": "2026-10-27T00:00:00Z", "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()},
           "k0": {p: {"passed": True} for p in ("73", "54", "71")}}
    (e5 / "registration" / "addendum_2.json").write_text(json.dumps(add))
    for m in ("manifest_p0b.json", "manifest_v2.json"):
        (e5 / m).write_bytes((E5 / m).read_bytes())
    for p in (73, 54, 71):
        _k0_file(e5, p, True)
    monkeypatch.setenv("E5_DIR", str(e5))
    monkeypatch.setenv("E5_RUN", str(run))
    monkeypatch.setenv("DACAPO_ROOT", str(ROOT))
    monkeypatch.setenv("DACAPO_SCALE", "full")
    _a0_passed("full")
    return {"e5": e5, "run": run, "tmp": tmp_path}


def _a0_passed(scale="full"):
    C.write_json(C.a0_path(), {"go": True, "scale": scale, "addendum_sha256": C.addendum_status()["addendum_sha256"]})


A0_SCRIPT = {"a0_checks.py:main": [0], "a0_checks.py:fresh": [0], "a0_checks.py:verdict": [0]}


def _a0_side(a, rc):
    if a[0] == "a0_checks.py" and a[2] == "verdict":
        _a0_passed(C.scale())


class Fake(S.Runner):
    """Plays back scripted exit codes per command name; side(cmd) may write the files a command writes."""

    def __init__(self, tmp: Path, script: dict, side=None):
        super().__init__(tmp / "stage.log", "py", "py", "cuda")
        self.script, self.side, self.calls = {k: list(v) for k, v in script.items()}, side, []

    def call(self, args, ace=False, cwd=E5):
        name = args[0] if args[0] not in ("score_analogy.py", "a0_checks.py") else f"{args[0]}:{args[1] if args[0] == 'score_analogy.py' else args[2]}"
        self.calls.append(args)
        rc = self.script[name].pop(0)
        if self.side:
            self.side(args, rc)
        return rc


def _sel(decision, cell="rep-rend-flute"):
    C.write_json(C.selection_path(), {"decision": decision, "cell": cell if decision == "selected" else None, "attempt": 1})


def _gens(r):
    return [c for c in r.calls if c[0] == "run_analogy.py"]


def test_p0b_selected(env):
    r = Fake(env["tmp"], {"run_analogy.py": [0], "score_analogy.py:quality": [0]},
             side=lambda a, rc: _sel("selected") if a[1] == "quality" else None)
    st = S.p0b(r)
    assert st["state"] == "ok" and st["cell"] == "rep-rend-flute"
    assert [c[0] for c in r.calls] == ["run_analogy.py", "score_analogy.py"]
    assert "--verify-audio" in r.calls[0]                                  # the first launch hashes every file


def test_p0b_relaunch_after_75_and_resume_pass_after_incomplete(env):
    def side(a, rc):
        if a[1] == "quality" and rc == 4:
            C.write_json(C.results_dir() / "e5v2_p0b_incomplete.json", {"lost": 0, "resumable_missing": 3})
        if a[1] == "quality" and rc == 0:
            _sel("selected")
    r = Fake(env["tmp"], {"run_analogy.py": [75, 75, 4, 0], "score_analogy.py:quality": [4, 0]}, side=side)
    assert S.p0b(r)["state"] == "ok"
    assert ["--new-pass" in c for c in _gens(r)] == [False, False, False, True]
    assert ["--verify-audio" in c for c in _gens(r)] == [True, False, False, False]


def test_lost_audio_is_repaired_inside_the_current_pass(env):
    def side(a, rc):
        if a[1] == "quality" and rc == 4:
            C.write_json(C.results_dir() / "e5v2_p0b_incomplete.json", {"lost": 2, "resumable_missing": 0})
        if a[1] == "quality" and rc == 0:
            _sel("selected")
    r = Fake(env["tmp"], {"run_analogy.py": [0, 0], "score_analogy.py:quality": [4, 0]}, side=side)
    assert S.p0b(r)["state"] == "ok"
    assert not any("--new-pass" in c for c in r.calls)


@pytest.mark.parametrize("rc", [1, -9, 2, 6])
def test_a_crash_refusal_or_halt_is_a_technical_stop_without_a_quality_stage(env, rc):
    """An infrastructure fault (an uncaught exception, a kill, a refusal, five errors in a row) never
    reaches the quality stage, so it can use no resume pass and freeze no missing draw as a gate failure."""
    r = Fake(env["tmp"], {"run_analogy.py": [rc]})
    st = S.p0b(r)
    assert st["state"] == "technical_stop" and [c[0] for c in r.calls] == ["run_analogy.py"]
    assert not C.terminal_path().exists() and not C.void_path().exists()


def test_repeated_crashes_never_add_resume_passes(env):
    for _ in range(3):
        r = Fake(env["tmp"], {"run_analogy.py": [1]})
        assert S.p0b(r)["state"] == "technical_stop"
        assert not any("--new-pass" in c for c in r.calls)


def test_a_frozen_p0b_writes_its_missing_selection_and_generates_nothing(env):
    """A reset between the quality freeze and the selection file: Run all only reads the outcome back."""
    C.write_json(C.quality_path("p0b", 1), {"frozen": True})
    r = Fake(env["tmp"], {"score_analogy.py:quality": [0]}, side=lambda a, rc: _sel("selected"))
    st = S.p0b(r)
    assert st["state"] == "ok" and st["cell"] == "rep-rend-flute" and not _gens(r)


def test_a_frozen_main_grid_goes_on_to_scoring(env):
    """A reset after the main quality freeze, before the score stage wrote its files."""
    _sel("selected")
    C.write_json(C.quality_path("main", 1), {"frozen": True})

    def side(a, rc):
        if a[1] == "score":
            C.write_terminal("negative_benchmark", "score", None, attempt=1)
    r = Fake(env["tmp"], {"score_analogy.py:quality": [0], "score_analogy.py:score": [0]}, side=side)
    st = S.main_grid(r)
    assert st["state"] == "terminal" and st["paper_level"] == "negative_benchmark" and not _gens(r)
    assert [c[1] for c in r.calls] == ["quality", "score"]


def test_a_runtime_unlike_a0s_runs_a0_again_once_then_generation_continues(env):
    def side(a, rc):
        _a0_side(a, rc)
        if a[0] == "score_analogy.py" and a[1] == "quality":
            _sel("selected")
    r = Fake(env["tmp"], {"run_analogy.py": [7, 0], **A0_SCRIPT, "score_analogy.py:quality": [0]}, side=side)
    assert S.p0b(r)["state"] == "ok"
    assert [c[0] for c in r.calls] == ["run_analogy.py", "a0_checks.py", "a0_checks.py", "a0_checks.py",
                                       "run_analogy.py", "score_analogy.py"]
    assert list(C.results_dir().glob("e5v2_a0_superseded_*.json"))         # the earlier A0 is kept
    r = Fake(env["tmp"], {"run_analogy.py": [7, 7], **A0_SCRIPT}, side=side)
    C.selection_path().unlink()
    assert S.p0b(r)["state"] == "technical_stop"                         # still different after A0: stop


def test_a0_from_another_addendum_runs_again_before_generation(env):
    C.write_json(C.a0_path(), {"go": True, "scale": "full", "addendum_sha256": "old"})
    r = Fake(env["tmp"], {**A0_SCRIPT, "run_analogy.py": [0], "score_analogy.py:quality": [0]},
             side=lambda a, rc: (_a0_side(a, rc), _sel("selected") if a[1] == "quality" else None))
    assert S.p0b(r)["state"] == "ok" and [c[0] for c in r.calls][:3] == ["a0_checks.py"] * 3
    r = Fake(env["tmp"], {"a0_checks.py:main": [1]})
    C.selection_path().unlink()
    C.write_json(C.a0_path(), {"go": True, "scale": "full", "addendum_sha256": "old"})
    assert S.p0b(r)["state"] == "technical_stop" and not _gens(r)


def test_k1_is_terminal_at_full_scale(env):
    def side(a, rc):
        if a[1] == "quality":
            _sel("k1_triggered")
            C.write_terminal("infill_failure_v2_K1", "K1_phase0b", 2)
    r = Fake(env["tmp"], {"run_analogy.py": [0], "score_analogy.py:quality": [3]}, side=side)
    st = S.p0b(r)
    assert st["state"] == "terminal" and st["paper_level"] == "infill_failure_v2_K1" and st["strike"] == 2
    assert S.main_grid(Fake(env["tmp"], {}))["state"] == "terminal"     # nothing more runs


def test_tiny_scale_records_k1_and_runs_the_main_grid(env, monkeypatch):
    monkeypatch.setenv("DACAPO_SCALE", "tiny")
    _a0_passed("tiny")

    def side(a, rc):
        if a[0] == "score_analogy.py" and a[1] == "quality" and a[2] == "p0b":
            _sel("k1_triggered")
            C.write_terminal("infill_failure_v2_K1", "K1_phase0b", 2)
        if a[0] == "score_analogy.py" and a[1] == "quality" and a[2] == "main":
            C.write_terminal("infill_failure_v2_gate", "main_quality", 2)
    r = Fake(env["tmp"], {"run_analogy.py": [0, 0], "score_analogy.py:quality": [3, 3], "score_analogy.py:score": [0]}, side=side)
    assert S.p0b(r)["k1"] is True
    assert (C.results_dir() / "e5_terminal_tiny_K1.json").exists()
    st = S.main_grid(r)
    assert st["state"] == "ok" and st["diagnostic_exit"] == 0
    main_gen = _gens(r)[-1]
    assert main_gen[main_gen.index("--cell") + 1] == C.SCALES["tiny"]["p0b_cells"][0]
    assert "--diagnostic" in r.calls[-1]


def test_full_main_grid_scores_after_a_passing_quality_stage(env):
    _sel("selected")

    def side(a, rc):
        if a[1] == "score":
            C.write_terminal("negative_benchmark", "score", None)
    r = Fake(env["tmp"], {"run_analogy.py": [0], "score_analogy.py:quality": [0], "score_analogy.py:score": [0]}, side=side)
    st = S.main_grid(r)
    assert st["state"] == "terminal" and st["paper_level"] == "negative_benchmark" and st["strike"] is None
    assert not any("--cell" in c for c in r.calls)


def _amend(env, failed, files):
    reg = env["e5"] / "registration" / "addendum_2.json"
    a = json.loads(reg.read_text())
    a["amendments"] = [{"id": "A1", "void_attempt": 1, "failed_checks": failed, "files": files}]
    reg.write_text(json.dumps(a))


def test_a_broken_configuration_voids_attempt_1_then_attempt_2_needs_a_linked_amendment(env):
    r = Fake(env["tmp"], {"run_analogy.py": [5]})
    assert S.p0b(r)["state"] == "void"
    v = json.loads(C.void_path().read_text())
    assert v["attempt"] == 1 and v["phase"] == "p0b" and v["registered_sha256"]
    assert S.attempt_now(r) is None and S.p0b(Fake(env["tmp"], {}))["state"] == "void"
    reg = env["e5"] / "registration" / "addendum_2.json"
    reg.write_text(reg.read_text().replace("2026-10-27T00:00:00Z", "2026-10-28T00:00:00Z"))
    assert S.attempt_now(r) is None                                      # a changed addendum alone is not enough
    _amend(env, ["some other check"], [])
    assert S.attempt_now(r) is None                                      # the amendment must name the failed check
    _amend(env, ["per-run hard check"], [])
    assert S.attempt_now(r) == 2
    r2 = Fake(env["tmp"], {**A0_SCRIPT, "run_analogy.py": [5]}, side=_a0_side)
    st = S.p0b(r2)                                                       # a void attempt 2 is strike two
    assert st["state"] == "terminal" and st["strike"] == 2
    assert [c[0] for c in r2.calls][:3] == ["a0_checks.py"] * 3          # A0 runs again under the amended addendum
    g = _gens(r2)[0]
    assert g[g.index("--attempt") + 1] == "2"


def test_attempt2_amendment_must_list_every_changed_registered_file(env):
    C.write_json(C.void_path(), {"attempt": 1, "phase": "main", "failed_checks": ["lossless_audio"],
                                 "addendum_sha256": "old", "registered_sha256": {"dacapo/seeds.py": "before",
                                                                                 "dacapo/ace.py": "same"}})
    st = C.addendum_status()
    st["addendum"]["code_sha256"]["dacapo/ace.py"] = "same"
    st["addendum"]["amendments"] = [{"void_attempt": 1, "failed_checks": ["lossless_audio"], "files": []}]
    assert not C.attempt2_allowed(C.read_json(C.void_path()), st)[0]     # seeds.py changed, not listed
    st["addendum"]["amendments"][0]["files"] = ["dacapo/seeds.py"]
    assert C.attempt2_allowed(C.read_json(C.void_path()), st) == (True, "")


def test_tiny_scale_labels_a_void_second_attempt_as_such(env, monkeypatch):
    monkeypatch.setenv("DACAPO_SCALE", "tiny")
    C.write_json(C.void_path(), {"attempt": 1, "phase": "p0b", "failed_checks": ["per-run hard check"],
                                 "addendum_sha256": "old", "registered_sha256": C.addendum_status()["addendum"]["code_sha256"]})
    _amend(env, ["per-run hard check"], [])
    _a0_passed("tiny")
    st = S.p0b(Fake(env["tmp"], {"run_analogy.py": [5]}))
    assert st["state"] == "ok" and st["k1"] is False and "p0b_integrity" in st["detail"]
    assert (C.results_dir() / "e5_terminal_tiny_p0b_integrity.json").exists()


def test_setup_refuses_unregistered_code(env):
    reg = env["e5"] / "registration" / "addendum_2.json"
    a = json.loads(reg.read_text())
    a["code_sha256"]["dacapo/seeds.py"] = "0" * 64
    reg.write_text(json.dumps(a))
    st = S.setup(Fake(env["tmp"], {}), 1)
    assert st["state"] == "technical_stop" and st["reason"] == "unregistered code"


SETUP_SCRIPT = {"make_items.py": [0, 0], "score_analogy.py:regress-v1": [0], "score_analogy.py:measurement": [0],
                "score_analogy.py:gate": [0, 0, 0]}


@pytest.mark.parametrize("break_k0", ["failed", "missing", "addendum disagrees"])
def test_setup_stops_unless_the_registered_flute_k0_file_passed(env, monkeypatch, break_k0):
    if break_k0 == "failed":
        _k0_file(env["e5"], 73, False)
    elif break_k0 == "missing":
        (env["e5"] / "results" / "e5_ruler_gate_p73.json").unlink()
    else:
        reg = env["e5"] / "registration" / "addendum_2.json"
        a = json.loads(reg.read_text())
        a["k0"]["73"]["passed"] = False
        reg.write_text(json.dumps(a))
    st = S.setup(Fake(env["tmp"], SETUP_SCRIPT), 1)
    assert st["state"] == "technical_stop" and any("flute" in d for d in st["detail"])
    monkeypatch.setenv("DACAPO_SCALE", "tiny")
    st = S.setup(Fake(env["tmp"], SETUP_SCRIPT), 1)
    assert st["state"] == "ok" and any("flute" in d for d in st["detail"])


def _setup_world(monkeypatch, git=(), free=10**13):
    import collections
    import run_analogy
    du = collections.namedtuple("du", "total used free")
    monkeypatch.setattr(run_analogy.shutil, "disk_usage", lambda p: du(10**13, 0, free))
    monkeypatch.setattr(S, "_git_changes", lambda files: list(git))

    def side(a, rc):
        if a[0] == "score_analogy.py" and a[1] == "gate":
            out = Path(a[a.index("--out") + 1])
            C.write_json(out, {"results": {"summary": {"passed": True, "overall": 1.0}}})
    return side


@pytest.mark.parametrize("problem", [None, "git", "space"])
def test_setup_at_full_scale_needs_committed_files_and_room_on_the_state_disk(env, monkeypatch, problem):
    side = _setup_world(monkeypatch, git=["dacapo/seeds.py"] if problem == "git" else (),
                        free=10**9 if problem == "space" else 10**13)
    st = S.setup(Fake(env["tmp"], SETUP_SCRIPT, side=side), 1)
    if problem is None:
        assert st["state"] == "ok" and st["detail"] == []
    else:
        assert st["state"] == "technical_stop" and len(st["detail"]) == 1
        assert ("not committed" if problem == "git" else "disk space") in st["detail"][0]


def test_setup_stops_when_the_measurement_software_differs(env):
    script = dict(SETUP_SCRIPT, **{"score_analogy.py:measurement": [2]})
    st = S.setup(Fake(env["tmp"], script), 1)
    assert st["state"] == "technical_stop" and any("measurement" in d for d in st["detail"])


def test_a_failed_second_timbre_drops_its_cells_only(env):
    _k0_file(env["e5"], 71, False)
    assert C.registered_k0(71)["passed"] is False and C.registered_k0(73)["passed"] is True
    assert C.k0_excluded_cells("full") == tuple(c for c in C.CELLS if c.endswith("clarinet"))
    assert C.k0_excluded_cells("tiny") == ()
    reg = env["e5"] / "registration" / "addendum_2.json"
    a = json.loads(reg.read_text())
    a["k0"]["54"]["passed"] = False                                      # file and addendum disagree: not passed
    reg.write_text(json.dumps(a))
    assert C.registered_k0(54) == {"program": 54, "file_exists": True, "addendum_agrees": False, "passed": False}


def test_echo_hides_ace_step_records_and_prompt_dumps():
    e = S.Echo()
    lines = ["2026-10-27 13:21:06.893 | INFO     | x:y:122 - text_prompt:\n", "# Instruction\n", "Repaint ...\n",
             "2026-10-27 13:21:06.893 | INFO     | x:y:125 - =====\n", "  mel00__lit.flac   74.8s\n",
             "2026-10-27 13:21:06.893 | ERROR    | x:y:9 - out of memory\n", "WARNING:root:Coremltools is not installed\n",
             "\n", "Traceback (most recent call last):\n"]
    assert [l for l in lines if e.show(l)] == ["  mel00__lit.flac   74.8s\n", lines[5], "Traceback (most recent call last):\n"]


def test_run_all_keeps_each_scale_in_its_own_folders(tmp_path):
    import run_all
    tiny, full = run_all.Paths("tiny", str(tmp_path)), run_all.Paths("full", str(tmp_path))
    keys = ("E5_RUN", "E6_OUT", "E6_RES", "E6_A0", "ACESTEP_PROJECT_ROOT")
    a = {v for k, v in tiny.env().items() if k in keys}
    b = {v for k, v in full.env().items() if k in keys}
    assert not a & b and len(a) == len(b) == len(keys)
    assert all(str(tmp_path / "tiny") in tiny.env()[k] for k in keys[:4]) and all(str(tmp_path / "full") in full.env()[k] for k in keys[:4])
    assert tiny.env()["DACAPO_SCALE"] == "tiny" and full.env()["DACAPO_SCALE"] == "full"
    assert tiny.status != full.status
    assert tiny.key.parent == tmp_path / "tiny" and not str(tiny.key).startswith(str(tiny.pkg))   # never zipped


def test_package_withholds_confirmatory_audio_until_the_terminal_outcome(tmp_path):
    import run_all
    p = run_all.Paths("full", str(tmp_path))
    for a in (1, 2):
        d = p.e5_run / ("outputs_v2_main" + ("" if a == 1 else "_a2"))
        d.mkdir(parents=True)
        (d / "design.json").write_text("{}")
    assert run_all.e5_main_dir(p) is None
    (p.e5_run / "results").mkdir()
    (p.e5_run / "results" / "e5_terminal.json").write_text(json.dumps({"stage": "score", "attempt": 2}))
    assert run_all.e5_main_dir(p).name == "outputs_v2_main_a2"            # never the void attempt 1


def test_stage_minutes_add_up_across_resets(tmp_path):
    import os
    import run_all
    log = tmp_path / "main.log"
    log.write_text("x")
    os.utime(log, (1000.0 + 150, 1000.0 + 150))
    assert run_all.earlier_minutes({"state": "running", "started_epoch": 1000.0, "minutes_total": 1.0}, log) == 3.5
    assert run_all.earlier_minutes({"state": "ok", "minutes": 2.0}, log) == 2.0
    assert run_all.earlier_minutes({}, log) == 0.0
