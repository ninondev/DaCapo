"""E5 v2 stages of Run A, applied without a person: setup, A0, Phase 0b (K1), and the confirmatory grid
with its quality and accuracy stages. run_all calls one function per stage; each returns a status dict.

States: ok; technical_stop (setup, A0, a refusal, a crash or a halted generation: fixed and run again, not
a strike, and no resume pass is used); terminal (the E5 v2 outcome is written to results/e5_terminal.json:
K1, a failed Q1/Q2, a void second attempt, or the scored result); void (attempt 1 failed an integrity
check before any accuracy existed: attempt 2 needs a fix recorded in an amended addendum); blocked (an
earlier stage did not pass).

Generation and scoring never interleave: a phase's quality stage runs only when its generation can do no
more, and the accuracy stage only after the frozen quality decision passed. A phase whose quality stage has
frozen generates nothing more; Run all only reads its outcome back. At the tiny scale an outcome that stops
the registered run is recorded and the chain goes on, so every stage runs once.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_config as C  # noqa: E402
from dacapo.motif_integrity import read_ledger  # noqa: E402

# run_analogy exit codes
R_DONE, R_REFUSED, R_PENDING, R_VOID, R_HALT, R_A0_STALE, R_RETRY = 0, 2, 4, 5, 6, 7, 75
MAX_RELAUNCH = 1344 + 10   # a relaunch follows a raised generation, which uses that job's start in the pass
MAX_ROUNDS = 1 + C.MAX_RESUME_PASSES + 3
RUNNER_STOPS = {R_REFUSED: "the runner refused before generating (see the stage log)",
                R_HALT: f"{C.HALT_AFTER_ERRORS} generations in a row raised an error; generation halted (see the ledger)",
                R_A0_STALE: "the runtime or addendum still differs from the one A0 checked"}

_LOG_LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ \| (\w+)\s*\|[^-]*- ?(.*)$")
_QUIET = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING")


class Echo:
    """Which subprocess lines reach the notebook (the stage log keeps every line): no ACE-Step log
    records below ERROR, none of the multi-line prompt dumps that follow them, no blank lines."""

    def __init__(self):
        self.block = False

    def show(self, line: str) -> bool:
        m = _LOG_LINE.match(line.rstrip("\n"))
        if m:
            quiet = m.group(1) in _QUIET
            msg = m.group(2).strip()
            self.block = quiet and (msg == "" or msg.endswith(":"))
            return not quiet
        if line.startswith("WARNING:root:") or not line.strip():
            return False
        return not self.block


class Runner:
    """Runs commands with the right interpreter, streaming output to the stage log."""

    def __init__(self, log: Path, py_main: str, py_ace: str, device: str, env: dict | None = None):
        self.log, self.py_main, self.py_ace, self.device = Path(log), py_main, py_ace, device
        self.env = {**os.environ, **(env or {})}
        self.log.parent.mkdir(parents=True, exist_ok=True)

    def say(self, msg: str) -> None:
        with self.log.open("a") as fh:
            fh.write(msg + "\n")
        print(f"[e5] {msg}", flush=True)

    def call(self, args: list[str], ace: bool = False, cwd: Path = HERE) -> int:
        cmd = [self.py_ace if ace else self.py_main, *args]
        echo = Echo()
        with self.log.open("a") as fh:
            fh.write(f"\n$ (cd {cwd}) {' '.join(cmd)}\n")
            fh.flush()
            p = subprocess.Popen(cmd, cwd=cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line)
                if echo.show(line):
                    print(f"[e5] {line}", end="", flush=True)
            p.wait()
        return p.returncode


def _status(state: str, **kw) -> dict:
    return {"state": state, "scale": C.scale(), "utc": C.utc(), **kw}


def _terminal_status() -> dict | None:
    t = C.read_json(C.terminal_path())
    return None if t is None else _status("terminal", paper_level=t["paper_level"], strike=t.get("strike"), stage=t.get("stage"))


def _tiny_continue(r: Runner, what: str) -> None:
    """Test scales only: keep the outcome on record and let the next stage run."""
    t = C.terminal_path()
    if t.exists():
        dst = t.with_name(f"e5_terminal_{C.scale()}_{what}.json")
        os.replace(t, dst)
        r.say(f"{C.scale()} scale: the registered run would stop here ({what}); outcome kept in {dst.name}, chain continues")


def attempt_now(r: Runner) -> int | None:
    """1, or 2 after a void attempt 1 once an amendment links the fix to it; None if E5 must wait."""
    v = C.read_json(C.void_path())
    if v is None:
        return 1
    ok, why = C.attempt2_allowed(v, C.addendum_status())
    if ok:
        return 2
    r.say(f"attempt 1 is void; attempt 2 needs an amendment of addendum 2 ({why}), then register.py")
    return None


def _git_changes(files: list[str]) -> list[str] | None:
    """Registered files with uncommitted changes (or untracked); None if git cannot tell."""
    try:
        out = subprocess.run(["git", "-C", str(C.repo_root()), "status", "--porcelain", "--", *files],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    return [line[3:] for line in out.splitlines() if line.strip()]


def setup(r: Runner, workers: int) -> dict:
    """Check that the registered code, data and measurement software are in place, render the sources and
    hypothesis renders from the committed manifests and check their sha256, rerun the v1 regression, and
    run K0 for every caption timbre on these renders. Not a strike when it fails."""
    st = C.addendum_status()
    rec = {"utc_start": C.utc(), "registration_ok": st["ok"], "changed": st["changed"], "scale": C.scale()}
    if not st["ok"]:
        r.say(f"code differs from addendum 2: {st['changed']}")
        return _status("technical_stop", reason="unregistered code", detail=st["changed"])
    add = st["addendum"] or {}
    registered = sorted({*(add.get("code_sha256") or {}), *(add.get("data_sha256") or {}),
                         "experiments/e5_motif/registration/addendum_2.json"})
    rec["git_uncommitted"] = _git_changes(registered)
    sc = C.SCALES[C.scale()]
    want: dict = {}
    for man, ids in ((sc["p0b_manifest"], sc["p0b_ids"]), (C.P0B_MANIFEST.name, C.A0_IDS), (sc["main_manifest"], sc["main_ids"])):
        want.setdefault(man, set()).update(ids)
    renders = {man: r.call(["make_items.py", "render", "--manifest", man, "--ids", ",".join(sorted(ids))])
               for man, ids in want.items()}
    rec["render_exit_codes"] = renders
    rec["regress_v1"] = r.call(["score_analogy.py", "regress-v1"])
    rec["measurement"] = r.call(["score_analogy.py", "measurement"])
    k0 = {}
    for cap, prog in C.PROGRAMS.items():
        out = C.results_dir() / f"setup_k0_p{prog}.json"
        if not (C.read_json(out) or {}).get("results"):
            r.call(["score_analogy.py", "gate", "--program", str(prog), "--manifest", sc["k0_manifest"],
                    "--ids", ",".join(sc["k0_ids"]), "--out", str(out), "--workers", str(workers)])
        s = ((C.read_json(out) or {}).get("results") or {}).get("summary", {})
        k0[str(prog)] = {"passed": bool(s.get("passed")), "registered": C.registered_k0(prog),
                         "overall": s.get("overall"), "wrong_note_auc": s.get("wrong_note_auc"),
                         "render_mismatch": len(s.get("render_mismatch", [])), "n_items": s.get("n_items")}
    rec["k0"] = k0
    rec["space"] = _space_problem(sc)
    rec["environment"] = _environment()
    rec["utc_end"] = C.utc()
    C.write_json(C.results_dir() / "e5v2_setup.json", rec)
    problems = [f"render {k}" for k, v in renders.items() if v != 0]
    if not k0[str(C.PROGRAMS["flute"])]["registered"]["passed"]:
        problems.append("the registered K0 has not passed for flute, the registered timbre: E5 v2 waits for the "
                        "remedy recorded in addendum 2")
    problems += ["v1 regression"] if rec["regress_v1"] != 0 else []
    problems += ["measurement software differs from addendum 2"] if rec["measurement"] != 0 else []
    problems += [f"K0 program {p} on these renders" for p, v in k0.items() if v["registered"]["passed"] and not v["passed"]]
    problems += [f"disk space: {rec['space']}"] if rec["space"] else []
    if rec["git_uncommitted"] is None or rec["git_uncommitted"]:
        problems.append(f"registered files are not committed: {rec['git_uncommitted']}")
    if problems and C.scale() == "full":
        r.say(f"setup failed: {problems}")
        return _status("technical_stop", reason="setup", detail=problems)
    if problems:
        r.say(f"{C.scale()} scale: setup problems recorded, chain continues: {problems}")
    return _status("ok", detail=problems)


def _space_problem(sc: dict) -> str | None:
    """The runner's free-space rule, applied before A0 so a full Drive stops the run early."""
    from run_analogy import space_shortfall
    man = C.read_json(C.e5_dir() / sc["p0b_manifest"])
    if man is None or not (C.e5_dir() / sc["main_manifest"]).exists():
        return "a manifest is missing"
    C.run_dir().mkdir(parents=True, exist_ok=True)
    return space_shortfall(C.run_dir(), len(C.jobs("p0b", man, C.scale())), "p0b")


def _environment() -> dict:
    out = {}
    try:
        out["fluidsynth"] = subprocess.run(["fluidsynth", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        out["fluidsynth"] = None
    sf2 = Path(os.environ.get("DACAPO_SF2", "/usr/share/sounds/sf2/FluidR3_GM.sf2"))
    out["sf2_sha256"] = C.sha256_file(sf2) if sf2.exists() else None
    out["sf2_registered"] = out["sf2_sha256"] == C.SF2_SHA256
    return out


def _a0_current() -> bool:
    done = C.read_json(C.a0_path())
    return bool(done and done.get("go") and done.get("scale") == C.scale()
                and done.get("addendum_sha256") == C.addendum_status()["addendum_sha256"])


def _ensure_a0(r: Runner) -> dict | None:
    """A status to return when A0 cannot pass now; None when A0 holds for this addendum (it runs again
    after an amendment, or when an earlier rerun did not pass)."""
    if _a0_current():
        return None
    st = a0(r)
    return None if st["state"] == "ok" else st


def a0(r: Runner, force: bool = False) -> dict:
    """A0 generation checks; a failure is a technical stop, not a strike. A0 counts only for the addendum
    and the runtime it checked: the runner asks for a new A0 (force) when either differs."""
    if not force and _a0_current():
        return _status("ok", detail="A0 passed earlier")
    if C.a0_path().exists():
        os.replace(C.a0_path(), C.a0_path().with_name(f"e5v2_a0_superseded_{int(time.time())}.json"))
    rc = r.call(["a0_checks.py", "--part", "main", "--device", r.device], ace=True)
    if rc == 0 and C.scale() == "full":
        rc = r.call(["a0_checks.py", "--part", "fresh", "--device", r.device], ace=True)
    if rc != 0:
        return _status("technical_stop", reason="A0 could not run", returncode=rc)
    rc = r.call(["a0_checks.py", "--part", "verdict"])
    return _status("ok") if rc == 0 else _status("technical_stop", reason="A0 hard check failed")


def _generate(r: Runner, phase: str, attempt: int, extra: list[str]) -> tuple[int | None, str]:
    """Generation, then the quality stage, for one phase and attempt. Returns (0, "") when the quality rule
    passed, (3, "") for a terminal outcome, (5, "") for a void attempt 1, or (None, reason) for a technical
    stop. The runner starts again after every exit 75; any exit other than 0 or 4 stops before the quality
    stage, so a crash never uses a resume pass. A quality exit 4 means generation can still do something:
    lost audio is repaired inside the current pass, never-completed draws get an automatic resume pass (at
    most two). A frozen phase generates nothing: only its outcome is read back."""
    q_args = ["score_analogy.py", "quality", phase, "--attempt", str(attempt)]
    if C.quality_path(phase, attempt).exists():
        q = r.call(q_args)
        return (q, "") if q in (0, 3, 5) else (None, f"the frozen quality stage ended with code {q}")
    new_pass, verify, a0_again = False, True, True
    for _ in range(MAX_ROUNDS):
        relaunches = 0
        while True:
            rc = r.call(["run_analogy.py", "--phase", phase, "--device", r.device, "--attempt", str(attempt), *extra]
                        + (["--new-pass"] if new_pass else []) + (["--verify-audio"] if verify else []), ace=True)
            new_pass = verify = False
            if rc == R_A0_STALE and a0_again:
                a0_again = False
                r.say("A0 runs again: the addendum or the runtime differs from the one it checked")
                if a0(r, force=True)["state"] != "ok":
                    return None, "A0 ran again for a changed runtime or addendum and did not pass"
                continue
            if rc != R_RETRY or relaunches >= MAX_RELAUNCH:
                break
            relaunches += 1
        if rc == R_VOID:
            ev = [e for e in read_ledger(C.phase_dir(phase, attempt) / "ledger.jsonl") if e.get("event") == "check_failed"]
            return (3 if C.record_void(phase, attempt, ["per-run hard check"], ev[-1:]) else 5), ""
        if rc not in (R_DONE, R_PENDING):
            return None, RUNNER_STOPS.get(rc, f"the runner ended with code {rc} (a crash; see the stage log)")
        q = r.call(q_args)
        if q != 4:
            return (q, "") if q in (0, 3, 5) else (None, f"the quality stage ended with code {q}")
        inc = C.read_json(C.results_dir() / f"e5v2_{phase}_incomplete.json") or {}
        new_pass = not inc.get("lost")
    return None, "generation did not finish within the round limit"


def _terminal_label() -> str:
    stage = (C.read_json(C.terminal_path()) or {}).get("stage", "terminal")
    return {"K1_phase0b": "K1", "main_quality": "Q"}.get(stage, stage)


def p0b(r: Runner) -> dict:
    t = _terminal_status()
    if t:
        return t
    attempt = attempt_now(r)
    if attempt is None:
        return _status("void", reason="attempt 1 void, fix not recorded")
    void = C.read_json(C.void_path())
    sel = C.read_json(C.selection_path())
    if sel and sel.get("decision") == "selected" and not (attempt == 2 and void and void["phase"] == "p0b"):
        return _status("ok", detail=f"Phase 0b selected {sel['cell']} earlier")
    p_attempt = attempt if (void and void["phase"] == "p0b") else 1
    if not C.quality_path("p0b", p_attempt).exists():
        stop = _ensure_a0(r)
        if stop:
            return stop
    rc, why = _generate(r, "p0b", p_attempt, [])
    if rc is None:
        return _status("technical_stop", reason=why)
    sel = C.read_json(C.selection_path()) or {}
    if rc == 0:
        return _status("ok", cell=sel.get("cell"), attempt=p_attempt)
    if rc == 3:
        if C.scale() != "full":
            label = _terminal_label()
            _tiny_continue(r, label)
            return _status("ok", detail=f"{C.scale()} scale: terminal outcome {label} recorded", k1=label == "K1")
        return _terminal_status() or _status("terminal")
    return _status("void", phase="p0b", attempt=p_attempt)


def main_grid(r: Runner, n_boot: int | None = None) -> dict:
    t = _terminal_status()
    if t:
        return t
    attempt = attempt_now(r)
    if attempt is None:
        return _status("void", reason="attempt 1 void, fix not recorded")
    sel = C.read_json(C.selection_path()) or {}
    extra = []
    if sel.get("decision") != "selected":
        if C.scale() == "full":
            return _status("blocked", reason=f"Phase 0b decision {sel.get('decision')}")
        extra = ["--cell", C.SCALES[C.scale()]["p0b_cells"][0]]
    if not C.quality_path("main", attempt).exists():
        stop = _ensure_a0(r)
        if stop:
            return stop
    rc, why = _generate(r, "main", attempt, extra)
    if rc is None:
        return _status("technical_stop", reason=why)
    boot = ["--n-boot", str(n_boot)] if n_boot else []
    if rc == 0:
        s = r.call(["score_analogy.py", "score", "main", "--attempt", str(attempt), *boot])
        if s == 0:
            return _terminal_status() or _status("ok")
        if s == 5:
            return _status("void", phase="main", attempt=attempt)
        return _terminal_status() or _status("technical_stop", reason=f"score ended with code {s}")
    if rc == 3:
        if C.scale() != "full":
            label = _terminal_label()
            s = r.call(["score_analogy.py", "score", "main", "--attempt", str(attempt), "--diagnostic", *boot]) \
                if label == "Q" else None
            r.say(f"{C.scale()} scale: terminal outcome {label}; diagnostic scoring exit {s}")
            return _status("ok", detail=f"{C.scale()} scale: terminal outcome {label} recorded", diagnostic_exit=s)
        return _terminal_status() or _status("terminal")
    return _status("void", phase="main", attempt=attempt)


def run_b_eligibility() -> dict:
    """Run B's E5 stages need a passed quality gate, a scored result and no K2 trigger (addendum 2, S7)."""
    res = C.read_json(C.results_dir() / "e5v2_main.json")
    if not res or "results" not in res:
        return {"e5_run_b": False, "reason": "no scored E5 v2 result"}
    s = res["results"]["summary"]
    k2 = s.get("K2", {}).get("status")
    ok = s["quality"]["decision"] == "pass" and k2 == "not_triggered" and res["config"].get("scale") == "full"
    return {"e5_run_b": ok, "K2": k2, "K3": s.get("K3_status"), "paper_level": s.get("paper_level")}
