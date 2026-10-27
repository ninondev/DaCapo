"""The E5 v2 runner on CPU with ACE-Step replaced by a stub (no model): design, ledger, passes, resumes,
lost audio, refusals and the hard per-run check. Each run is a subprocess, as run_all starts it."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5_motif"
sys.path.insert(0, str(E5))
from dacapo import motif_integrity as I  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not installed")

FAKE = '''
import os, signal, sys, types, hashlib
import numpy as np
mode = os.environ.get("FAKE_ACE")
if mode:
    import soundfile as sf
    m = types.ModuleType("dacapo.ace")
    def setup_determinism():
        os.environ["ACESTEP_VAE_DECODE_CHUNK_SIZE"] = "256"
        return {"fake": True}
    m.setup_determinism = setup_determinism
    m.load = lambda device: object()
    m.runtime_facts = lambda h: {"attn_implementation": os.environ.get("FAKE_ATTN", "sdpa")}
    def generate(h, out_wav, caption, duration, seed, steps=8, shift=3.0, src=None, extra=None, rng_seed=None,
                 out_subtype=None, **kw):
        with open(os.environ["FAKE_ACE_LOG"], "a") as fh:
            fh.write(f"{os.path.basename(src)}|{seed}|{rng_seed}\\n")
        if mode.startswith("raise:") and mode[6:] in src:
            raise TimeoutError("Music generation timed out after 600 seconds.")
        if mode.startswith("die:") and mode[4:] in src:
            os.kill(os.getpid(), signal.SIGKILL)
        if mode.startswith("nofile:") and mode[7:] in src:
            return {}
        n = sf.info(src).frames
        rng = np.random.default_rng(seed + (1 if "ctl" in src else 0))
        sf.write(str(out_wav), (rng.standard_normal(((n // 1920) * 1920, 2)) * 1000).astype(np.int16), 48000,
                 subtype=out_subtype)
        ins = extra["instruction"]
        return {"seconds": 0.0, "instruction": ins, "instruction_logged": "other" if mode == "badinstr" else ins,
                "dcw_enabled": False, "dcw_active_lines": 0, "frames": n // 1920, "duration_meta": int(n / 48000),
                "repaint_cache_used": 0, "dit_seed": seed, "rng_seed": rng_seed, "decode_chunk": 256, "trace": []}
    m.generate = generate
    sys.modules["dacapo.ace"] = m
    import dacapo
    dacapo.ace = m
'''


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    """Rendered mel00 and v2c00 sources (both lengths), shared by every test."""
    import make_items
    root = tmp_path_factory.mktemp("src")
    make_items.render_from_manifest(E5 / "manifest_p0b.json", root, ["mel00", "mel01"])
    return root


def _setup(tmp: Path, sources: Path, a0=True, files=("dacapo/seeds.py",)) -> dict:
    e5, run, fake = tmp / "e5", tmp / "run", tmp / "fake"
    for d in (e5 / "registration", run / "results", fake):
        d.mkdir(parents=True, exist_ok=True)
    for f in ("manifest_p0b.json", "manifest_v2.json"):
        shutil.copy(E5 / f, e5 / f)
    for d in ("sources", "sources_t18"):
        shutil.copytree(sources / d, e5 / d)
    shutil.copy(E5 / "registration" / "e5v2_seeds.json", e5 / "registration" / "e5v2_seeds.json")
    add = {"addendum_utc": "2026-10-27T00:00:00Z",
           "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
           "k0": {p: {"passed": True} for p in ("73", "54", "71")}}
    (e5 / "registration" / "addendum_2.json").write_text(json.dumps(add))
    if a0:
        _a0(run, e5)
    (fake / "sitecustomize.py").write_text(FAKE)
    env = {**os.environ, "E5_DIR": str(e5), "E5_RUN": str(run), "DACAPO_ROOT": str(ROOT), "DACAPO_SCALE": "tiny",
           "PYTHONPATH": f"{fake}{os.pathsep}{ROOT}", "FAKE_ACE": "ok", "FAKE_ACE_LOG": str(tmp / "calls.log")}
    return {"e5": e5, "run": run, "env": env, "calls": tmp / "calls.log"}


def _a0(run: Path, e5: Path, **kw) -> None:
    sha = hashlib.sha256((e5 / "registration" / "addendum_2.json").read_bytes()).hexdigest()
    rec = {"go": True, "scale": "tiny", "addendum_sha256": sha, "facts": {"attn_implementation": "sdpa", "setup": {}}}
    (run / "results" / "e5v2_a0.json").write_text(json.dumps({**rec, **kw}))


def _run(w, *args, **env) -> int:
    p = subprocess.run([sys.executable, str(E5 / "run_analogy.py"), "--device", "cpu", *args],
                       env={**w["env"], **env}, capture_output=True, text=True)
    (w["run"] / "runner.log").open("a").write(f"$ {args} {env} -> {p.returncode}\n{p.stdout}\n{p.stderr[-3000:]}\n")
    return p.returncode


def _calls(w) -> list[str]:
    return w["calls"].read_text().splitlines() if w["calls"].exists() else []


def _events(w, phase="p0b"):
    return I.read_ledger(w["run"] / f"outputs_v2_{phase}" / "ledger.jsonl")


def test_first_pass_then_resume_generates_nothing(tmp_path, sources):
    w = _setup(tmp_path, sources)
    assert _run(w, "--phase", "p0b") == 0, (w["run"] / "runner.log").read_text()
    assert len(_calls(w)) == 2
    d = json.loads((w["run"] / "outputs_v2_p0b" / "design.json").read_text())
    assert [j["cond"] for j in d["jobs"]] == ["lit", "ctl"] and d["arm_order"] == ["rep-rend-flute"]
    assert d["scale"] == "tiny" and d["addendum_sha256"]
    assert _run(w, "--phase", "p0b") == 0 and len(_calls(w)) == 2
    ev = [e["event"] for e in _events(w)]
    assert ev.count("done") == 2 and ev.count("restart") == 1 and ev[0] == "pass"
    st = I.draw_status(d, _events(w), w["run"] / "outputs_v2_p0b")
    assert set(st.values()) == {"ok"}


def test_lost_audio_before_the_freeze_is_generated_again(tmp_path, sources):
    w = _setup(tmp_path, sources)
    assert _run(w, "--phase", "p0b") == 0
    od = w["run"] / "outputs_v2_p0b"
    wav = sorted(od.glob(f"*__lit__*{I.AUDIO_EXT}"))[0]
    wav.unlink()
    assert _run(w, "--phase", "p0b") == 0 and len(_calls(w)) == 3
    ev = _events(w)
    assert [e["event"] for e in ev].count("lost") == 1
    st, _ = I.ledger_state(ev)
    assert I.sha256_file(wav) == st[I.job_key(next(e for e in ev if e["event"] == "lost"))]["done"][-1]["wav_sha256"]
    d = json.loads((od / "design.json").read_text())
    assert I.check_ledger(d, ev, {I.job_key(j): I.sha256_file(od / I.audio_name(j)) for j in d["jobs"]})["ok"]


def test_a_wav_without_a_done_event_is_moved_aside(tmp_path, sources):
    w = _setup(tmp_path, sources)
    assert _run(w, "--phase", "p0b") == 0
    od = w["run"] / "outputs_v2_p0b"
    lines = (od / "ledger.jsonl").read_text().splitlines()
    keep = [l for l in lines if not (json.loads(l)["event"] == "done" and json.loads(l)["cond"] == "ctl")]
    (od / "ledger.jsonl").write_text("\n".join(keep) + "\n")              # the done line never reached the disk
    assert _run(w, "--phase", "p0b") == 0 and len(_calls(w)) == 3      # ctl's start had no outcome: run again
    ev = [e["event"] for e in _events(w)]
    assert ev.count("orphan") == 1 and ev.count("interrupted") == 1 and ev.count("resume") == 0
    assert list((od / "_orphans").glob(f"*{I.AUDIO_EXT}"))


def test_errors_exit_75_and_each_job_has_at_most_three_attempts(tmp_path, sources):
    w = _setup(tmp_path, sources)
    bad = {"FAKE_ACE": "raise:mel00_lit"}
    assert _run(w, "--phase", "p0b", **bad) == 75                       # lit fails, the process ends at once
    assert _run(w, "--phase", "p0b", **bad) == 4                        # same pass: lit skipped, ctl done
    assert _run(w, "--phase", "p0b", "--new-pass", **bad) == 75         # resume pass 1
    assert _run(w, "--phase", "p0b", **bad) == 4
    assert _run(w, "--phase", "p0b", "--new-pass", **bad) == 75         # resume pass 2: third attempt
    assert _run(w, "--phase", "p0b", **bad) == 4
    assert _run(w, "--phase", "p0b", "--new-pass", **bad) == 2          # no third resume pass
    st, n_resume = I.ledger_state(_events(w))
    lit = next(k for k in st if k[1] == "lit")
    assert st[lit]["starts"] == 3 and st[lit]["errors"] == 3 and n_resume == 2
    assert len(_calls(w)) == 4


def test_refusals_before_any_generation(tmp_path, sources):
    w = _setup(tmp_path, sources, a0=False)
    assert _run(w, "--phase", "p0b") == 2                               # A0 has not passed
    _a0(w["run"], w["e5"], scale="full")
    assert _run(w, "--phase", "p0b") == 2                               # A0 passed at another scale
    _a0(w["run"], w["e5"])
    add = json.loads((w["e5"] / "registration" / "addendum_2.json").read_text())
    (w["e5"] / "registration" / "addendum_2.json").write_text(json.dumps({**add, "code_sha256": {"dacapo/seeds.py": "0" * 64}}))
    assert _run(w, "--phase", "p0b") == 2                               # unregistered code
    (w["e5"] / "registration" / "addendum_2.json").write_text(json.dumps(add))
    seeds = json.loads((w["e5"] / "registration" / "e5v2_seeds.json").read_text())
    seeds["p0b"]["mel00"][0][0] += 1
    (w["e5"] / "registration" / "e5v2_seeds.json").write_text(json.dumps(seeds))
    assert _run(w, "--phase", "p0b") == 2                               # seeds differ from the committed table
    shutil.copy(E5 / "registration" / "e5v2_seeds.json", w["e5"] / "registration" / "e5v2_seeds.json")
    src = w["e5"] / "sources" / "mel00_lit.wav"                         # the tiny cell uses the rendered length
    keep = src.read_bytes()
    src.unlink()
    assert _run(w, "--phase", "p0b") == 2                               # a source is not rendered
    import numpy as np
    import soundfile as sf
    sf.write(str(src), np.zeros((980800, 2), np.int16), 48000, subtype="PCM_16")
    assert _run(w, "--phase", "p0b") == 2 and _calls(w) == []          # a source differs from the manifest
    src.write_bytes(keep)
    (w["run"] / "results" / "e5_terminal.json").write_text("{}")
    assert _run(w, "--phase", "p0b") == 2                               # a terminal outcome exists
    (w["run"] / "results" / "e5_terminal.json").unlink()
    (w["run"] / "results" / "e5v2_p0b_quality.json").write_text("{}")
    assert _run(w, "--phase", "p0b") == 2                               # the phase is frozen
    (w["run"] / "results" / "e5v2_p0b_quality.json").unlink()
    assert _run(w, "--phase", "p0b", "--cell", "rep-rend-flute") == 2   # --cell is for the main phase
    assert _run(w, "--phase", "main") == 2                              # Phase 0b selected nothing
    assert _run(w, "--phase", "p0b", "--attempt", "2") == 2             # no void attempt 1
    assert _calls(w) == []


def test_main_phase_at_the_tiny_scale_can_name_its_cell(tmp_path, sources):
    w = _setup(tmp_path, sources)
    assert _run(w, "--phase", "main", "--cell", "rep-t18-clarinet") == 0
    d = json.loads((w["run"] / "outputs_v2_main" / "design.json").read_text())
    assert d["config"]["cell"] == "rep-t18-clarinet" and d["ruler_program"] == 71 and len(d["jobs"]) == 2
    assert [j["src"] for j in d["jobs"]] == ["sources_t18/mel01_lit.wav", "sources_t18/mel01_ctl.wav"]
    assert d["tag"] == "e5v2-tiny-main" and d["manifest"] == "manifest_p0b.json"
    w["env"]["DACAPO_SCALE"] = "full"
    assert _run(w, "--phase", "main", "--cell", "rep-t18-clarinet") == 2


def test_a_broken_configuration_voids_the_attempt(tmp_path, sources):
    w = _setup(tmp_path, sources)
    assert _run(w, "--phase", "p0b", FAKE_ACE="badinstr") == 5
    ev = _events(w)
    assert [e["event"] for e in ev].count("check_failed") == 1 and not [e for e in ev if e["event"] == "done"]
    assert not list((w["run"] / "outputs_v2_p0b").glob(f"*{I.AUDIO_EXT}"))


def test_a0_from_another_addendum_or_runtime_asks_for_a_new_a0(tmp_path, sources):
    w = _setup(tmp_path, sources)
    _a0(w["run"], w["e5"], addendum_sha256="0" * 64)
    assert _run(w, "--phase", "p0b") == 7 and _calls(w) == []
    _a0(w["run"], w["e5"])
    assert _run(w, "--phase", "p0b", FAKE_ATTN="eager") == 7 and _calls(w) == []
    assert _run(w, "--phase", "p0b") == 0 and len(_calls(w)) == 2


def test_a_process_killed_mid_generation_does_not_use_an_attempt(tmp_path, sources):
    """A runtime reset (here SIGKILL) during a generation: no outcome was seen, so the next process runs
    that job again in the same pass and the start is not counted."""
    w = _setup(tmp_path, sources)
    assert _run(w, "--phase", "p0b", FAKE_ACE="die:mel00_ctl") == -9
    assert _run(w, "--phase", "p0b") == 0
    ev = _events(w)
    assert [e["event"] for e in ev].count("interrupted") == 1 and [e["event"] for e in ev].count("resume") == 0
    st, n_resume = I.ledger_state(ev)
    ctl = next(k for k in st if k[1] == "ctl")
    assert st[ctl]["starts"] == 1 and st[ctl]["not_counted"] == 1 and len(st[ctl]["done"]) == 1 and n_resume == 0


def test_a_crash_outside_the_generation_call_is_not_an_attempt(tmp_path, sources):
    w = _setup(tmp_path, sources)
    assert _run(w, "--phase", "p0b", FAKE_ACE="nofile:mel00_lit") == 1   # the copy to the run folder fails
    assert _run(w, "--phase", "p0b") == 0
    st, _ = I.ledger_state(_events(w))
    lit = next(k for k in st if k[1] == "lit")
    assert st[lit]["starts"] == 1 and st[lit]["errors"] == 0 and st[lit]["not_counted"] == 1


def test_five_errors_in_a_row_halt_generation_and_are_not_counted(tmp_path, sources):
    w = _setup(tmp_path, sources)
    bad = {"FAKE_ACE": "raise:mel00"}
    codes = [_run(w, "--phase", "p0b", **bad), _run(w, "--phase", "p0b", **bad), _run(w, "--phase", "p0b", **bad),
             _run(w, "--phase", "p0b", "--new-pass", **bad), _run(w, "--phase", "p0b", **bad), _run(w, "--phase", "p0b", **bad),
             _run(w, "--phase", "p0b", "--new-pass", **bad), _run(w, "--phase", "p0b", **bad)]
    assert codes == [75, 75, 4, 75, 75, 4, 75, 6]                         # the eighth process halts before loading
    ev = _events(w)
    halt = next(e for e in ev if e["event"] == "halt")
    assert len(halt["jobs"]) == 5 and "TimeoutError" in halt["errors"][0]
    st, _ = I.ledger_state(ev)
    assert all(s["starts"] == 0 for s in st.values())                    # the five errors used no attempt
    assert _run(w, "--phase", "p0b") == 0 and len(_calls(w)) == 7        # fixed: both jobs run in the same pass


def test_space_shortfall_is_refused_before_generating(tmp_path, sources, monkeypatch):
    import collections
    import run_analogy
    w = _setup(tmp_path, sources)
    for k, v in w["env"].items():
        if k in ("E5_DIR", "E5_RUN", "DACAPO_ROOT", "DACAPO_SCALE"):
            monkeypatch.setenv(k, v)
    du = collections.namedtuple("du", "total used free")
    monkeypatch.setattr(run_analogy.shutil, "disk_usage", lambda p: du(10**13, 0, 10**9))
    assert "GB needed" in run_analogy.space_shortfall(w["run"], 2, "p0b")
    monkeypatch.setattr(run_analogy.shutil, "disk_usage", lambda p: du(10**13, 0, 10**12))
    assert run_analogy.space_shortfall(w["run"], 2, "p0b") is None
    monkeypatch.setattr(run_analogy.shutil, "disk_usage", lambda p: du(0, 0, 0))
    assert run_analogy.space_shortfall(w["run"], 2, "p0b") is None     # a mount that reports no size
