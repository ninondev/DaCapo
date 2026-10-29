"""run_b2.py and run_b.py without a model: select and decide on fake tiny-scale rows (the selection line, the lock
once Stage T has rows, a "test only" verdict), packaging with the pack and key outside the package, the space check,
the A0 timing, the refusals of the full scale, and run_b.py's stop rules, A0 reuse and resume."""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments"))
import plan_b2 as P  # noqa: E402
import run_b2 as RB  # noqa: E402


def args(**kw):
    base = {"scale": "tiny", "device": "cpu", "test_whisper": None, "stage": "dev", "only": "", "package": "", "key": "",
            "pack": "", "part": "", "b1_out": ""}
    return argparse.Namespace(**{**base, **kw})


def fake_run(paths, run, text, npa=0.8):
    """A generated and scored clip: real files with their sha256 in the ledger rows."""
    import soundfile as sf
    out = paths.out(run["stage"])
    out.mkdir(parents=True, exist_ok=True)
    sf.write(str(out / f"{run['run_id']}.wav"), np.zeros((4800, 2), np.float32), 48000, subtype="PCM_16")
    np.save(out / f"{run['run_id']}.pred.npy", np.zeros((3, 64), np.float32))
    np.save(out / f"{run['run_id']}.target.npy", np.ones((3, 64), np.float32))
    rid = run["run_id"]
    g = {**{k: run[k] for k in ("run_id", "stage", "item", "draw", "config", "labels", "guide", "k", "clock", "head_set",
                                "heads", "beta", "shuffle", "hooked", "fallback")},
         "status": "ok", "hook_calls": 40 if run["hooked"] else 0, "max_key": 20 if run["hooked"] else None, "n_lyr": 27,
         "peak_before_norm": 0.5, "wav_sha256": RB.sha256(out / f"{rid}.wav"), "pred_sha256": RB.sha256(out / f"{rid}.pred.npy"),
         "target_sha256": RB.sha256(out / f"{rid}.target.npy"), "guide_latent_sha256": "lat",
         "code_sha256": {}, "test_flags": ["scale tiny"], "a0_sha256": RB.sha256(paths.a0_report)}
    if run["stage"] == "test":
        g["selection_sha256"] = RB.selection_head(paths)["selection_sha256"]
    RB.append_jsonl(paths.ledger(run["stage"]), g)
    words = text.split()
    s = {"run_id": rid, "stage": run["stage"], "wav_sha256": g["wav_sha256"], "stem_share": {"vocals": 0.9, "other": 0.1},
         "vocal_active_s": 4.0, "whisper": "openai/whisper-small.en", "whisper_revision": "x", "whisper_text": text,
         "whisper_chunks": [[" " + w, 2.4 + 0.6 * k, 2.6 + 0.6 * k] for k, w in enumerate(words)], "npa_v": npa,
         "npa_v_oct": npa, "npa_mix": npa, "npa_mix_oct": npa, "ctc_text": text.upper(), "ctc": "c", "ctc_revision": "r",
         "test_flags": ["test whisper openai/whisper-small.en on cpu"]}
    RB.append_jsonl(paths.scores(run["stage"]), s)


@pytest.fixture
def paths(tmp_path):
    p = RB.Paths(str(tmp_path / "b2"), str(tmp_path / "guides"))
    RB.write(p.a0_report, {"pass": True, "recorded": True, "fallback": False, "trim": dict(P.NO_TRIM), "test_flags": []})
    for run in P.dev_runs("tiny"):
        it = P.all_items()[run["item"]]
        fake_run(p, run, it["lyrics"].split("\n", 1)[1].lower(), npa=0.8 if run["clock"] != "none" else 0.3)
    return p


def test_select_writes_the_line_and_locks_once_stage_t_has_rows(paths, capsys):
    RB.select(args(), paths)
    sel = json.loads(paths.selection.read_text())
    assert sel["s_star"] == "oohs_k4_A2" and sel["b_star"] == "oohs_k3_none" and sel["stage_t"]
    head = RB.selection_head(paths)
    assert head["selection_sha256"] == RB.sha256(paths.selection)
    RB.select(args(), paths)
    assert "unchanged" in capsys.readouterr().out
    run = P.test_runs(P.parse(sel["s_star"]), P.parse(sel["b_star"]), "tiny")[0]
    fake_run(paths, run, "the dogs bark")
    dev = P.dev_runs("tiny")[1]
    RB.append_jsonl(paths.scores("dev"), {**[r for r in RB.read_jsonl(paths.scores("dev")) if r["run_id"] == dev["run_id"]][0],
                                          "npa_v": 0.0})
    with pytest.raises(RB.SetupError, match="fixed"):
        RB.select(args(), paths)


def test_stage_t_refuses_without_the_selection_line(paths):
    with pytest.raises(RB.SetupError, match="selection"):
        RB.gen(args(stage="test"), paths)
    RB.select(args(), paths)
    paths.ledger("test").write_text("")
    with pytest.raises(RB.SetupError, match="first line"):
        RB.gen(args(stage="test"), paths)


def test_decide_and_package_on_fake_tiny_rows(paths, tmp_path):
    RB.select(args(), paths)
    sel = json.loads(paths.selection.read_text())
    lyr = P.all_items()["item01"]["lyrics"].split("\n", 1)[1].lower()
    for run in P.test_runs(P.parse(sel["s_star"]), P.parse(sel["b_star"]), "tiny"):
        fake_run(paths, run, lyr if run["clock"] != "none" else "oh oh", npa=0.9)
    RB.decide(args(), paths)
    res = json.loads((paths.res / "b2_result.json").read_text())["results"]
    assert res["verdict"] == "test only" and res["arms"]["S*"]["J"] == 1.0 and res["arms"]["S*_b0"]["J"] == 0.0
    assert res["p0"]["selection"]["held"] and res["p0"]["hooks"]["held"] and res["p0"]["guide_latents"]["held"]
    assert not res["p0"]["hashes"]["held"]      # fake rows record no code hashes
    assert not res["p0"]["reference_and_peak"]["held"]   # scored with a test Whisper, in Stage D and Stage T
    assert res["p0"]["reference_and_peak"]["detail"]["other_reference"] == {"dev": 2, "test": 2}
    assert not res["p0"]["a0"]["detail"]["ran_on_the_registered_code"]
    assert len(res["clips"]) == 2 and res["demo"][0]["s_star_q"] == 0.9
    assert res["reel"] == {"label": "top 1 of 1 by rule", "items": ["item01"]}
    key, pack = tmp_path / "key.json", tmp_path / "pack.zip"
    RB.package(args(package=str(tmp_path / "pkg"), key=str(key), pack=str(pack)), paths)
    out = tmp_path / "pkg" / "b2"
    assert (out / "b2_result.json").exists() and pack.exists() and key.exists() and not (out / "blind_pack.zip").exists()
    assert len(list((out / "demo").glob("item01__*"))) == 2 and not list((tmp_path / "pkg").rglob("*key*"))
    assert json.loads((out / "demo" / "demo.json").read_text())["reel"]["items"] == ["item01"]
    pair = json.loads(key.read_text())["pairs"]["pair_01"]
    assert {pair["A"]["arm"], pair["B"]["arm"]} == {"S*", "S*_b0"} and json.loads(key.read_text())["order_seed"] == 20261029
    sides = {p: (v["A"]["run_id"], v["B"]["run_id"]) for p, v in json.loads(key.read_text())["pairs"].items()}
    for _ in range(4):   # packaging again keeps the sides, so a pack already handed out still matches its key
        RB.package(args(package=str(tmp_path / "pkg"), key=str(key), pack=str(pack)), paths)
        assert {p: (v["A"]["run_id"], v["B"]["run_id"]) for p, v in json.loads(key.read_text())["pairs"].items()} == sides
    with zipfile.ZipFile(pack) as z:   # the pack shares no audio file with the demo
        packed = {hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith((".mp3", ".wav"))}
    demo = {hashlib.sha256(f.read_bytes()).hexdigest() for f in (out / "demo").glob("*") if f.suffix in (".mp3", ".wav")}
    assert packed and demo and not packed & demo
    for bad in ({"key": str(tmp_path / "pkg" / "k.json"), "pack": str(pack)},
                {"key": str(key), "pack": str(tmp_path / "pkg" / "p.zip")}):
        with pytest.raises(ValueError):
            RB.package(args(package=str(tmp_path / "pkg"), **bad), paths)


def test_space_check_counts_what_is_left(paths, tmp_path):
    a = args(b1_out=str(tmp_path / "b1"))
    left = RB.runs_to_generate(paths, a)
    assert left == {"b1": 2, "dev": 0, "test": 2}   # the fixture generated both tiny Stage D runs
    (tmp_path / "b1").mkdir()
    for rid in ("item00_r0_nohook", "item00_r0_align2"):
        (tmp_path / "b1" / f"{rid}.wav").write_bytes(b"x")
    assert RB.runs_to_generate(paths, a)["b1"] == 0
    full = RB.runs_to_generate(RB.Paths(str(tmp_path / "f"), str(tmp_path / "g")), args(scale="full"))
    assert full == {"b1": 180, "dev": 1080, "test": 768}
    RB.select(args(), paths)
    assert RB.runs_to_generate(paths, a)["test"] == 2


def test_a0_timing_by_kind():
    runs = {"t1": {"guide": None, "gen_seconds": 30.0}, "t2": {"guide": None, "gen_seconds": 2.0},
            "c1": {"guide": "oohs", "gen_seconds": 3.0}, "c2": {"guide": "oohs", "gen_seconds": 4.0}}
    rows = {"t1": {"seconds_total": 5.0}, "t2": {"seconds_total": 7.0}, "c1": {"seconds_total": 6.0},
            "c2": {"seconds_total": 50.0}}
    t = RB.a0_timing(runs, list(runs), rows)
    assert t["medians"]["text2music"]["gen_seconds_median"] == 2.0   # the first run warms up and is left out
    assert t["seconds_per_run"]["guided"] == pytest.approx(3.5 + 28.0 + 0.25 * 46, abs=0.01)
    assert t["seconds_per_run"]["text2music"] == pytest.approx(2.0 + 6.0 + 0.08 * 46, abs=0.01)
    only = RB.a0_timing({k: runs[k] for k in ("c1", "c2")}, ["c1", "c2"], rows)   # the tiny scale: no text2music run
    assert only["medians"]["text2music"]["from"] == "guided"
    assert RB.a0_timing(runs, list(runs), {})["seconds_per_run"]["guided"] == float("inf")


def test_resume_counts_a_run_only_while_its_files_hash_to_the_ledger(paths):
    run = P.dev_runs("tiny")[0]
    gens = RB.generated(paths.dev)
    assert gens[run["run_id"]]["status"] == "ok"
    (paths.dev / f"{run['run_id']}.wav").write_bytes(b"changed")
    assert RB.generated(paths.dev)[run["run_id"]]["status"] == "files_missing"
    (paths.dev / f"{run['run_id']}.pred.npy").unlink()
    assert RB.generated(paths.dev)[run["run_id"]]["status"] == "files_missing"
    for k in range(3):
        RB.append_jsonl(paths.ledger("dev"), {"run_id": "x", "status": "failed", "attempt": k + 1})
    assert RB.failed_attempts(paths.dev)["x"] == 3
    with open(paths.ledger("dev"), "ab") as fh:
        fh.write(b'{"run_id": "cut sh')   # a line cut short by a reset
    assert "x" in RB.generated(paths.dev) and RB.read_jsonl(paths.ledger("dev"))[-1]["run_id"] == "x"
    RB.append_jsonl(paths.ledger("dev"), {"run_id": "y", "status": "failed", "attempt": 1})
    assert RB.read_jsonl(paths.ledger("dev"))[-1]["run_id"] == "y"


def test_full_scale_refusals(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "experiments" / "run_b.py"), "--device", "cpu", "--state", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "full scale runs on CUDA" in r.stderr
    r = subprocess.run([sys.executable, str(HERE / "run_b2.py"), "score", "--stage", "dev", "--test-whisper", "openai/whisper-tiny",
                        "--root", str(tmp_path / "x")], capture_output=True, text=True)
    assert r.returncode == 2 and "no test Whisper" in r.stdout
    with pytest.raises(RB.SetupError, match="CUDA"):
        RB.refuse_full_scale_tests(args(scale="full", device="cpu"), needs_cuda=True)


# run_b.py stop rules

import run_b  # noqa: E402


class FakeChain:
    def __init__(self, states: dict):
        self.states, self.calls = states, []

    def _r(self, name):
        self.calls.append(name)
        st = self.states.get(name, "ok")
        return dict(st) if isinstance(st, dict) else {"state": st}

    def setup(self):
        return self._r("setup")

    def a0(self):
        return self._r("a0")

    def b1(self, decide_only=False):
        if decide_only:
            self.calls.append("b1 decide")
            return {"state": "stopped", "verdict": "technical"}
        return self._r("b1")

    def b2_stage(self, stage, which):
        return self._r(stage)

    def select(self):
        return self._r("select")

    def decide(self):
        return self._r("decide")

    def package(self, status):
        return self._r("package")


@pytest.fixture
def rpaths(tmp_path):
    return run_b.Paths("tiny", str(tmp_path))


def go(rpaths, states, status=None, monkeypatch=None):
    chain = FakeChain(states)
    st = run_b.run(chain, rpaths, run_b.STAGES, status or {}, "tiny")
    return chain.calls, st


def test_all_stages_run_in_order(rpaths):
    calls, st = go(rpaths, {})
    assert calls == run_b.STAGES and all(st[s]["state"] == "ok" for s in run_b.STAGES)


@pytest.mark.parametrize("bad", ["setup", "a0"])
def test_setup_or_a0_failure_stops_the_run(rpaths, bad):
    calls, st = go(rpaths, {bad: "technical_stop"})
    assert calls == run_b.STAGES[:run_b.STAGES.index(bad) + 1] + ["package"]
    assert st["decide"]["state"] == "skipped"


def test_b1_failure_does_not_stop_b2(rpaths, monkeypatch):
    monkeypatch.setattr(run_b, "shared_code_problems", lambda: [])
    calls, _ = go(rpaths, {"b1": "failed"})
    assert calls == run_b.STAGES
    monkeypatch.setattr(run_b, "shared_code_problems", lambda: ["dacapo/attn_bias.py differs from addendum 3"])
    calls, st = go(rpaths, {"b1": "failed"})
    assert calls == ["setup", "a0", "b1", "decide", "package"] and st["b2dev"]["state"] == "skipped"


def test_b2_failures_and_no_stage_t(rpaths):
    calls, st = go(rpaths, {"b2dev": "technical_stop"})
    assert calls == ["setup", "a0", "b1", "b2dev", "decide", "package"]
    calls, st = go(rpaths, {"select": {"state": "ok", "stage_t": False}})
    assert "b2test" not in calls and st["b2test"]["state"] == "skipped" and "decide" in calls


def test_resume_skips_finished_stages_but_setup_a0_decide_and_package(rpaths):
    done = {s: {"state": "ok", "scale": "tiny"} for s in ("setup", "a0", "b1", "b2dev")}
    calls, _ = go(rpaths, {}, status=done)
    assert calls == ["setup", "a0", "select", "b2test", "decide", "package"]   # a0 reuses a current report


def ledger_row(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": "item00_r0_nohook", "status": "ok"}) + "\n")


@pytest.mark.parametrize("bad", ["setup", "a0"])
def test_a_failure_after_generation_started_still_decides(rpaths, bad):
    ledger_row(rpaths.ledgers()["b1"])   # B1 cut short by a reset; the next runtime fails setup or A0
    calls, st = go(rpaths, {bad: "technical_stop"})
    assert calls == STAGES_UNTIL(bad) + ["b1 decide", "decide", "package"]
    assert st["b1"]["state"] == "stopped" and st["b2dev"]["state"] == "skipped" and st["decide"]["state"] == "ok"
    done = {s: {"state": "ok", "scale": "tiny"} for s in ("a0", "b1", "b2dev", "select")}
    ledger_row(rpaths.ledgers()["test"])
    calls, st = go(rpaths, {"setup": "technical_stop"}, status=done)
    assert calls == ["setup", "decide", "package"] and st["b1"]["state"] == "ok" and st["b2test"]["state"] == "skipped"


def STAGES_UNTIL(name):
    return run_b.STAGES[:run_b.STAGES.index(name) + 1]


def test_status_is_written_whole(rpaths):
    run_b.write_json(rpaths.status, {"setup": {"state": "ok"}})
    assert json.loads(rpaths.status.read_text()) == {"setup": {"state": "ok"}}
    assert not list(rpaths.root.glob("*.tmp"))


def test_b1_counts_as_done_only_when_gen_score_and_decide_pass(rpaths, monkeypatch):
    chain = run_b.Chain(rpaths, "cpu")
    codes = iter([1, 0, 0])
    monkeypatch.setattr(chain, "call", lambda *a, **k: next(codes))
    assert chain.b1()["state"] == "failed"
    codes = iter([0, 0, 0])
    assert chain.b1()["state"] == "ok"
    codes = iter([0])
    assert chain.b1(decide_only=True)["state"] == "stopped"


def a0_files(rpaths, code):
    a0 = rpaths.b2 / "a0"
    RB.write(a0 / "a0_gen.json", {"code_sha256": code})
    RB.write(a0 / "a0_scores.json", {"rows": {}})
    RB.write(a0 / "a0_report.json", {"pass": True, "scale": "tiny", "trim": dict(P.NO_TRIM), "fallback": False,
                                      "a0_gen_sha256": RB.sha256(a0 / "a0_gen.json"),
                                      "a0_scores_sha256": RB.sha256(a0 / "a0_scores.json")})


def test_a0_reuses_a_current_report_and_never_replaces_one_b2_follows(rpaths, monkeypatch):
    import register_b2
    chain = run_b.Chain(rpaths, "cpu")
    ran = []
    monkeypatch.setattr(chain, "b2", lambda *a, **k: ran.append(a) or 0)
    a0_files(rpaths, register_b2.hashes(register_b2.runtime_files()))
    res = chain.a0()
    assert res["state"] == "ok" and res["reused"] and not ran
    a0_files(rpaths, {"other": "code"})
    assert run_b.a0_current(rpaths)[0] is None and "other code" in run_b.a0_current(rpaths)[1]
    ledger_row(rpaths.ledgers()["dev"])
    res = chain.a0()
    assert res["state"] == "technical_stop" and not ran
    rpaths.ledgers()["dev"].unlink()
    chain.a0()
    assert ran and ran[0][:4] == ("a0", "a0", "--part", "gen")


def test_env_disables_the_safetensors_conversion(rpaths):
    assert rpaths.env()["DISABLE_SAFETENSORS_CONVERSION"] == "true"
