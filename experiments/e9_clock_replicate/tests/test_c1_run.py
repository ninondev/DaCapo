"""run_c1.py and run_c.py without a model: decide and package on fake tiny-scale rows (a "test only" verdict, P0's
checks, the demo, the blind pack outside the package with its pairs from the fixed seed and none of B2's items), the
A0 checks on fake tensors, resume and rescoring, the block loop, the refusals of the full scale, run_c.py's stop
rules and A0 reuse; setup's guide comparison against an addendum; gen's attempts (one per run per process, the
stop on failures in a row, attempts whose process died) and its runtime check; the gen passes of a block, the
heartbeat and the resume line."""
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
import package_c1 as PK  # noqa: E402
import plan_c1 as P  # noqa: E402
import run_c1 as RC  # noqa: E402


def args(**kw):
    base = {"scale": "tiny", "device": "cpu", "test_whisper": None, "items": "", "package": "", "key": "", "pack": "",
            "part": "", "block_items": 24}
    return argparse.Namespace(**{**base, **kw})


def fake_run(paths, run, text, npa=0.8, a0="A"):
    """A generated and scored clip: real files with their sha256 in the ledger rows."""
    import soundfile as sf
    out = paths.test
    out.mkdir(parents=True, exist_ok=True)
    rid = run["run_id"]
    sf.write(str(out / f"{rid}.wav"), np.zeros((4800, 2), np.float32), 48000, subtype="PCM_16")
    np.save(out / f"{rid}.pred.npy", np.zeros((3, 64), np.float32))
    np.save(out / f"{rid}.target.npy", np.ones((3, 64), np.float32))
    g = {**{k: run[k] for k in ("run_id", "stage", "item", "draw", "config", "labels", "guide", "k", "clock", "head_set",
                                "heads", "beta", "shuffle", "hooked", "fallback", "task", "cover_strength", "dit_seed",
                                "rng_seed")},
         "status": "ok", "hook_calls": 40 if run["hooked"] else 0, "max_key": 20 if run["hooked"] else None, "n_lyr": 27,
         "peak_before_norm": 0.5, "wav_sha256": RC.sha256(out / f"{rid}.wav"), "pred_sha256": RC.sha256(out / f"{rid}.pred.npy"),
         "target_sha256": RC.sha256(out / f"{rid}.target.npy"), "guide_latent_sha256": "lat", "guide_sha256": "g",
         "code_sha256": {}, "test_flags": ["scale tiny"], "a0_sha256": RC.sha256(paths.a0_report) if a0 == "A" else a0}
    RC.append_jsonl(paths.ledger(), g)
    words = text.split()
    s = {"run_id": rid, "stage": "test", "wav_sha256": g["wav_sha256"], "stem_share": {"vocals": 0.9, "other": 0.1},
         "vocal_active_s": 4.0, "whisper": "openai/whisper-small.en", "whisper_revision": "x", "whisper_text": text,
         "whisper_chunks": [[" " + w, 2.4 + 0.6 * k, 2.6 + 0.6 * k] for k, w in enumerate(words)], "npa_v": npa,
         "npa_v_oct": npa, "npa_mix": npa, "npa_mix_oct": npa, "ctc_text": text.upper(), "ctc_stem_text": text.upper(),
         "ctc": "c", "ctc_revision": "r", "test_flags": ["test whisper openai/whisper-small.en on cpu"]}
    RC.append_jsonl(paths.scores(), s)
    return g, s


@pytest.fixture
def paths(tmp_path):
    p = RC.Paths(str(tmp_path / "c1"), str(tmp_path / "guides"))
    RC.write(p.a0_report, {"pass": True, "recorded": True, "test_flags": [], "code_sha256": {}})
    return p


def lyric(iid="item01"):
    return P.all_items()[iid]["lyrics"].split("\n", 1)[1].lower()


def checks_ok():
    return {n: {"returncode": 0, "output": "current"} for n, _ in RC.REGISTER_SCRIPTS}


def test_decide_and_package_on_fake_tiny_rows(paths, tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "register_checks", checks_ok)
    for run in P.test_runs("tiny"):
        fake_run(paths, run, lyric() if run["hooked"] else "oh oh", npa=0.9)
    RC.decide(args(), paths)
    res = json.loads((paths.res / "c1_result.json").read_text())["results"]
    assert res["verdict"] == "test only" and res["arms"]["S*"]["J"] == 1.0 and res["arms"]["S*_b0"]["J"] == 0.0
    p0 = res["p0"]
    assert p0["plan"]["held"] and p0["hooks"]["held"] and p0["guide_latents"]["held"] and p0["complete"]["held"]
    assert not p0["hashes"]["held"]                 # no addendum 4 yet, and fake rows record no code hashes
    assert not p0["reference_and_peak"]["held"] and p0["reference_and_peak"]["detail"]["other_reference"] == 2
    assert not p0["a0"]["detail"]["ran_on_the_registered_code"] or not p0["hashes"]["held"]
    assert res["rule"]["R1"]["delta_j"] == 1.0 and res["labels"]["independent_asr"]["J_ctc_stem"] == 1.0
    assert len(res["clips"]) == 2 and res["clips"][0]["ctc_stem_text"] and res["demo"][0]["s_star_q"] == 0.9
    assert res["pooled_with_b2"]["n_items"] == 25 and res["counts"] == {"planned": 2, "generated": 2, "scored": 2}
    key, pack = tmp_path / "key.json", tmp_path / "pack.zip"
    RC.package(args(package=str(tmp_path / "pkg"), key=str(key), pack=str(pack)), paths)
    out = tmp_path / "pkg" / "c1"
    assert (out / "c1_result.json").exists() and (out / "ledgers" / "test_generation.jsonl").exists()
    assert pack.exists() and key.exists() and not list((tmp_path / "pkg").rglob("*key*"))
    assert len(list((out / "demo").glob("item01__*"))) == 2
    k = json.loads(key.read_text())
    assert k["order_seed"] == 20261031 and k["n_pairs"] == 1
    pair = k["pairs"]["pair_01"]
    assert {pair["A"]["arm"], pair["B"]["arm"]} == {"S*", "S*_b0"} and pair["item"] == "item01"
    sides = (pair["A"]["run_id"], pair["B"]["run_id"])
    for _ in range(4):   # packaging again keeps the sides
        RC.package(args(package=str(tmp_path / "pkg"), key=str(key), pack=str(pack)), paths)
        p = json.loads(key.read_text())["pairs"]["pair_01"]
        assert (p["A"]["run_id"], p["B"]["run_id"]) == sides
    with zipfile.ZipFile(pack) as z:   # the pack shares no audio file with the demo
        names = z.namelist()
        packed = {hashlib.sha256(z.read(n)).hexdigest() for n in names if n.endswith((".mp3", ".wav"))}
    demo = {hashlib.sha256(f.read_bytes()).hexdigest() for f in (out / "demo").glob("*") if f.suffix in (".mp3", ".wav")}
    assert packed and demo and not packed & demo and "ratings.csv" in names and "README.txt" in names
    for bad in ({"key": str(tmp_path / "pkg" / "k.json"), "pack": str(pack)},
                {"key": str(key), "pack": str(tmp_path / "pkg" / "p.zip")}):
        with pytest.raises(ValueError, match="inside the package"):
            RC.package(args(package=str(tmp_path / "pkg"), **bad), paths)


def demo96():
    return [{"item": f"c1t{i:02d}", "s_star_run": f"c1t{i:02d}_r0_oohs_k3_A8", "comparator": "S*_b0",
             "comparator_run": f"c1t{i:02d}_r1_oohs_k3_none"} for i in range(96)]


def test_pack_pairs_come_from_the_fixed_seed():
    d = demo96()
    got = PK.pack_pairs(list(reversed(d)))                 # the input order does not matter: item order first
    want = [d[int(j)] for j in np.random.default_rng(20261031).permutation(96)[:24]]
    assert got == want and len(got) == 24 and len({x["item"] for x in got}) == 24
    assert PK.pack_pairs(d) == got                         # the same pairs every time


def test_pack_shares_no_item_with_b2():
    b2 = json.loads(P.B2_RESULT.read_text())["results"]["demo"]
    b2_items = {x["item"] for x in b2}
    ours = {x["item"] for x in PK.pack_pairs(demo96())}
    assert len(b2_items) == 24 and not ours & b2_items and all(i.startswith("c1t") for i in ours)
    assert not {x["s_star_run"] for x in PK.pack_pairs(demo96())} & {x["s_star_run"] for x in b2}


def test_full_pack_on_fake_files(tmp_path):
    """24 pairs from 96 demo items, the key outside the package, every clip copied."""
    paths = RC.Paths(str(tmp_path / "c1"), str(tmp_path / "guides"))
    import soundfile as sf
    paths.test.mkdir(parents=True)
    paths.guides.mkdir(parents=True)
    for d in demo96():
        for rid in (d["s_star_run"], d["comparator_run"]):
            sf.write(str(paths.test / f"{rid}.wav"), np.random.default_rng(len(rid)).standard_normal((480, 2)).astype(np.float32) * 0.1,
                     48000, subtype="PCM_16")
        sf.write(str(paths.guides / f"{d['item']}_oohs.wav"), np.zeros((480, 2), np.float32), 48000, subtype="PCM_16")
    RC.write(paths.res / "c1_result.json", {"results": {"verdict": "x", "demo": demo96()}})
    key, pack = tmp_path / "out" / "key.json", tmp_path / "out" / "pack.zip"
    res = PK.package(paths, tmp_path / "pkg", key, pack, P.all_items())
    assert res["demo_clips"] == 192 and res["blind_pack"]["n_pairs"] == 24
    assert res["blind_pack"]["items"] == [x["item"] for x in PK.pack_pairs(demo96())]
    k = json.loads(key.read_text())
    arms = [{v["A"]["arm"], v["B"]["arm"]} for v in k["pairs"].values()]
    assert all(a == {"S*", "S*_b0"} for a in arms) and len(k["pairs"]) == 24
    with zipfile.ZipFile(pack) as z:
        assert len([n for n in z.namelist() if n.startswith("pair_")]) == 72


def test_resume_counts_a_run_only_while_its_files_hash_to_the_ledger(paths):
    run = P.test_runs("tiny")[0]
    fake_run(paths, run, lyric())
    assert RC.generated(paths.test)[run["run_id"]]["status"] == "ok"
    (paths.test / f"{run['run_id']}.wav").write_bytes(b"changed")
    assert RC.generated(paths.test)[run["run_id"]]["status"] == "files_missing"
    for k in range(3):
        RC.append_jsonl(paths.ledger(), {"run_id": "x", "status": "failed", "attempt": k + 1})
    assert RC.failed_attempts(paths.test)["x"] == 3


def test_score_rows_are_current_only_for_the_same_file_and_models():
    import score_c1 as S
    g = {"wav_sha256": "w"}
    ok = {"wav_sha256": "w", "whisper": "m", "whisper_revision": RC.R.WHISPER_REVISION, "ctc_revision": S.CTC_REVISION,
          "ctc_stem_text": "A"}
    assert RC.score_current(ok, g, "m", False)
    assert not RC.score_current({**ok, "wav_sha256": "v"}, g, "m", False)
    assert not RC.score_current({**ok, "ctc_stem_text": None}, g, "m", False)
    assert not RC.score_current({**ok, "error": "x"}, g, "m", False)
    assert not RC.score_current({**ok, "whisper_revision": "main"}, g, "m", False)
    assert RC.score_current({**ok, "whisper_revision": "main"}, g, "m", True)


def test_decide_counts_a_score_of_another_file_as_unscored(paths):
    run = P.test_runs("tiny")[0]
    g, s = fake_run(paths, run, lyric())
    RC.append_jsonl(paths.scores(), {**s, "wav_sha256": "other"})
    used, clips = RC.clips_of(P.test_runs("tiny"), RC.generated(paths.test), {s["run_id"]: {**s, "wav_sha256": "other"}})
    assert run["run_id"] not in used and not clips[run["run_id"]]["scored"] and clips[run["run_id"]]["q"] == 0.0


def test_a0_checks_on_fake_tensors():
    import torch
    runs = P.a0_runs("full")
    t = {"pred": torch.zeros(3, 4), "target": torch.ones(3, 4)}
    rows = {r["name"]: {"status": "ok", "peak_before_norm": 0.5, "hooked": r["hooked"],
                        "hook_calls": 40 if r["hooked"] else 0, "max_key": 20 if r["hooked"] else None, "n_lyr": 27,
                        "guide_latent_sha256": "L"} for r in runs}
    tens = {"s_b0": t, "beta0": {"pred": torch.zeros(3, 4)}, "s_star": {"pred": torch.ones(3, 4)}}
    rep = {"hard": {"setup": True}, "soft": {}, "runs": rows, "bit_identity": {}}
    RC.a0_checks(runs, rep, tens, "full")
    assert all(rep["hard"].values()) and rep["bit_identity"] == {"beta0_vs_no_hook": True}
    tens["beta0"] = {"pred": torch.full((3, 4), 1e-3)}
    rep = {"hard": {}, "soft": {}, "runs": rows, "bit_identity": {}}
    RC.a0_checks(runs, rep, tens, "full")
    assert not rep["hard"]["beta0_bit_identical"] and rep["soft"]["beta0_max_abs_diff"] == pytest.approx(1e-3)
    tens = {"s_b0": t, "beta0": {"pred": torch.zeros(3, 4)}, "s_star": {"pred": torch.zeros(3, 4)}}
    rep = {"hard": {}, "soft": {}, "runs": rows, "bit_identity": {}}
    RC.a0_checks(runs, rep, tens, "full")
    assert not rep["hard"]["s_star_changes_output"]
    bad = {**rows, "s_star": {**rows["s_star"], "hook_calls": 39}}
    rep = {"hard": {}, "soft": {}, "runs": bad, "bit_identity": {}}
    RC.a0_checks(runs, rep, {"s_b0": t, "beta0": {"pred": torch.zeros(3, 4)}, "s_star": {"pred": torch.ones(3, 4)}}, "full")
    assert not rep["hard"]["hook_calls_40"]
    lat = {**rows, "beta0": {**rows["beta0"], "guide_latent_sha256": "M"}}
    rep = {"hard": {}, "soft": {}, "runs": lat, "bit_identity": {}}
    RC.a0_checks(runs, rep, tens, "full")
    assert not rep["hard"]["guide_latents_equal"]


def test_a0_projection_is_recorded_not_applied():
    runs = {"s_b0": {"status": "ok", "gen_seconds": 30.0}, "beta0": {"status": "ok", "gen_seconds": 4.0},
            "s_star": {"status": "ok", "gen_seconds": 6.0}}
    proj = RC.a0_projection(runs, {"s_b0": {"seconds_total": 6.0}, "s_star": {"seconds_total": 8.0}}, 1536, 4)
    assert proj["seconds_per_run_a0"] == 12.0 and proj["from_a0"] == round(P.projected_hours(12.0, 1536, 4), 2)
    assert proj["from_run_b"] == round(P.projected_hours(12.1, 1536, 4), 2)


def test_gen_and_score_need_a_passing_a0(tmp_path):
    p = RC.Paths(str(tmp_path / "c1"), str(tmp_path / "g"))
    with pytest.raises(RC.SetupError, match="A0"):
        RC.gen(args(), p)
    with pytest.raises(RC.SetupError, match="A0"):
        RC.score(args(), p)


def test_full_scale_refusals(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "experiments" / "run_c.py"), "--device", "cpu", "--state", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "full scale runs on CUDA" in r.stderr
    r = subprocess.run([sys.executable, str(HERE / "run_c1.py"), "score", "--test-whisper", "openai/whisper-tiny",
                        "--root", str(tmp_path / "x")], capture_output=True, text=True)
    assert r.returncode == 2 and "no test Whisper" in r.stdout
    with pytest.raises(RC.SetupError, match="CUDA"):
        RC.refuse_full_scale_tests(args(scale="full", device="cpu"), needs_cuda=True)
    with pytest.raises(RC.SetupError, match="CUDA"):
        RC.refuse_full_scale_tests(args(scale="full", device="cuda", test_whisper="x"), needs_cuda=False)


# run_c.py

import run_c  # noqa: E402


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

    def c1test(self):
        return self._r("c1test")

    def decide(self):
        return self._r("decide")

    def package(self, status):
        return self._r("package")


@pytest.fixture
def rpaths(tmp_path):
    return run_c.Paths("tiny", str(tmp_path))


def go(rpaths, states, status=None):
    chain = FakeChain(states)
    st = run_c.run(chain, rpaths, run_c.STAGES, status or {}, "tiny")
    return chain.calls, st


def ledger_row(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": "item01_r0_oohs_k3_A8", "status": "ok"}) + "\n")


def test_all_stages_run_in_order(rpaths):
    calls, st = go(rpaths, {})
    assert calls == run_c.STAGES == ["setup", "a0", "c1test", "decide", "package"]
    assert all(st[s]["state"] == "ok" for s in run_c.STAGES)


@pytest.mark.parametrize("bad", ["setup", "a0"])
def test_setup_or_a0_failure_stops_the_run(rpaths, bad):
    calls, st = go(rpaths, {bad: "technical_stop"})
    assert calls == run_c.STAGES[:run_c.STAGES.index(bad) + 1] + ["package"]
    assert st["decide"]["state"] == "skipped" and st["c1test"]["state"] == "skipped"


@pytest.mark.parametrize("bad", ["setup", "a0"])
def test_a_failure_after_generation_started_still_decides(rpaths, bad):
    ledger_row(rpaths.ledger())
    calls, st = go(rpaths, {bad: "technical_stop"})
    assert calls == run_c.STAGES[:run_c.STAGES.index(bad) + 1] + ["decide", "package"]
    assert st["c1test"]["state"] == "skipped" and st["decide"]["state"] == "ok"


def test_c1test_failure_still_decides(rpaths):
    calls, st = go(rpaths, {"c1test": "technical_stop"})
    assert calls == run_c.STAGES and st["c1test"]["state"] == "technical_stop"


def test_resume_skips_a_finished_c1test(rpaths):
    done = {s: {"state": "ok", "scale": "tiny"} for s in ("setup", "a0", "c1test")}
    calls, _ = go(rpaths, {}, status=done)
    assert calls == ["setup", "a0", "decide", "package"]


def test_c1test_runs_gen_then_score_block_by_block(rpaths, monkeypatch):
    chain = run_c.Chain(rpaths, "cpu")
    seen = []
    monkeypatch.setattr(chain, "c1", lambda stage, cmd, *a, **k: seen.append((cmd, a[a.index("--items") + 1])) or 0)
    monkeypatch.setattr(run_c, "ledger_counts", lambda p: {})
    rpaths.scale = "full"
    res = chain.c1test()
    blocks = [",".join(b) for b in P.blocks("full", 24)]
    assert res["state"] == "ok" and res["blocks_done"] == ["c1t00-c1t23", "c1t24-c1t47", "c1t48-c1t71", "c1t72-c1t95"]
    assert seen == [(c, b) for b in blocks for c in ("gen", "score")]
    calls = iter([0, 0, 0, 0, 7])
    monkeypatch.setattr(chain, "c1", lambda *a, **k: next(calls))
    res = chain.c1test()
    assert res["state"] == "technical_stop" and res["failed"].startswith("gen c1t48-c1t71") and len(res["blocks_done"]) == 2


def test_score_errors_rescore_once_per_block(rpaths, monkeypatch):
    chain = run_c.Chain(rpaths, "cpu")
    rpaths.scores().parent.mkdir(parents=True, exist_ok=True)
    rpaths.scores().write_text(json.dumps({"run_id": "item01_r0_oohs_k3_A8", "error": "x"}) + "\n")
    seen = []
    monkeypatch.setattr(chain, "c1", lambda stage, cmd, *a, **k: seen.append(cmd) or 0)
    monkeypatch.setattr(run_c, "ledger_counts", lambda p: {})
    assert chain.c1test()["state"] == "ok" and seen == ["gen", "score", "score"]
    assert run_c.score_errors(rpaths, ["item01"]) and not run_c.score_errors(rpaths, ["c1t00"])


def a0_files(rpaths, code):
    a0 = rpaths.c1 / "a0"
    RC.write(a0 / "a0_gen.json", {"code_sha256": code})
    RC.write(a0 / "a0_scores.json", {"rows": {}})
    RC.write(a0 / "a0_report.json", {"pass": True, "scale": "tiny", "a0_gen_sha256": RC.sha256(a0 / "a0_gen.json"),
                                      "a0_scores_sha256": RC.sha256(a0 / "a0_scores.json")})


def test_a0_reuses_a_current_report_and_never_replaces_one_c1_follows(rpaths, monkeypatch):
    import register_c1
    chain = run_c.Chain(rpaths, "cpu")
    ran = []
    monkeypatch.setattr(chain, "c1", lambda *a, **k: ran.append(a) or 0)
    a0_files(rpaths, register_c1.hashes(register_c1.runtime_files()))
    res = chain.a0()
    assert res["state"] == "ok" and res["reused"] and not ran
    a0_files(rpaths, {"other": "code"})
    assert run_c.a0_current(rpaths)[0] is None and "other code" in run_c.a0_current(rpaths)[1]
    ledger_row(rpaths.ledger())
    res = chain.a0()
    assert res["state"] == "technical_stop" and not ran
    rpaths.ledger().unlink()
    chain.a0()
    assert ran and ran[0][:4] == ("a0", "a0", "--part", "gen")


def test_status_is_written_whole_and_env(rpaths):
    run_c.write_json(rpaths.status, {"setup": {"state": "ok"}})
    assert json.loads(rpaths.status.read_text()) == {"setup": {"state": "ok"}} and not list(rpaths.root.glob("*.tmp"))
    env = rpaths.env()
    assert env["DISABLE_SAFETENSORS_CONVERSION"] == "true" and env["ACESTEP_VAE_DECODE_CHUNK_SIZE"] == "256"
    assert rpaths.root.name == "tiny" and rpaths.key.parent == rpaths.root and not rpaths.key.is_relative_to(rpaths.pkg)


# setup against an addendum (review: setup_check called a function make_items_c1 did not have)

def draft_addendum(tmp_path, change=None):
    g = P.read_json(P.GUIDES_JSON)
    sha = json.loads(json.dumps(g["sha256"]))
    if change:
        sha[change][P.GUIDE] = "0" * 64
    p = tmp_path / "addendum_4.json"
    p.write_text(json.dumps({"plan": {"guides_sha256": sha, "soundfont_sha256": g["soundfont_sha256"],
                                      "seed_table_sha256": RC.sha256(P.SEEDS_JSON)}}))
    return p


@pytest.mark.parametrize("change", [None, "c1t05"])
def test_setup_check_compares_the_guides_with_addendum_4(paths, tmp_path, monkeypatch, change):
    import make_items_c1
    monkeypatch.setattr(RC.REG, "ADDENDUM", draft_addendum(tmp_path, change))
    monkeypatch.setattr(make_items_c1, "render_all", lambda out, workers=4: P.read_json(P.GUIDES_JSON))
    monkeypatch.setattr(RC, "register_checks", checks_ok)
    RC.setup_check(args(), paths)
    rec = json.loads((paths.setup / "setup_check.json").read_text())
    assert rec["guides"] == ([] if change is None else ["c1t05 oohs"]) and rec["seeds"] == []
    assert rec["register_checks"] == checks_ok() and isinstance(rec["packages"], list)
    assert len(list((paths.setup / "history").glob("setup_check_*.json"))) == 1
    if change:
        with pytest.raises(RC.SetupError, match="guide c1t05 oohs"):
            RC.setup_check(args(scale="full"), paths)


# gen without a model

FP = {k: "x" for k in P.RUNTIME_KEYS}


def gen_env(paths, monkeypatch, fail=lambda n, run: None, runtime=FP):
    """gen with a fake model and a fake generate_run; fail(call number, run) raises to make a run fail."""
    RC.write(paths.a0 / "a0_gen.json", {"runtime": FP})
    monkeypatch.setattr(RC, "load_model", lambda a, items: (None, {it["id"]: {} for it in items}, {}))
    monkeypatch.setattr(RC, "check_guides", lambda p, ids: None)
    monkeypatch.setattr(RC, "registration", lambda a: {"problems": [], "registered": True, "code_sha256": {}})
    monkeypatch.setattr(RC, "runtime_now", lambda h: dict(runtime))
    calls = []

    def fake(h, run, lay, it, out, guides):
        calls.append(run["run_id"])
        fail(len(calls), run)
        rid = run["run_id"]
        for sfx in (".wav", ".pred.npy", ".target.npy"):
            (out / f"{rid}{sfx}").write_bytes((rid + sfx).encode())
        return {"run_id": rid, "item": run["item"], "draw": run["draw"], "status": "ok", "gen_seconds": 0.1,
                "hook_calls": 0, "peak_before_norm": 0.5, "wav_sha256": RC.sha256(out / f"{rid}.wav"),
                "pred_sha256": RC.sha256(out / f"{rid}.pred.npy"), "target_sha256": RC.sha256(out / f"{rid}.target.npy")}, None
    monkeypatch.setattr(RC.RB, "generate_run", fake)
    return calls


def run_gen(a, paths):
    try:
        RC.gen(a, paths)
        return 0
    except SystemExit as e:
        return e.code


BLOCK = "c1t00,c1t01"   # 32 runs, 16 per item


def test_gen_tries_a_failed_run_once_per_process(paths, monkeypatch):
    bad = "c1t00_r1_oohs_k3_none"

    def fail(n, run):
        if run["run_id"] == bad:
            raise RuntimeError("this run always fails")
    calls = gen_env(paths, monkeypatch, fail)
    a = args(scale="full", device="cuda", items=BLOCK)
    assert run_gen(a, paths) == RC.EXIT_AGAIN and len(calls) == 32 and calls.count(bad) == 1
    assert run_gen(a, paths) == RC.EXIT_AGAIN and calls[32:] == [bad]
    assert run_gen(a, paths) == 0 and calls[33:] == [bad]            # its third and last attempt
    assert run_gen(a, paths) == 0 and len(calls) == 34                # nothing left to try
    assert RC.attempts_used(paths.test)[bad] == 3 and RC.failed_attempts(paths.test)[bad] == 3
    rows = RC.read_jsonl(paths.ledger())
    assert [r["attempt"] for r in rows if r["run_id"] == bad and r["status"] == "started"] == [1, 2, 3]
    gens = RC.generated(paths.test)
    assert sum(g["status"] == "ok" for g in gens.values()) == 31 and gens[bad]["status"] == "failed"


def test_an_error_that_sticks_stops_the_process_before_it_burns_attempts(paths, monkeypatch):
    state = {"broken": True}

    def fail(n, run):
        if n >= 10 and state["broken"]:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
    calls = gen_env(paths, monkeypatch, fail)
    a = args(scale="full", device="cuda", items=BLOCK)
    assert run_gen(a, paths) == RC.EXIT_STICKY
    used = RC.attempts_used(paths.test)
    assert len(calls) == 17 and sum(used.values()) == 8 and max(used.values()) == 1   # c1t00's last 7 runs, c1t01's first
    state["broken"] = False                                                           # a fresh process works
    assert run_gen(a, paths) == 0
    assert sum(g["status"] == "ok" for g in RC.generated(paths.test).values()) == 32


def test_one_item_that_always_fails_does_not_stop_the_block(paths, monkeypatch):
    def fail(n, run):
        if run["item"] == "c1t00":
            raise RuntimeError("item-level failure")
    gen_env(paths, monkeypatch, fail)
    a = args(scale="full", device="cuda", items=BLOCK)
    assert run_gen(a, paths) == RC.EXIT_AGAIN                       # c1t01 succeeds after c1t00's 16 failures
    assert sum(g["status"] == "ok" for g in RC.generated(paths.test).values()) == 16
    assert [run_gen(a, paths) for _ in range(3)] == [RC.EXIT_AGAIN, 0, 0]
    assert sorted(set(RC.attempts_used(paths.test).values())) == [3]


def test_attempts_whose_process_died_count_from_the_second(paths, monkeypatch):
    victim = "c1t00_r0_oohs_k3_A8"

    def fail(n, run):
        if run["run_id"] == victim:
            raise KeyboardInterrupt   # not an Exception: the process ends inside the attempt, as on a kill
    calls = gen_env(paths, monkeypatch, fail)
    a = args(scale="full", device="cuda", items=BLOCK)
    for k in range(4):
        with pytest.raises(KeyboardInterrupt):
            RC.gen(a, paths)
        assert RC.attempts_used(paths.test).get(victim, 0) == k            # the first death is free
    assert run_gen(a, paths) == 0 and calls.count(victim) == 4               # used up: the block goes on without it
    assert sum(g["status"] == "ok" for g in RC.generated(paths.test).values()) == 31
    used = RC.attempts_used
    RC.append_jsonl(paths.ledger(), {"run_id": "x", "status": "started"})
    RC.append_jsonl(paths.ledger(), {"run_id": "x", "status": "ok"})
    assert used(paths.test).get("x", 0) == 0                                 # a finished attempt closes its start


def test_gen_refuses_a_runtime_other_than_a0s(paths, monkeypatch):
    other = {**FP, "gpu": "NVIDIA A100-SXM4-40GB"}
    gen_env(paths, monkeypatch, runtime=other)
    a = args(scale="full", device="cuda", items="c1t00")
    with pytest.raises(RC.SetupError, match="differs from the one A0 ran on"):
        RC.gen(a, paths)
    assert not RC.read_jsonl(paths.ledger())
    monkeypatch.setenv(RC.ALLOW_RUNTIME_CHANGE, "1")
    assert run_gen(a, paths) == 0
    rows = [r for r in RC.read_jsonl(paths.ledger()) if r.get("status") == "ok"]
    assert len(rows) == 16 and all(r["runtime_change"] == {"gpu": ["x", other["gpu"]]} for r in rows)
    rec = RC.runtime_record(paths, P.test_runs("full", ["c1t00"]), RC.generated(paths.test))
    assert rec["runs_on_another_runtime"] == 16 and rec["changes"] == [{"gpu": ["x", other["gpu"]]}]
    monkeypatch.delenv(RC.ALLOW_RUNTIME_CHANGE)
    gen_env(paths, monkeypatch)                                              # A0's runtime: no change recorded
    a = args(scale="full", device="cuda", items="c1t01")
    assert run_gen(a, paths) == 0
    assert all(r["runtime_change"] is None for r in RC.read_jsonl(paths.ledger()) if r.get("item") == "c1t01")


def test_generated_of_is_run_b2s_generated_for_a_block(paths, monkeypatch):
    gen_env(paths, monkeypatch)
    assert run_gen(args(scale="full", device="cuda", items=BLOCK), paths) == 0
    (paths.test / "c1t01_r2_oohs_k3_R8.wav").write_bytes(b"changed")
    ids = [r["run_id"] for r in P.test_runs("full", ["c1t01"])]
    full = RC.generated(paths.test)
    assert RC.generated_of(paths.test, ids) == {k: v for k, v in full.items() if k in ids}
    assert RC.generated_of(paths.test, ids)["c1t01_r2_oohs_k3_R8"]["status"] == "files_missing"


def test_decide_p0_needs_the_registration_checks_at_setup_and_decision(paths, monkeypatch):
    for run in P.test_runs("tiny"):
        fake_run(paths, run, lyric())
    bad = {**checks_ok(), "addendum_2": {"returncode": 1, "output": "stale"}}
    monkeypatch.setattr(RC, "register_checks", lambda: bad)
    RC.write(paths.setup / "setup_check.json", {"register_checks": checks_ok(), "guides": [], "seeds": [], "packages": []})
    RC.decide(args(), paths)
    res = json.loads((paths.res / "c1_result.json").read_text())["results"]
    assert "decision: addendum_2 --check failed" in res["p0"]["hashes"]["detail"]
    assert res["registration_checks_at_decision"]["addendum_2"] == 1
    monkeypatch.setattr(RC, "register_checks", checks_ok)
    RC.write(paths.setup / "setup_check.json", {"register_checks": bad, "guides": [], "seeds": [], "packages": []})
    RC.decide(args(), paths)
    res = json.loads((paths.res / "c1_result.json").read_text())["results"]
    assert "setup: addendum_2 --check failed" in res["p0"]["hashes"]["detail"]
    assert not any(d.startswith("decision:") for d in res["p0"]["hashes"]["detail"])


def test_package_problems():
    import numpy
    assert RC.package_problems({"numpy": numpy.__version__}) == []
    assert RC.package_problems({"numpy": "0.0.1"}) == [f"numpy {numpy.__version__}, pinned 0.0.1"]
    assert RC.package_problems({"no-such-package-dacapo": "1.0"}) == ["no-such-package-dacapo None, pinned 1.0"]


# run_c.py: gen passes, heartbeat, resume line, A0 and the runtime

@pytest.mark.parametrize("codes,ok,n", [([0], True, 1), ([5, 0], True, 2), ([4, 0], True, 2), ([-9, 5, 0], True, 3),
                                        ([4, 5, 4], False, 3), ([1], False, 1), ([2], False, 1), ([5] * 8, False, 8)])
def test_gen_block_goes_on_in_fresh_processes(rpaths, monkeypatch, codes, ok, n):
    chain = run_c.Chain(rpaths, "cpu")
    it = iter(codes)
    seen = []
    monkeypatch.setattr(chain, "c1", lambda stage, cmd, *a, **k: seen.append(cmd) or next(it))
    res = chain.gen_block("c1t00")
    assert res["ok"] is ok and len(seen) == n and seen == ["gen"] * n and res["codes"] == codes[:n]


def test_heartbeat_keeps_a_second_writer_out(tmp_path):
    root = tmp_path / "tiny"
    root.mkdir()
    hb = run_c.Heartbeat(root)
    assert hb.other() is None
    run_c.write_json(hb.path, {"host": "elsewhere", "pid": 1, "epoch": run_c.time.time() - 10})
    assert hb.other()["host"] == "elsewhere"
    r = subprocess.run([sys.executable, str(ROOT / "experiments" / "run_c.py"), "--scale", "tiny", "--device", "cpu",
                        "--state", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 3 and "ANOTHER run_c.py" in r.stdout and not (root / "status.json").exists()
    run_c.write_json(hb.path, {"host": "elsewhere", "pid": 1, "epoch": run_c.time.time() - run_c.STALE_S - 1})
    assert hb.other() is None
    hb.start()
    assert run_c.read(hb.path)["pid"] == run_c.os.getpid() and hb.other() is None
    hb.stop()
    assert not hb.path.exists()


def test_resume_line(rpaths):
    assert run_c.resume_note(rpaths, {"c1test": {"state": "ok"}}) is None and run_c.resume_note(rpaths, {}) is None
    st = {"c1test": {"state": "skipped", "reason": "setup failed: Run C stopped"}}
    assert "before any C1 generation" in run_c.resume_note(rpaths, st)
    ledger_row(rpaths.ledger())
    note = run_c.resume_note(rpaths, {"c1test": {"state": "technical_stop", "failed": "gen item01-item01: x"}})
    assert "incomplete" in note and "1 of 2 runs generated" in note and "Run all again" in note


def runtime_files(rpaths, a0_rt, now_rt, epoch=None):
    import register_c1
    a0_files(rpaths, register_c1.hashes(register_c1.runtime_files()))
    gen = json.loads((rpaths.c1 / "a0" / "a0_gen.json").read_text())
    RC.write(rpaths.c1 / "a0" / "a0_gen.json", {**gen, "runtime": a0_rt})
    a0 = rpaths.c1 / "a0"
    rep = json.loads((a0 / "a0_report.json").read_text())
    RC.write(a0 / "a0_report.json", {**rep, "a0_gen_sha256": RC.sha256(a0 / "a0_gen.json")})
    RC.write(rpaths.c1 / "setup" / "setup_model.json", {"runtime_fingerprint": now_rt,
                                                        "epoch": run_c.time.time() if epoch is None else epoch})


def test_a0_is_reused_only_on_its_runtime(rpaths, monkeypatch):
    chain = run_c.Chain(rpaths, "cpu")
    ran = []
    monkeypatch.setattr(chain, "c1", lambda *a, **k: ran.append(a) or 0)
    other = {**FP, "torch_cuda": "13.1"}
    runtime_files(rpaths, FP, FP)
    assert chain.a0()["reused"] and not ran
    runtime_files(rpaths, FP, other, epoch=chain.started - 60)    # a setup record from an earlier session: not compared
    assert chain.a0()["reused"] and not ran
    runtime_files(rpaths, FP, other)
    rep, why = run_c.a0_current(rpaths, chain.started)
    assert rep is None and "another runtime" in why and "torch_cuda" in why
    chain.a0()                                                    # no C1 run yet: A0 runs again on this runtime
    assert ran and ran[0][:4] == ("a0", "a0", "--part", "gen")
    ran.clear()
    runtime_files(rpaths, FP, other)
    ledger_row(rpaths.ledger())
    res = chain.a0()
    assert res["state"] == "technical_stop" and "ALLOW_RUNTIME_CHANGE" in res["failed"] and not ran
    monkeypatch.setenv(run_c.ALLOW_RUNTIME_CHANGE, "1")
    assert chain.a0()["state"] == "ok" and not ran


def test_notebook_is_the_builders_output():
    """notebook_runC.ipynb is build_notebook_c.py's output (with or without --commit), so the hashed builder fixes it."""
    import ast
    import re
    import build_notebook_c
    nb = json.loads((ROOT / "notebook_runC.ipynb").read_text())
    pinned = re.search(r"^REGISTERED = (None|'[0-9a-f]{40}')", "".join(nb["cells"][1]["source"]), re.M)
    assert pinned and nb["cells"] == build_notebook_c.cells(ast.literal_eval(pinned.group(1)))
    first = "".join(nb["cells"][1]["source"])
    assert 'shutil.which("nvidia-smi")' in first and r're.search(r"\bL4\b", gpu)' in first and "ALLOW_OTHER_GPU" in first
    run = "".join(nb["cells"][-3]["source"])
    assert "except KeyboardInterrupt" in run and "p.terminate()" in run
    assert "DACAPO_ALLOW_RUNTIME_CHANGE" in "".join(nb["cells"][-4]["source"])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux only")
def test_a_child_ends_with_its_parent(tmp_path):
    """run_c1.py's processes end when run_c.py ends (DACAPO_DIE_WITH_PARENT=1, which run_c.py sets)."""
    import os
    import time
    pidfile = tmp_path / "child.pid"
    child = (f"import os, sys, time; sys.path.insert(0, {str(ROOT / 'experiments')!r}); import run_c; run_c.die_with_parent(); "
             f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(60)")
    parent = (f"import os, subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}], "
              f"env={{**os.environ, 'DACAPO_DIE_WITH_PARENT': '1'}})\n"
              f"while not os.path.exists({str(pidfile)!r}): time.sleep(0.1)\n")
    subprocess.run([sys.executable, "-c", parent], timeout=60, check=True)
    pid = int(pidfile.read_text())
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        pytest.fail("the child outlived its parent")
    src = (HERE / "run_c1.py").read_text()
    assert "die_with_parent()\n    ap = argparse" in src and run_c.Paths("tiny", str(tmp_path)).env()["DACAPO_DIE_WITH_PARENT"] == "1"
