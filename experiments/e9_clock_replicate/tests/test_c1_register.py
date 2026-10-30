"""Addendum 4: written and checked (a draft before review, the registered file after), complete, refuses a seed
collision, flags a changed file at run time, names the earlier addenda; and nothing that an earlier registration
hashes has changed (addendum 2, the B1 code addendum, addendum 3 and E7 Stage S)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import plan_c1 as P  # noqa: E402
import register_c1 as REG  # noqa: E402

SCRIPT = HERE / "register_c1.py"


@pytest.fixture(scope="module")
def add(tmp_path_factory):
    """The registered addendum 4 if it exists, else a draft written now (before review it is not in the repo)."""
    if REG.ADDENDUM.exists():
        return json.loads(REG.ADDENDUM.read_text()), REG.ADDENDUM
    out = tmp_path_factory.mktemp("reg") / "addendum_4.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(out.read_text()), out


def test_check_passes(add):
    r = subprocess.run([sys.executable, str(SCRIPT), "--check", "--addendum", str(add[1])], capture_output=True, text=True,
                       cwd=ROOT)
    assert r.returncode == 0 and "current" in r.stdout, r.stdout + r.stderr


def test_check_fails_on_a_stale_addendum(add, tmp_path):
    bad = json.loads(json.dumps(add[0]))
    bad["data_sha256"]["experiments/e9_clock_replicate/lines_c1.txt"] = "0" * 64
    p = tmp_path / "stale.json"
    p.write_text(json.dumps(bad))
    r = subprocess.run([sys.executable, str(SCRIPT), "--check", "--addendum", str(p)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "data_sha256" in r.stdout
    r = subprocess.run([sys.executable, str(SCRIPT), "--check", "--addendum", str(tmp_path / "none.json")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1


def test_check_fails_on_a_changed_title_or_an_added_key(add, tmp_path):
    for change in ({"title": "Addendum 4: something else"}, {"deviation": "R1 now needs Delta_J >= 0.30"},
                   {"addendum_utc": "yesterday"}):
        bad = {**json.loads(json.dumps(add[0])), **change}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        r = subprocess.run([sys.executable, str(SCRIPT), "--check", "--addendum", str(p)], capture_output=True, text=True,
                           cwd=ROOT)
        assert r.returncode == 1 and "stale" in r.stdout, (change, r.stdout)
        assert REG.form_problems(bad) and REG.form_problems(bad)[0] in REG.registration_problems(bad)
    assert REG.form_problems(add[0]) == []
    gone = {k: v for k, v in add[0].items() if k != "fill_ins"}
    assert REG.form_problems(gone) == ["addendum 4 has no 'fill_ins'"]


def test_keys_are_the_computed_ones():
    assert set(REG.computed()) | {"addendum_utc", "title"} == set(REG.KEYS)


def test_notebook_pins_and_addendum_2s_check_are_hashed(add):
    assert "experiments/build_notebook_c.py" in REG.CODE and "experiments/build_notebook.py" in REG.REUSED
    assert "experiments/e5_motif/register.py" in REG.REUSED
    assert set(REG.runtime_files()) >= {"experiments/build_notebook.py", "experiments/build_notebook_c.py",
                                        "experiments/e5_motif/register.py"}
    env = add[0]["plan"]["environment"]
    assert env == REG.pins() and env["main"]["librosa"] == "1.0.0" and env["ace"]["diffusers"] == "0.39.0"
    assert env["main"]["transformers"] == env["ace"]["transformers"] and "basic-pitch" in env["main"]
    assert add[0]["plan"]["runtime_keys"] == list(P.RUNTIME_KEYS) and add[0]["plan"]["attempts"]["max_attempts"] == 3


def test_every_c1_module_is_registered():
    mods = {f"experiments/e9_clock_replicate/{p.name}" for p in HERE.glob("*.py")}
    assert mods == {f for f in REG.CODE if f.startswith("experiments/e9")} and "experiments/run_c.py" in REG.CODE
    assert all((ROOT / f).exists() for f in REG.runtime_files())
    for f in ("experiments/e8_clock_sing/run_b2.py", "experiments/e8_clock_sing/score_b2.py", "dacapo/attn_bias.py",
              "experiments/e6_clock/clock_score.py", "experiments/e5_motif/stage.py"):
        assert f in REG.REUSED
    assert "results/gpu_run3/b2_result.json" in REG.DATA and "experiments/e9_clock_replicate/power_c1.json" in REG.DATA


def test_hashes_and_earlier_addenda(add):
    a = add[0]
    assert a["code_sha256"] == REG.hashes(REG.CODE + REG.REUSED) and a["data_sha256"] == REG.hashes(REG.DATA)
    assert set(a["tests_sha256"]) == set(REG.test_files()) and len(a["tests_sha256"]) >= 7
    assert a["earlier_addenda_sha256"] == {str(p.relative_to(ROOT)): REG.sha256(p) for p in (REG.REG2, REG.REG_B1, REG.REG_3)}
    assert REG.registration_problems(a) == []


def test_plan_record(add):
    pl = add[0]["plan"]
    assert pl["n_runs"] == 1536 and len(pl["test_items"]) == 96 and len(pl["lines"]) == 96
    assert [x["config"] for x in pl["arms"]] == ["oohs_k3_A8", "oohs_k3_none", "oohs_k3_R8", "oohs_k3_A8_shuf"]
    assert len(pl["guides_sha256"]) == 98 and pl["seed_table_sha256"] == REG.sha256(P.SEEDS_JSON)
    assert pl["rule"]["max_missing_runs"] == 76 and pl["rule"]["j_min"] == 0.5 and pl["rule"]["boot_seed"] == 0
    assert pl["power"]["sims_per_cell"] == 2000 and set(pl["power"]["cells"]) == {"observed", "halved", "halved_s_down", "null"}
    assert pl["tags"]["full"] == {"a0": "runC-a0", "test": "runC-test"} and pl["package"]["blind_seed"] == 20261031
    assert 5.5 < pl["time"]["session_hours"] < 6.2 and pl["chance_group"] == 24


def test_text(add):
    a = add[0]
    assert a["title"].startswith("Addendum 4") and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", a["addendum_utc"])
    ids = [s["id"] for s in a["sections"]]
    assert ids[:3] == ["question", "design", "generation"] and {"rule", "labels", "power", "disclosures", "run"} <= set(ids)
    text = json.dumps([a["sections"], a["fill_ins"], a["custody"]])
    assert "written by the agents" in text and "before any Run C generation" in text and "deliberately" in text
    power = next(s["text"] for s in a["sections"] if s["id"] == "power")
    assert str(a["plan"]["power"]["cells"]["observed"]["P_R1"]) in power
    assert f"{100 * a['plan']['power']['cells']['null']['P_R1']:.1f}% of simulated runs" in power
    # the repo names no AI assistant or its maker in prose; the names are kept hex-encoded so this file names none
    banned = [bytes.fromhex(h).decode() for h in ("436c61756465", "436f646578", "43686174475054", "47656d696e69",
                                                   "436f70696c6f74", "416e7468726f706963")]
    assert not re.search(r"\b(" + "|".join(banned) + r")\b", text)


def test_registration_refuses_a_seed_collision(monkeypatch, tmp_path):
    tab = P.seed_table()
    monkeypatch.setattr(P, "earlier_seeds", lambda: {"b2": {tab["full"]["test"]["c1t05"][2][1]}})
    out = tmp_path / "addendum_4.json"
    monkeypatch.setattr(sys, "argv", ["register_c1.py", "--out", str(out)])
    assert REG.main() == 1 and not out.exists()


def test_run_time_check_flags_a_changed_file(add):
    a = add[0]
    bad = json.loads(json.dumps(a))
    bad["code_sha256"]["experiments/e8_clock_sing/run_b2.py"] = "0" * 64
    assert any(p.startswith("experiments/e8_clock_sing/run_b2.py") for p in REG.registration_problems(bad))
    bad = json.loads(json.dumps(a))
    bad["earlier_addenda_sha256"]["experiments/e5_motif/registration/addendum_3.json"] = "0" * 64
    assert any("addendum_3.json" in p for p in REG.registration_problems(bad))
    bad = json.loads(json.dumps(a))
    bad["data_sha256"]["experiments/e9_clock_replicate/seeds_c1.json"] = "0" * 64
    assert any("seeds_c1.json" in p for p in REG.registration_problems(bad))


@pytest.mark.parametrize("script", ["experiments/e5_motif/register.py", "experiments/e6_clock/register_b1code.py",
                                    "experiments/e8_clock_sing/register_b2.py"])
def test_nothing_registered_changed(script):
    r = subprocess.run([sys.executable, str(ROOT / script), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "current" in r.stdout, r.stdout + r.stderr


def test_e7_registration_unchanged():
    sys.path.insert(0, str(ROOT))
    from experiments.e7_melody_adapter import registration
    assert registration.verify_registration()["code_sha256"]


def test_no_new_b2_module():
    import register_b2
    mods = {f"experiments/e8_clock_sing/{p.name}" for p in (ROOT / "experiments" / "e8_clock_sing").glob("*.py")}
    assert mods <= set(register_b2.CODE)
    tests = {str(p.relative_to(ROOT)) for p in (ROOT / "experiments" / "e8_clock_sing" / "tests").glob("*.py")}
    assert tests == set(json.loads(register_b2.ADDENDUM.read_text())["tests_sha256"])
    assert not list(ROOT.glob("tests/test_c1*.py")) and not list(ROOT.glob("tests/test_e9*.py"))
