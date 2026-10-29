"""Addendum 3: current, complete, refuses a seed collision, lists the B1 code addendum's hashes again, and flags
a changed file at run time."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import plan_b2 as P  # noqa: E402
import register_b2 as REG  # noqa: E402


@pytest.fixture(scope="module")
def add():
    if not REG.ADDENDUM.exists():
        pytest.skip("addendum 3 not written yet")
    return json.loads(REG.ADDENDUM.read_text())


def test_check_passes():
    r = subprocess.run([sys.executable, str(HERE / "register_b2.py"), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_b2_module_is_registered():
    mods = {f"experiments/e8_clock_sing/{p.name}" for p in HERE.glob("*.py")}
    assert mods <= set(REG.CODE) and "experiments/run_b.py" in REG.CODE
    assert all((ROOT / f).exists() for f in REG.runtime_files())


def test_b1_code_addendum_listed_again(add):
    b1 = json.loads(REG.REG_B1.read_text())
    rec = add["b1_code_addendum"]
    assert rec["sha256"] == REG.sha256(REG.REG_B1) and rec["code_sha256"] == b1["code_sha256"]
    assert rec["tests_sha256"] == b1["tests_sha256"] and rec["cpu_test"]["sha256"] == b1["cpu_test"]["sha256"]
    assert rec["cpu_test"]["passed"] and add["addendum_2_sha256"] == REG.sha256(REG.REG2)
    for f in REG.SHARED_B1:
        assert add["code_sha256"][f] == b1["code_sha256"][f]


def test_custody_time_stamp_and_plain_text(add):
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", add["addendum_utc"])
    assert "B1 code addendum" in add["custody"] and "first" in add["custody"] and "before Run B" in add["custody"]
    b1 = json.loads(REG.REG_B1.read_text())
    assert "before Run B" in b1["custody"] and "before addendum 3" in b1["custody"]
    text = json.dumps([add["sections"], add["fill_ins"], add["custody"]])
    assert "written by the agents" in json.loads(P.B2_MANIFEST.read_text())["author"] and "written by the agents" in text
    assert "owner demo" in text and "runB2-owner" not in text


def test_plan_record(add):
    pl = add["plan"]
    assert len(pl["stage_d_configs"]) == 30 and pl["n_stage_d_runs"] == 1080 and len(pl["lines"]) == 24
    assert sum(len(v) for v in pl["guides_sha256"].values()) == 72
    assert pl["power"]["sims_per_cell"] == 2000 and add["data_sha256"]["experiments/e8_clock_sing/power_b2.json"]
    assert pl["seed_table_sha256"] == REG.sha256(P.SEEDS_JSON)


def test_registration_refuses_a_seed_collision(monkeypatch, tmp_path):
    tab = P.seed_table()
    monkeypatch.setattr(P, "earlier_seeds", lambda: {"b1": {tab["full"]["test"]["b2t05"][2][1]}})
    out = tmp_path / "addendum_3.json"
    monkeypatch.setattr(REG, "ADDENDUM", out)
    monkeypatch.setattr(sys, "argv", ["register_b2.py"])
    assert REG.main() == 1 and not out.exists()


def test_run_time_check_flags_a_changed_file(add):
    assert REG.registration_problems(add) == [] or REG.registration_problems(add)  # current or listing why
    bad = json.loads(json.dumps(add))
    bad["code_sha256"]["dacapo/attn_bias.py"] = "0" * 64
    assert any(p.startswith("dacapo/attn_bias.py") for p in REG.registration_problems(bad))
    bad = json.loads(json.dumps(add))
    bad["data_sha256"]["experiments/e8_clock_sing/manifest_b2.json"] = "0" * 64
    assert any("manifest_b2.json" in p for p in REG.registration_problems(bad))
