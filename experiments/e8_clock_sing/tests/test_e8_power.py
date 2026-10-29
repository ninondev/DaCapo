"""B2 power simulation: the registered output has 2,000 simulations per cell and reruns identically."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import power_b2 as PW  # noqa: E402


def test_registered_output():
    doc = json.loads(PW.OUT.read_text())
    assert doc["sims_per_cell"] == 2000 and list(doc["cells"]) == ["0.0", "0.1", "0.2", "0.3", "0.4"]
    c = doc["cells"]
    assert c["0.0"]["P1"] <= 0.05 and c["0.0"]["P2"] <= 0.05
    assert c["0.1"]["P1"] < c["0.2"]["P1"] < c["0.3"]["P1"] < c["0.4"]["P1"]
    assert all(v["P1_and_P2"] <= min(v["P1"], v["P2"]) for v in c.values())


def test_small_run():
    cells = PW.simulate(40, ps=(0.0, 0.4))
    assert cells["0.0"]["P1"] < cells["0.4"]["P1"] and cells["0.4"]["joint_per_clip"] > cells["0.0"]["joint_per_clip"]


@pytest.mark.slow
def test_registered_output_reruns_identically(tmp_path, monkeypatch):
    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", ["power_b2.py", "--out", str(out)])
    assert PW.main() == 0
    assert out.read_text() == PW.OUT.read_text()
