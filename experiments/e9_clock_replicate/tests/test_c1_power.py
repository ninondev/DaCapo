"""The power simulation: B2's per-item outcomes, the rule it applies (the registered bootstrap index), its cells, and
power_c1.json as written by power_c1.py."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import decide_c1 as D  # noqa: E402
import power_c1 as PW  # noqa: E402


def test_b2_pairs():
    x = PW.b2_pairs()
    assert x.shape == (24, 2) and x[:, 0].sum() == 14 and x[:, 1].sum() == 8
    assert int(((x[:, 0] == 1) & (x[:, 1] == 0)).sum()) == 8 and int(((x[:, 0] == 0) & (x[:, 1] == 1)).sum()) == 2


def test_cells_change_what_they_say():
    rng = np.random.default_rng(1)
    pairs = PW.b2_pairs()
    obs = np.mean([PW.draw(rng, pairs, "observed")[:, 0].mean() - PW.draw(rng, pairs, "observed")[:, 1].mean()
                   for _ in range(400)])
    half = [PW.draw(rng, pairs, "halved") for _ in range(400)]
    down = [PW.draw(rng, pairs, "halved_s_down") for _ in range(400)]
    null = [PW.draw(rng, pairs, "null") for _ in range(400)]
    assert abs(obs - 0.25) < 0.02
    assert abs(np.mean([h[:, 0].mean() - h[:, 1].mean() for h in half]) - 0.125) < 0.015
    assert abs(np.mean([h[:, 0].mean() for h in half]) - 14 / 24) < 0.015
    assert abs(np.mean([h[:, 0].mean() for h in down]) - 11 / 24) < 0.015
    assert abs(np.mean([h[:, 0].mean() - h[:, 1].mean() for h in null])) < 0.015


def test_rule_matches_the_decision_code():
    """power_c1.decide applies the same rule as decide_c1 (the registered index, rounding, R1 and R2)."""
    idx = np.random.default_rng(0).integers(0, 96, (10000, 96))
    for w, l, r1 in ((10, 3, False), (11, 3, True)):
        x = np.zeros((96, 2))
        x[:w, 0] = 1
        x[w:w + l, 1] = 1
        got = PW.decide(x, idx)
        bos = {a: {f"c1t{i:02d}": {"bo4": float(x[i, k]), "j": bool(x[i, k])} for i in range(96)}
               for k, a in enumerate(("S*", "S*_b0"))}
        con = D.D.contrasts(bos, sorted(bos["S*"]), ["S*_b0"], D.D.boot_index(96))
        assert got["r1"] == r1 == D.rule({"S*": {"J": x[:, 0].mean(), "n_items_joint": w, "n_items": 96},
                                         "S*_b0": {"J": x[:, 1].mean()}}, con)["R1"]["held"]
        assert got["p25"] == con["delta_j"]["p2_5"]


def test_simulation_is_reproducible():
    a = PW.simulate(40, cells=("observed", "null"))
    assert a == PW.simulate(40, cells=("observed", "null")) and set(a) == {"observed", "null"}


def test_power_file():
    doc = json.loads(PW.OUT.read_text())
    assert doc["sims_per_cell"] == 2000 and doc["n_items"] == 96 and set(doc["cells"]) == set(PW.CELLS)
    assert doc["b2"] == {"n_items": 24, "J_s_star": 0.583333, "J_s_b0": 0.333333, "wins": 8, "losses": 2, "ties": 14}
    import hashlib
    assert doc["source_sha256"] == hashlib.sha256(PW.B2_RESULT.read_bytes()).hexdigest()
    c = doc["cells"]
    assert c["observed"]["P_R1"] > 0.9 and c["null"]["P_R1"] < 0.05 and c["halved"]["P_R1"] < c["observed"]["P_R1"]
    assert all(v["P_size_bar_and_R1"] <= v["P_R1"] and v["P_R1_and_R2"] <= min(v["P_R1"], v["P_R2"]) for v in c.values())
    assert PW.simulate(2000, cells=("null",))["null"] == c["null"]
