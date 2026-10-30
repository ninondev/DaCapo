"""The post-hoc Run B checks: the bootstrap agrees with decide_b2, the exact ideal bootstrap agrees with brute force
and with a large Monte Carlo, and the script is deterministic and matches the committed output."""
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "e8_clock_sing"))
import decide_b2 as D  # noqa: E402  (imported read-only)
import runb_secondary as R  # noqa: E402

RES, _ = R.load(R.SRC)
ITEMS = sorted(RES["arms"]["S*"]["items"])
COMPARATORS = list(RES["contrasts"]["delta_j"]["comparators"])
J = {a: np.array([float(RES["arms"][a]["items"][i]["j"]) for i in ITEMS]) for a in RES["arms"]}
OUT = R.analyse(RES)


def test_bootstrap_reproduces_decide_b2():
    bos = {a: RES["arms"][a]["items"] for a in RES["arms"]}
    con = D.contrasts(bos, ITEMS, COMPARATORS, D.boot_index(len(ITEMS)))
    assert con["delta_j"]["p2_5"] == RES["rule"]["P1"]["p2_5"] == OUT["p1_bootstrap"]["p2_5"]["value"] == 0.0
    assert con["delta_j"]["value"] == OUT["p1_bootstrap"]["delta_j"]["value"] == 0.25
    # the counts come from decide_b2's own resamples
    idx = D.boot_index(len(ITEMS))
    best = np.max(np.stack([J[b][idx].mean(1) for b in COMPARATORS]), axis=0)
    d = np.round(J["S*"][idx].mean(1) - best, 6)
    n_le, n_lt = OUT["p1_bootstrap"]["n_le_0"]["value"], OUT["p1_bootstrap"]["n_lt_0"]["value"]
    assert (int((d <= 0).sum()), int((d < 0).sum())) == (n_le, n_lt) == (286, 104)
    # numpy's linear 2.5th percentile of 10,000 values lies between the 250th and 251st smallest, both 0 here
    assert n_lt <= 249 and n_le >= 251
    assert OUT["p1_bootstrap"]["seeds_0_19_p2_5"]["value"] == [0.0] * 20


def test_ideal_bootstrap_brute_force_on_a_small_case():
    js = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    comps = [np.array([0.0, 1.0, 1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0, 1.0])]
    le = lt = 0
    for s in itertools.product(range(5), repeat=5):
        s = list(s)
        dlt = js[s].sum() - max(c[s].sum() for c in comps)
        le, lt = le + (dlt <= 0), lt + (dlt < 0)
    got = R.ideal_bootstrap(js, comps)
    assert abs(got["le"] - le / 5 ** 5) < 1e-12 and abs(got["lt"] - lt / 5 ** 5) < 1e-12


def test_ideal_bootstrap_agrees_with_large_monte_carlo():
    comps = [J[b] for b in COMPARATORS]
    rng = np.random.default_rng(12345)
    d = np.concatenate([R.delta_boot(J["S*"], comps, rng.integers(0, len(ITEMS), size=(100000, len(ITEMS))))
                        for _ in range(10)])
    exact_le = OUT["p1_bootstrap"]["exact_p_le_0"]["value"]
    exact_lt = OUT["p1_bootstrap"]["exact_p_lt_0"]["value"]
    assert abs((d <= 0).mean() - exact_le) < 0.003
    assert abs((d < 0).mean() - exact_lt) < 0.003
    assert OUT["p1_bootstrap"]["exact_p2_5_is_0"]["value"] is True


def test_constants_match_the_registered_scorer():
    assert R.HALLUCINATIONS == D.S.HALLUCINATIONS
    for a in ("S*", "S*_b0"):
        rows = R.arm_rows(RES)[a]
        assert all(R.kept(rows[i])["run_id"] == D.S.kept(rows[i])["run_id"] for i in ITEMS)


def test_script_is_deterministic(tmp_path):
    outs = []
    for name in ("a.json", "b.json"):
        p = tmp_path / name
        subprocess.run([sys.executable, str(HERE / "runb_secondary.py"), "--out", str(p)], check=True,
                       capture_output=True)
        outs.append(p.read_bytes())
    assert outs[0] == outs[1]
    assert outs[0] == R.OUT.read_bytes()
    doc = json.loads(outs[0])
    assert doc["source"]["file"] == "results/gpu_run3/b2_result.json"
