"""The post-hoc Run C checks: the script agrees with c1_result.json's own J, Delta_J and bootstrap p2.5 and with
decide_c1's code, the shifted-count score reproduces the registered J, the sign test is exact, and the script is
deterministic and matches the committed output."""
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # the registered folders imported below stay as they are
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "e9_clock_replicate"))
import decide_c1 as C  # noqa: E402  (imported read-only)
import runc_secondary as R  # noqa: E402

C1, _ = R.load(R.C1_SRC)
B2, _ = R.load(R.B2_SRC)
ITEMS = sorted(C1["arms"]["S*"]["items"])
J = {a: np.array([float(C1["arms"][a]["items"][i]["j"]) for i in ITEMS]) for a in R.C1_ARMS}
OUT = R.analyse(C1, B2)


def v(section, key):
    return OUT[section][key]["value"]


def test_j_delta_j_and_p2_5_agree_with_c1_result():
    rule = C1["rule"]
    assert C1["bootstrap"]["n_boot"] == R.N_BOOT and C1["bootstrap"]["seed"] == R.C1_SEED
    for a in R.C1_ARMS:
        assert R.rnd(J[a].mean()) == C1["arms"][a]["J"]
    assert v("j_s_star_bootstrap", "j") == rule["R1"]["J_s_star"] == rule["R2"]["J"] == 0.489583
    assert v("c1_vs_b2", "delta_j_c1") == rule["R1"]["delta_j"] == C1["contrasts"]["delta_j"]["value"] == 0.3125
    # the bootstrap from c1_result's own per-item outcomes and the registered index
    idx = R.boot_index(len(ITEMS), R.C1_SEED)
    d = np.round(J["S*"][idx].mean(1) - J["S*_b0"][idx].mean(1), 6)
    assert R.rnd(np.percentile(d, 2.5)) == rule["R1"]["p2_5"] == C1["contrasts"]["delta_j"]["p2_5"] == 0.197917
    assert R.ci(d) == rule["R1"]["ci95"]
    assert all(OUT["consistency"][k]["value"] is True for k in
               ("kept_draws_reproduced", "j_reproduced", "bootstrap_index_reproduced", "r1_reproduced",
                "pooled_reproduced"))


def test_agrees_with_decide_c1_code():
    bos = {a: C1["arms"][a]["items"] for a in R.C1_ARMS}
    con = C.D.contrasts(bos, ITEMS, ["S*_b0"], C.D.boot_index(len(ITEMS)))
    assert con["delta_j"]["value"] == v("c1_vs_b2", "delta_j_c1")
    assert con["delta_j"]["p2_5"] == C1["rule"]["R1"]["p2_5"]
    assert con["paired"]["S*_b0"]["mean_bo4_q"]["value"] == v("c1_vs_b2", "mean_kept_q_diff_c1")
    pooled = C.pooled(bos, C.b2_item_outcomes({"results": B2}))
    assert pooled["delta_j"] == C1["pooled_with_b2"]["delta_j"]
    b2_con = B2["contrasts"]["paired"]["S*_b0"]
    assert b2_con["J"]["value"] == v("c1_vs_b2", "delta_j_b2") == B2["rule"]["P1"]["delta_j"] == 0.25
    assert b2_con["mean_bo4_q"]["value"] == v("c1_vs_b2", "mean_kept_q_diff_b2")


def test_shifted_score_at_zero_is_the_registered_j():
    rows = R.arm_rows(C1)["S*"]
    assert np.array_equal(R.item_j(rows, ITEMS, R.shifted(0, 0)), J["S*"])
    assert v("consistency", "npa_v_times_notes_max_deviation") < 1e-5
    assert v("consistency", "wr_times_words_max_deviation") < 1e-5
    m = OUT["r2_margin"]
    assert m["n_items_joint"]["value"] == 47
    assert m["n_items_joint_minus_1_note"]["value"] <= 47 <= m["n_items_joint_plus_1_note"]["value"]
    assert m["n_items_joint_minus_1_word"]["value"] <= 47 <= m["n_items_joint_plus_1_word"]["value"]


def test_sign_test_is_exact():
    for w, lo in ((8, 2), (5, 5), (0, 3), (9, 4), (0, 0)):   # every sign sequence counted
        n, obs = w + lo, abs(w - lo)
        brute = sum(1 for s in itertools.product((0, 1), repeat=n) if abs(2 * sum(s) - n) >= obs) / 2 ** n if n else 1.0
        assert abs(R.sign_test(w, lo) - brute) < 1e-12
    # C1's 36 against 6, both tails summed exactly
    tails = sum(math.comb(42, k) for k in range(43) if abs(2 * k - 42) >= 30)
    assert R.sign_test(36, 6) == tails / 2 ** 42
    assert v("sign_test_s_star_vs_b0", "p_two_sided") == R.sig(tails / 2 ** 42)
    diff = J["S*"] - J["S*_b0"]
    s = OUT["sign_test_s_star_vs_b0"]
    assert (s["wins"]["value"], s["losses"]["value"], s["ties"]["value"]) == (
        int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum()))
    assert s["wins"]["value"] + s["losses"]["value"] + s["ties"]["value"] == len(ITEMS)


def test_c1_and_b2_resamples_are_independent_streams():
    assert R.B2_SEED_JOINT != R.C1_SEED
    lo, hi = v("c1_vs_b2", "delta_j_c1_minus_b2_ci95")
    assert lo <= v("c1_vs_b2", "delta_j_c1_minus_b2") <= hi
    assert v("c1_vs_b2", "delta_j_c1_minus_b2") == R.rnd(
        v("b2_to_c1_change", "change_s_star") - v("b2_to_c1_change", "change_s_b0"))


def test_script_is_deterministic(tmp_path):
    outs = []
    for name in ("a.json", "b.json"):
        p = tmp_path / name
        subprocess.run([sys.executable, str(HERE / "runc_secondary.py"), "--out", str(p)], check=True,
                       capture_output=True)
        outs.append(p.read_bytes())
    assert outs[0] == outs[1]
    assert outs[0] == R.OUT.read_bytes()
    doc = json.loads(outs[0])
    assert doc["sources"]["c1"]["file"] == "results/gpu_run4/c1_result.json"
    assert doc["sources"]["b2"]["file"] == "results/gpu_run3/b2_result.json"
