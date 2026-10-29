"""B2 selection: a pure, deterministic function of the Stage D rows, with the registered key, tie order, gate
eligibility and the no-eligible case."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import plan_b2 as P  # noqa: E402
import select_b2 as SEL  # noqa: E402

ITEMS = [f"item{i:02d}" for i in range(12)]


def clips(q: dict, gate: dict | None = None, missing=()) -> list[dict]:
    """q: {config: q of every clip, or {item: [q per draw]}}; gate: {config: gate of every clip}."""
    out = []
    for c in P.dev_configs():
        cid = c["config"]
        for i in ITEMS:
            for r in P.DEV_DRAWS:
                if (cid, i, r) in missing:
                    continue
                v = q.get(cid, 0.0)
                v = v[i][r] if isinstance(v, dict) else v
                out.append({"item": i, "draw": r, "config": cid, "q": v, "scored": True,
                            "gate": (gate or {}).get(cid, True)})
    return out


def pick(q, gate=None, missing=(), waive=False):
    tab = SEL.table(clips(q, gate, missing), P.dev_configs(), ITEMS)
    return tab, SEL.select(tab, waive_gate=waive)


def test_highest_j_dev_wins():
    tab, s = pick({"oohs_k4_A4": 0.6, "oohs_k3_A2": 0.4, "clarinet_k3_none": 0.55, "oohs_k5_none": 0.3})
    assert s["s_star"] == "oohs_k4_A4" and s["b_star"] == "clarinet_k3_none" and s["stage_t"]
    assert tab["oohs_k4_A4"]["j_dev"] == 12 and tab["oohs_k3_A2"]["j_dev"] == 0 and tab["oohs_k4_A4"]["n_clips"] == 36


def test_bo3_is_the_best_draw_and_missing_counts_as_zero():
    per = {i: [0.0, 0.7, 0.1] if k < 5 else [0.0, 0.0, 0.0] for k, i in enumerate(ITEMS)}
    tab, s = pick({"oohs_k3_A8": per, "clarinet_k4_W2": 0.45})
    assert tab["oohs_k3_A8"]["j_dev"] == 5 and tab["oohs_k3_A8"]["bo3"]["item00"] == 0.7
    assert s["s_star"] == "oohs_k3_A8"
    missing = {("oohs_k3_A8", i, 1) for i in ITEMS}
    tab, s = pick({"oohs_k3_A8": per, "clarinet_k4_W2": 0.45}, missing=missing)
    assert tab["oohs_k3_A8"]["j_dev"] == 0 and s["s_star"] == "clarinet_k4_W2"


def test_ties_go_to_mean_bo3_then_mean_q_then_fewer_heads_beta_k_and_oohs():
    per_a = {i: [0.6, 0.0, 0.0] for i in ITEMS}
    per_b = {i: [0.6, 0.6, 0.0] for i in ITEMS}
    _, s = pick({"oohs_k3_A2": per_a, "oohs_k4_A2": per_b})
    assert s["s_star"] == "oohs_k4_A2"  # same J and bo3, higher mean q
    _, s = pick({"oohs_k5_W2": 0.6, "oohs_k5_A8": 0.6})
    assert s["s_star"] == "oohs_k5_A8"  # fewer steered heads (7 < 80)
    _, s = pick({"oohs_k5_A8": 0.6, "oohs_k5_A4": 0.6})
    assert s["s_star"] == "oohs_k5_A4"  # smaller beta
    _, s = pick({"oohs_k5_A2": 0.6, "oohs_k4_A2": 0.6})
    assert s["s_star"] == "oohs_k4_A2"  # smaller k
    _, s = pick({"clarinet_k4_A2": 0.6, "oohs_k4_A2": 0.6})
    assert s["s_star"] == "oohs_k4_A2"  # oohs before clarinet
    _, s = pick({"clarinet_k3_none": 0.2, "oohs_k3_none": 0.2, "oohs_k3_A2": 0.1})
    assert s["b_star"] == "oohs_k3_none"
    _, s = pick({})
    assert s["s_star"] == "oohs_k3_A2" and s["b_star"] == "oohs_k3_none"  # all zero: the tie order alone


def test_gate_eligibility_and_the_no_eligible_case():
    gate = {c["config"]: False for c in P.dev_configs() if c["clock"] != "none"}
    gate["clarinet_k5_A2"] = True
    tab, s = pick({"oohs_k4_A4": 0.9, "clarinet_k5_A2": 0.1}, gate)
    assert not tab["oohs_k4_A4"]["eligible"] and s["s_star"] == "clarinet_k5_A2" and s["eligible"] == ["clarinet_k5_A2"]
    gate["clarinet_k5_A2"] = False
    tab, s = pick({"oohs_k4_A4": 0.9}, gate)
    assert s["s_star"] is None and not s["stage_t"] and s["label"] == "clock silences the voice"
    assert tab["oohs_k4_none"]["eligible"]  # the gate rule applies to clocked configs only
    _, s = pick({"oohs_k4_A4": 0.9}, gate, waive=True)
    assert s["s_star"] == "oohs_k4_A4" and s["gate_waived"]


def test_gate_rate_threshold_is_0_60():
    gates = {i: True for i in ITEMS}
    out = clips({"oohs_k4_A4": 0.9})
    passed = 0
    for c in out:
        if c["config"] == "oohs_k4_A4":
            c["gate"] = passed < 22  # 22 of 36 = 0.611
            passed += 1
    tab = SEL.table(out, P.dev_configs(), ITEMS)
    assert tab["oohs_k4_A4"]["gate_rate"] == 0.611111 and tab["oohs_k4_A4"]["eligible"]
    passed = 0
    for c in out:
        if c["config"] == "oohs_k4_A4":
            c["gate"] = passed < 21  # 0.583
            passed += 1
    tab = SEL.table(out, P.dev_configs(), ITEMS)
    assert not tab["oohs_k4_A4"]["eligible"] and gates


def test_deterministic_document_and_dev_futility():
    q = {"oohs_k4_A4": 0.6, "clarinet_k3_none": 0.7}
    tab, s = pick(q)
    assert s["dev_futility"] is True
    a = SEL.doc_bytes(SEL.selection_doc(tab, s, P.dev_configs(), {"x": "1"}, {"y": "2"}, "full", P.NO_TRIM))
    tab2, s2 = pick(q)
    b = SEL.doc_bytes(SEL.selection_doc(tab2, s2, P.dev_configs(), {"x": "1"}, {"y": "2"}, "full", P.NO_TRIM))
    assert a == b and b"utc" not in a
    _, s = pick({"oohs_k4_A4": 0.7, "clarinet_k3_none": 0.6})
    assert s["dev_futility"] is False
