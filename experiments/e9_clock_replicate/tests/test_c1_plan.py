"""The C1 plan: 1,536 runs in the order item, draw, arm; B2's four recipes; one seed pair per (item, draw) from the
runC-test tag, shared by the arms; the random heads per draw; the rotated windows; A0 and the tiny scale; the blocks;
and the seed table with its collision check."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import plan_b2 as P2  # noqa: E402
import plan_c1 as P  # noqa: E402
import readout as R  # noqa: E402

from dacapo.seeds import run_seeds  # noqa: E402

RUNS = P.test_runs("full")
ITEMS = [it["id"] for it in P.test_items()]


def test_runs_order_and_count():
    assert len(RUNS) == 96 * 4 * 4 == 1536 and len({r["run_id"] for r in RUNS}) == 1536
    want = [(i, r, cid) for i in ITEMS for r in P.DRAWS for _, cid in P.ARMS]
    assert [(x["item"], x["draw"], x["config"]) for x in RUNS] == want
    assert [x["labels"] for x in RUNS[:4]] == [["S*"], ["S*_b0"], ["S*_rand"], ["S*_shuf"]]


def test_arms_are_b2s_recipes():
    s, b0, rnd, shuf = (P2.parse(c) for _, c in P.ARMS)
    assert (s["guide"], s["k"], s["head_set"], s["beta"], s["shuffle"]) == ("oohs", 3, "A", 8.0, False)
    assert (b0["guide"], b0["k"], b0["head_set"], b0["clock"]) == ("oohs", 3, None, "none") and P2.unclocked(s) == b0
    assert rnd == P2.config("oohs", 3, "A8", random=True) and shuf == P2.config("oohs", 3, "A8", shuffle=True)
    b2 = P2.label_configs(P2.test_arms(s, b0))
    assert all(b2[lab] == cid for lab, cid in P.ARMS)
    sel = json.loads((ROOT / "results" / "gpu_run3" / "b2_selection.json").read_text())
    assert (sel["s_star"], sel["b_star"]) == ("oohs_k3_A8", "oohs_k3_none")


def test_every_run_is_a_k3_voice_oohs_cover():
    for x in RUNS:
        assert x["guide"] == "oohs" and x["k"] == 3 and x["cover_strength"] == 3 / 8 and x["task"] == "cover-nofsq"
        it = P.all_items()[x["item"]]
        assert x["lyrics"] == it["lyrics"] and x["keyscale"] == it["key"] and x["bpm"] == 100 and x["duration"] == 14.0
        assert x["caption"] == R.CAPTION_F and not x["fallback"]


def test_seeds_shared_by_the_arms_of_an_item_and_draw():
    by: dict = {}
    for x in RUNS:
        by.setdefault((x["item"], x["draw"]), set()).add((x["dit_seed"], x["rng_seed"]))
        assert (x["dit_seed"], x["rng_seed"]) == run_seeds("runC-test", x["item"], x["draw"])
    assert all(len(v) == 1 for v in by.values()) and len({next(iter(v)) for v in by.values()}) == 384


def test_heads_per_draw():
    for x in RUNS:
        if x["config"] == "oohs_k3_none":
            assert not x["hooked"] and x["heads"] is None and x["beta"] == 0.0 and x["head_set"] is None
            continue
        assert x["hooked"] and x["beta"] == 8.0
        if x["config"] == "oohs_k3_R8":
            want = R.random_head_sets()[3 + x["draw"]]
            assert x["head_set"] == f"random {3 + x['draw']}"
        else:
            want = R.ALIGN
            assert x["head_set"] == "A"
        assert x["heads"] == {str(L): list(h) for L, h in want.items()}
        assert x["shuffle"] == (x["config"] == "oohs_k3_A8_shuf")
    sets = {json.dumps(x["heads"], sort_keys=True) for x in RUNS if x["config"] == "oohs_k3_R8"}
    assert len(sets) == 4


def test_shuffled_windows_rotate_by_half_the_words():
    tok = P.read_json(P.TOK_JSON)["items"]
    for iid in ("c1t00", "c1t01", "c1t95"):
        it = P.all_items()[iid]
        keys = tok[iid]["keys"]
        plain, shuf = P.spans(it, keys), P.spans(it, keys, shuffle=True)
        n = len(plain)
        assert n == len(it["words"]) == len(shuf)
        spb = 60.0 / it["melody"]["bpm"]
        assert [s[1] for s in plain] == [round(nt["start"] * spb, 6) for nt in it["melody"]["notes"]]
        for j in range(n):
            assert shuf[j][0] == tuple(keys[j]) == plain[j][0]
            assert shuf[j][1:] == plain[(j + n // 2) % n][1:]


def test_a0_runs():
    full = P.a0_runs("full")
    assert [r["name"] for r in full] == ["s_b0", "beta0", "s_star"] and {r["item"] for r in full} == {"item00"}
    assert len({(r["dit_seed"], r["rng_seed"]) for r in full}) == 1
    assert (full[0]["dit_seed"], full[0]["rng_seed"]) == run_seeds("runC-a0", "item00", 0)
    b0, beta0, s = full
    assert not b0["hooked"] and beta0["hooked"] and beta0["beta"] == 0.0 and beta0["head_set"] == "A"
    assert beta0["heads"] == {str(L): h for L, h in R.ALIGN.items()} and beta0["config"] == "oohs_k3_none"
    assert s["hooked"] and s["beta"] == 8.0 and s["config"] == "oohs_k3_A8"
    assert [r["score"] for r in full] == [True, False, True]
    tiny = P.a0_runs("tiny")
    assert [r["name"] for r in tiny] == ["s_b0", "beta0"] and {r["item"] for r in tiny} == {"item01"}
    assert all(r["score"] for r in tiny) and (tiny[0]["dit_seed"], tiny[0]["rng_seed"]) == run_seeds("runC-a0-tiny", "item01", 0)


def test_tiny_scale_touches_no_run_c_item():
    tiny = P.test_runs("tiny")
    assert [x["run_id"] for x in tiny] == ["item01_r0_oohs_k3_A8", "item01_r0_oohs_k3_none"]
    assert all((x["dit_seed"], x["rng_seed"]) == run_seeds("runC-test-tiny", "item01", 0) for x in tiny)


def test_blocks_and_item_filter():
    bl = P.blocks("full", P.BLOCK_ITEMS)
    assert len(bl) == 4 and [b[0] for b in bl] == ["c1t00", "c1t24", "c1t48", "c1t72"] and sum(map(len, bl)) == 96
    sub = P.test_runs("full", bl[1])
    assert len(sub) == 24 * 16 and sub == [x for x in RUNS if x["item"] in set(bl[1])]
    assert P.blocks("tiny", 24) == [["item01"]] and len(P.blocks("full", 50)) == 2


def test_chance_sets_are_groups_of_24():
    s = P.item_set("c1t30")
    assert [it["id"] for it in s] == [f"c1t{i:02d}" for i in range(24, 48)]
    assert len(P.item_set("item01")) == 12 and P.item_set("c1t95")[-1]["id"] == "c1t95"


def test_seed_table_file_and_no_collision():
    tab = P.seed_table()
    assert P.read_json(P.SEEDS_JSON) == json.loads(json.dumps(tab))
    assert P.seed_problems(tab) == []
    assert len(tab["full"]["test"]) == 96 and all(len(v) == 4 for v in tab["full"]["test"].values())
    assert set(tab["tiny"]["test"]) == {"item01"} and set(tab["full"]["a0"]) == {"item00"}
    early = P.earlier_seeds()
    assert {"b2", "e4", "e5_v2", "e6", "b1", "e7_tags"} <= set(early)
    b2 = P2.read_json(P2.SEEDS_JSON)
    assert b2["full"]["test"]["b2t00"][0][0] in early["b2"] and b2["tiny"]["dev"]["item00"][0][1] in early["b2"]


def test_seed_check_refuses_a_collision():
    tab = P.seed_table()
    hit = tab["full"]["test"]["c1t40"][2][1]
    bad = P.seed_problems(tab, {"b2": {hit}})
    assert bad and "b2" in bad[0]
    dup = json.loads(json.dumps(tab))
    dup["tiny"]["test"]["item01"][0] = dup["full"]["test"]["c1t00"][0]
    assert any("repeat" in p for p in P.seed_problems(dup, {}))


def test_time_projection():
    assert P.projected_hours(12.1, 1536, 4) == pytest.approx(35 / 60 + (1536 * 12.1 + 4 * 120) / 3600)
    assert 5.5 < P.projected_hours(n_blocks=4) < 6.2
