"""C1 of Run C, registered in experiments/e5_motif/registration/addendum_4.json: a replication of B2's two fixed
recipes on 96 new lines and melodies. The items, seeds, arms, runs, the clock's windows and the seed check.

No ACE-Step import and nothing heavier than numpy, so the ACE environment, the measurement environment, the
registration and the tests all share it. B2's plan (experiments/e8_clock_sing/plan_b2.py) is imported and never
edited: the recipes are its configs, parsed by its own parse().

  S*       oohs_k3_A8       Voice Oohs guide, cover for the first 3 of 8 steps, clock beta 8 on head set A
  S*_b0    oohs_k3_none     the same without the hook (the context manager is not entered)
  S*_rand  oohs_k3_R8       S*'s guide, k and beta on readout.random_head_sets()[3 + r] for draw r
  S*_shuf  oohs_k3_A8_shuf  S*'s guide, k, heads and beta, windows rotated by floor(n/2) words

  python plan_c1.py seeds   write seeds_c1.json (refuses on any collision)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E8 = ROOT / "experiments" / "e8_clock_sing"
for p in (ROOT, E8, ROOT / "experiments" / "e6_clock"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import plan_b2 as P  # noqa: E402
import readout as R  # noqa: E402

from dacapo.seeds import DERIVATION, run_seeds  # noqa: E402

C1_MANIFEST = HERE / "manifest_c1.json"
LINES = HERE / "lines_c1.txt"
GUIDES_JSON = HERE / "guides_c1.json"
TOK_JSON = HERE / "tok_check_c1.json"
SEEDS_JSON = HERE / "seeds_c1.json"
GUIDES_DIR = HERE / "guides"
B2_RESULT = ROOT / "results" / "gpu_run3" / "b2_result.json"   # B2's per-item outcomes: power and the pooled estimate

STEPS = P.STEPS
GUIDE = "oohs"
N_ITEMS = 96
GROUP = 24                                      # chance NPA: the other melodies of the group of 24
BLOCK_ITEMS = 24                                # run_c.py generates, then scores, 24 items at a time
ARMS = (("S*", "oohs_k3_A8"), ("S*_b0", "oohs_k3_none"), ("S*_rand", "oohs_k3_R8"), ("S*_shuf", "oohs_k3_A8_shuf"))
ARM_ORDER = tuple(a for a, _ in ARMS)
COMPARATOR = "S*_b0"
DRAWS, A0_DRAWS = (0, 1, 2, 3), (0,)
HOOK_CALLS = P.HOOK_CALLS                       # 40: 5 layers x 8 steps
TAGS = {"full": {"a0": "runC-a0", "test": "runC-test"}, "tiny": {"a0": "runC-a0-tiny", "test": "runC-test-tiny"}}
A0_ITEM = {"full": "item00", "tiny": "item01"}
# The tiny CPU scale touches no Run C item and no registered seed: every stage runs on the dev item item01 under
# its own seed tags.
TINY = {"test_items": ("item01",), "test_draws": (0,), "test_labels": ("S*", "S*_b0")}
EXTRA_ITEMS = ("item00", "item01")              # E4 dev items with a Run C guide: A0 (item00) and the tiny scale
# A0 on the A0 item with draw 0 of the a0 tag: name, config, beta (None: the config's own clock), why
A0_RUNS = (
    ("s_b0", "oohs_k3_none", None, "S*_b0 once, the reference of the bit-identity check"),
    ("beta0", "oohs_k3_none", 0.0, "the hook at beta 0 on set A: bit-identical to no hook"),
    ("s_star", "oohs_k3_A8", None, "S* once: 40 hook calls, and beta 8 changes the output"),
)
A0_TINY = ("s_b0", "beta0")
A0_SCORED = {"full": ("s_star", "s_b0"), "tiny": ("s_b0", "beta0")}
# Run B's measured Stage T cost on the L4 (generation and scoring, results/gpu_run3/status.json: 96.9 min for 480
# runs) and the fixed time of a session (install, setup, A0, decide, package).
STAGE_T_SECONDS_B2 = 12.1
FIXED_HOURS = (25 + 10) / 60
# Attempts: a failed run gets at most R.MAX_ATTEMPTS (3) attempts, each in its own gen process. run_c.py starts at
# most GEN_PASSES gen processes per block in one session, and stops the block after STICKY_TRIPS gen processes that
# stopped on failures in a row spanning two items.
GEN_PASSES = 8
STICKY_TRIPS = 2
# The runtime a C1 generation depends on (fields of dacapo.ace.runtime_facts): A0 records it, every later gen process
# must match it, so a resumed session on another GPU or software stack cannot mix runs into the record.
RUNTIME_KEYS = ("device", "gpu", "gpu_sm_count", "torch_cuda", "cudnn", "versions", "attn_implementation", "dit_dtype",
                "handler_dtype", "vae_dtype", "text_encoder_dtype", "vae_encode_chunk_samples")


# items

def read_json(p: Path):
    return json.loads(Path(p).read_text())


def test_items() -> list[dict]:
    return read_json(C1_MANIFEST)["items"]


def extra_items() -> list[dict]:
    dev = {it["id"]: it for it in P.dev_items()}
    return [dev[i] for i in EXTRA_ITEMS]


def all_items() -> dict[str, dict]:
    return {it["id"]: it for it in extra_items() + test_items()}


def item_set(item_id: str) -> list[dict]:
    """The set an item belongs to; its other melodies give the chance level of pitch accuracy. A test item's set is
    its group of 24 (c1t00-c1t23, c1t24-c1t47, c1t48-c1t71, c1t72-c1t95), so each clip is compared with 23 other
    melodies, as in B2; the A0 and tiny items belong to the 12 E4 lines."""
    dev = P.dev_items()
    if item_id in {it["id"] for it in dev}:
        return dev
    its = test_items()
    i = [it["id"] for it in its].index(item_id)
    return its[GROUP * (i // GROUP):GROUP * (i // GROUP + 1)]


def guide_path(guides_dir: Path, item_id: str, guide: str = GUIDE) -> Path:
    return P.guide_path(guides_dir, item_id, guide)


# arms and runs

def arms() -> list[dict]:
    """The four arms in their registered order, each one config of plan_b2 with its label."""
    return [{**P.parse(cid), "labels": [lab]} for lab, cid in ARMS]


def label_configs() -> dict[str, str]:
    return dict(ARMS)


def seeds(scale: str, stage: str, item_id: str, draw: int) -> tuple[int, int]:
    return run_seeds(TAGS[scale][stage], item_id, draw)


def make_run(stage: str, scale: str, it: dict, draw: int, c: dict, labels: list[str] | None = None) -> dict:
    """One run, with exactly the fields of plan_b2.make_run (run_b2.generate_run reads them); the seeds come from
    the Run C tags."""
    dit, rng = seeds(scale, stage, it["id"], draw)
    heads = P.heads_for(c, draw)
    return {"run_id": f"{it['id']}_r{draw}_{c['config']}", "stage": stage, "item": it["id"], "draw": draw,
            "config": c["config"], "labels": list(labels or [c["config"]]), "guide": c["guide"], "k": c["k"],
            "clock": c["clock"],
            "head_set": c["head_set"] if c["head_set"] != "R" else f"random {P.RANDOM_FIRST + draw}",
            "heads": None if heads is None else {str(L): list(hs) for L, hs in heads.items()},
            "beta": c["beta"], "shuffle": c["shuffle"], "hooked": heads is not None, "fallback": False,
            "task": "cover-nofsq", "cover_strength": c["k"] / STEPS, "dit_seed": dit, "rng_seed": rng,
            "lyrics": it["lyrics"], "caption": P.CAPTION, "duration": float(it["duration"]),
            "bpm": int(it["melody"]["bpm"]), "keyscale": it["key"]}


def test_runs(scale: str = "full", items: list[str] | None = None) -> list[dict]:
    """C1 test: 96 items x 4 draws x 4 arms = 1,536 runs, in the order item, draw, arm. items: only these item
    ids (a block of the run), in plan order."""
    its, draws, cand = test_items(), DRAWS, arms()
    if scale == "tiny":
        its = [all_items()[i] for i in TINY["test_items"]]
        draws = TINY["test_draws"]
        cand = [a for a in cand if set(a["labels"]) & set(TINY["test_labels"])]
    if items is not None:
        keep = set(items)
        its = [it for it in its if it["id"] in keep]
    out = []
    for it in its:
        for r in draws:
            out += [make_run("test", scale, it, r, a, a["labels"]) for a in cand]
    return out


def a0_runs(scale: str = "full") -> list[dict]:
    """A0 on item00 (item01 at the tiny scale) with draw 0 of the a0 tag: S*_b0, the hook at beta 0 on set A, S*."""
    it = all_items()[A0_ITEM[scale]]
    out = []
    for name, cid, beta, why in A0_RUNS:
        if scale == "tiny" and name not in A0_TINY:
            continue
        run = make_run("a0", scale, it, A0_DRAWS[0], P.parse(cid))
        run.update(run_id=f"a0_{name}", name=name, why=why, score=name in A0_SCORED[scale])
        if beta is not None:
            run.update(hooked=True, head_set="A", beta=float(beta), heads={str(L): list(hs) for L, hs in R.ALIGN.items()})
        out.append(run)
    return out


def blocks(scale: str, size: int) -> list[list[str]]:
    """The test items in plan order, cut into blocks of size items (generation, then scoring, block by block)."""
    ids = list(dict.fromkeys(r["item"] for r in test_runs(scale)))
    size = max(1, int(size))
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def spans(it: dict, keys: list[list[int]], shuffle: bool = False) -> list:
    return P.spans(it, keys, shuffle)


def projected_hours(sec_per_run: float = STAGE_T_SECONDS_B2, n_runs: int | None = None, n_blocks: int = 1,
                    block_overhead_s: float = 120.0) -> float:
    """Session hours: the fixed time, every run at sec_per_run, and the model loads of each block (ACE-Step for
    generation and the scorer's models for scoring)."""
    n = len(test_runs("full")) if n_runs is None else n_runs
    return FIXED_HOURS + (n * sec_per_run + n_blocks * block_overhead_s) / 3600


# seeds

def b2_seeds() -> set[int]:
    """Every seed and pin of B2 (both scales, every stage), from B2's registered seed table."""
    tab = P.read_json(P.SEEDS_JSON)
    return {s for sc in P.TAGS for st in tab[sc].values() for v in st.values() for pair in v for s in pair}


def earlier_seeds() -> dict[str, set[int]]:
    """Every DiT seed and global pin used before Run C, by source: plan_b2.earlier_seeds() (E4, E5 v1 and v2 with
    the tiny tags, E6, B1 with its tiny runs, the CPU test, the fixed seeds and the E7 tags) and B2."""
    out = dict(P.earlier_seeds())
    out["b2"] = b2_seeds()
    return out


def seed_table() -> dict:
    """Every Run C seed pair, by scale and stage (committed as seeds_c1.json and hashed in addendum 4)."""
    tab: dict = {"derivation": DERIVATION, "tags": TAGS,
                 "note": "(dit_seed, rng_seed) per item and draw; every arm of an (item, draw) shares the pair; the "
                         "tiny scale runs on " + ", ".join(TINY["test_items"])}
    ids = [it["id"] for it in test_items()]
    plan = {"full": {"a0": ([A0_ITEM["full"]], A0_DRAWS), "test": (ids, DRAWS)},
            "tiny": {"a0": ([A0_ITEM["tiny"]], A0_DRAWS), "test": (list(TINY["test_items"]), TINY["test_draws"])}}
    for scale, stages in plan.items():
        tab[scale] = {st: {i: [list(seeds(scale, st, i, r)) for r in draws] for i in its}
                      for st, (its, draws) in stages.items()}
    return tab


def seed_problems(tab: dict | None = None, earlier: dict[str, set[int]] | None = None) -> list[str]:
    """Collisions among the new seeds, with any earlier seed, and between a0 and test. Empty when clean."""
    tab = tab or seed_table()
    earlier = earlier_seeds() if earlier is None else earlier
    vals = [s for sc in TAGS for per in tab[sc].values() for pairs in per.values() for p in pairs for s in p]
    out = []
    dup = sorted({s for s in vals if vals.count(s) > 1})
    if dup:
        out.append(f"new seeds repeat: {dup[:10]}")
    for src, old in earlier.items():
        hit = sorted(set(vals) & old)
        if hit:
            out.append(f"new seeds equal {src} seeds: {hit[:10]}")
    old_tags = {t for sc in P.TAGS.values() for t in sc.values()}
    clash = sorted({t for sc in TAGS.values() for t in sc.values()} & old_tags)
    if clash:
        out.append(f"tags already used by B2: {clash}")
    return out


def main() -> int:
    """python plan_c1.py seeds   write seeds_c1.json (refuses on any collision)"""
    if sys.argv[1:] != ["seeds"]:
        print(main.__doc__)
        return 2
    tab = seed_table()
    bad = seed_problems(tab)
    if bad:
        print("seed collision, nothing written:", bad)
        return 1
    SEEDS_JSON.write_text(json.dumps(tab, indent=1) + "\n")
    n = sum(len(p) for sc in TAGS for st in tab[sc].values() for p in st.values())
    print(f"wrote {SEEDS_JSON.relative_to(ROOT)}: {n} seed pairs, no collision with {len(earlier_seeds())} earlier sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
