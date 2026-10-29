"""B2 of Run B, registered in experiments/e5_motif/registration/addendum_3.json: the items, seeds, Stage D
grid, Stage T arms, the clock's windows, the registered timing trim and the seed collision check.

No ACE-Step import and nothing heavier than numpy, so the ACE environment, the measurement environment,
the registration and the tests all share it.

A config is one recipe: a guide timbre and cover depth k (or no guide), and a clock (no hook, or beta on
a head set). Its id reads oohs_k3_A2: Voice Oohs guide, k = 3 of 8 steps under the guide, beta 2 on the
seven alignment heads. W is all 16 heads of layers 2-6, R is E6's random set 3 + r for draw r, and the
suffix _shuf rotates the windows so that word j's tokens get the window of word (j + floor(n/2)) mod n.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "e6_clock"))
import readout as R  # noqa: E402

from dacapo.seeds import DERIVATION, run_seeds  # noqa: E402

E4_MANIFEST = ROOT / "experiments" / "e4_sing" / "manifest.json"
B2_MANIFEST = HERE / "manifest_b2.json"
LINES = HERE / "lines_b2.txt"
GUIDES_JSON = HERE / "guides_b2.json"
TOK_JSON = HERE / "tok_check_b2.json"
SEEDS_JSON = HERE / "seeds_b2.json"
GUIDES_DIR = HERE / "guides"

STEPS, SHIFT = R.STEPS, R.SHIFT
CAPTION = R.CAPTION_F
GUIDES = {"oohs": 53, "clarinet": 71}          # GM Voice Oohs and clarinet, as in E4
KS = (3, 4, 5)                                  # cover steps under the guide: audio_cover_strength k / 8
CLOCKS = {"none": (None, 0.0), "A2": ("A", 2.0), "A4": ("A", 4.0), "A8": ("A", 8.0), "W2": ("W", 2.0)}
LAYERS = (2, 3, 4, 5, 6)
N_HEADS = 16
HEAD_SETS = {"A": R.ALIGN, "W": {L: list(range(N_HEADS)) for L in LAYERS}}
RANDOM_FIRST = 3                                # S*_rand uses readout.random_head_sets()[3 + r] for draw r
HOOK_CALLS = len(LAYERS) * STEPS                # 40
DEV_DRAWS, TEST_DRAWS, A0_DRAWS = (0, 1, 2), (0, 1, 2, 3), (0, 1)
TAGS = {"full": {"a0": "runB2-a0", "dev": "runB2-dev", "test": "runB2-test"},
        "tiny": {"a0": "runB2-a0-tiny", "dev": "runB2-dev-tiny", "test": "runB2-test-tiny"}}
# The unclocked guided arms of Stage T that P1 compares S* with (noguide is a reference, not a comparator).
COMPARATORS = ("S*_b0", "ctx_oohs_k3", "ctx_oohs_k4", "B*")
ARM_ORDER = ("S*", "S*_b0", "S*_rand", "S*_shuf", "ctx_oohs_k3", "ctx_oohs_k4", "noguide", "B*")
# Registered timing trim (A0): projected hours = fixed hours + planned runs x measured seconds per run of their kind.
BUDGET_HOURS = 7.0
FIXED_HOURS = (25 + 10) / 60                    # install, setup and A0 (25 min); decide and package (10 min)
N_B1 = 180
# Whisper large-v3 sometimes repeats "oh, oh, ..." up to its 448-token limit on wordless audio. On Run A's L4 such
# a clip took about 50 s to score against a median of 4.1 s (results/gpu_run2/e6_scores.json), and loops hit 22 of
# 96 guided E4 clips at k3 and k4 and 6 of 88 E6 text2music clips. A0 scores too few clips to see this rate, so
# the projection adds it to every run as a registered allowance.
LOOP_EXTRA_S = 46.0
LOOP_RATE = {"guided": 0.25, "text2music": 0.08}
# The tiny CPU scale touches no test item and no registered seed: Stage T runs on the dev item item01 under
# its own seed tag. Its Stage D configs differ in k, so decide compares guide latents across k.
TINY = {"dev": (("item00", 0, "oohs_k3_none"), ("item00", 0, "oohs_k4_A2")),
        "test_items": ("item01",), "test_draws": (0,), "test_labels": ("S*", "S*_b0")}
NO_TRIM = {"drop_clarinet_k3": False, "drop_noguide_test": False}


# items

def read_json(p: Path):
    return json.loads(Path(p).read_text())


def dev_items() -> list[dict]:
    """item00-item11 of the E4 manifest (registered in addendum 2)."""
    return read_json(E4_MANIFEST)["items"]


def test_items() -> list[dict]:
    return read_json(B2_MANIFEST)["items"]


def items(stage: str) -> list[dict]:
    return dev_items() if stage in ("dev", "a0") else test_items()


def all_items() -> dict[str, dict]:
    return {it["id"]: it for it in dev_items() + test_items()}


def guide_path(guides_dir: Path, item_id: str, guide: str) -> Path:
    return Path(guides_dir) / f"{item_id}_{guide}.wav"


def written_onsets(it: dict) -> list[float]:
    spb = 60.0 / float(it["melody"]["bpm"])
    return [round(n["start"] * spb, R.DIGITS) for n in it["melody"]["notes"]]


# configs

def config(guide: str | None, k: int | None, clock: str = "none", shuffle: bool = False, random: bool = False) -> dict:
    """One recipe. random: the clock's beta on E6 random set 3 + r instead of its head set."""
    if guide is None:
        if clock != "none" or shuffle or random:
            raise ValueError("the unguided arm has no clock")
        return {"config": "noguide", "guide": None, "k": None, "clock": "none", "head_set": None, "beta": 0.0,
                "shuffle": False}
    if guide not in GUIDES or k not in KS or clock not in CLOCKS:
        raise ValueError(f"unknown config {guide} k{k} {clock}")
    hs, beta = CLOCKS[clock]
    if (shuffle or random) and hs is None:
        raise ValueError("only a clocked config can be shuffled or moved to random heads")
    if random:
        hs, clock = "R", f"R{int(beta)}"
    cid = f"{guide}_k{k}_{clock}" + ("_shuf" if shuffle else "")
    return {"config": cid, "guide": guide, "k": k, "clock": clock, "head_set": hs, "beta": float(beta),
            "shuffle": bool(shuffle)}


def parse(cid: str) -> dict:
    if cid == "noguide":
        return config(None, None)
    parts = cid.split("_")
    guide, k, clock = parts[0], int(parts[1][1:]), parts[2]
    shuffle = len(parts) > 3 and parts[3] == "shuf"
    if clock.startswith("R"):
        return config(guide, k, "A" + clock[1:], shuffle, random=True)
    return config(guide, k, clock, shuffle)


def clocked(c: dict) -> bool:
    return c["head_set"] is not None


def n_steered(c: dict) -> int:
    """Steered heads (the first tie-break of the selection)."""
    if c["head_set"] is None:
        return 0
    return sum(len(v) for v in HEAD_SETS.get(c["head_set"], R.ALIGN).values())


def heads_for(c: dict, draw: int) -> dict[int, list[int]] | None:
    if c["head_set"] is None:
        return None
    if c["head_set"] == "R":
        return R.random_head_sets()[RANDOM_FIRST + draw]
    return HEAD_SETS[c["head_set"]]


def unclocked(c: dict) -> dict:
    return config(c["guide"], c["k"]) if c["guide"] else config(None, None)


def dev_configs(trim: dict | None = None) -> list[dict]:
    """The 30 Stage D configs in the order guide, k, clock (25 after the trim's first step)."""
    trim = trim or NO_TRIM
    out = [config(g, k, c) for g in GUIDES for k in KS for c in CLOCKS]
    if trim.get("drop_clarinet_k3"):
        out = [c for c in out if not (c["guide"] == "clarinet" and c["k"] == 3)]
    return out


def test_arms(s_star: dict, b_star: dict, trim: dict | None = None) -> list[dict]:
    """Stage T arms in the order of addendum 3 (section 3.9), coinciding arms merged into one entry that
    carries every label: [{"config": ..., "labels": [...]}]."""
    trim = trim or NO_TRIM
    cand = [("S*", s_star), ("S*_b0", unclocked(s_star))]
    if s_star["head_set"] == "A":
        cand.append(("S*_rand", config(s_star["guide"], s_star["k"], s_star["clock"], random=True)))
    cand.append(("S*_shuf", config(s_star["guide"], s_star["k"], s_star["clock"], shuffle=True)))
    cand += [("ctx_oohs_k3", config("oohs", 3)), ("ctx_oohs_k4", config("oohs", 4))]
    if not trim.get("drop_noguide_test"):
        cand.append(("noguide", config(None, None)))
    cand.append(("B*", b_star))
    merged: dict[str, dict] = {}
    for label, c in cand:
        merged.setdefault(c["config"], {**c, "labels": []})["labels"].append(label)
    return list(merged.values())


def label_configs(arms: list[dict]) -> dict[str, str]:
    """arm label -> config id."""
    return {lab: a["config"] for a in arms for lab in a["labels"]}


# runs

def seeds(scale: str, stage: str, item_id: str, draw: int) -> tuple[int, int]:
    return run_seeds(TAGS[scale][stage], item_id, draw)


def make_run(stage: str, scale: str, it: dict, draw: int, c: dict, labels: list[str] | None = None) -> dict:
    dit, rng = seeds(scale, stage, it["id"], draw)
    heads = heads_for(c, draw)
    return {"run_id": f"{it['id']}_r{draw}_{c['config']}", "stage": stage, "item": it["id"], "draw": draw,
            "config": c["config"], "labels": list(labels or [c["config"]]), "guide": c["guide"], "k": c["k"],
            "clock": c["clock"], "head_set": c["head_set"] if c["head_set"] != "R" else f"random {RANDOM_FIRST + draw}",
            "heads": None if heads is None else {str(L): list(hs) for L, hs in heads.items()},
            "beta": c["beta"], "shuffle": c["shuffle"], "hooked": heads is not None, "fallback": False,
            "task": "cover-nofsq" if c["guide"] else "text2music",
            "cover_strength": c["k"] / STEPS if c["guide"] else 1.0, "dit_seed": dit, "rng_seed": rng,
            "lyrics": it["lyrics"], "caption": CAPTION, "duration": float(it["duration"]),
            "bpm": int(it["melody"]["bpm"]), "keyscale": it["key"]}


def with_fallback(run: dict) -> dict:
    """The registered fallback after a GPU A0 difference: an unclocked run gets the hook at beta 0 on set A."""
    if run["hooked"]:
        return run
    return {**run, "hooked": True, "fallback": True, "head_set": "A", "beta": 0.0,
            "heads": {str(L): list(hs) for L, hs in R.ALIGN.items()}}


def dev_runs(scale: str = "full", trim: dict | None = None) -> list[dict]:
    """Stage D: 12 items x 3 draws x 30 configs = 1,080 runs, in the order item, draw, config."""
    cfgs = dev_configs(trim)
    out = []
    for it in dev_items():
        for r in DEV_DRAWS:
            out += [make_run("dev", scale, it, r, c) for c in cfgs]
    if scale == "tiny":
        keep = {f"{i}_r{r}_{c}" for i, r, c in TINY["dev"]}
        out = [x for x in out if x["run_id"] in keep]
    return out


def test_runs(s_star: dict, b_star: dict, scale: str = "full", trim: dict | None = None) -> list[dict]:
    """Stage T: 24 items x 4 draws x 5-8 arms, in the order item, draw, arm."""
    arms = test_arms(s_star, b_star, trim)
    its, draws = test_items(), TEST_DRAWS
    if scale == "tiny":
        its = [all_items()[i] for i in TINY["test_items"]]
        draws = TINY["test_draws"]
        arms = [a for a in arms if set(a["labels"]) & set(TINY["test_labels"])]
    out = []
    for it in its:
        for r in draws:
            out += [make_run("test", scale, it, r, a, a["labels"]) for a in arms]
    return out


A0_RUNS = (  # name, draw, config, beta (None: no hook), why
    ("t2m_nohook", 0, "noguide", None, "reference"),
    ("t2m_nohook_repeat", 0, "noguide", None, "determinism repeat"),
    ("t2m_A0", 0, "noguide", 0.0, "beta 0 bit-identity, text2music"),
    ("cover_nohook", 1, "oohs_k4_none", None, "reference"),
    ("cover_A0", 1, "oohs_k4_none", 0.0, "beta 0 bit-identity, cover-nofsq k4"),
    ("cover_A2", 1, "oohs_k4_none", 2.0, "beta 2 changes the output"),
    ("cover_k3", 1, "oohs_k3_none", None, "the same guide latent at k3"),
    ("cover_k5", 1, "oohs_k5_none", None, "the same guide latent at k5"),
)
A0_TINY = ("cover_nohook", "cover_A0")


def a0_runs(scale: str = "full") -> list[dict]:
    """GPU A0 on item00 with the runB2-a0 seeds (draw 0 text2music, draw 1 Voice Oohs k4)."""
    it = next(x for x in dev_items() if x["id"] == "item00")
    out = []
    for name, r, cid, beta, why in A0_RUNS:
        if scale == "tiny" and name not in A0_TINY:
            continue
        run = make_run("a0", scale, it, r, parse(cid))
        run.update(run_id=f"a0_{name}", name=name, why=why)
        if beta is not None:
            run.update(hooked=True, head_set="A", beta=float(beta), heads={str(L): list(hs) for L, hs in R.ALIGN.items()})
        out.append(run)
    return out


# windows

def spans(it: dict, keys: list[list[int]], shuffle: bool = False) -> list:
    """The clock's windows for one item: word j's keys get +beta from its written onset to the next onset
    (the last word to its note end); shuffle rotates them by floor(n/2) words."""
    from dacapo.attn_bias import rotate_spans, word_spans

    sp = word_spans(it["melody"]["notes"], it["melody"]["bpm"], keys)
    return rotate_spans(sp, len(sp) // 2) if shuffle else sp


# timing trim

def seconds_per_run(gen_s: float, score_s: float, kind: str) -> float:
    """Measured generation and scoring seconds of one run of a kind (guided or text2music), plus the
    registered allowance for Whisper loops."""
    return gen_s + score_s + LOOP_RATE[kind] * LOOP_EXTRA_S


def projected_hours(sec: dict[str, float], trim: dict) -> float:
    """sec: seconds per run by kind. B1 and Stage T's noguide arm are text2music; Stage D and the other seven
    Stage T arms are guided. Stage T is projected at its largest, 8 arms."""
    n_dev = len(dev_configs(trim)) * len(dev_items()) * len(DEV_DRAWS)
    n_cell = 24 * len(TEST_DRAWS)
    n_t2m = N_B1 + (0 if trim.get("drop_noguide_test") else n_cell)
    return FIXED_HOURS + (n_t2m * sec["text2music"] + (n_dev + 7 * n_cell) * sec["guided"]) / 3600


def timing_trim(sec: dict[str, float]) -> dict:
    """The registered trim: past 7.0 projected hours, drop the five clarinet-k3 configs (-180 runs), and if
    still past, also drop noguide from Stage T (-96 runs)."""
    trim = dict(NO_TRIM)
    steps = [round(projected_hours(sec, trim), 3)]
    if steps[-1] > BUDGET_HOURS:
        trim["drop_clarinet_k3"] = True
        steps.append(round(projected_hours(sec, trim), 3))
        if steps[-1] > BUDGET_HOURS:
            trim["drop_noguide_test"] = True
            steps.append(round(projected_hours(sec, trim), 3))
    return {**trim, "seconds_per_run": {k: round(float(v), 2) for k, v in sec.items()}, "projected_hours": steps,
            "budget_hours": BUDGET_HOURS, "loop_allowance": {"extra_s": LOOP_EXTRA_S, "rate": dict(LOOP_RATE)}}


# seeds

def e7_ids() -> list[str]:
    try:
        return [it["id"] for it in read_json(ROOT / "experiments" / "e7_melody_adapter" / "melodies_suno.json")["items"]]
    except (OSError, KeyError, ValueError):
        return [f"mel{i:02d}" for i in range(12)]


def earlier_seeds() -> dict[str, set[int]]:
    """Every DiT seed and global pin used before B2, by source (addendum 3, section seeds)."""
    sys.path.insert(0, str(ROOT / "experiments" / "e5_motif"))
    import v2_config as C

    out: dict[str, set[int]] = {"e4": {1, 2}, "e5_v1": {1001, 2002}, "e5_reserved": set(C.RESERVED),
                                "posthoc_and_heads": {42, 43, 44, 20261027, 20261029}}
    tab = C.seed_table()
    out["e5_v2"] = {s for ph in ("a0", "p0b", "main") for v in tab[ph].values() for p in v for s in p}
    ids = list(C.P0B_IDS) + list(C.MAIN_IDS) + list(C.A0_IDS)
    out["e5_v2_tiny"] = {s for tag in C.TINY_TAGS.values() for i in ids for d in C.DRAWS for s in run_seeds(tag, i, d)}
    e6 = {61000 + 10 * i + r for i in range(12) for r in range(4)}
    e6 |= {62000 + 100 * v + 10 * w + r for v in range(4) for w in range(2) for r in range(5)}
    out["e6"] = e6 | {s + R.RNG_OFFSET for s in e6}
    out["e6_acceptance"] = {6000, 6001, 6000 + R.RNG_OFFSET, 6001 + R.RNG_OFFSET}
    b1 = {63000 + 10 * i + r for i in range(12) for r in range(3)}
    out["b1"] = b1 | {s + R.RNG_OFFSET for s in b1}
    import run_b1
    out["b1_tiny"] = {s for x in run_b1.plan("tiny") for s in (x["dit_seed"], x["rng_seed"])}
    out["runb_cpu_test"] = {6002, 906002}
    e7_tags = ("e7-suno", "e7-smoke", "e7-data", "e7-dev", "e7-test", "e7-listen")
    out["e7_tags"] = {s for t in e7_tags for i in e7_ids() for d in range(4) for s in run_seeds(t, i, d)}
    return out


def seed_table() -> dict:
    """Every B2 seed pair, by scale and stage (committed as seeds_b2.json and hashed in addendum 3)."""
    tab: dict = {"derivation": DERIVATION, "tags": TAGS, "note": "(dit_seed, rng_seed) per item and draw; every arm "
                 "and config of an (item, draw) shares the pair; the tiny scale's Stage T runs on " +
                 ", ".join(TINY["test_items"])}
    test_ids = [it["id"] for it in test_items()]
    for scale in TAGS:
        plan = {"a0": (["item00"], A0_DRAWS), "dev": ([it["id"] for it in dev_items()], DEV_DRAWS),
                "test": (test_ids + (list(TINY["test_items"]) if scale == "tiny" else []), TEST_DRAWS)}
        tab[scale] = {st: {i: [list(seeds(scale, st, i, r)) for r in draws] for i in ids}
                      for st, (ids, draws) in plan.items()}
    return tab


def seed_problems(tab: dict | None = None) -> list[str]:
    """Collisions among the new seeds, with any earlier seed, and between dev and test. Empty when clean."""
    tab = tab or seed_table()
    out = []
    new: list[tuple[str, int]] = []
    for scale in TAGS:
        for st, per in tab[scale].items():
            for i, pairs in per.items():
                for p in pairs:
                    new += [(f"{scale}/{st}/{i}", s) for s in p]
    vals = [s for _, s in new]
    dup = sorted({s for s in vals if vals.count(s) > 1})
    if dup:
        out.append(f"new seeds repeat: {dup[:10]}")
    for src, old in earlier_seeds().items():
        hit = sorted(set(vals) & old)
        if hit:
            out.append(f"new seeds equal {src} seeds: {hit[:10]}")
    for scale in TAGS:
        dev = {s for v in tab[scale]["dev"].values() for p in v for s in p}
        test = {s for v in tab[scale]["test"].values() for p in v for s in p}
        if dev & test:
            out.append(f"{scale}: dev and test share seeds {sorted(dev & test)[:10]}")
    return out


def main() -> int:
    """python plan_b2.py seeds   write seeds_b2.json (refuses on any collision)"""
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
    print(f"wrote {SEEDS_JSON.relative_to(ROOT)}: {n} seed pairs, no collision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
