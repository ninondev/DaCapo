"""E5 v2: every generation setting, seed rule, cell, run list and decision constant in one place.

The A0 checks, the runner, the scorer, the stage driver and the seed-table writer import this module,
so a value can only differ between them by editing this file, whose sha256 addendum 2 registers.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from dacapo.seeds import DERIVATION, run_seeds

HERE = Path(__file__).resolve().parent

ACE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
HF_REPO, HF_REVISION = "ACE-Step/Ace-Step1.5", "19671f406d603126926c1b7e2adc169acbcade22"
WEIGHTS_SHA256 = {  # files at HF_REVISION; ACE-Step copies the modeling file from its repo at ACE_COMMIT
    "acestep-v15-turbo/model.safetensors": "3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7",
    "acestep-v15-turbo/silence_latent.pt": "a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b",
    "acestep-v15-turbo/modeling_acestep_v15_turbo.py": "71b22e01b490bd9e149e4c0c5da2e1f512126f3ab3316e0ae97779af84d19512",
    "vae/diffusion_pytorch_model.safetensors": "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0",
    "Qwen3-Embedding-0.6B/model.safetensors": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"}
GPU_REQUIRED = "L4"
SF2_SHA256 = "74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0"  # FluidR3_GM.sf2

# Generation settings shared by every v2 repaint.
REP = "Repaint the mask area based on the given conditions:"       # acestep TASK_INSTRUCTIONS["repaint"]
T2M = "Fill the audio semantic mask based on the given conditions:"  # acestep DEFAULT_DIT_INSTRUCTION
INSTRUCTIONS = {"rep": REP, "t2m": T2M}
REPAINT = {"repainting_start": 12.0, "repainting_end": 16.0, "chunk_mask_mode": "explicit",
           "repaint_mode": "balanced", "repaint_strength": 0.5, "dcw_enabled": False,
           "infer_method": "ode", "sampler_mode": "euler"}
GEN = {"duration": 18.0, "steps": 8, "shift": 3.0, "lyrics": "[Instrumental]", "bpm": 120, "keyscale": "",
       "timesignature": "", "task": "repaint"}  # duration is ignored for repaint: ACE-Step follows the source
CAPTION_KEYS = ("flute", "synth_voice", "clarinet")
CAPTIONS = {c: f"solo {c.replace('_', ' ')} melody, monophonic, no accompaniment" for c in CAPTION_KEYS}
PROGRAMS = {"flute": 73, "synth_voice": 54, "clarinet": 71}  # GM programs of the hypothesis renders (K0)
WINDOW_FRAMES = (300, 400)  # int(12*48000//1920), int(16*48000//1920)

# Source lengths. t18: the rendered source cut to 864000 samples (18.000 s, 450 latent frames) with a
# 50 ms half-cosine fade-out; the cut ends inside D's last note or its release. rend: as rendered.
T18_SAMPLES, T18_FADE = 864000, 2400
LENGTH_KEYS = ("rend", "t18")

TAGS = {"a0": "e5v2-a0", "p0b": "e5v2-p0b", "main": "e5v2-main"}
DRAWS = (0, 1, 2, 3)
RESERVED = {1001, 2002, 20261027}  # v1 seeds and the CPU pilot's global pin

P0B_MANIFEST = HERE / "manifest_p0b.json"   # v1 items mel00-mel13 with both source lengths
MAIN_MANIFEST = HERE / "manifest_v2.json"   # fresh items v2c00-v2c35, make_items generator seed 303
SEED_TABLE = HERE / "registration" / "e5v2_seeds.json"
ADDENDUM = HERE / "registration" / "addendum_2.json"
P0B_IDS = tuple(f"mel{i:02d}" for i in range(14))
MAIN_IDS = tuple(f"v2c{i:02d}" for i in range(36))
A0_IDS = ("mel00", "mel01")
CONDITIONS = ("lit", "ctl", "rel+2", "rel+5", "rel-3", "seq", "ana", "null")

# Phase 0b cells in tie order: repaint before text2music, as rendered before 18 s, then flute, synth
# voice, clarinet. The selected cell is the K1-passing cell with the most passing gaps.
CELLS = tuple(f"{i}-{L}-{c}" for i in ("rep", "t2m") for L in ("rend", "t18") for c in CAPTION_KEYS)

K1_FRAC = 0.75          # a cell passes K1 with >= 84 of its 112 gaps passing
Q1_FRAC = 0.75          # >= 756 of 1008 non-null gaps pass
Q2_FRAC = 0.75          # >= 189 of 252 non-null (melody, condition) cells ...
Q2_MIN_SHARE = 0.5      # ... have >= 2 of 4 passing draws
MAX_STARTS = 3          # generation attempts per job
MAX_RESUME_PASSES = 2   # automatic resume passes; afterwards still-missing draws count as gate failures
HALT_AFTER_ERRORS = 5   # this many generations raising in a row is a broken environment: technical stop
AUDIO_SUBTYPE = "PCM_16"
# Free space a phase needs per remaining job: a 21 s FLAC (1.3-1.6 MB measured) with headroom, plus its
# tracker output; refused below that, before any generation.
BYTES_PER_JOB = 2_600_000
SPACE_MARGIN = 1_000_000_000

# Run sizes. "full" is the registered design and the only scale whose results count. "tiny" exercises
# every stage on CPU with one or two generations per stage, in its own folders, on pilot melodies only and
# with seeds of its own tags, so it never generates a confirmatory melody or uses a registered seed.
TINY_TAGS = {"a0": "e5v2-tiny-a0", "p0b": "e5v2-tiny-p0b", "main": "e5v2-tiny-main"}
SCALES = {
    "full": {"tags": TAGS, "p0b_manifest": "manifest_p0b.json", "p0b_ids": P0B_IDS, "p0b_conds": ("lit", "ctl"),
             "p0b_cells": CELLS, "p0b_draws": DRAWS, "main_manifest": "manifest_v2.json", "main_ids": MAIN_IDS,
             "main_conds": CONDITIONS, "main_draws": DRAWS, "k0_manifest": "manifest_v2.json", "k0_ids": MAIN_IDS,
             "a0_runs": "full"},
    "tiny": {"tags": TINY_TAGS, "p0b_manifest": "manifest_p0b.json", "p0b_ids": ("mel00",), "p0b_conds": ("lit", "ctl"),
             "p0b_cells": (CELLS[0],), "p0b_draws": (0,), "main_manifest": "manifest_p0b.json", "main_ids": ("mel01",),
             "main_conds": ("lit", "ctl"), "main_draws": (0,), "k0_manifest": "manifest_p0b.json", "k0_ids": P0B_IDS,
             "a0_runs": "tiny"},
}


def cell_parts(cell: str) -> tuple[str, str, str]:
    i, L, c = cell.split("-")
    return i, L, c


def expected_frames(source_samples: int) -> int:
    return source_samples // 1920


def expected_duration_meta(source_samples: int) -> int:
    return int(source_samples / 48000)


def min_pass_per_cell(n_draws: int) -> int:
    return max(1, math.ceil(Q2_MIN_SHARE * n_draws))


def jobs(phase: str, manifest: dict, scale: str = "full", cell: str | None = None,
         excluded_cells: tuple = ()) -> list[dict]:
    """Ordered job list (design.json 'jobs'). Melodies in id order; every job of a melody runs before the
    next melody starts; within a melody: draw, then cell, then condition.
    p0b: the cells not excluded by K0 x {lit, ctl} x draws 0-3 per pilot melody (1344 jobs at full scale).
    main: the selected cell x 8 conditions x draws 0-3 per confirmatory melody (1152 jobs)."""
    sc = SCALES[scale]
    items = {it["id"]: it for it in manifest["items"]}
    if phase == "p0b":
        ids, conds, draws = sc["p0b_ids"], sc["p0b_conds"], sc["p0b_draws"]
        cells = tuple(c for c in sc["p0b_cells"] if c not in set(excluded_cells))
        split = "pilot"
    elif phase == "main":
        if cell not in CELLS:
            raise ValueError(f"unknown cell {cell!r}")
        ids, conds, draws, cells, split = sc["main_ids"], sc["main_conds"], sc["main_draws"], (cell,), "confirm"
    else:
        raise ValueError(phase)
    out = []
    for mid in ids:
        it = items[mid]
        for d in draws:
            dit, rng = run_seeds(sc["tags"][phase], mid, d)
            for c in cells:
                ins, L, cap = cell_parts(c)
                for cond in conds:
                    out.append({"id": mid, "split": split, "cond": cond, "arm": c if phase == "p0b" else "base",
                                "seed": f"d{d}", "draw": d, "dit_seed": dit, "rng_seed": rng, "cell": c,
                                "instruction": INSTRUCTIONS[ins], "caption": CAPTIONS[cap], "length": L,
                                "src": it["sources"][L][cond], "source_samples": it["source_samples"][L]})
    return out


def seed_table() -> dict:
    """Committed with addendum 2; the runner refuses seeds that differ from it."""
    tab = {"derivation": DERIVATION, "tags": TAGS, "reserved": sorted(RESERVED)}
    for ph, ids in (("a0", A0_IDS), ("p0b", P0B_IDS), ("main", MAIN_IDS)):
        tab[ph] = {mid: [list(run_seeds(TAGS[ph], mid, d)) for d in DRAWS] for mid in ids}
    flat = [s for ph in ("a0", "p0b", "main") for v in tab[ph].values() for p in v for s in p]
    assert len(flat) == len(set(flat)), "seed collision across tags"
    assert not RESERVED & set(flat), "a v2 seed equals a v1 or pilot seed"
    return tab


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    """Write through a temporary file and os.replace, so a reader never sees half a file."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# Where things live. E5_DIR holds the committed data (manifests, registration, rendered sources);
# E5_RUN holds one run's state (designs, ledgers, outputs, results). run_all points E5_RUN at a
# folder per scale, on Google Drive when it is mounted.

def repo_root() -> Path:
    import os
    return Path(os.environ.get("DACAPO_ROOT", HERE.parents[1]))


def e5_dir() -> Path:
    import os
    return Path(os.environ.get("E5_DIR", HERE))


def scale() -> str:
    import os
    s = os.environ.get("DACAPO_SCALE", "full")
    if s not in SCALES:
        raise ValueError(f"DACAPO_SCALE={s!r}")
    return s


def run_dir() -> Path:
    import os
    return Path(os.environ.get("E5_RUN", e5_dir() / ("run_v2" if scale() == "full" else f"run_v2_{scale()}")))


def results_dir() -> Path:
    return run_dir() / "results"


def phase_dir(phase: str, attempt: int) -> Path:
    return run_dir() / (f"outputs_v2_{phase}" + ("" if attempt == 1 else f"_a{attempt}"))


def quality_path(phase: str, attempt: int) -> Path:
    return results_dir() / (f"e5v2_{phase}_quality" + ("" if attempt == 1 else f"_a{attempt}") + ".json")


def terminal_path() -> Path:
    return results_dir() / "e5_terminal.json"


def void_path() -> Path:
    return results_dir() / "e5v2_void.json"


def selection_path() -> Path:
    return results_dir() / "e5v2_phase0b.json"


def a0_path() -> Path:
    return results_dir() / "e5v2_a0.json"


def ruler_gate_path(program: int) -> Path:
    """The registered K0 file (run locally before addendum 2 was committed); a test scale uses the K0 its own
    setup stage ran on its own items."""
    if scale() != "full":
        return results_dir() / f"setup_k0_p{program}.json"
    return e5_dir() / "results" / f"e5_ruler_gate_p{program}.json"


def read_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def addendum_path() -> Path:
    return e5_dir() / "registration" / "addendum_2.json"


def addendum_status() -> dict:
    """Registered code and data check (addendum 2, S10 check 10): every file listed in code_sha256 and
    data_sha256 must hash to its registered value. Returns {ok, addendum, addendum_sha256, current,
    current_data, changed}; 'current' holds the code hashes, 'current_data' the data hashes."""
    from dacapo.motif_integrity import code_sha256

    add_path = addendum_path()
    add = read_json(add_path)
    if add is None or not add.get("code_sha256"):
        return {"ok": False, "addendum": add, "addendum_sha256": sha256_file(add_path) if add else None,
                "current": {}, "current_data": {}, "changed": ["no registered code hashes"]}
    cur = code_sha256(sorted(add["code_sha256"]), repo_root())
    data = code_sha256(sorted(add.get("data_sha256") or {}), repo_root())
    changed = sorted(f for f, h in add["code_sha256"].items() if cur.get(f) != h)
    changed += sorted(f for f, h in (add.get("data_sha256") or {}).items() if data.get(f) != h)
    return {"ok": not changed, "addendum": add, "addendum_sha256": sha256_file(add_path), "current": cur,
            "current_data": data, "changed": changed}


def registered_k0(program: int) -> dict:
    """The registered K0 of one timbre, read from its result file (hashed in data_sha256); the k0 section
    of addendum 2 must agree with it. A missing file or a disagreement counts as not passed."""
    path = e5_dir() / "results" / f"e5_ruler_gate_p{program}.json"
    s = (((read_json(path) or {}).get("results") or {}).get("summary")) or {}
    reg = (((read_json(addendum_path()) or {}).get("k0") or {}).get(str(program))) or {}
    agrees = bool(s) and bool(reg) and reg.get("passed") == bool(s.get("passed"))
    return {"program": program, "file_exists": bool(s), "addendum_agrees": agrees,
            "passed": bool(s.get("passed")) and agrees}


def k0_excluded_cells(scale_name: str) -> tuple:
    """Phase 0b cells whose caption timbre has no passed registered K0 (deviation aa); none at a test scale,
    which keeps its one cell."""
    if scale_name != "full":
        return ()
    return tuple(c for c in CELLS if not registered_k0(PROGRAMS[cell_parts(c)[2]])["passed"])


def attempt2_allowed(void: dict | None, st: dict) -> tuple[bool, str]:
    """Attempt 2 follows a void attempt 1 only under an amended addendum 2 whose amendments list links the
    void: an entry with void_attempt 1 that names every failed check, and every registered file whose
    sha256 changed since the void."""
    if not (void and void.get("attempt") == 1):
        return False, "no void attempt 1"
    if void.get("addendum_sha256") == st["addendum_sha256"]:
        return False, "addendum 2 is unchanged since attempt 1 became void"
    add = st["addendum"] or {}
    now = {**(add.get("code_sha256") or {}), **(add.get("data_sha256") or {})}
    then = void.get("registered_sha256") or {}
    changed = sorted(f for f in set(now) | set(then) if now.get(f) != then.get(f))
    for a in add.get("amendments") or []:
        if a.get("void_attempt") == 1 and set(void.get("failed_checks", [])) <= set(a.get("failed_checks", [])) \
                and set(changed) <= set(a.get("files", [])):
            return True, ""
    return False, (f"no amendment in addendum 2 links the failed checks {void.get('failed_checks')} "
                   f"to the changed files {changed}")


def write_terminal(paper_level: str, stage: str, strike: int | None, **extra) -> bool:
    """One terminal outcome per E5 v2 run, written once. While it exists no E5 v2 generation starts."""
    import time

    p = terminal_path()
    if p.exists():
        return False
    write_json(p, {"paper_level": paper_level, "stage": stage, "strike": strike, "scale": scale(),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})
    return True


def utc() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def record_void(phase: str, attempt: int, failed: list, detail=None) -> bool:
    """An integrity failure found before any hypothesis accuracy exists. Attempt 1 becomes void (one
    attempt 2 may follow a fix recorded in an amended addendum, see attempt2_allowed); a void attempt 2 is
    strike two. Idempotent. Returns True when the terminal outcome exists."""
    write_json(results_dir() / f"e5v2_{phase}_a{attempt}_void_detail.json", {"failed_checks": failed, "detail": detail})
    if attempt >= 2:
        write_terminal("invalid_run", f"{phase}_integrity", 2, failed_checks=failed, attempt=attempt)
        return True
    if not void_path().exists():
        st = addendum_status()
        add = st["addendum"] or {}
        write_json(void_path(), {"attempt": attempt, "phase": phase, "failed_checks": failed, "utc": utc(),
                                 "addendum_sha256": st["addendum_sha256"],
                                 "registered_sha256": {**(add.get("code_sha256") or {}), **(add.get("data_sha256") or {})}})
    return False
