"""The registered selection of B2 (addendum 3): S*, the clocked recipe, and B*, the unclocked one, as a pure
function of the Stage D rows, and the writer that fixes it before Stage T.

For config c and item i, bo3_ci is the largest q over the 3 draws (a run that is missing, failed or
unscored has q 0). J_dev(c) is the number of items with bo3_ci >= 0.5. A clocked config is eligible if its
vocal-gate rate over its 36 clips is at least 0.60. S* is the eligible clocked config with the highest
(J_dev, mean_i bo3_ci, mean per-clip q); remaining ties go to fewer steered heads, smaller beta, smaller k,
and oohs before clarinet. B* is the unclocked config with the highest value of the same key. Values are
rounded to 1e-6 before any comparison. With no eligible clocked config, Stage T does not run and B2 is
"not supported" with the label "clock silences the voice".
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan_b2 as P  # noqa: E402

GATE_MIN = 0.60
JOINT = 0.5
DIGITS = 6


def rnd(x) -> float:
    return round(float(x), DIGITS)


def table(clips: list[dict], configs: list[dict], items: list[str], draws=P.DEV_DRAWS) -> dict:
    """The dev table: {config: {...}}. clips: per-clip values (score_b2.clip) with run keys item, draw and
    config; a planned clip without a value counts as q 0 and gate failed."""
    by = {(c["item"], c["draw"], c["config"]): c for c in clips}
    out = {}
    for cfg in configs:
        cid = cfg["config"]
        qs = {i: [float(by.get((i, r, cid), {}).get("q", 0.0)) for r in draws] for i in items}
        gates = [bool(by.get((i, r, cid), {}).get("gate", False)) for i in items for r in draws]
        bo3 = {i: rnd(max(v)) for i, v in qs.items()}
        out[cid] = {"config": cid, "guide": cfg["guide"], "k": cfg["k"], "clock": cfg["clock"],
                    "head_set": cfg["head_set"], "beta": cfg["beta"], "n_steered": P.n_steered(cfg),
                    "clocked": P.clocked(cfg), "n_clips": len(gates),
                    "n_scored": sum(bool(by.get((i, r, cid), {}).get("scored")) for i in items for r in draws),
                    "j_dev": sum(b >= JOINT for b in bo3.values()), "mean_bo3": rnd(np.mean(list(bo3.values()))),
                    "mean_q": rnd(np.mean([q for v in qs.values() for q in v])),
                    "gate_rate": rnd(np.mean(gates)), "bo3": bo3}
        out[cid]["eligible"] = (not out[cid]["clocked"]) or out[cid]["gate_rate"] >= GATE_MIN
    return out


def key(row: dict) -> tuple:
    """Higher is better: (J_dev, mean bo3, mean q), then fewer heads, smaller beta, smaller k, oohs first."""
    return (row["j_dev"], row["mean_bo3"], row["mean_q"], -row["n_steered"], -row["beta"], -(row["k"] or 0),
            row["guide"] == "oohs")


def select(tab: dict, waive_gate: bool = False) -> dict:
    """S* and B* from the dev table. waive_gate (the tiny test scale only): with no eligible clocked config,
    pick S* among all clocked configs so the test chain reaches Stage T."""
    clocked = [r for r in tab.values() if r["clocked"]]
    eligible = [r for r in clocked if r["eligible"]]
    unclocked = [r for r in tab.values() if not r["clocked"]]
    waived = False
    if not eligible and waive_gate:
        eligible, waived = clocked, True
    b = max(unclocked, key=key) if unclocked else None
    if not eligible:
        return {"s_star": None, "b_star": b["config"] if b else None, "stage_t": False,
                "label": "clock silences the voice", "eligible": [], "gate_waived": False}
    s = max(eligible, key=key)
    return {"s_star": s["config"], "b_star": b["config"] if b else None, "stage_t": True, "label": None,
            "eligible": sorted(r["config"] for r in eligible), "gate_waived": waived,
            "dev_futility": not key(s)[:3] > key(b)[:3] if b else None}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str | None:
    return sha256_bytes(Path(p).read_bytes()) if Path(p).exists() else None


def selection_doc(tab: dict, sel: dict, configs: list[dict], ledgers: dict, code: dict, scale: str, trim: dict) -> dict:
    """The content of b2_selection.json (no time stamp, so the same rows give the same file)."""
    return {"experiment": "b2_selection", "scale": scale, "s_star": sel["s_star"], "b_star": sel["b_star"],
            "stage_t": sel["stage_t"], "label": sel["label"], "eligible": sel["eligible"],
            "gate_waived": sel["gate_waived"], "dev_futility": sel.get("dev_futility"),
            "configs": [c["config"] for c in configs], "trim": trim,
            "rule": {"joint": JOINT, "gate_min": GATE_MIN, "digits": DIGITS,
                     "key": "J_dev, mean bo3, mean q; fewer steered heads, smaller beta, smaller k, oohs first"},
            "dev_table": tab, "stage_d_ledgers_sha256": ledgers, "code_sha256": code}


def doc_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
