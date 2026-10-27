"""Write the computed sections of registration/addendum_2.json: the sha256 of every code file the E5 v2
and E6 runtime checks verify, of the committed data, of the tests, the K0 results and the sentence that
states them (k0_outcome). Run it as the very last step before the commit; after any later edit of a listed
file the runtime refuses to start.

  python experiments/e5_motif/register.py           recompute and write
  python experiments/e5_motif/register.py --check   exit 1 if the written hashes are not current
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import v2_config as C  # noqa: E402
from dacapo.motif_integrity import sha256_file  # noqa: E402

ADDENDUM = HERE / "registration" / "addendum_2.json"
E5_CODE = ["experiments/e5_motif/v2_config.py", "experiments/e5_motif/make_items.py",
           "experiments/e5_motif/run_analogy.py", "experiments/e5_motif/score_analogy.py",
           "experiments/e5_motif/a0_checks.py", "experiments/e5_motif/stage.py",
           "dacapo/__init__.py", "dacapo/ace.py", "dacapo/seeds.py", "dacapo/motif_stats.py",
           "dacapo/motif_features.py", "dacapo/motif_integrity.py", "dacapo/metrics.py", "dacapo/analysis.py",
           "dacapo/render.py", "dacapo/symbolic.py", "dacapo/provenance.py", "experiments/run_all.py"]
E6_CODE = ["experiments/e6_clock/readout.py", "dacapo/__init__.py", "dacapo/ace.py", "dacapo/attn_tap.py",
           "dacapo/provenance.py", "dacapo/seeds.py", "dacapo/words.py", "experiments/e4_sing/manifest.json"]
DATA = ["experiments/e5_motif/manifest.json", "experiments/e5_motif/manifest_p0b.json",
        "experiments/e5_motif/manifest_v2.json", "experiments/e5_motif/registration/e5v2_seeds.json",
        "experiments/e5_motif/registration/prereg.json", "experiments/e5_motif/registration/addendum_1.json",
        "experiments/e5_motif/registration/v1_outcome.json",
        *[f"experiments/e5_motif/results/e5_ruler_gate_p{p}.json" for p in (73, 54, 71)],
        "experiments/e5_motif/k0_first_run/manifest_v2.json",
        *[f"experiments/e5_motif/k0_first_run/e5_ruler_gate_p{p}.json" for p in (73, 54, 71)]]
TESTS_GLOB = "tests/test_*.py"


def k0_summary() -> dict:
    out = {}
    for cap, p in C.PROGRAMS.items():
        f = HERE / "results" / f"e5_ruler_gate_p{p}.json"
        if not f.exists():
            raise SystemExit(f"{f} missing: run score_analogy.py gate --program {p} first")
        s = json.loads(f.read_text())["results"]["summary"]
        out[str(p)] = {"timbre": cap, "passed": s["passed"], "overall": s["overall"], "n_correct": s["n_correct"],
                       "n_rows": s["n_rows"], "n_tie_rows": s["n_tie_rows"], "tau": s["tau"],
                       "wrong_note_auc": s["wrong_note_auc"], "per_condition": s["per_condition"],
                       "manifest_problems": s["manifest_problems"], "render_mismatch": s["render_mismatch"]}
    return out


def k0_outcome(k0: dict) -> str:
    parts = [f"{v['timbre'].replace('_', ' ')} (GM {p}) {'passed' if v['passed'] else 'failed'}: classification "
             f"{v['overall']} ({v['n_correct']} of {v['n_rows']} rows), wrong-note AUC {v['wrong_note_auc']['1_wrong']} "
             f"with one wrong note and {v['wrong_note_auc']['2_wrong']} with two" for p, v in k0.items()]
    return ("Written by register.py from results/e5_ruler_gate_p*.json (tau 0.7, ties counted as errors; a timbre "
            "passes with classification >= 0.95 and both AUCs >= 0.9): " + "; ".join(parts) + ".")


def computed() -> dict:
    seeds = HERE / "registration" / "e5v2_seeds.json"
    if not seeds.exists():
        C.write_json(seeds, C.seed_table())
    if json.loads(seeds.read_text()) != json.loads(json.dumps(C.seed_table())):
        raise SystemExit("registration/e5v2_seeds.json differs from v2_config.seed_table()")
    sha = lambda files: {f: sha256_file(ROOT / f) for f in files}
    k0 = k0_summary()
    return {"code_sha256": sha(E5_CODE), "e6_code_sha256": sha(E6_CODE), "data_sha256": sha(DATA),
            "tests_sha256": sha(sorted(str(p.relative_to(ROOT)) for p in ROOT.glob(TESTS_GLOB))),
            "k0": k0, "k0_outcome": k0_outcome(k0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    add = json.loads(ADDENDUM.read_text())
    new = computed()
    current = {"code_sha256": add.get("code_sha256"), "e6_code_sha256": add.get("e6_readout", {}).get("code_sha256"),
               "data_sha256": add.get("data_sha256"), "tests_sha256": add.get("tests_sha256"), "k0": add.get("k0"),
               "k0_outcome": add.get("k0_outcome")}
    if a.check:
        stale = sorted(k for k in new if new[k] != current[k])
        print("addendum 2 hashes:", "current" if not stale else f"stale {stale}")
        return 1 if stale else 0
    add["addendum_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    add["code_sha256"] = new["code_sha256"]
    add["data_sha256"] = new["data_sha256"]
    add["tests_sha256"] = new["tests_sha256"]
    add["k0"] = new["k0"]
    add["k0_outcome"] = new["k0_outcome"]
    add.setdefault("e6_readout", {})["code_sha256"] = new["e6_code_sha256"]
    ADDENDUM.write_text(json.dumps(add, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {ADDENDUM.relative_to(ROOT)}: {len(new['code_sha256'])} E5 code files, "
          f"{len(new['e6_code_sha256'])} E6 files, {len(new['data_sha256'])} data files, addendum_utc {add['addendum_utc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
