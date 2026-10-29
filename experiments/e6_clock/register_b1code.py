"""Write or check experiments/e5_motif/registration/addendum_2_b1code.json, the B1 code addendum: the sha256
of the B1 code, of its tests and of the CPU test result, the details the code fills in, and the eight E6
files of addendum 2 listed again with their unchanged hashes. Run it as the last step before the commit.

  python experiments/e6_clock/register_b1code.py           write it (refuses if an addendum 2 file changed,
                                                           or if the CPU test did not pass on this code)
  python experiments/e6_clock/register_b1code.py --check   exit 1 if the written file is not current
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import readout as R  # noqa: E402

REG2 = ROOT / "experiments" / "e5_motif" / "registration" / "addendum_2.json"
ADDENDUM = ROOT / "experiments" / "e5_motif" / "registration" / "addendum_2_b1code.json"
CODE = ("dacapo/attn_bias.py", "experiments/e6_clock/clock_score.py", "experiments/e6_clock/run_b1.py",
        "experiments/e6_clock/runb_cpu_test.py", "experiments/e6_clock/register_b1code.py")
TESTS = "experiments/e6_clock/tests"
CPU_TEST = "results/cpu/runb_cpu_test.json"

CUSTODY = ("This addendum is committed and pushed before Run B and before addendum 3 (the B2 registration), and that "
           "push is its external timestamp. It changes nothing that addendum 2 registers: items, seeds, arms, "
           "intervention, measurement and rule stay as section e6_readout ('B1, fixed now') states them.")
FILL_INS = [
    ("arms", "The five arms of a draw are named nohook (beta 0: no hook is entered), align1, align2 and align4 "
             "(beta 1, 2 and 4 on the seven alignment heads) and random2 (beta 2 on registered random set r for draw "
             "r, which equals readout.random_head_sets()[r]). Runs go in the order item, draw, arm."),
    ("recall", "Recall of a clip: the lyric words found by the registered in-order exact matching "
               "(dacapo.words.match_words over Whisper's normalised word chunks; chunks with an empty word or no "
               "start time are dropped, as in readout.py), divided by the number of lyric words. An arm's recall "
               "and vocal-gate rate are means over its 36 planned clips. A missing, failed or unscored clip counts "
               "as recall 0 and gate failed."),
    ("evaluable", "A clip is evaluable if it passes E6's vocal gate and at least 4 lyric words are matched, as in the "
                  "E6 read-out. Its error is the median over the matched words of |Whisper onset - written onset|. "
                  "An item's error in an arm is the median of the errors of its evaluable draws."),
    ("written_onset", "Word j sings note j. Its written onset is the note's start in beats x 0.6 s (100 BPM), from "
                      "experiments/e4_sing/manifest.json. The last word's window ends at its note end, (start + dur) "
                      "x 0.6 s; dur in the manifest is already 0.9 of the note's length, so item00 ends at 11.94 s."),
    ("frames_and_keys", "DiT query frame q covers latent frames 2q and 2q+1 and is placed at its centre, 0.08q + 0.04 s. "
                        "It lies in word j's window if onset_j <= centre < onset_(j+1). Word j's keys are s plus its "
                        "token indices in the pure lyric, with s from ACE-Step's _extract_lyric_segment (9 for these "
                        "lines, so the first word is key 12). The bias goes to the steered heads of layers 2-6 at all "
                        "8 steps; the hook asserts that every biased key lies below n_lyr and n_lyr below Lk."),
    ("not_improved", "An item with no evaluable draw in the arm or at beta 0, or with a beta 0 error of 0, is not "
                     "improved: it fails the 30% count and enters the medians with reduction 0. A computed reduction "
                     "enters as it is, negative values included."),
    ("rounding", "Seconds and reductions are rounded to 1e-6 before any comparison, as in E6. A rounded reduction of "
                 "0.30 counts as 30%. The guards compare the rounded change in recall and in gate rate with -0.10. The "
                 "random arm's rounded median reduction must be strictly below the rounded half of beta 2's."),
    ("integrity", "Integrity can block a verdict but never create one. The verdict is 'technical' unless all of these "
                  "hold: the eight addendum 2 files and the B1 code hash to their registered values at generation "
                  "(every run row records them) and at decision; readout.py check-gate accepts Run A's gate "
                  "(E6_RES=results/gpu_run2); every scored clip used Whisper large-v3 at the pinned revision; every "
                  "generated clip has its pre-normalisation peak; at most 9 of the 180 runs are missing, failed or "
                  "unscored; every hooked run made 40 hook calls with every biased key inside the lyric, and every "
                  "no-hook run made none."),
    ("attempts", "A run that raises is tried at most 3 times with the same seeds. The ledgers are append-only, and a "
                 "run counts as generated only while its WAV and pred_latents hash to its ledger row."),
    ("verdicts", "'supported' if conditions 1-3 hold and 'not supported' otherwise, with the failed conditions named; "
                 "'technical' if integrity fails; 'test only' for any run with a test flag (the tiny CPU scale, a test "
                 "Whisper model, unregistered code). Labels describe each condition and never change the verdict."),
    ("exploratory", "From addendum 2's exploratory list, the parts that apply to B1, per arm: the share of matched "
                    "words within 0.25 s of the written onset, the median signed error, the error without each clip's "
                    "first matched word, and 95% intervals from a bootstrap over items (10,000 resamples, seed 0) for "
                    "the median reduction and the items improved."),
    ("generation", "Every run goes through dacapo.ace.generate at batch 1 with sdpa attention and the E5 v2 "
                   "determinism settings. Clips are written by soundfile as 16-bit PCM WAV, pred_latents as float32 "
                   ".npy."),
    ("tiny_scale", "The tiny CPU scale (run_b1.py --scale tiny, part of experiments/run_b.py --scale tiny) runs item00 "
                   "draw 0 without the hook and at beta 2 on the alignment heads, with its own seeds "
                   "(dacapo.seeds.run_seeds('runB1-tiny', item, draw)) and a test Whisper, and every verdict it writes "
                   "is 'test only'. Before this addendum, the tiny scale used the registered seeds of those two runs "
                   "(DiT seed 63000, pin 963000): the code tests on this machine's CPU generated these two planned "
                   "runs several times and scored them with whisper-tiny and Whisper small.en. Nobody listened to "
                   "them, their transcripts were read only to check the code, and they were deleted."),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_test_summary() -> dict:
    p = ROOT / CPU_TEST
    if not p.exists():
        raise SystemExit(f"{CPU_TEST} missing: run experiments/e6_clock/runb_cpu_test.py first")
    d = json.loads(p.read_text())
    res, cfg = d["results"], d["config"]
    runs = res["runs"]
    return {"path": CPU_TEST, "sha256": sha256(p), "finished_utc": d["finished_utc"], "passed": res["passed"],
            "checks": res["checks"], "recorded": res["recorded"], "n_lyr": res["n_lyr"], "word_keys": res["word_keys"],
            "guide_sha256": res["guide_sha256"], "soundfont_sha256": res["soundfont_sha256"],
            "hook_calls": {n: r["hook"]["hook_calls"] for n, r in runs.items() if r["hook"]},
            "code_sha256": cfg["code_sha256"], "dit_seed": cfg["dit_seed"], "rng_seed": cfg["rng_seed"]}


def computed() -> dict:
    tests = sorted(str(p.relative_to(ROOT)) for p in (ROOT / TESTS).glob("*.py"))
    return {"refers_to": ["experiments/e5_motif/registration/addendum_2.json"],
            "addendum_2_sha256": sha256(REG2), "custody": CUSTODY,
            "fill_ins": [{"id": k, "text": t} for k, t in FILL_INS],
            "code_sha256": {f: sha256(ROOT / f) for f in CODE},
            "tests_sha256": {f: sha256(ROOT / f) for f in tests},
            "cpu_test": cpu_test_summary(),
            "e6_files_sha256": R.code_hashes()}


def problems(new: dict) -> list[str]:
    out = []
    reg = R.registered_hashes(REG2)
    if new["e6_files_sha256"] != reg:
        out.append(f"E6 files differ from addendum 2: {sorted(f for f in reg if new['e6_files_sha256'].get(f) != reg[f])}")
    cpu = new["cpu_test"]
    if not cpu["passed"]:
        out.append("the CPU test did not pass")
    now = {f: sha256(ROOT / f) for f in cpu["code_sha256"]}
    if now != cpu["code_sha256"]:
        out.append(f"the CPU test ran on other code: {sorted(f for f in now if now[f] != cpu['code_sha256'][f])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    new = computed()
    bad = problems(new)
    if a.check:
        try:
            cur = json.loads(ADDENDUM.read_text())
        except (OSError, ValueError):
            cur = {}
        stale = sorted(k for k in new if new[k] != cur.get(k))
        print("B1 code addendum:", "current" if not (stale or bad) else f"stale {stale} {bad}")
        return 1 if stale or bad else 0
    if bad:
        print("B1 code addendum not written:", "; ".join(bad))
        return 1
    out = {"addendum_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "title": "B1 code addendum to addendum 2: the code, tests and CPU test of the lyric clock B1", **new}
    ADDENDUM.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {ADDENDUM.relative_to(ROOT)}: {len(new['code_sha256'])} code files, {len(new['tests_sha256'])} tests, "
          f"CPU test {'passed' if new['cpu_test']['passed'] else 'FAILED'}, addendum_utc {out['addendum_utc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
