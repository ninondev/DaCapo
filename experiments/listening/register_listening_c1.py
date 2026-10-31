"""Write or check experiments/listening/registration/listening_plan_c1.json, the analysis plan of the C1 blind pack
(Run C), with the sha256 of the analysis code (unchanged from listening_plan.json) and of this script. Run it as the
last step before the commit, and commit and push before anyone rates the C1 pack.

  python experiments/listening/register_listening_c1.py           write it (refuses to overwrite a different plan)
  python experiments/listening/register_listening_c1.py --check   exit 1 if the written file is not current
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
PLAN = HERE / "registration" / "listening_plan_c1.json"
CODE = ("experiments/listening/analyze_ratings.py", "experiments/listening/register_listening_c1.py")
TESTS = "experiments/listening/tests"
PRIVATE = "the owner's private repository ninondev/project-ryuichi"

PACK = {"pack": "c1_blind_pack.zip: 24 pairs, written by experiments/e9_clock_replicate/package_c1.py in Run C",
        "pack_sha256": "4989f1d083dce353c802d52258ab1150a0ca8ace04cbfe09426c2be441f9a4d4",
        "key": f"runs/colab_l4_runC_20261031/c1_blind_key_SEALED.json in {PRIVATE}, added at commit "
               "f29df5d1335f185d877e505f75fa16cacf2582e6",
        "salt": "runs/colab_l4_runC_20261031/c1_key_salt_SEALED.txt, added at the same commit",
        "key_salted_sha256": "35f4d0fcb035c890ec3a9c0e81dca1bef6cd931b03ab17fbd3fc7294234be591",
        "questions": ["melody_right", "words_clear"], "answers": ["A", "B", "same", "neither"],
        "scores": "results/gpu_run4/c1_result.json"}

SECTIONS = [
    ("scope", "The C1 pack is exploratory (addendum 4, section package): its result never changes C1's registered "
              "verdict, 'Clock helps, capability not reached'. This plan fixes its analysis before anyone rates it. It "
              "is the analysis of the B2 pack in listening_plan.json, applied to the C1 pack with the same code "
              "(analyze_ratings.py in its 'b2' mode, with --scores results/gpu_run4/c1_result.json). S* is the clock "
              "recipe and S*_b0 the same melody guide without the clock."),
    ("raters", "A rater of the C1 pack confirms, in one line when handing in a sheet, that they have not seen the C1 "
               "key, have not heard any labelled demo clip of Run C (in the result package or in the folder of demo clips "
               "sent apart from it), and have not read Run C's per-clip results (results/gpu_run4/ or the same files in "
               "the result package; their transcripts, read with the demo list, tell which clip of a pair has the "
               "clock). A 'no', or no confirmation by closing, excludes the rater from the C1 analysis. "
               "listening_plan.json's other rules on raters, sheets and formats apply unchanged. A rater may rate the "
               "B2 pack and the C1 pack; the two analyses are reported apart."),
    ("closing", "listening_plan.json's closing section applies unchanged to the C1 sheets, with its own exclusions file "
                "(exclusions_c1.json) and its own deadline: ratings close when the owner says no more C1 sheets are "
                "coming, at the latest on 2026-11-30 at 23:59 UTC."),
    ("test", "listening_plan.json's b2_test section, unchanged: the centred conditional randomization test over the "
             "side vectors with the key's number of pairs with S* on side A, per question, with the labels "
             "'listeners favour the clock', 'listeners favour no clock' and 'no clear preference' at p < 0.05, two "
             "sided, and no correction for the two questions."),
    ("report", "listening_plan.json's b2_report section, unchanged, with the machine scores taken from "
               "results/gpu_run4/c1_result.json: Whisper word recall, CTC word recall on the mix, and NPA on the "
               "separated vocal. CTC word recall on the separated vocal is not part of the agreement report. "
               "analyze_ratings.py names listening_plan.json as its plan in the output; for the C1 pack the plan is "
               "this file."),
    ("custody", "The C1 sides came from the operating system's random source when Run C packaged its clips. The key "
                "and its salt are in the private repository at the path and commit in pack; the published fingerprint "
                "is sha256 of the salt file's 64 ASCII hex characters (no newline) followed by the key file's bytes. "
                "The owner uploaded the key and the pack to the agents on 2026-10-31; the key's contents were shown to "
                "the agents when it arrived, and an agent opened it again to audit the pack (pair order, that every "
                "clip matches its run, and that nothing in the files shows a side). Facts from that audit, the side balance "
                "among them, appeared in the agents' working output, which the owner can open; the test does not depend "
                "on a rater not knowing the side balance. The owner was asked not to open the key file; whether it was "
                "opened before the upload is asked at closing and recorded. File metadata in the zip of 140 demo clips "
                "that the owner sent apart from the result package shows that two labelled demo clips of item c1t00, "
                "which is not in the pack, were accessed by an application on the owner's computer on 2026-10-31 (one "
                "carries a last-opened time of 05:01Z); the owner says no demo clip was listened to, only moved. Both "
                "facts go into the closing record. The agents also ran the analysis "
                "once on the real key with a sheet that answers 'same' everywhere, with the code registered here, to "
                "check it, and deleted the output. No output of analyze_ratings.py with the real key is shared before the "
                "key is published, because each output holds the key's sha256. No rating of the C1 pack existed when "
                "this plan was written."),
    ("reporting", "The analysis writes results/listening/c1_ratings.json. README.md and LOG.md report the labels, D, "
                  "the centred margin, m, the p values and the counts. The key and its salt are published with the "
                  "results. A change to this plan after this file is committed goes into a new file."),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def computed() -> dict:
    tests = sorted((ROOT / TESTS).rglob("*.py"))
    refs = ("experiments/listening/registration/listening_plan.json",
            "experiments/e5_motif/registration/addendum_4.json", "results/gpu_run4/c1_result.json")
    return {"title": "Analysis plan of the C1 blind listening pack of Run C (exploratory)",
            "refers_to": {r: sha256(ROOT / r) for r in refs},
            "pack": PACK,
            "sections": [{"id": i, "text": t} for i, t in SECTIONS],
            "code_sha256": {c: sha256(ROOT / c) for c in CODE},
            "tests_sha256": {str(t.relative_to(ROOT)): sha256(t) for t in tests}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    want = computed()
    if PLAN.exists():
        have = json.loads(PLAN.read_text(encoding="utf-8"))
        utc = have.pop("plan_utc", None)
        same = have == want
        if a.check:
            print("C1 listening plan: current" if same else "C1 listening plan: NOT current")
            return 0 if same else 1
        if same:
            print(f"C1 listening plan: already current ({utc})")
            return 0
        print("C1 listening plan: a different plan is written; a change after registration needs a new file")
        return 1
    if a.check:
        print("C1 listening plan: missing")
        return 1
    PLAN.write_text(json.dumps({"plan_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **want},
                               indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"C1 listening plan: written {PLAN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
