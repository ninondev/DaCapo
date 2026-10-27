"""Pilot: can ACE-Step 1.5 turbo make clean a cappella singing?

A cappella output is where vocal pitch can be measured reliably (E0: solo voice-like leads
measure at RPA ~0.93; in band mixes 1-2 wrong notes are invisible). Generates a few captions x
seeds with the same lyrics and saves audio and timing. Scoring is in acappella_score.py (main
environment).

  ACESTEP_REPO=... python acappella_pilot.py --out outputs_acappella
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dacapo import ace as ace_lab

LYRICS = """[Verse]
Morning light upon the river
Carry me where the willows grow
Every step a quiet whisper
Every breath a song I know"""

CAPTIONS = {
    "acap_female": "a cappella, solo female vocal, no instruments, no drums, dry studio recording, clear melody",
    "acap_male": "a cappella, solo male vocal, no instruments, no drums, dry studio recording, clear melody",
    "folk_female": "soft folk ballad, female vocal, acoustic guitar",  # control: normal song with accompaniment
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_acappella")
    ap.add_argument("--seeds", default="11,12")
    ap.add_argument("--duration", type=float, default=16.0)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(__file__).parent / args.out
    out.mkdir(parents=True, exist_ok=True)
    h = ace_lab.load(args.device)
    log = {"settings": vars(args), "lyrics": LYRICS, "captions": CAPTIONS, "runs": []}
    for key, cap in CAPTIONS.items():
        for seed in [int(s) for s in args.seeds.split(",")]:
            wav = out / f"{key}_s{seed}.wav"
            r = ace_lab.generate(h, wav, cap, args.duration, seed, lyrics=LYRICS, shift=args.shift, bpm=90)
            log["runs"].append({"key": key, "seed": seed, "wav": wav.name, "seconds": r["seconds"]})
            (out / "log.json").write_text(json.dumps(log, indent=1))
            print(f"{key:>12s} seed={seed} {r['seconds']:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
