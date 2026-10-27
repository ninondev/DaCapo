"""E5 runner (ACE-Step environment): ACE-Step repaint fills the silent gap after three context units.

  python run_analogy.py --phase gate --device cuda    # pilot melodies x 3 captions x {lit, ctl}
  python run_analogy.py --phase main --device cuda    # confirmatory grid with the caption that passed

Repaint uses only the dense VAE context (no LM codes), so this shows what the renderer alone does
with in-context motifs. Seeds are paired across conditions within a melody.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dacapo import ace as ace_lab

HERE = Path(__file__).parent
CAPTIONS = {"cap1": "solo flute melody, monophonic, no accompaniment",
            "cap2": "solo flute, classical, dry room",
            "cap3": "unaccompanied solo flute, a single melodic line, no other instruments, no chords, no drums"}
REPAINT = {"repainting_start": 12.0, "repainting_end": 16.0, "chunk_mask_mode": "explicit", "repaint_mode": "balanced"}
MAIN = {"s1": ["lit", "ctl", "rel+2", "rel+5", "rel-3", "seq", "ana", "null"], "s2": ["lit", "ctl", "rel+5"]}
SEEDS = {"s1": 1001, "s2": 2002}


def gate_choice() -> str:
    """Caption whose pilot gaps pass the pre-registered quality gate most often (>= 75% required)."""
    p = HERE / "results" / "e5_outputs_gate.json"
    rows = json.loads(p.read_text())["results"]["rows"]
    rate = {}
    for cap in CAPTIONS:
        sel = [r for r in rows if r["arm"] == cap]
        ok = [r["gap_voiced"] >= 0.6 and r["gap_polyphony"] <= 1.3 and abs(r["gap_rms_db_vs_context"]) <= 6 for r in sel]
        rate[cap] = sum(ok) / max(1, len(ok))
    best = max(rate, key=rate.get)
    print("gate pass rates:", rate, "->", best, flush=True)
    if rate[best] < 0.75:
        print("WARNING: no caption passed the 75% gate (kill criterion K1); running anyway for diagnosis", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gate", "main"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    man = json.loads((HERE / "manifest.json").read_text())
    h = ace_lab.load(args.device)
    out = HERE / f"outputs_{args.phase}"
    out.mkdir(exist_ok=True)
    log_path = out / "log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"runs": []}

    if args.phase == "gate":
        jobs = [(it, c, cap, "s1") for it in man["items"] if it["split"] == "pilot" for c in ("lit", "ctl") for cap in CAPTIONS]
    else:
        cap = gate_choice()
        log["caption"] = CAPTIONS[cap]
        items = [it for it in man["items"] if it["split"] == "confirm"]
        if args.limit:
            items = items[: args.limit]
        jobs = [(it, c, cap, s) for s, conds in MAIN.items() for it in items for c in conds]
    for it, cond, cap, s in jobs:
        arm = cap if args.phase == "gate" else "base"
        wav = out / f"{it['id']}__{cond}__{arm}__{s}.wav"
        if wav.exists():
            continue
        r = ace_lab.generate(h, wav, CAPTIONS[cap], 18.0, SEEDS[s], lyrics="[Instrumental]", bpm=120, keyscale="",
                             task="repaint", src=str(HERE / it["sources"][cond]), extra=REPAINT)
        log["runs"].append({"id": it["id"], "cond": cond, "arm": arm, "seed": s, "seconds": r["seconds"]})
        log_path.write_text(json.dumps(log, indent=1))
        print(f"{wav.stem:>36s} {r['seconds']:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
