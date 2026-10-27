"""Score E1 outputs with the calibrated ruler.

Each (stimulus, pathway) output is scored against
  * its own melody ("matched"), and
  * the other melodies at the same tempo ("mismatched", the chance level for this audio).
The main number is matched minus mismatched: the melody information that survived the pathway.
The input stimulus is scored the same way (pathway "input") to give the ruler's ceiling on this kind of mix.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dacapo.analysis import chroma_argmax, f0, notes_basic_pitch, skyline
from dacapo.metrics import frame_metrics, note_metrics, pitch_class_accuracy
from dacapo.provenance import write_result
from dacapo.symbolic import Melody, Note

HERE = Path(__file__).parent
METRICS = ["swift.RPA", "swift.RCA", "chroma.pc_acc", "skyline.F"]


def melody_of(s: dict) -> Melody:
    return Melody(tuple(Note(**n) for n in s["melody"]["notes"]), s["melody"]["bpm"])


def features(wav: str) -> dict:
    return {"f0": f0(wav, "swift"), "chroma": chroma_argmax(wav), "notes": skyline(notes_basic_pitch(wav))}


def score(feat: dict, m: Melody) -> dict:
    rt, rf = m.f0_track()
    iv, hz = m.note_events_seconds()
    fr = frame_metrics(rt, rf, *feat["f0"])
    return {"swift.RPA": fr["RPA"], "swift.RCA": fr["RCA"],
            "chroma.pc_acc": pitch_class_accuracy(rt, rf, *feat["chroma"])["pc_acc"],
            "skyline.F": note_metrics(iv, hz, feat["notes"])["F"]}


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(outputs: str = "outputs"):
    t0 = time.time()
    stimuli = json.loads((HERE / "stimuli" / "manifest.json").read_text())["stimuli"]
    runs = json.loads((HERE / outputs / "driver_log.json").read_text())
    by_id = {s["id"]: s for s in stimuli}
    # stimulus wavs are relative to stimuli/ (absolute paths pass through); output wavs keep only their file name
    items = [(s["id"], "input", str(HERE / "stimuli" / s["wav"])) for s in stimuli] + \
            [(r["id"], r["pathway"], str(HERE / outputs / Path(r["wav"]).name)) for r in runs["runs"]]

    rows = []
    for sid, pathway, wav in items:
        s = by_id[sid]
        feat = features(wav)
        matched = score(feat, melody_of(s))
        others = [o for o in stimuli if o["bpm"] == s["bpm"] and o["id"] != sid]
        mism = [score(feat, melody_of(o)) for o in others]
        row = {"id": sid, "pathway": pathway, "bpm": s["bpm"], "lead": s["lead"]}
        for k in METRICS:
            row[k] = matched[k]
            row[k + ".mismatch"] = float(np.mean([x[k] for x in mism]))
            row[k + ".delta"] = matched[k] - row[k + ".mismatch"]
        rows.append(row)
        print(f"{sid:>16s} {pathway:>12s} " + " ".join(f"{k}={row[k]:.2f}({row[k + '.delta']:+.2f})" for k in METRICS), flush=True)

    summary = {}
    for pathway in dict.fromkeys(r["pathway"] for r in rows):
        for bpm in sorted({r["bpm"] for r in rows}):
            sel = [r for r in rows if r["pathway"] == pathway and r["bpm"] == bpm]
            if not sel:
                continue
            entry = {"n": len(sel)}
            for k in METRICS:
                d = np.array([r[k + ".delta"] for r in sel])
                entry[k] = round(float(np.mean([r[k] for r in sel])), 3)
                entry[k + ".delta"] = round(float(d.mean()), 3)
                entry[k + ".delta_ci95"] = [round(v, 3) for v in bootstrap_ci(d)]
            summary[f"{pathway}@{bpm}"] = entry
    cfg = {"experiment": "e1_code_bottleneck", "driver_settings": runs.get("settings"), "f0": "swift",
           "chroma_fmin_hz": 250, "note_extractor": "basic_pitch+skyline", "mismatch": "other stimuli, same tempo"}
    cfg["outputs"] = outputs
    write_result(HERE / "results" / f"e1_scores_{outputs}.json", cfg, {"summary": summary, "rows": rows}, t0)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs")
