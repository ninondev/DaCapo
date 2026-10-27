"""Score E4 outputs (main environment; no ACE-Step code).

Per output:
  npa / npa_oct       per-note pitch accuracy vs the item's melody (absolute / octave-tolerant)
  npa_mismatch        same, against the other items' melodies (chance level for this audio)
  rpa                 frame raw pitch accuracy (SwiftF0) at the best lag
  wer / wer_mismatch  Whisper word error rate vs the item's lyrics / vs other items' lyrics
  voiced_frac, polyphony   a cappella check (is it one voice?)
Swap outputs also get npa_other and wer_other (vs the swapped-in item). The summary's
frac_follows_swapped_in is the share with npa_other > npa (melody swap) or wer_other < wer (lyrics swap).

  python score_sing.py sweep      python score_sing.py swap
"""
from __future__ import annotations

import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from dacapo.analysis import f0_swift, notes_basic_pitch
from dacapo.metrics import frame_metrics, levenshtein, note_pitch_accuracy
from dacapo.provenance import write_result
from dacapo.symbolic import Melody, Note

HERE = Path(__file__).parent


def whisper_name() -> str:
    import torch

    # singing is hard for ASR: use the large model whenever a GPU is available
    return "openai/whisper-large-v3" if torch.cuda.is_available() else "openai/whisper-small.en"


@lru_cache(maxsize=1)
def _pipe():
    import torch
    from transformers import pipeline

    gpu = torch.cuda.is_available()
    return pipeline("automatic-speech-recognition", model=whisper_name(), device=0 if gpu else -1,
                    torch_dtype=torch.float16 if gpu else torch.float32)


def asr():
    p = _pipe()
    kw = {} if whisper_name().endswith(".en") else {"generate_kwargs": {"language": "en", "task": "transcribe"}}
    return lambda path: p(path, **kw)


def norm_words(s: str) -> list[str]:
    s = s.split("\n", 1)[-1] if s.startswith("[") else s
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    return levenshtein(ref, hyp) / max(1, len(ref))


def melody(it: dict) -> Melody:
    return Melody(tuple(Note(**n) for n in it["melody"]["notes"]), it["melody"]["bpm"])


def pitch_scores(et, ef, m: Melody) -> dict:
    rt, rf = m.f0_track()
    fm = frame_metrics(rt, rf, et, ef, max_lag=0.5)
    iv, _ = m.note_events_seconds()
    notes = [(s, e, n.pitch) for (s, e), n in zip(iv, m.notes)]
    a = note_pitch_accuracy(notes, et, ef, lag=fm["lag_s"])
    b = note_pitch_accuracy(notes, et, ef, lag=fm["lag_s"], octave_tolerant=True)
    return {"npa": a["npa"], "npa_oct": b["npa"], "voiced_notes": a["voiced_notes"], "rpa": fm["RPA"], "rca": fm["RCA"]}


def score_file(wav: Path, own: dict, others: list[dict]) -> dict:
    et, ef = f0_swift(wav)
    p = pitch_scores(et, ef, melody(own))
    mis = [pitch_scores(et, ef, melody(o)) for o in others]
    hyp = norm_words(asr()(str(wav))["text"])
    notes = notes_basic_pitch(wav)
    grid = np.arange(0, et[-1], 0.05)
    poly = float(np.mean([sum(s <= g < e for s, e, _ in notes) for g in grid])) if notes else 0.0
    w = wer(own["words"], hyp)
    # WER > 1 means ASR hallucinated extra words (typical on wordless "oooh" singing): cap for averaging
    return {**p, "npa_mismatch": float(np.mean([m["npa"] for m in mis])), "npa_oct_mismatch": float(np.mean([m["npa_oct"] for m in mis])),
            "wer": w, "wer_capped": min(w, 1.0), "wer_mismatch": float(np.mean([min(wer(o["words"], hyp), 1.0) for o in others])),
            "hyp": " ".join(hyp), "voiced_frac": float((ef > 0).mean()), "polyphony": poly}


def main(mode: str = "sweep"):
    t0 = time.time()
    man = json.loads((HERE / "manifest.json").read_text())
    items = {it["id"]: it for it in man["items"]}
    out = HERE / f"outputs_{mode}"
    rows = []
    for wav in sorted(out.glob("*.wav")):
        parts = wav.stem.split("__")
        own = items[parts[0]]
        others = [o for k, o in items.items() if k != own["id"]]
        r = {"file": wav.name, "id": own["id"], "split": own["split"], "arm": parts[1], "seed": parts[-1]}
        r.update(score_file(wav, own, others))
        m = re.match(r"(mel|lyr)swap_to_(item\d+)_at(\d)", parts[1])
        if m:
            other = items[m.group(2)]
            o = pitch_scores(*f0_swift(wav), melody(other))
            r.update(kind=m.group(1), other=other["id"], at=int(m.group(3)), npa_other=o["npa"],
                     wer_other=wer(other["words"], r["hyp"].split()))
        rows.append(r)
        print(f"{wav.stem:>48s} npa {r['npa']:.2f} (mis {r['npa_mismatch']:.2f}) wer {r['wer']:.2f} "
              f"(mis {r['wer_mismatch']:.2f}) poly {r['polyphony']:.2f}", flush=True)

    summary = {}
    if mode == "sweep":
        for arm in sorted({r["arm"] for r in rows}):
            for split in ("dev", "test"):
                sel = [r for r in rows if r["arm"] == arm and r["split"] == split]
                if sel:
                    summary[f"{arm}|{split}"] = {k: round(float(np.mean([r[k] for r in sel])), 3) for k in
                                                ["npa", "npa_mismatch", "npa_oct", "rpa", "wer_capped", "wer_mismatch", "voiced_frac", "polyphony"]}
                    summary[f"{arm}|{split}"]["n"] = len(sel)
    else:
        for kind in ("mel", "lyr"):
            for s in range(1, 8):
                sel = [r for r in rows if r.get("kind") == kind and r.get("at") == s]
                if not sel:
                    continue
                if kind == "mel":
                    follow = [r["npa_other"] > r["npa"] for r in sel]
                else:
                    follow = [r["wer_other"] < r["wer"] for r in sel]
                summary[f"{kind}_swap_at{s}"] = {"frac_follows_swapped_in": round(float(np.mean(follow)), 3), "n": len(sel),
                                                 "npa_own": round(float(np.mean([r["npa"] for r in sel])), 3),
                                                 "npa_other": round(float(np.mean([r["npa_other"] for r in sel])), 3),
                                                 "wer_own": round(float(np.mean([r["wer"] for r in sel])), 3),
                                                 "wer_other": round(float(np.mean([r["wer_other"] for r in sel])), 3)}
    write_result(HERE / "results" / f"e4_{mode}.json", {"experiment": f"e4_sing_{mode}", "asr": whisper_name(), "f0": "swift"},
                 {"summary": summary, "rows": rows}, t0)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
