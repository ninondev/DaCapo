"""E1b analysis (main env): what do ACE-Step's 5 Hz FSQ codes encode?

1. Sensitivity: for the same melody, how often does the code index stay identical when we change
   the lead timbre, transpose by +2, change 2 notes, or delay by 200 ms? The floor is a different
   melody on the same timbre.
2. Linear probes (held-out melodies): decode pitch class / MIDI pitch at each 5 Hz frame from
   (a) code quantized vectors, (b) FSQ digits, (c) VAE latents mean-pooled to 5 Hz.
   (c) shows what is there before the semantic bottleneck, at the same frame rate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dacapo.provenance import write_result

HERE = Path(__file__).parent
FPS = 5.0
LEVELS = [8, 8, 8, 5, 5, 5]  # ResidualFSQ levels in acestep-v15-turbo config


def digits(codes: np.ndarray) -> np.ndarray:
    basis = np.cumprod([1] + LEVELS[:-1])
    return np.stack([(codes // b) % l for b, l in zip(basis, LEVELS)], axis=1)


def labels(item: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per 5 Hz frame: MIDI pitch at the frame centre (-1 if none), pitch class, and whether the
    melody changes note inside the frame (sub-frame motion the code would have to summarise)."""
    spb = 60.0 / item["melody"]["bpm"]
    notes = [(x["start"] * spb, (x["start"] + x["dur"]) * spb, x["pitch"]) for x in item["melody"]["notes"]]
    pitch, change = np.full(n, -1), np.zeros(n, dtype=bool)
    for f in range(n):
        c, a, b = (f + 0.5) / FPS, f / FPS, (f + 1) / FPS
        for s, e, p in notes:
            if s <= c < e:
                pitch[f] = p
        change[f] = any(a < s < b for s, _, _ in notes)
    return pitch, np.where(pitch >= 0, pitch % 12, -1), change


def agreement(a: np.ndarray, b: np.ndarray, lo: int) -> float:
    n = min(len(a), len(b))
    return float((a[lo:n] == b[lo:n]).mean())


def main():
    t0 = time.time()
    man = json.loads((HERE / "probe_stimuli" / "manifest.json").read_text())
    z = np.load(HERE / "results" / "e1b_codes.npz")
    items = {it["id"]: it for it in man["items"]}
    lead_in = int(2 * 60.0 / man["bpm"] * FPS) + 1  # skip band-only lead-in frames

    sens = {k: [] for k in ["same_audio_reencode", "timbre_flute", "timbre_violin", "transp+2", "2wrong", "delay200ms", "delay200ms_realigned", "other_melody"]}
    ids = sorted({it["melody_id"] for it in man["items"]})
    for m in ids:
        ref = z[f"p{m:02d}_piano_orig/codes"]
        if f"p{m:02d}_piano_orig/codes_rep" in z:
            sens["same_audio_reencode"].append(agreement(ref, z[f"p{m:02d}_piano_orig/codes_rep"], lead_in))
        sens["timbre_flute"].append(agreement(ref, z[f"p{m:02d}_flute_orig/codes"], lead_in))
        sens["timbre_violin"].append(agreement(ref, z[f"p{m:02d}_violin_orig/codes"], lead_in))
        sens["other_melody"].append(agreement(ref, z[f"p{(m + 1) % len(ids):02d}_piano_orig/codes"], lead_in))
        for v in ["transp+2", "2wrong", "delay200ms"]:
            key = f"p{m:02d}_piano_{v}/codes"
            if key in z:
                sens[v].append(agreement(ref, z[key], lead_in))
        if f"p{m:02d}_piano_delay200ms/codes" in z:
            sens["delay200ms_realigned"].append(agreement(ref, z[f"p{m:02d}_piano_delay200ms/codes"][1:], lead_in))
    sens_summary = {k: {"mean_identical_frac": round(float(np.mean(v)), 3), "n": len(v)} for k, v in sens.items() if v}

    feats = {"quant": [], "digits": [], "latent_pool5": []}
    rows = []
    for sid, it in items.items():
        if it["variant"] != "orig":
            continue
        codes = z[f"{sid}/codes"]
        n = len(codes)
        lat = z[f"{sid}/latent"]
        pooled = lat[: n * 5].reshape(-1, 5, lat.shape[1]).mean(axis=1) if len(lat) >= n * 5 else \
            np.vstack([lat, np.repeat(lat[-1:], n * 5 - len(lat), 0)]).reshape(n, 5, -1).mean(axis=1)
        pitch, pc, change = labels(it, n)
        for f in range(n):
            if pitch[f] < 0:
                continue
            rows.append((it["melody_id"], it["timbre"], pitch[f], pc[f], change[f]))
            feats["quant"].append(z[f"{sid}/quant"][f])
            feats["digits"].append(digits(codes[f:f + 1])[0])
            feats["latent_pool5"].append(pooled[f])
    mel = np.array([r[0] for r in rows])
    timbre = np.array([r[1] for r in rows])
    y_pc = np.array([r[3] for r in rows])
    y_pitch = np.array([r[2] for r in rows])
    chg = np.array([r[4] for r in rows])
    test_mel = set(ids[-6:])
    te = np.array([m in test_mel for m in mel])

    def clf(name):
        if name == "digits":
            return make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000, C=1.0))
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))

    probes = {}
    for target, y in [("pitch_class", y_pc), ("midi_pitch", y_pitch)]:
        majority = float((y[te] == np.bincount(y[~te]).argmax()).mean())
        probes[target] = {"majority_baseline": round(majority, 3)}
        for name, X in feats.items():
            X = np.asarray(X)
            model = clf(name).fit(X[~te], y[~te])
            pred = model.predict(X[te])
            ok = pred == y[te]
            probes[target][name] = {"acc": round(float(ok.mean()), 3),
                                    "acc_frames_with_note_change": round(float(ok[chg[te]].mean()), 3),
                                    "acc_frames_steady": round(float(ok[~chg[te]].mean()), 3)}
            # cross-timbre: train piano+flute, test violin (still held-out melodies)
            tr = ~te & (timbre != "violin")
            tv = te & (timbre == "violin")
            m2 = clf(name).fit(X[tr], y[tr])
            probes[target][name]["acc_unseen_timbre"] = round(float((m2.predict(X[tv]) == y[tv]).mean()), 3)
    res = {"sensitivity": sens_summary, "probes": probes,
           "n_train_frames": int((~te).sum()), "n_test_frames": int(te.sum()),
           "frac_test_frames_with_note_change": round(float(chg[te].mean()), 3)}
    cfg = {"experiment": "e1b_code_probe", "levels": LEVELS, "fps": FPS, "test_melodies": sorted(test_mel),
           "classifier": "logreg (C=0.1 standardized; digits one-hot C=1.0)"}
    write_result(HERE / "results" / "e1b_probe.json", cfg, res, t0)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
