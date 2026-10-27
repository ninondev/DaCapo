"""E3 analysis: is there a timbre-invariant, editable pitch subspace in the VAE latent space?

  python subspace.py fit     # LDA on training timbres -> held-out-timbre decoding + latent edits
  python subspace.py score   # after vae_io.py decode: measure what the edits did to the audio

Edit variants for a source latent z (frames inside sustained notes only):
  identity        z unchanged (VAE round trip; floor for "did anything change")
  replace_k{K}    U U^T coordinates replaced by those of the target pitch's class mean (K-dim LDA basis U)
  shift_k{K}      z + U U^T (mu_target - mu_source)  (source pitch known from the score: oracle)
  meanshift       z + (mu_target - mu_source)        (full 64-d difference of class means)
The target pitch per frame comes from the target melody (same rhythm, new pitches).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FPS = 25.0
KS = (4, 8, 16, 31)


def frame_labels(mel: dict, n: int, attack: float = 0.08, release: float = 0.04) -> np.ndarray:
    spb = 60.0 / mel["bpm"]
    lab = np.full(n, -1)
    t = (np.arange(n) + 0.5) / FPS
    for x in mel["notes"]:
        s, e = x["start"] * spb + attack, (x["start"] + x["dur"]) * spb - release
        lab[(t >= s) & (t < e)] = x["pitch"]
    return lab


def lda(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3):
    classes = np.unique(y)
    mu = {c: X[y == c].mean(0) for c in classes}
    g = X.mean(0)
    Sw = sum(((X[y == c] - mu[c]).T @ (X[y == c] - mu[c])) for c in classes) / len(X)
    Sb = sum((y == c).mean() * np.outer(mu[c] - g, mu[c] - g) for c in classes)
    Sw += ridge * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    evals, evecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
    order = np.argsort(-evals.real)
    return mu, evecs.real[:, order], evals.real[order]


def fit():
    t0 = time.time()
    man = json.loads((HERE / "manifest.json").read_text())
    z = np.load(HERE / "results" / "latents.npz")
    items = {it["id"]: it for it in man["items"]}
    Xtr, ytr, Xte, yte, tte = [], [], [], [], []
    for sid, it in items.items():
        if it["kind"] != "scale":
            continue
        lat = z[f"{sid}/wav"]
        lab = frame_labels(it["melody"], len(lat))
        keep = lab >= 0
        (Xte if it["heldout"] else Xtr).append(lat[keep])
        (yte if it["heldout"] else ytr).append(lab[keep])
        if it["heldout"]:
            tte += [it["timbre"]] * int(keep.sum())
    Xtr, ytr, Xte, yte, tte = np.vstack(Xtr), np.concatenate(ytr), np.vstack(Xte), np.concatenate(yte), np.array(tte)
    mu, V, ev = lda(Xtr, ytr)
    classes = np.array(sorted(mu))
    M = np.stack([mu[c] for c in classes])

    diag = {"n_train_frames": int(len(Xtr)), "n_test_frames": int(len(Xte)), "lda_top_eigenvalues": [round(float(e), 2) for e in ev[:10]]}
    for k in KS:
        P = V[:, :k]
        d = ((Xte @ P)[:, None, :] - (M @ P)[None]) ** 2
        pred = classes[d.sum(-1).argmin(1)]
        diag[f"heldout_pitch_acc_k{k}"] = round(float((pred == yte).mean()), 3)
        diag[f"heldout_pitchclass_acc_k{k}"] = round(float((pred % 12 == yte % 12).mean()), 3)
        diag[f"per_timbre_k{k}"] = {t: round(float((pred[tte == t] == yte[tte == t]).mean()), 3) for t in np.unique(tte)}
    # nearest class mean in the full 64-d space, for comparison
    d = ((Xte[:, None, :] - M[None]) ** 2).sum(-1)
    diag["heldout_pitch_acc_full64_nearest_mean"] = round(float((classes[d.argmin(1)] == yte).mean()), 3)

    edits = {}
    for sid, it in items.items():
        if it["kind"] != "melody":
            continue
        lat = z[f"{sid}/wav"].astype(np.float64)
        src = frame_labels(it["melody"], len(lat))
        tgt = frame_labels(it["target"], len(lat))
        ok = (src >= 0) & (tgt >= 0) & np.isin(src, classes) & np.isin(tgt, classes)
        edits[f"{sid}/identity"] = lat.astype(np.float32)
        edits[f"{sid}/target_render"] = z[f"{sid}/target_wav"].astype(np.float32)
        mt = np.stack([mu[p] if p in mu else np.zeros(64) for p in np.where(ok, tgt, classes[0])])
        ms = np.stack([mu[p] if p in mu else np.zeros(64) for p in np.where(ok, src, classes[0])])
        for k in KS:
            U, _ = np.linalg.qr(V[:, :k])
            rep = lat.copy()
            rep[ok] = lat[ok] + (mt[ok] - lat[ok]) @ U @ U.T
            edits[f"{sid}/replace_k{k}"] = rep.astype(np.float32)
            sh = lat.copy()
            sh[ok] = lat[ok] + (mt[ok] - ms[ok]) @ U @ U.T
            edits[f"{sid}/shift_k{k}"] = sh.astype(np.float32)
        ms_full = lat.copy()
        ms_full[ok] = lat[ok] + (mt[ok] - ms[ok])
        edits[f"{sid}/meanshift"] = ms_full.astype(np.float32)
        diag.setdefault("edited_frame_fraction", {})[sid] = round(float(ok.mean()), 3)
    np.savez_compressed(HERE / "results" / "edits.npz", **edits)
    np.savez_compressed(HERE / "results" / "subspace.npz", V=V, classes=classes, M=M)
    (HERE / "results" / "fit_diag.json").write_text(json.dumps({"wall_seconds": round(time.time() - t0, 1), **diag}, indent=1))
    print(json.dumps(diag, indent=1))


def score():
    from dacapo.analysis import f0_swift
    from dacapo.attributes import features, similarity
    from dacapo.metrics import frame_metrics
    from dacapo.provenance import write_result
    from dacapo.symbolic import Melody, Note

    t0 = time.time()
    man = json.loads((HERE / "manifest.json").read_text())
    dec = HERE / "edited"
    rows = []
    for it in man["items"]:
        if it["kind"] != "melody":
            continue
        src_m = Melody(tuple(Note(**n) for n in it["melody"]["notes"]), it["melody"]["bpm"])
        tgt_m = Melody(tuple(Note(**n) for n in it["target"]["notes"]), it["target"]["bpm"])
        orig = features(HERE / it["wav"])
        for w in sorted(dec.glob(f"{it['id']}__*.wav")):
            variant = w.stem.split("__")[1]
            et, ef = f0_swift(w)
            row = {"id": it["id"], "timbre": it["timbre"], "variant": variant}
            for name, m in (("target", tgt_m), ("source", src_m)):
                rt, rf = m.f0_track()
                fm = frame_metrics(rt, rf, et, ef, max_lag=0.1)
                row[f"RPA_vs_{name}"], row[f"RCA_vs_{name}"] = fm["RPA"], fm["RCA"]
            sim = similarity(orig, features(w))
            row["timbre_cos_vs_source_audio"] = sim["timbre_cos"]
            row["rms_corr_vs_source_audio"] = sim["rms_corr"]
            rows.append(row)
            print(f"{it['id']:>28s} {variant:>14s} RPA->tgt {row['RPA_vs_target']:.2f} RPA->src {row['RPA_vs_source']:.2f} "
                  f"timbre {row['timbre_cos_vs_source_audio']:.3f}", flush=True)
    summary = {}
    for v in sorted({r["variant"] for r in rows}):
        sel = [r for r in rows if r["variant"] == v]
        summary[v] = {k: round(float(np.mean([r[k] for r in sel])), 3) for k in
                      ["RPA_vs_target", "RCA_vs_target", "RPA_vs_source", "timbre_cos_vs_source_audio", "rms_corr_vs_source_audio"]}
        summary[v]["n"] = len(sel)
    write_result(HERE / "results" / "e3_scores.json", {"experiment": "e3_pitch_subspace_vae_only", "ks": list(KS)},
                 {"summary": summary, "rows": rows}, t0)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"fit": fit, "score": score}[sys.argv[1]]()
