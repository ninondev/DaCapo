"""Runs the GPU experiment suite (inside the Colab notebook, one environment).

Each stage is a list of commands run in order, with logs in package/logs/<stage>.log. A failed
stage is recorded in package/status.json and the suite moves on. The experiment scripts skip
outputs that already exist, so a re-run after a disconnect continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
PKG = Path("/content/work/package") if Path("/content").exists() else REPO / "package"
PY = sys.executable


def stages(scale: str) -> dict[str, list[tuple[str, list[str]]]]:
    q = scale == "quick"
    e1, e1b, e2, e3, e4, e5 = (EXP / "e1_melody_paths", EXP / "e1_melody_paths", EXP / "e2_hooks",
                               EXP / "e3_pitch_readout", EXP / "e4_sing", EXP / "e5_motif")
    drv = [PY, "ace_driver.py", "--manifest", "stimuli/manifest.json", "--device", "cuda"]
    return {
        "smoke": [(e2, [PY, "smoke_test.py", "--device", "cuda"])],
        "acap": [(e2, [PY, "acappella_pilot.py", "--device", "cuda", "--seeds", "11,12" if q else "11,12,13"]),
                 (e2, [PY, "acappella_score.py"])],
        "sing": [(e4, [PY, "make_items.py"]),
                 (e4, [PY, "run_sing.py", "--device", "cuda", "--mode", "sweep", "--seeds", "1" if q else "1,2"] + (["--limit", "2"] if q else [])),
                 (e4, [PY, "score_sing.py", "sweep"])],
        "swap": [(e4, [PY, "make_items.py"]),
                 (e4, [PY, "run_sing.py", "--device", "cuda", "--mode", "swap", "--seeds", "1"] + (["--limit", "1"] if q else [])),
                 (e4, [PY, "score_sing.py", "swap"])],
        "analogy": [(e5, [PY, "make_items.py"]),
                    (e5, [PY, "score_analogy.py", "gate"]),
                    (e5, [PY, "run_analogy.py", "--phase", "gate", "--device", "cuda"]),
                    (e5, [PY, "score_analogy.py", "score", "outputs_gate"]),
                    (e5, [PY, "run_analogy.py", "--phase", "main", "--device", "cuda"] + (["--limit", "3"] if q else [])),
                    (e5, [PY, "score_analogy.py", "score", "outputs_main"])],
        "e1": [(e1, [PY, "make_stimuli.py", "--n", "4" if q else "16", "--tempi", "90,150" if q else "80,120,160"]),
               (e1, drv + ["--out", "outputs_s1", "--shift", "1.0"]),
               (e1, drv + ["--out", "outputs_s3", "--shift", "3.0"]),
               (e1, [PY, "score.py", "outputs_s1"]),
               (e1, [PY, "score.py", "outputs_s3"])],
        "e1b": [(e1b, [PY, "probe_stimuli.py", "--n", "12" if q else "48"]),
                (e1b, [PY, "ace_codes.py", "--manifest", "probe_stimuli/manifest.json", "--out", "results/e1b_codes.npz", "--device", "cuda"]),
                (e1b, [PY, "probe.py"])],
        "e3": [(e3, [PY, "make_stimuli.py"]),
               (e3, [PY, "vae_io.py", "encode", "--manifest", "manifest.json", "--out", "results/latents.npz"]),
               (e3, [PY, "subspace.py", "fit"]),
               (e3, [PY, "vae_io.py", "decode", "--latents", "results/edits.npz", "--outdir", "edited"]),
               (e3, [PY, "subspace.py", "score"])],
    }


ORDER = ["smoke", "acap", "sing", "analogy", "swap", "e1", "e1b", "e3"]


def run_stage(name: str, cmds: list[tuple[Path, list[str]]], status: dict) -> None:
    log = PKG / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    status[name] = {"state": "running"}
    with log.open("a") as fh:
        for cwd, cmd in cmds:
            fh.write(f"\n$ (cd {cwd}) {' '.join(cmd)}\n")
            fh.flush()
            p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line)
                if not any(tag in line for tag in ("| INFO", "| DEBUG", "| WARNING")):
                    print(f"[{name}] {line}", end="", flush=True)
            p.wait()
            if p.returncode != 0:
                status[name] = {"state": "failed", "cmd": " ".join(cmd), "returncode": p.returncode,
                                "minutes": round((time.time() - t0) / 60, 1)}
                print(f"[{name}] FAILED (see logs/{name}.log); continuing with the next stage", flush=True)
                return
    status[name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}


def package() -> None:
    """Copy result and log JSON plus a small listening subset (mp3) into PKG."""
    for f in EXP.glob("*/results/*.json"):
        dst = PKG / "results" / f.parent.parent.name / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
    for f in EXP.glob("*/outputs_*/log.json"):
        dst = PKG / "logs" / f"{f.parent.parent.name}_{f.parent.name}_log.json"
        shutil.copy(f, dst)
    for f in EXP.glob("*/outputs*/driver_log.json"):
        shutil.copy(f, PKG / "logs" / f"{f.parent.parent.name}_{f.parent.name}_driver_log.json")
    picks = []
    for it in ("item03", "item05", "item08"):
        for arm in ("noguide", "ctx_oohs_k2", "ctx_oohs_k3", "ctx_oohs_k4", "ctx_clarinet_k3", "sde_oohs_n0.25"):
            picks += list((EXP / "e4_sing" / "outputs_sweep").glob(f"{it}__{arm}__s1.wav"))
        picks += list((EXP / "e4_sing" / "guides").glob(f"{it}_oohs.wav"))
    picks += list((EXP / "e2_hooks" / "outputs_acappella").glob("*.wav"))
    picks += list((EXP / "e4_sing" / "outputs_swap").glob("item03__*__s1.wav"))[:16]
    for cond in ("lit", "ctl", "rel+5", "seq", "ana"):
        picks += list((EXP / "e5_motif" / "outputs_main").glob(f"mel0[23]__{cond}__base__s1.wav"))
        picks += list((EXP / "e5_motif" / "sources").glob(f"mel02_{cond}.wav"))
    out = PKG / "audio"
    out.mkdir(parents=True, exist_ok=True)
    for w in picks:
        mp3 = out / f"{w.parent.parent.name}__{w.parent.name}__{w.stem}.mp3"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(w), "-b:a", "96k", str(mp3)], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=["quick", "full"])
    ap.add_argument("--stages", default=",".join(ORDER))
    args = ap.parse_args()
    PKG.mkdir(parents=True, exist_ok=True)
    status_path = PKG / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status["_meta"] = {"scale": args.scale, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                             capture_output=True, text=True).stdout.strip()}
    all_stages = stages(args.scale)
    for name in [s for s in ORDER if s in args.stages.split(",")]:
        if status.get(name, {}).get("state") == "ok":
            print(f"[{name}] already done, skipping", flush=True)
            continue
        print(f"\n===== stage {name} =====", flush=True)
        run_stage(name, all_stages[name], status)
        status_path.write_text(json.dumps(status, indent=1))
        if name == "smoke" and status[name]["state"] != "ok":
            print("Smoke test failed: stopping before spending GPU time on the full suite.", flush=True)
            break
    package()
    status_path.write_text(json.dumps(status, indent=1))
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
