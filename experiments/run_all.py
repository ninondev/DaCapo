"""Runs the GPU experiments: inside the Colab notebook, or on this machine's CPU at the tiny test scale.

Run A (the default stages, in order, each gated automatically):
  setup    render the E5 v2 sources and hypothesis renders from the committed manifests, check their
           sha256, rerun the v1 regression and K0 for every caption timbre
  a0       generation checks on pilot melodies (a failure is a technical stop, not a strike)
  p0b      E5 Phase 0b and the K1 decision (quality metrics only)
  main     E5 confirmatory grid, frozen quality stage, accuracy stage, terminal outcome file
  e6       E6 lyric-alignment read-out (decides whether B1 may run)
  package  results, logs, a listening subset and a blind-listening pack with a separate key, in one
           folder that the notebook zips
The v1 stages (smoke, acap, sing, swap, e1, e1b, e3) are not part of Run A; --stages runs them.

  python experiments/run_all.py                                 Run A at full scale on CUDA
  python experiments/run_all.py --scale tiny --device cpu       every Run A stage with 1-2 generations each
  python experiments/run_all.py --stages smoke,e1 --scale quick v1 stages, for reproduction

--state DIR keeps all run state (ledgers, designs, results, audio, stage status, package) under
DIR/<scale>/, e.g. on Google Drive, so a new runtime resumes where the last one stopped. Each scale keeps
its run state in its own folders. The rendered sources are shared, and every run checks them against the
sha256 in the manifests. Only "full" can produce confirmatory results. DACAPO_PY_MAIN and DACAPO_PY_ACE
name the interpreters for measurement and for ACE-Step commands when they differ (on Colab both are this
interpreter). The blind-listening answer key is written next to the package folder, never inside it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
E5, E6 = EXP / "e5_motif", EXP / "e6_clock"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(E5))
from stage import Echo  # noqa: E402

PY = sys.executable
RUN_A = ["setup", "a0", "p0b", "main", "e6", "package"]
V1_STAGES = ["smoke", "acap", "sing", "swap", "e1", "e1b", "e3"]
E5_STAGES = ("setup", "a0", "p0b", "main")
BLIND_SEED = 20261028
E6_TINY_WHISPER = "openai/whisper-tiny"


def v1_stages(scale: str) -> dict[str, list[tuple[Path, list[str]]]]:
    q = scale == "quick"
    e1, e1b, e2, e3, e4 = (EXP / "e1_melody_paths", EXP / "e1_melody_paths", EXP / "e2_hooks",
                           EXP / "e3_pitch_readout", EXP / "e4_sing")
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


def run_v1_stage(name: str, cmds: list[tuple[Path, list[str]]], log: Path) -> dict:
    t0 = time.time()
    with log.open("a") as fh:
        for cwd, cmd in cmds:
            fh.write(f"\n$ (cd {cwd}) {' '.join(cmd)}\n")
            fh.flush()
            echo = Echo()
            p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line)
                if echo.show(line):
                    print(f"[{name}] {line}", end="", flush=True)
            p.wait()
            if p.returncode != 0:
                return {"state": "failed", "cmd": " ".join(cmd), "returncode": p.returncode,
                        "minutes": round((time.time() - t0) / 60, 1)}
    return {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}


class Paths:
    """Every folder a scale writes to: under --state DIR/<scale>/ when given, else local per-scale folders.
    ACE-Step's progress-estimate cache goes to a local folder per scale (ACESTEP_PROJECT_ROOT)."""

    def __init__(self, scale: str, state: str | None):
        self.scale = scale
        sfx = "" if scale == "full" else f"_{scale}"
        if state:
            root = Path(state) / scale
            self.root, self.e5_run, self.e6_out = root, root / "e5_run_v2", root / "e6_outputs_readout"
            self.e6_res, self.e6_a0, self.pkg = root / "e6_results", root / "e6_outputs_readout_a0", root / "package"
            self.key = root / "listening_key.json"
        else:
            local_pkg = Path("/content/work") if Path("/content").exists() else REPO
            self.root = local_pkg / f"package{sfx}"
            self.e5_run, self.e6_out = E5 / f"run_v2{sfx}", E6 / f"outputs_readout{sfx}"
            self.e6_res, self.e6_a0, self.pkg = E6 / f"results{sfx}", E6 / f"outputs_readout_a0{sfx}", self.root
            self.key = local_pkg / f"listening_key{sfx}.json"
        self.ace = Path(tempfile.gettempdir()) / f"dacapo_acestep_{scale}"
        for p in (self.root, self.e5_run, self.e6_out, self.e6_res, self.e6_a0, self.pkg / "logs", self.ace):
            p.mkdir(parents=True, exist_ok=True)
        self.status = self.root / "status.json"

    def env(self) -> dict:
        return {"DACAPO_SCALE": "tiny" if self.scale == "tiny" else "full", "E5_RUN": str(self.e5_run),
                "E6_SCALE": self.scale, "E6_OUT": str(self.e6_out), "E6_RES": str(self.e6_res), "E6_A0": str(self.e6_a0),
                "ACESTEP_PROJECT_ROOT": str(self.ace),
                "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}


def e5_stage(name: str, paths: Paths, device: str, workers: int, status: dict) -> dict:
    os.environ.update(paths.env())
    import stage as S
    py_main = os.environ.get("DACAPO_PY_MAIN", PY)
    py_ace = os.environ.get("DACAPO_PY_ACE", PY)
    r = S.Runner(paths.pkg / "logs" / f"{name}.log", py_main, py_ace, device, env=paths.env())
    full = paths.scale == "full"
    needs = {"a0": ["setup"], "p0b": ["setup", "a0"], "main": ["setup", "a0", "p0b"]}.get(name, [])
    blocked = [n for n in needs if status.get(n, {}).get("state") != "ok"]
    if blocked and full:
        return S._status("blocked", reason=f"needs {blocked}")
    if name == "setup":
        return S.setup(r, workers)
    if name == "a0":
        return S.a0(r)
    if name == "p0b":
        return S.p0b(r)
    res = S.main_grid(r, n_boot=None if full else 2000)
    res["run_b"] = S.run_b_eligibility()
    return res


def e6_stage(paths: Paths, device: str) -> dict:
    """E6 read-out: acceptance checks on two clips outside the plan, the registered scoring path on them,
    the planned clips, scoring, the decision, and the Run B gate check."""
    t0 = time.time()
    env = {**os.environ, **paths.env()}
    py_main, py_ace = os.environ.get("DACAPO_PY_MAIN", PY), os.environ.get("DACAPO_PY_ACE", PY)
    tiny = paths.scale == "tiny"
    whisper = ["--whisper", E6_TINY_WHISPER] if tiny else []
    steps = [("a0", py_ace, ["a0", "--device", device]),
             ("score_a0", py_main, ["score", "--a0", "--device", device, *whisper]),
             # tiny: the one planned clip already generated on CPU while the read-out was tested (disclosed)
             ("gen", py_ace, ["gen", "--device", device] + (["--only", "item00_r0"] if tiny else [])),
             ("score", py_main, ["score", "--device", device, *whisper]),
             ("decide", py_main, ["decide"])]
    log = paths.pkg / "logs" / "e6.log"
    out = {}
    for label, py, args in steps:
        rc = _call([py, "readout.py", *args], E6, env, log, "e6")
        out[label] = rc
        if rc != 0:
            return {"state": "technical_stop", "scale": paths.scale, "failed": label, "returncode": rc, "codes": out,
                    "minutes": round((time.time() - t0) / 60, 1)}
    gate = json.loads((paths.e6_res / "e6_gate.json").read_text())
    out["check_gate"] = _call([py_main, "readout.py", "check-gate"], E6, env, log, "e6")
    return {"state": "ok", "scale": paths.scale, "b1_allowed": gate["b1_allowed"], "labels": gate["labels"],
            "reasons": gate["reasons"], "codes": out, "minutes": round((time.time() - t0) / 60, 1)}


def _call(cmd: list[str], cwd: Path, env: dict, log: Path, tag: str) -> int:
    echo = Echo()
    with log.open("a") as fh:
        fh.write(f"\n$ (cd {cwd}) {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            fh.write(line)
            if echo.show(line):
                print(f"[{tag}] {line}", end="", flush=True)
        p.wait()
    return p.returncode


def _mp3(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-b:a", "96k", str(dst)], check=False)
    return p.returncode == 0


def e5_main_dir(paths: Paths) -> Path | None:
    """The confirmatory audio that may leave the run: at the full scale only once the terminal outcome
    exists, and only from the attempt that produced it."""
    term = _read(paths.e5_run / "results" / "e5_terminal.json")
    if term is None and paths.scale == "full":
        return None
    attempt = (term or {}).get("attempt", 1)
    d = paths.e5_run / ("outputs_v2_main" + ("" if attempt == 1 else f"_a{attempt}"))
    return d if (d / "design.json").exists() else None


def _read(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def blind_pack(paths: Paths, out: Path) -> dict:
    """About 60 clips under random names, chosen from the plans with a fixed seed (never from scores):
    5 confirmatory E5 draws per condition and 20 E6 read-out clips. The key goes to paths.key, outside
    the package folder."""
    import v2_config as C
    from dacapo.motif_integrity import audio_name
    rng = random.Random(BLIND_SEED)
    picks = []
    d = e5_main_dir(paths)
    if d is not None:
        jobs = json.loads((d / "design.json").read_text())["jobs"]
        for cond in C.CONDITIONS:
            js = sorted((j for j in jobs if j["cond"] == cond), key=lambda j: (j["id"], j["seed"]))
            for j in rng.sample(js, min(5, len(js))):
                picks.append(("e5", d / audio_name(j), {k: j[k] for k in ("id", "cond", "seed")} | {"phase_dir": d.name}))
    sys.path.insert(0, str(E6))
    import readout
    plan = readout.items()
    e4 = [x for x in plan if x["set"] == "e4"]
    extra = [x for x in plan if x["set"] == "extra"]
    for x in rng.sample(e4, 12) + rng.sample(extra, 8):
        picks.append(("e6", paths.e6_out / f"{x['id']}.wav", {k: x[k] for k in ("id", "set", "lyric_id", "voice", "seed")}))
    present = [p for p in picks if p[1].exists()]
    rng.shuffle(present)
    pack = out / "listening_pack"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    key, rows = {}, ["clip,vocals_present,words_intelligible,accompaniment,gap_repeats_earlier_motif,notes"]
    for i, (exp, wav, meta) in enumerate(present, 1):
        cid = f"clip_{i:02d}"
        if _mp3(wav, pack / f"{cid}.mp3"):
            key[cid] = {"experiment": exp, "file": wav.name, **meta}
            rows.append(f"{cid},,,,,")
    (pack / "ratings.csv").write_text("\n".join(rows) + "\n")
    (pack / "README.txt").write_text(
        "Listen to each clip once or twice, in any order, and fill in ratings.csv with yes, no or unsure.\n"
        "vocals_present: is anyone singing?\n"
        "words_intelligible: can you make out the sung words?\n"
        "accompaniment: is there any instrument besides the main voice or melody?\n"
        "gap_repeats_earlier_motif: for instrumental clips, a short melody plays, then something fills\n"
        "seconds 12 to 16. Does that part repeat a phrase heard earlier in the clip? Leave it empty for songs.\n")
    paths.key.write_text(json.dumps({"seed": BLIND_SEED, "clips": key}, indent=1))
    (out / "listening_key.json").unlink(missing_ok=True)
    shutil.make_archive(str(out / "listening_pack"), "zip", pack)
    shutil.rmtree(pack)
    return {"n_clips": len(key), "n_planned": len(picks), "e5_dir": d.name if d else None, "key": str(paths.key)}


def package(paths: Paths, status: dict) -> dict:
    out = paths.pkg
    for sub in ("e5", "e6"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for f in paths.e5_run.glob("results/*.json"):
        shutil.copy(f, out / "e5" / f.name)
    for f in list(paths.e5_run.glob("outputs*/design.json")) + list(paths.e5_run.glob("outputs*/ledger.jsonl")) \
            + list(paths.e5_run.glob("outputs*/provenance_*.json")) + list(paths.e5_run.glob("outputs_a0/a0_*.json")):
        shutil.copy(f, out / "e5" / f"{f.parent.name}__{f.name}")
    for f in paths.e6_res.glob("*.json"):
        shutil.copy(f, out / "e6" / f.name)
    for f in list(paths.e6_out.glob("log.json")) + list(paths.e6_out.glob("*.failed.json")) + list(paths.e6_a0.glob("a0_*.json")):
        shutil.copy(f, out / "e6" / f"{f.parent.name}__{f.name}")
    listen = out / "audio"
    n = 0
    d = e5_main_dir(paths)
    if d is not None:
        from dacapo.motif_integrity import AUDIO_EXT
        for w in sorted(d.glob(f"*__d0{AUDIO_EXT}"))[:16]:
            n += _mp3(w, listen / f"e5__{d.name}__{w.stem}.mp3")
    for cid in ("item03_r0", "item05_r0", "item08_r0", "item10_r1", "v0_f_r0", "v0_m_r0", "v2_f_r0", "v3_m_r0"):
        w = paths.e6_out / f"{cid}.wav"
        if w.exists():
            n += _mp3(w, listen / f"e6__{cid}.mp3")
    blind = blind_pack(paths, out)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    return {"state": "ok", "scale": paths.scale, "listening_subset": n, "blind_pack": blind,
            "e5_audio": d.name if d else "withheld: no terminal E5 outcome yet"}


def earlier_minutes(prev: dict, log: Path) -> float:
    """Minutes this stage already ran in earlier runtimes. A stage cut off by a reset is still 'running' in
    status.json; its time ends at the last line its log received."""
    total = float(prev.get("minutes_total", prev.get("minutes", 0.0)) or 0.0)
    if prev.get("state") == "running" and prev.get("started_epoch"):
        end = log.stat().st_mtime if log.exists() else prev["started_epoch"]
        total = float(prev.get("minutes_total") or 0.0) + max(0.0, end - prev["started_epoch"]) / 60
    return round(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=["full", "tiny", "quick"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--stages", default=",".join(RUN_A))
    ap.add_argument("--state", default=os.environ.get("DACAPO_STATE", ""))
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    # inherited by every stage; cuBLAS reads the first when CUDA starts
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("ACESTEP_VAE_DECODE_CHUNK_SIZE", "256")
    names = [s for s in args.stages.split(",") if s]
    unknown = [s for s in names if s not in RUN_A + V1_STAGES]
    if unknown:
        sys.exit(f"unknown stages {unknown}")
    if args.scale == "quick" and set(names) & set(RUN_A):
        sys.exit("--scale quick applies to the v1 stages; Run A runs at full or tiny scale")
    if args.scale == "full" and args.device != "cuda" and set(names) & set(RUN_A):
        sys.exit("the full scale runs on CUDA; use --scale tiny for a CPU test")
    if args.state and not Path(args.state).exists():
        print(f"state folder {args.state} not found: run state stays on the local disk and is lost if the runtime resets",
              flush=True)
        args.state = ""
    paths = Paths(args.scale, args.state or None)
    print(f"scale {args.scale}, device {args.device}, state in {paths.root}", flush=True)
    status = json.loads(paths.status.read_text()) if paths.status.exists() else {}
    status["_meta"] = {"scale": args.scale, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "state": str(paths.root),
                       "durable": bool(args.state),
                       "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                             capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else None}
    v1 = v1_stages(args.scale)
    for name in [s for s in RUN_A + V1_STAGES if s in names]:
        prev = status.get(name, {})
        # setup and package always run: setup re-renders sources a reset removed (it skips what is in place)
        if prev.get("state") in ("ok", "terminal") and prev.get("scale") == args.scale and name not in ("setup", "package"):
            print(f"[{name}] done earlier ({prev['state']}), skipping", flush=True)
            continue
        print(f"\n===== stage {name} =====", flush=True)
        t0 = time.time()
        earlier = earlier_minutes(prev, paths.pkg / "logs" / f"{name}.log")
        status[name] = {"state": "running", "scale": args.scale, "started_epoch": t0, "minutes_total": earlier}
        paths.status.write_text(json.dumps(status, indent=1, default=str))
        if name in E5_STAGES:
            res = e5_stage(name, paths, args.device, args.workers, status)
        elif name == "e6":
            res = e6_stage(paths, args.device)
        elif name == "package":
            res = package(paths, status)
        else:
            res = run_v1_stage(name, v1[name], paths.pkg / "logs" / f"{name}.log")
        res.setdefault("scale", args.scale)
        res["minutes"] = round((time.time() - t0) / 60, 1)
        res["minutes_total"] = round(earlier + (time.time() - t0) / 60, 1)
        status[name] = res
        paths.status.write_text(json.dumps(status, indent=1, default=str))
        print(f"[{name}] {res['state']}", flush=True)
        if name == "smoke" and res["state"] != "ok":
            print("Smoke test failed: stopping before spending GPU time on the v1 stages.", flush=True)
            break
    paths.status.write_text(json.dumps(status, indent=1, default=str))
    print(json.dumps({k: v.get("state") if isinstance(v, dict) else v for k, v in status.items() if k != "_meta"}, indent=1))


if __name__ == "__main__":
    main()
