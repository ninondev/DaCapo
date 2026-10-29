"""Run B: B1, the lyric clock as addendum 2 and the B1 code addendum fix it, and B2, the clock with a melody
guide, as addendum 3 registers it. Every stage resumes from its ledgers.

Stages, in order (each gated automatically):
  setup    every registration hash, readout.py check-gate for B1, the soundfont, the 72 guides rendered again
           and compared with their sha256, the seed table, disk space, and the model facts (sdpa, alignment
           heads, determinism flags, ACE-Step commit, weights, ACE-Step's own lyric keys)
  a0       GPU checks on item00 with the runB2-a0 seeds: determinism, beta 0 bit-identity (else the registered
           fallback for B2), encoder states across the cover switch, 40 hook calls, one guide latent across k3-k5,
           the scorer, and the registered timing trim
  b1       experiments/e6_clock/run_b1.py gen, score, decide
  b2dev    Stage D: generation and scoring
  select   S* and B* from the Stage D rows; the selection's sha256 becomes the first line of the Stage T ledger
  b2test   Stage T: generation and scoring
  decide   the B2 rule (always runs; "Technical" when P0 fails)
  package  results, ledgers, logs and the demo by rule; the blind A/B pack and its key go next to the package

  python experiments/run_b.py --device cuda --state /content/drive/MyDrive/dacapo_runB   the registered run
  python experiments/run_b.py --scale tiny --device cpu                                   every stage, 1-2 runs each

--state DIR keeps the run state under DIR/<scale>/, e.g. on Google Drive, so a new runtime resumes where the last
one stopped; each scale has its own folders. DACAPO_PY_MAIN and DACAPO_PY_ACE name the interpreters for measuring
and for ACE-Step when they differ (on Colab both are this interpreter). The tiny scale scores with Whisper
small.en on the CPU, and every verdict it writes is "test only". The full scale refuses the CPU.

Stop rules: a setup or A0 failure before any B1 or B2 generation stops the run (package still collects the logs).
The same failure on a resumed run stops generation, but B1's decision, decide and package still run. A B1 problem
does not stop B2 unless the code B1 and B2 share fails its hash check. Stage T needs the selection line in its
ledger. decide always runs once B1 or B2 is under way, and writes "Technical" when P0 fails. A0 runs once: later
runtimes reuse its report while the report and the code still hash to what A0 recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
E6, E8 = EXP / "e6_clock", EXP / "e8_clock_sing"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EXP / "e5_motif"))
sys.path.insert(0, str(E8))
from stage import Echo  # noqa: E402

PY = sys.executable
STAGES = ["setup", "a0", "b1", "b2dev", "select", "b2test", "decide", "package"]
ALWAYS = ("setup", "a0", "decide", "package")   # a0 returns at once while its report is current
TINY_WHISPER = "openai/whisper-small.en"
SHARED = ("dacapo/attn_bias.py", "experiments/e6_clock/clock_score.py")
BANNER = "#" * 78


class Paths:
    """Every folder a scale writes to: DIR/<scale>/ with --state, else a local folder per scale. The guides are
    shared by the scales (setup renders them again and checks every sha256)."""

    def __init__(self, scale: str, state: str | None):
        sfx = "" if scale == "full" else f"_{scale}"
        if state:
            self.root = Path(state) / scale
        elif Path("/content").exists():
            self.root = Path("/content/work") / f"runB{sfx}"
        else:
            self.root = E8 / f"outputs_runb{sfx}"
        self.scale = scale
        self.b1_out, self.b1_res = self.root / "b1" / "outputs", self.root / "b1" / "results"
        self.b2, self.pkg, self.logs = self.root / "b2", self.root / "package", self.root / "logs"
        self.key, self.pack = self.root / "b2_blind_key.json", self.root / "b2_blind_pack.zip"
        self.status = self.root / "status.json"
        self.guides = E8 / "guides"
        self.ace = Path(tempfile.gettempdir()) / f"dacapo_acestep_runb_{scale}"
        for p in (self.root, self.b1_out, self.b1_res, self.b2, self.logs, self.ace):
            p.mkdir(parents=True, exist_ok=True)

    def env(self) -> dict:
        # DISABLE_SAFETENSORS_CONVERSION: the CTC model loads pytorch_model.bin at its pinned revision, and this
        # stops transformers from also fetching a converted copy from another revision in the background
        return {**os.environ, "ACESTEP_PROJECT_ROOT": str(self.ace), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "ACESTEP_VAE_DECODE_CHUNK_SIZE": "256", "DISABLE_SAFETENSORS_CONVERSION": "true",
                "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}

    def ledgers(self) -> dict[str, Path]:
        return {"b1": self.b1_out / "ledger.jsonl", "dev": self.b2 / "dev" / "ledger.jsonl",
                "test": self.b2 / "test" / "ledger.jsonl"}


class Chain:
    def __init__(self, paths: Paths, device: str):
        self.paths, self.device = paths, device
        self.py_main = os.environ.get("DACAPO_PY_MAIN", PY)
        self.py_ace = os.environ.get("DACAPO_PY_ACE", PY)
        self.tiny = paths.scale == "tiny"

    def call(self, stage: str, py: str, script: Path, args: list[str]) -> int:
        """Run one command, streaming every line to the stage log and the filtered lines to the console."""
        cmd = [py, str(script), *args]
        echo = Echo()
        log = self.paths.logs / f"{stage}.log"
        with log.open("a") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.flush()
            p = subprocess.Popen(cmd, cwd=script.parent, env=self.paths.env(), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line)
                if echo.show(line):
                    print(f"[{stage}] {line}", end="", flush=True)
            p.wait()
        return p.returncode

    def b2(self, stage: str, cmd: str, *extra: str, ace: bool = False) -> int:
        args = [cmd, "--scale", self.paths.scale, "--root", str(self.paths.b2), "--guides", str(self.paths.guides), *extra]
        return self.call(stage, self.py_ace if ace else self.py_main, E8 / "run_b2.py", args)

    def whisper(self) -> list[str]:
        return ["--test-whisper", TINY_WHISPER] if self.tiny else []

    def dev(self) -> list[str]:
        return ["--device", self.device]

    # stages

    def setup(self) -> dict:
        rc_check = self.b2("setup", "setup", "--part", "check", "--b1-out", str(self.paths.b1_out))
        rc_model = self.b2("setup", "setup", "--part", "model", *self.dev(), ace=True)
        rec = read(self.paths.b2 / "setup" / "setup_check.json") or {}
        model = read(self.paths.b2 / "setup" / "setup_model.json") or {}
        out = {"codes": {"check": rc_check, "model": rc_model}, "problems": rec.get("problems", []) + model.get("problems", []),
               "b1_gate": (rec.get("b1_gate") or {}).get("ok")}
        return {"state": "ok" if rc_check == 0 and rc_model == 0 else "technical_stop", **out}

    def a0(self) -> dict:
        keep = ("hard", "bit_identity", "fallback", "b1_note", "timing", "trim")
        rep, why = a0_current(self.paths)
        if rep is not None:
            return {"state": "ok", "reused": "A0 passed earlier on this code", **{k: rep.get(k) for k in keep}}
        if any(has_runs(self.paths.ledgers()[s]) for s in ("dev", "test")):
            # the B2 rows follow a report that is gone or no longer matches; a new A0 cannot replace it
            return {"state": "technical_stop", "failed": f"the A0 report the B2 rows follow is not current ({why})"}
        rc = self.b2("a0", "a0", "--part", "gen", *self.dev(), ace=True)
        if rc != 0:
            return {"state": "technical_stop", "failed": "a0 gen", "returncode": rc}
        rc = self.b2("a0", "a0", "--part", "score", *self.dev(), *self.whisper())
        if rc != 0:
            return {"state": "technical_stop", "failed": "a0 score", "returncode": rc}
        rc = self.b2("a0", "a0", "--part", "report")
        rep = read(self.paths.b2 / "a0" / "a0_report.json") or {}
        return {"state": "ok" if rc == 0 else "technical_stop", "returncode": rc, **{k: rep.get(k) for k in keep}}

    def b1(self, decide_only: bool = False) -> dict:
        """gen, score and decide; the stage is done only when all three exit 0, so a later runtime resumes a
        stage cut short. decide_only: generation has stopped, but B1's decision is written from its ledgers."""
        common = ["--scale", self.paths.scale, "--out", str(self.paths.b1_out), "--res", str(self.paths.b1_res)]
        codes = {}
        if not decide_only:
            codes["gen"] = self.call("b1", self.py_ace, E6 / "run_b1.py", ["gen", *self.dev(), *common])
            codes["score"] = self.call("b1", self.py_main, E6 / "run_b1.py", ["score", *self.dev(), *common, *self.whisper()])
        codes["decide"] = self.call("b1", self.py_main, E6 / "run_b1.py", ["decide", *common])
        res = (read(self.paths.b1_res / "b1_result.json") or {}).get("results", {})
        a0 = read(self.paths.b2 / "a0" / "a0_report.json") or {}
        state = "stopped" if decide_only else ("ok" if not any(codes.values()) else "failed")
        return {"state": state, "codes": codes, "verdict": res.get("verdict"), "reasons": res.get("reasons"),
                "labels": res.get("labels"), "note": a0.get("b1_note")}

    def b2_stage(self, stage: str, which: str) -> dict:
        rc = self.b2(stage, "gen", "--stage", which, *self.dev(), ace=True)
        if rc != 0:
            return {"state": "technical_stop", "failed": f"gen {which}", "returncode": rc}
        codes = []
        for _ in range(2):  # a second pass rescores clips whose scoring raised
            codes.append(self.b2(stage, "score", "--stage", which, *self.dev(), *self.whisper()))
            if codes[-1] != 0 or not score_errors(self.paths.b2, which):
                break
        if codes[-1] != 0:
            return {"state": "technical_stop", "failed": f"score {which}", "returncode": codes[-1]}
        return {"state": "ok", "score_passes": len(codes), **ledger_counts(self.paths.b2, which)}

    def select(self) -> dict:
        rc = self.b2("select", "select")
        sel = read(self.paths.b2 / "results" / "b2_selection.json") or {}
        return {"state": "ok" if rc == 0 else "technical_stop", "returncode": rc, "s_star": sel.get("s_star"),
                "b_star": sel.get("b_star"), "stage_t": sel.get("stage_t"), "gate_waived": sel.get("gate_waived")}

    def decide(self) -> dict:
        rc = self.b2("decide", "decide")
        s = read(self.paths.b2 / "results" / "b2_summary.json") or {}
        return {"state": "ok" if rc == 0 else "failed", "returncode": rc, "verdict": s.get("verdict"),
                "reasons": s.get("reasons"), "labels": s.get("labels"), "J": s.get("J")}

    def package(self, status: dict) -> dict:
        p = self.paths
        rc = self.b2("package", "package", "--package", str(p.pkg), "--key", str(p.key), "--pack", str(p.pack))
        b1 = p.pkg / "b1"
        if b1.exists():
            shutil.rmtree(b1)
        b1.mkdir(parents=True)
        for f in list(p.b1_res.glob("*.json")) + list(p.b1_res.glob("*.jsonl")) + list(p.b1_out.glob("*.json")) \
                + list(p.b1_out.glob("ledger.jsonl")):
            shutil.copy(f, b1 / f.name)
        logs = p.pkg / "logs"
        if logs.exists():
            shutil.rmtree(logs)
        shutil.copytree(p.logs, logs)
        write_json(p.pkg / "status.json", status)
        return {"state": "ok" if rc == 0 else "failed", "returncode": rc, "package": str(p.pkg), "key": str(p.key),
                "pack": str(p.pack)}


def read(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def write_json(p: Path, obj) -> None:
    """Written to a temporary file and renamed, so a reset mid-write never leaves a cut-off status."""
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def sha256(p: Path) -> str | None:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def has_runs(ledger: Path) -> bool:
    """The ledger holds at least one run row (the Stage T ledger's selection line is not one)."""
    return any("run_id" in r for r in jsonl(ledger))


def generation_started(paths: Paths) -> bool:
    return any(has_runs(p) for p in paths.ledgers().values())


def a0_current(paths: Paths) -> tuple[dict | None, str]:
    """(the A0 report, "") when it passed at this scale and it, its inputs and the code still hash to what A0
    recorded; else (None, why)."""
    import register_b2
    a0 = paths.b2 / "a0"
    rep, gen = read(a0 / "a0_report.json"), read(a0 / "a0_gen.json")
    if not rep or not rep.get("pass") or rep.get("scale") != paths.scale:
        return None, "no passing A0 report at this scale"
    if rep.get("a0_gen_sha256") != sha256(a0 / "a0_gen.json") or rep.get("a0_scores_sha256") != sha256(a0 / "a0_scores.json"):
        return None, "the A0 files differ from the report"
    if not gen or gen.get("code_sha256") != register_b2.hashes(register_b2.runtime_files()):
        return None, "A0 ran on other code"
    return rep, ""


def jsonl(p: Path) -> list[dict]:
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def score_errors(b2: Path, which: str) -> bool:
    last = {r["run_id"]: r for r in jsonl(b2 / f"scores_{which}.jsonl") if "run_id" in r}
    return any("error" in r for r in last.values())


def ledger_counts(b2: Path, which: str) -> dict:
    last = {r["run_id"]: r for r in jsonl(b2 / which / "ledger.jsonl") if "run_id" in r}
    scores = {r["run_id"]: r for r in jsonl(b2 / f"scores_{which}.jsonl") if "run_id" in r}
    return {"generated": sum(r.get("status") == "ok" for r in last.values()),
            "failed": sum(r.get("status") == "failed" for r in last.values()),
            "scored": sum("error" not in r for r in scores.values())}


def shared_code_problems() -> list[str]:
    import register_b2
    return [p for p in register_b2.registration_problems() if any(p.startswith(f) for f in SHARED)]


def earlier_minutes(prev: dict, log: Path) -> float:
    """Minutes the stage ran in earlier runtimes; a stage cut off by a reset ends at its log's last line."""
    total = float(prev.get("minutes_total", prev.get("minutes", 0.0)) or 0.0)
    if prev.get("state") == "running" and prev.get("started_epoch"):
        end = log.stat().st_mtime if log.exists() else prev["started_epoch"]
        total = float(prev.get("minutes_total") or 0.0) + max(0.0, end - prev["started_epoch"]) / 60
    return round(total, 1)


def run(chain, paths: Paths, names: list[str], status: dict, scale: str) -> dict:
    """The stages in order, with the stop rules; status is updated and written after every stage."""
    halt = None   # setup or A0 failed before any generation: the run stops, and only package collects the logs
    stop = None   # generation stops (a B2 stage failed, or setup or A0 failed once B1 or B2 had started):
    #               B1's decision, decide and package still run
    for name in STAGES:
        if name not in names:
            continue
        prev = status.get(name, {})
        done = prev.get("state") == "ok" and prev.get("scale") == scale
        if (halt and name != "package") or (stop and name not in ("decide", "package")):
            why = f"{halt} failed: Run B stopped" if halt else f"{stop} did not pass"
            if done:  # a stage finished earlier keeps its record
                print(f"[{name}] done earlier; not run again ({why})", flush=True)
            elif stop and name == "b1" and has_runs(paths.ledgers()["b1"]):
                print(f"[b1] generation stopped ({why}); B1's decision is written from its ledgers", flush=True)
                status[name] = {**chain.b1(decide_only=True), "scale": scale, "reason": why}
                write_json(paths.status, status)
            else:
                status[name] = {"state": "skipped", "reason": why}
                print(f"[{name}] skipped: {why}", flush=True)
            continue
        if name == "b2test" and status.get("select", {}).get("stage_t") is False:
            status[name] = {"state": "skipped", "reason": "no clocked config passed the gate on dev: Stage T is not run"}
            print(f"[{name}] skipped: Stage T is not run", flush=True)
            continue
        if done and name not in ALWAYS:
            print(f"[{name}] done earlier, skipping", flush=True)
            continue
        print(f"\n===== stage {name} =====", flush=True)
        t0 = time.time()
        earlier = earlier_minutes(prev, paths.logs / f"{name}.log")
        status[name] = {"state": "running", "scale": scale, "started_epoch": t0, "minutes_total": earlier}
        write_json(paths.status, status)
        if name == "setup":
            res = chain.setup()
        elif name == "a0":
            res = chain.a0()
        elif name == "b1":
            res = chain.b1()
        elif name == "b2dev":
            res = chain.b2_stage(name, "dev")
        elif name == "select":
            res = chain.select()
        elif name == "b2test":
            res = chain.b2_stage(name, "test")
        elif name == "decide":
            res = chain.decide()
        else:
            res = chain.package(status)
        res.update(scale=scale, minutes=round((time.time() - t0) / 60, 1),
                   minutes_total=round(earlier + (time.time() - t0) / 60, 1))
        status[name] = res
        write_json(paths.status, status)
        print(f"[{name}] {res['state']}", flush=True)
        if res["state"] == "technical_stop" and name in ("setup", "a0"):
            if generation_started(paths):
                print(f"{name} failed: generation stops here; the decisions and the package still run "
                      f"(see {paths.logs / (name + '.log')}).", flush=True)
                stop = name
            else:
                print(f"{name} failed: Run B stops here (see {paths.logs / (name + '.log')}).", flush=True)
                halt = name
        if name == "b1" and res["state"] != "ok":
            bad = shared_code_problems()
            if bad:
                stop = "b1 (the shared code fails its hash check)"
                status[name]["shared_code"] = bad
        if name in ("b2dev", "select", "b2test") and res["state"] != "ok":
            stop = name
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=["full", "tiny"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--state", default=os.environ.get("DACAPO_STATE", ""))
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("ACESTEP_VAE_DECODE_CHUNK_SIZE", "256")
    names = [s for s in args.stages.split(",") if s]
    unknown = [s for s in names if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stages {unknown}")
    if args.scale == "full" and args.device != "cuda":
        sys.exit("the registered full scale runs on CUDA; use --scale tiny for a CPU test")
    if args.state and not Path(args.state).exists():
        print(f"{BANNER}\nSTATE FOLDER {args.state} NOT FOUND.\nTHE RUN STATE STAYS ON THE LOCAL DISK AND IS LOST IF THE "
              f"RUNTIME RESETS.\n{BANNER}", flush=True)
        args.state = ""
    if not args.state and args.scale == "full":
        print(f"{BANNER}\nNO --state FOLDER: THE RUN STATE IS NOT ON DRIVE AND CANNOT RESUME AFTER A RESET.\n{BANNER}",
              flush=True)
    paths = Paths(args.scale, args.state or None)
    chain = Chain(paths, args.device)
    print(f"Run B, scale {args.scale}, device {args.device}, state in {paths.root}", flush=True)
    status = read(paths.status) or {}
    status["_meta"] = {"scale": args.scale, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "state": str(paths.root),
                       "durable": bool(args.state),
                       "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                             text=True).stdout.strip() if shutil.which("nvidia-smi") else None}
    status = run(chain, paths, names, status, args.scale)
    write_json(paths.status, status)
    print(json.dumps({k: v.get("state") if isinstance(v, dict) else v for k, v in status.items() if k != "_meta"}, indent=1))
    if "decide" in status:
        print("B2 verdict:", status["decide"].get("verdict"), flush=True)
    if "b1" in status:
        print("B1 verdict:", status["b1"].get("verdict"), flush=True)


if __name__ == "__main__":
    main()
