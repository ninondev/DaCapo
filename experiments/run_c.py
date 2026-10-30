"""Run C: C1, a registered replication of B2's two fixed recipes on 96 new melodies (addendum 4). Every stage
resumes from its ledgers.

Stages, in order (each gated automatically):
  setup    every registration hash (addendum 2, the B1 code addendum, addendum 3, addendum 4), the soundfont, the
           98 guides rendered again and compared with their sha256, the seed table, disk space, and the model facts
           (sdpa, alignment heads, determinism flags, ACE-Step commit, weights, ACE-Step's own lyric keys)
  a0       GPU checks on item00 with the runC-a0 seeds: S*_b0, the hook at beta 0 (bit-identical to no hook), S*
           (40 hook calls, a changed output), one guide latent, and the scorer on S* and S*_b0
  c1test   the 1,536 runs, generated and then scored in blocks of 24 items, in plan order
  decide   the C1 rule (always runs once C1 generation has started; "Technical" when P0 fails)
  package  results, ledgers, logs and the demo by rule; the blind A/B pack and its key go next to the package

  python experiments/run_c.py --device cuda --state /content/drive/MyDrive/dacapo_runC   the registered run
  python experiments/run_c.py --scale tiny --device cpu                                  every stage, 1-2 runs each

--state DIR keeps the run state under DIR/<scale>/, e.g. on Google Drive, so a new runtime resumes where the last
one stopped; each scale has its own folders. DACAPO_PY_MAIN and DACAPO_PY_ACE name the interpreters for measuring
and for ACE-Step when they differ (on Colab both are this interpreter). The tiny scale runs on the dev item item01
with its own seed tags, scores with Whisper small.en on the CPU, and every verdict it writes is "test only". The
full scale refuses the CPU.

Stop rules: a setup or A0 failure before any C1 generation stops the run (package still collects the logs). The
same failure on a resumed run stops generation, but decide and package still run. A failed c1test block stops
generation; decide and package still run, and the run says how far C1 got and that Run all resumes it. A0 runs
once: later runtimes reuse its report while the report and the code still hash to what A0 recorded and the session
runs on A0's runtime (before any C1 generation, another runtime runs A0 again).

Within a block, gen runs in fresh processes until every run is generated or has used its 3 attempts: a process that
ends with failed runs that have attempts left (exit 5), that stopped on failures in a row across two items (exit 4)
or that died is followed by a fresh one, at most 8 per block in a session, and two exit-4 stops end the block.

A heartbeat file in the state folder (run_c.heartbeat.json, written every 30 s) keeps a second run_c.py from
working on the same state; one older than 3 minutes is stale. The processes run_c.py starts end when it ends.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
E9 = EXP / "e9_clock_replicate"
for p in (REPO, EXP / "e5_motif", E9):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from stage import Echo  # noqa: E402

PY = sys.executable
STAGES = ["setup", "a0", "c1test", "decide", "package"]
ALWAYS = ("setup", "a0", "decide", "package")   # a0 returns at once while its report is current
TINY_WHISPER = "openai/whisper-small.en"
BANNER = "#" * 78
EXIT_SETUP, EXIT_STICKY, EXIT_AGAIN = 2, 4, 5     # run_c1.py gen
HEARTBEAT_S, STALE_S = 30, 180
ALLOW_RUNTIME_CHANGE = "DACAPO_ALLOW_RUNTIME_CHANGE"


class Paths:
    """Every folder a scale writes to: DIR/<scale>/ with --state, else a local folder per scale. The guides are
    shared by the scales (setup renders them again and checks every sha256)."""

    def __init__(self, scale: str, state: str | None):
        sfx = "" if scale == "full" else f"_{scale}"
        if state:
            self.root = Path(state) / scale
        elif Path("/content").exists():
            self.root = Path("/content/work") / f"runC{sfx}"
        else:
            self.root = E9 / f"outputs_runc{sfx}"
        self.scale = scale
        self.c1, self.pkg, self.logs = self.root / "c1", self.root / "package", self.root / "logs"
        self.key, self.pack = self.root / "c1_blind_key.json", self.root / "c1_blind_pack.zip"
        self.status = self.root / "status.json"
        self.guides = E9 / "guides"
        self.ace = Path(tempfile.gettempdir()) / f"dacapo_acestep_runc_{scale}"
        for p in (self.root, self.c1, self.logs, self.ace):
            p.mkdir(parents=True, exist_ok=True)

    def env(self) -> dict:
        # DISABLE_SAFETENSORS_CONVERSION: the CTC model loads pytorch_model.bin at its pinned revision, and this
        # stops transformers from also fetching a converted copy from another revision in the background
        return {**os.environ, "ACESTEP_PROJECT_ROOT": str(self.ace), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "ACESTEP_VAE_DECODE_CHUNK_SIZE": "256", "DISABLE_SAFETENSORS_CONVERSION": "true",
                "DACAPO_DIE_WITH_PARENT": "1",
                "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}

    def ledger(self) -> Path:
        return self.c1 / "test" / "ledger.jsonl"

    def scores(self) -> Path:
        return self.c1 / "scores_test.jsonl"


class Chain:
    def __init__(self, paths: Paths, device: str):
        self.paths, self.device = paths, device
        self.py_main = os.environ.get("DACAPO_PY_MAIN", PY)
        self.py_ace = os.environ.get("DACAPO_PY_ACE", PY)
        self.tiny = paths.scale == "tiny"
        self.started = time.time()

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

    def c1(self, stage: str, cmd: str, *extra: str, ace: bool = False) -> int:
        args = [cmd, "--scale", self.paths.scale, "--root", str(self.paths.c1), "--guides", str(self.paths.guides), *extra]
        return self.call(stage, self.py_ace if ace else self.py_main, E9 / "run_c1.py", args)

    def whisper(self) -> list[str]:
        return ["--test-whisper", TINY_WHISPER] if self.tiny else []

    def dev(self) -> list[str]:
        return ["--device", self.device]

    # stages

    def setup(self) -> dict:
        rc_check = self.c1("setup", "setup", "--part", "check")
        rc_model = self.c1("setup", "setup", "--part", "model", *self.dev(), ace=True)
        rec = read(self.paths.c1 / "setup" / "setup_check.json") or {}
        model = read(self.paths.c1 / "setup" / "setup_model.json") or {}
        out = {"codes": {"check": rc_check, "model": rc_model}, "problems": rec.get("problems", []) + model.get("problems", [])}
        return {"state": "ok" if rc_check == 0 and rc_model == 0 else "technical_stop", **out}

    def a0(self) -> dict:
        keep = ("hard", "bit_identity", "projection")
        rep, why = a0_current(self.paths, self.started)
        if rep is not None:
            return {"state": "ok", "reused": "A0 passed earlier on this code and runtime", **{k: rep.get(k) for k in keep}}
        if has_runs(self.paths.ledger()):
            # the C1 rows follow a report that is gone or no longer matches; a new A0 cannot replace it
            return {"state": "technical_stop", "failed": f"the A0 report the C1 rows follow is not current ({why})"}
        rc = self.c1("a0", "a0", "--part", "gen", *self.dev(), ace=True)
        if rc != 0:
            return {"state": "technical_stop", "failed": "a0 gen", "returncode": rc}
        rc = self.c1("a0", "a0", "--part", "score", *self.dev(), *self.whisper())
        if rc != 0:
            return {"state": "technical_stop", "failed": "a0 score", "returncode": rc}
        rc = self.c1("a0", "a0", "--part", "report")
        rep = read(self.paths.c1 / "a0" / "a0_report.json") or {}
        return {"state": "ok" if rc == 0 else "technical_stop", "returncode": rc, **{k: rep.get(k) for k in keep}}

    def c1test(self) -> dict:
        """Generation and then scoring, block by block; a block's scoring runs a second pass for clips whose scoring
        raised. The stage is done when every block is."""
        import plan_c1
        done = []
        for block in plan_c1.blocks(self.paths.scale, plan_c1.BLOCK_ITEMS):
            ids = ",".join(block)
            g = self.gen_block(ids)
            if not g["ok"]:
                return {"state": "technical_stop", "failed": f"gen {block[0]}-{block[-1]}: {g['why']}",
                        "returncode": g["codes"][-1], "gen_codes": g["codes"], "blocks_done": done}
            codes = []
            for _ in range(2):
                codes.append(self.c1("c1test", "score", "--items", ids, *self.dev(), *self.whisper()))
                if codes[-1] != 0 or not score_errors(self.paths, block):
                    break
            if codes[-1] != 0:
                return {"state": "technical_stop", "failed": f"score {block[0]}-{block[-1]}", "returncode": codes[-1],
                        "blocks_done": done}
            done.append(f"{block[0]}-{block[-1]}")
            print(f"[c1test] block {block[0]}-{block[-1]} done ({len(done)} blocks)", flush=True)
        return {"state": "ok", "blocks_done": done, **ledger_counts(self.paths)}

    def gen_block(self, ids: str) -> dict:
        """gen in fresh processes until the block is done (exit 0): again after exit 5 (failed runs with attempts
        left), exit 4 (failures in a row across two items) or a process that died; at most GEN_PASSES processes, and
        the block stops at the STICKY_TRIPS-th exit 4 or on any other exit code."""
        import plan_c1
        codes, trips = [], 0
        for _ in range(plan_c1.GEN_PASSES):
            rc = self.c1("c1test", "gen", "--items", ids, *self.dev(), ace=True)
            codes.append(rc)
            if rc == 0:
                return {"ok": True, "codes": codes}
            if rc == EXIT_STICKY:
                trips += 1
                if trips >= plan_c1.STICKY_TRIPS:
                    return {"ok": False, "codes": codes, "why": f"runs failed in a row in {trips} fresh processes (an "
                                                                "error that sticks to the GPU or the runtime)"}
            elif rc != EXIT_AGAIN and rc >= 0:
                return {"ok": False, "codes": codes, "why": f"gen exited with {rc}"}
            print(f"[c1test] gen exited with {rc}: a fresh gen process goes on with the block", flush=True)
        return {"ok": False, "codes": codes, "why": f"the block was not done after {plan_c1.GEN_PASSES} gen processes"}

    def decide(self) -> dict:
        rc = self.c1("decide", "decide")
        s = read(self.paths.c1 / "results" / "c1_summary.json") or {}
        return {"state": "ok" if rc == 0 else "failed", "returncode": rc, "verdict": s.get("verdict"),
                "reasons": s.get("reasons"), "labels": s.get("labels"), "J": s.get("J")}

    def package(self, status: dict) -> dict:
        p = self.paths
        rc = self.c1("package", "package", "--package", str(p.pkg), "--key", str(p.key), "--pack", str(p.pack))
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
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def sha256(p: Path) -> str | None:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def jsonl(p: Path) -> list[dict]:
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def has_runs(ledger: Path) -> bool:
    return any("run_id" in r for r in jsonl(ledger))


def runtime_diff(paths: Paths, since: float | None) -> dict:
    """{field: [A0's, this session's]} where the runtime setup recorded in this session (setup_model.json written
    after since) differs from A0's (a0_gen.json); empty when they match or either is unknown (gen checks again)."""
    import plan_c1
    now, gen = read(paths.c1 / "setup" / "setup_model.json") or {}, read(paths.c1 / "a0" / "a0_gen.json") or {}
    a, b = gen.get("runtime"), now.get("runtime_fingerprint")
    if not a or not b or (since is not None and float(now.get("epoch") or 0) < since):
        return {}
    return {k: [a.get(k), b.get(k)] for k in plan_c1.RUNTIME_KEYS if a.get(k) != b.get(k)}


def a0_current(paths: Paths, since: float | None = None) -> tuple[dict | None, str]:
    """(the A0 report, "") when it passed at this scale, it, its inputs and the code still hash to what A0 recorded,
    and this session runs on A0's runtime (or, once C1 rows exist, DACAPO_ALLOW_RUNTIME_CHANGE=1 lets it go on);
    else (None, why)."""
    import register_c1
    a0 = paths.c1 / "a0"
    rep, gen = read(a0 / "a0_report.json"), read(a0 / "a0_gen.json")
    if not rep or not rep.get("pass") or rep.get("scale") != paths.scale:
        return None, "no passing A0 report at this scale"
    if rep.get("a0_gen_sha256") != sha256(a0 / "a0_gen.json") or rep.get("a0_scores_sha256") != sha256(a0 / "a0_scores.json"):
        return None, "the A0 files differ from the report"
    if not gen or gen.get("code_sha256") != register_c1.hashes(register_c1.runtime_files()):
        return None, "A0 ran on other code"
    diff = runtime_diff(paths, since)
    if diff and not (os.environ.get(ALLOW_RUNTIME_CHANGE) == "1" and has_runs(paths.ledger())):
        return None, (f"A0 ran on another runtime: {json.dumps(diff)}. Resume on the GPU and runtime version A0 ran on "
                      f"(an L4), or set ALLOW_RUNTIME_CHANGE = True in the notebook ({ALLOW_RUNTIME_CHANGE}=1) to go on "
                      "with the change recorded in every row")
    return rep, ""


def score_errors(paths: Paths, block: list[str]) -> bool:
    keep = set(block)
    last = {r["run_id"]: r for r in jsonl(paths.scores()) if "run_id" in r}
    return any("error" in r for rid, r in last.items() if rid.split("_r", 1)[0] in keep)


def ledger_counts(paths: Paths) -> dict:
    last = {r["run_id"]: r for r in jsonl(paths.ledger()) if "run_id" in r}
    scores = {r["run_id"]: r for r in jsonl(paths.scores()) if "run_id" in r}
    return {"generated": sum(r.get("status") == "ok" for r in last.values()),
            "failed": sum(r.get("status") == "failed" for r in last.values()),
            "scored": sum("error" not in r for r in scores.values())}


def resume_note(paths: Paths, status: dict) -> str | None:
    """A plain line when C1 generation is not finished: how far it got, and that Run all resumes it."""
    c = status.get("c1test") or {}
    if c.get("state") == "ok" or "c1test" not in status:
        return None
    import plan_c1
    n = ledger_counts(paths)
    planned = len(plan_c1.test_runs(paths.scale))
    why = c.get("failed") or c.get("reason") or c.get("state")
    if c.get("state") == "skipped" and not has_runs(paths.ledger()):
        return f"Run C stopped before any C1 generation ({why}); nothing was generated. The stage logs are in {paths.logs}."
    return (f"C1 generation incomplete ({why}): {n['generated']} of {planned:,} runs generated, {n['scored']} scored. "
            "A verdict written now covers only what exists. Choose Run all again to resume.")


class Heartbeat:
    """run_c.heartbeat.json in the state folder: host, pid and time, written every HEARTBEAT_S seconds. Another
    run_c.py refuses to start while a heartbeat from another process is younger than STALE_S seconds."""

    def __init__(self, root: Path):
        self.path = Path(root) / "run_c.heartbeat.json"
        self.me = {"host": socket.gethostname(), "pid": os.getpid()}
        self.stop_event = threading.Event()
        self.thread = None

    def other(self) -> dict | None:
        hb = read(self.path)
        if not hb or (hb.get("host"), hb.get("pid")) == (self.me["host"], self.me["pid"]):
            return None
        age = time.time() - float(hb.get("epoch") or 0)
        return {**hb, "age_s": round(age)} if age < STALE_S else None

    def beat(self) -> None:
        try:
            write_json(self.path, {**self.me, "epoch": time.time(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        except OSError:
            pass

    def start(self) -> None:
        self.beat()

        def loop():
            while not self.stop_event.wait(HEARTBEAT_S):
                self.beat()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        hb = read(self.path)
        if hb and (hb.get("host"), hb.get("pid")) == (self.me["host"], self.me["pid"]):
            try:
                self.path.unlink()
            except OSError:
                pass


def earlier_minutes(prev: dict, log: Path) -> float:
    """Minutes the stage ran in earlier runtimes; a stage cut off by a reset ends at its log's last line."""
    total = float(prev.get("minutes_total", prev.get("minutes", 0.0)) or 0.0)
    if prev.get("state") == "running" and prev.get("started_epoch"):
        end = log.stat().st_mtime if log.exists() else prev["started_epoch"]
        total = float(prev.get("minutes_total") or 0.0) + max(0.0, end - prev["started_epoch"]) / 60
    return round(total, 1)


def run(chain, paths: Paths, names: list[str], status: dict, scale: str) -> dict:
    """The stages in order, with the stop rules; status is updated and written after every stage."""
    halt = None   # setup or A0 failed before any C1 generation: only package runs
    stop = None   # generation stops (c1test failed, or setup or A0 failed once C1 generation had started)
    for name in STAGES:
        if name not in names:
            continue
        prev = status.get(name, {})
        done = prev.get("state") == "ok" and prev.get("scale") == scale
        if (halt and name != "package") or (stop and name not in ("decide", "package")):
            why = f"{halt} failed: Run C stopped" if halt else f"{stop} did not pass"
            if done:
                print(f"[{name}] done earlier; not run again ({why})", flush=True)
            else:
                status[name] = {"state": "skipped", "reason": why}
                print(f"[{name}] skipped: {why}", flush=True)
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
        elif name == "c1test":
            res = chain.c1test()
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
            if has_runs(paths.ledger()):
                print(f"{name} failed: generation stops here; decide and package still run "
                      f"(see {paths.logs / (name + '.log')}).", flush=True)
                stop = name
            else:
                print(f"{name} failed: Run C stops here (see {paths.logs / (name + '.log')}).", flush=True)
                halt = name
        if name == "c1test" and res["state"] != "ok":
            stop = name
    return status


def die_with_parent() -> None:
    """With DACAPO_DIE_WITH_PARENT=1 (the notebook sets it), this process ends when the process that started it ends,
    as run_c1.py's do when this one ends, so a restarted notebook kernel leaves nothing writing to the run state
    (Linux; elsewhere a no-op)."""
    if os.environ.get("DACAPO_DIE_WITH_PARENT") != "1" or not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
        if os.getppid() == 1:
            sys.exit(3)
    except (OSError, AttributeError):
        pass


def main():
    die_with_parent()
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
    hb = Heartbeat(paths.root)
    other = hb.other()
    if other and os.environ.get("DACAPO_IGNORE_HEARTBEAT") != "1":
        print(f"{BANNER}\nANOTHER run_c.py (pid {other.get('pid')} on {other.get('host')}) WROTE ITS HEARTBEAT TO "
              f"{paths.root} {other['age_s']} s AGO.\nTWO RUNS MUST NOT WRITE TO ONE STATE FOLDER. Stop the other one, or "
              f"wait {STALE_S // 60} minutes after it ended and choose Run all again.\n{BANNER}", flush=True)
        sys.exit(3)

    def on_term(signum, frame):   # a stopped cell: end here (finally removes the heartbeat; the children end with us)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_term)
    hb.start()
    try:
        go(args, paths, names)
    finally:
        hb.stop()


def go(args, paths: Paths, names: list[str]) -> None:
    chain = Chain(paths, args.device)
    print(f"Run C, scale {args.scale}, device {args.device}, state in {paths.root}", flush=True)
    status = read(paths.status) or {}
    status["_meta"] = {"scale": args.scale, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "state": str(paths.root),
                       "durable": bool(args.state),
                       "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                             text=True).stdout.strip() if shutil.which("nvidia-smi") else None}
    status = run(chain, paths, names, status, args.scale)
    note = resume_note(paths, status)
    status["_meta"]["resume"] = note
    write_json(paths.status, status)
    print(json.dumps({k: v.get("state") if isinstance(v, dict) else v for k, v in status.items() if k != "_meta"}, indent=1))
    if "decide" in status:
        print("C1 verdict:", status["decide"].get("verdict"), flush=True)
    if note:
        print(f"{BANNER}\n{note}\n{BANNER}", flush=True)


if __name__ == "__main__":
    main()
