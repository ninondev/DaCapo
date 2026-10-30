"""C1 of Run C, a registered replication of B2's two fixed recipes on 96 new melodies (addendum 4).

  python run_c1.py setup --part check                 main env: hashes, soundfont, guides, seeds, space
  python run_c1.py setup --part model --device cuda   ACE env: model facts and ACE-Step's own lyric keys
  python run_c1.py a0 --part gen --device cuda        ACE env: the A0 generations and their checks
  python run_c1.py a0 --part score --device cuda      main env: the registered scorer on the A0 clips
  python run_c1.py a0 --part report                   main env: pass or stop, and the time projection
  python run_c1.py gen [--items IDS] --device cuda    ACE env: WAV, latents and a ledger row per run
  python run_c1.py score [--items IDS] --device cuda  main env: one score row per generated clip
  python run_c1.py decide | package                   main env
  --root DIR      run state (default outputs_c1[_tiny] next to this file); --guides DIR (default guides/)
  --items IDS     gen and score: only these test items (a comma list), so run_c.py can work block by block
  --scale tiny    1-2 runs per stage on the dev item item01, any device, a test Whisper allowed; every verdict is
                  "test only"

B2's generation path is reused unchanged (experiments/e8_clock_sing/run_b2.py generate_run: ACE-Step inside
dacapo.attn_bias.CrossAttnBias when the run is hooked), and so are its ledger and resume helpers. A run is done
when its WAV and latents hash to its ledger row; a clip is scored when its score row names the WAV's sha256; a
failed run gets at most 3 attempts with the same seeds, each in a fresh process (one attempt per run per gen
process). Every attempt first appends a "started" row, so an attempt whose process died (a reset, a crash, a kill)
is seen by the next process: the first such attempt of a run is free, every later one counts as failed. gen hashes
the files of its own block only; decide checks every run.

Exit codes: 0 done (gen: every run of the block is generated or has used its attempts), 2 a setup problem (nothing
generated or scored), 3 an A0 hard failure, 4 gen stopped after failures in a row that span two items (an error
that sticks to the process or the GPU; a fresh process goes on), 5 gen finished a pass with failed runs that have
attempts left (a fresh process tries them again).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E8 = ROOT / "experiments" / "e8_clock_sing"
for p in (ROOT, HERE, E8, ROOT / "experiments" / "e6_clock"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import plan_b2 as P2  # noqa: E402
import plan_c1 as P  # noqa: E402
import readout as R  # noqa: E402
import register_c1 as REG  # noqa: E402
import run_b2 as RB  # noqa: E402
from run_b1 import append_jsonl, read_jsonl, sha256  # noqa: E402

BYTES_PER_RUN = RB.BYTES_PER_RUN
SPACE_MARGIN = RB.SPACE_MARGIN
read, write, utc = RB.read, RB.write, RB.utc
generated, failed_attempts, files_match = RB.generated, RB.failed_attempts, RB.files_match
EXIT_SETUP, EXIT_A0, EXIT_STICKY, EXIT_AGAIN = 2, 3, 4, 5
ALLOW_RUNTIME_CHANGE = "DACAPO_ALLOW_RUNTIME_CHANGE"   # "1": generate on a runtime other than A0's, recorded per row
RUNTIME_KEYS = P.RUNTIME_KEYS
REGISTER_SCRIPTS = (("addendum_2", "experiments/e5_motif/register.py"),
                    ("b1_code_addendum", "experiments/e6_clock/register_b1code.py"),
                    ("addendum_3", "experiments/e8_clock_sing/register_b2.py"),
                    ("addendum_4", "experiments/e9_clock_replicate/register_c1.py"))


class SetupError(RuntimeError):
    """A problem that affects every run (exit 2 before any generation or scoring)."""


class Paths:
    def __init__(self, root: str, guides: str):
        self.root, self.guides = Path(root), Path(guides)
        self.setup, self.a0, self.test = (self.root / n for n in ("setup", "a0", "test"))
        self.res = self.root / "results"
        self.a0_report = self.a0 / "a0_report.json"

    def ledger(self) -> Path:
        return self.test / "ledger.jsonl"

    def scores(self) -> Path:
        return self.root / "scores_test.jsonl"


# flags and registration

def test_flags(args) -> list[str]:
    flags = [f"scale {args.scale}"] if args.scale != "full" else []
    if getattr(args, "test_whisper", None):
        flags.append(f"--test-whisper {args.test_whisper}")
    return flags


def refuse_full_scale_tests(args, needs_cuda: bool) -> None:
    if args.scale == "full" and (getattr(args, "test_whisper", None) or (needs_cuda and args.device != "cuda")):
        raise SetupError("the registered full scale runs on CUDA without test flags")


def registration(args) -> dict:
    """The registration now. The full scale refuses unregistered files; the tiny scale records them."""
    problems = REG.registration_problems()
    if problems and args.scale == "full":
        raise SetupError("; ".join(problems))
    return {"problems": problems, "registered": not problems, "code_sha256": REG.hashes(REG.runtime_files())}


def a0_state(paths: Paths, need: bool = True) -> dict:
    rep = read(paths.a0_report)
    if rep is None or not rep.get("pass"):
        if need:
            raise SetupError("no passing A0 report: run the a0 stage first")
        return {"sha256": None, "report": rep}
    return {"sha256": sha256(paths.a0_report), "report": rep}


def plan_items(args) -> list[str] | None:
    return [x for x in args.items.split(",") if x] if getattr(args, "items", "") else None


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def keep_session_copy(paths: Paths, name: str, rec: dict) -> None:
    """setup/<name>.json is the latest session's; setup/history keeps every session's, so a resumed run shows where
    and on what each session ran."""
    write(paths.setup / f"{name}.json", rec)
    write(paths.setup / "history" / f"{name}_{stamp()}_{os.getpid()}.json", rec)


def register_checks() -> dict:
    """The four registration checks (addendum 2, the B1 code addendum, addendum 3, addendum 4), each run as its own
    --check: at setup and again at decision."""
    out = {}
    for name, script in REGISTER_SCRIPTS:
        try:
            p = subprocess.run([sys.executable, str(ROOT / script), "--check"], capture_output=True, text=True, cwd=ROOT)
            out[name] = {"returncode": p.returncode, "output": (p.stdout + p.stderr).strip()[-2000:]}
        except Exception as exc:
            out[name] = {"returncode": None, "output": f"did not run: {exc!r}"}
    return out


def failed_checks(checks: dict | None, when: str) -> list[str]:
    if not checks:
        return [f"{when}: no registration checks recorded"]
    return [f"{when}: {k} --check failed" for k, v in checks.items() if v.get("returncode") != 0]


def package_problems(want: dict[str, str]) -> list[str]:
    """Installed versions that differ from the notebook's pins (register_c1.pins())."""
    from importlib import metadata
    out = []
    for name, ver in sorted(want.items()):
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            got = None
        if got != ver:
            out.append(f"{name} {got}, pinned {ver}")
    return out


def runtime_fingerprint(facts: dict | None) -> dict | None:
    return {k: facts.get(k) for k in RUNTIME_KEYS} if facts else None


def runtime_change(a0: dict | None, now: dict | None) -> dict:
    """{field: [A0's value, this process's value]} for every runtime field that differs; empty when they match."""
    if not a0 or not now:
        return {"runtime": [None if not a0 else "recorded", None if not now else "recorded"]}
    return {k: [a0.get(k), now.get(k)] for k in RUNTIME_KEYS if a0.get(k) != now.get(k)}


def runtime_change_allowed() -> bool:
    return os.environ.get(ALLOW_RUNTIME_CHANGE) == "1"


def die_with_parent() -> None:
    """When run_c.py starts this process (DACAPO_DIE_WITH_PARENT=1), it ends with run_c.py, so a stopped notebook
    cell or a killed orchestrator leaves no generation or scoring writing to the ledgers (Linux; elsewhere a no-op)."""
    if os.environ.get("DACAPO_DIE_WITH_PARENT") != "1" or not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
        if os.getppid() == 1:   # the parent ended before the call
            sys.exit(EXIT_SETUP)
    except (OSError, AttributeError):
        pass


def generated_of(out: Path, run_ids) -> dict[str, dict]:
    """run_b2.generated for these runs only (the last ledger row of each; an ok row counts only while its files hash
    to it), so gen and score read one block's files rather than the whole run's."""
    keep = set(run_ids)
    last: dict[str, dict] = {}
    for row in read_jsonl(out / "ledger.jsonl"):
        if row.get("run_id") in keep:
            last[row["run_id"]] = row
    return {rid: {**row, "status": "files_missing"} if row.get("status") == "ok" and not files_match(out, row) else row
            for rid, row in last.items()}


def attempts_used(out: Path) -> dict[str, int]:
    """Attempts each run has used: its failed rows, plus every attempt after the first that started and never
    finished (its process died). One unfinished attempt is free, so a runtime reset costs a run nothing; a run that
    kills its process again and again uses up its attempts like a run that raises."""
    used: dict[str, int] = {}
    open_: dict[str, bool] = {}
    died: dict[str, int] = {}
    for r in read_jsonl(out / "ledger.jsonl"):
        rid = r.get("run_id")
        if rid is None:
            continue
        if r.get("status") == "started":
            if open_.get(rid):
                died[rid] = died.get(rid, 0) + 1
            open_[rid] = True
            continue
        open_[rid] = False
        if r.get("status") == "failed":
            used[rid] = used.get(rid, 0) + 1
    for rid, o in open_.items():
        if o:
            died[rid] = died.get(rid, 0) + 1
    for rid, n in died.items():
        if n > 1:
            used[rid] = used.get(rid, 0) + n - 1
    return used


def free_cuda(device: str) -> None:
    if device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:   # a broken CUDA context must not hide the run's own error
            pass


# generation (ACE env)

def load_model(args, items: list[dict]):
    """setup_determinism, the model, the alignment heads, and ACE-Step's own lyric keys for these items, which must
    equal tok_check_c1.json (run_b2.load_model, with Run C's tokenizer record)."""
    from dacapo import ace
    from dacapo.attn_bias import lyric_keys
    from acestep.core.scoring.dit_alignment import MusicStampsAligner

    det = ace.setup_determinism()
    h = ace.load(args.device)
    if {int(k): list(v) for k, v in h.custom_layers_config.items()} != R.ALIGN:
        raise SetupError(f"unexpected alignment heads {h.custom_layers_config}")
    if getattr(h.model.config, "_attn_implementation", None) != "sdpa":
        raise SetupError(f"attention implementation {h.model.config._attn_implementation}, registered sdpa")
    aligner = MusicStampsAligner(h.text_tokenizer)
    tok = P.read_json(P.TOK_JSON)["items"]
    lays = {}
    for it in items:
        lay = lyric_keys(h, aligner, it["lyrics"])
        want = tok[it["id"]]
        if (lay["keys"], lay["n_lyr"], lay["start"]) != (want["keys"], want["n_lyr"], want["start"]):
            raise SetupError(f"{it['id']}: ACE-Step's lyric keys differ from tok_check_c1.json")
        lays[it["id"]] = lay
    return h, lays, det


def check_guides(paths: Paths, item_ids) -> None:
    want = P.read_json(P.GUIDES_JSON)["sha256"]
    for iid in sorted(set(item_ids)):
        if sha256(P.guide_path(paths.guides, iid)) != want[iid][P.GUIDE]:
            raise SetupError(f"guide of {iid} differs from guides_c1.json: run setup")


def runtime_now(h) -> dict:
    from dacapo import ace
    return runtime_fingerprint(ace.runtime_facts(h))


def gen(args, paths: Paths) -> None:
    """One pass over the block: each run that is not done and has attempts left is tried once, in plan order."""
    refuse_full_scale_tests(args, needs_cuda=True)
    out = paths.test
    out.mkdir(parents=True, exist_ok=True)
    a0 = a0_state(paths)
    plan = P.test_runs(args.scale, plan_items(args))
    done = {rid for rid, row in generated_of(out, [x["run_id"] for x in plan]).items() if row.get("status") == "ok"}
    used = attempts_used(out)
    todo = [x for x in plan if x["run_id"] not in done and used.get(x["run_id"], 0) < R.MAX_ATTEMPTS]
    if not todo:
        print(f"C1 gen: every planned run{' of this block' if plan_items(args) else ''} is generated or has used its "
              "attempts", flush=True)
        return
    reg = registration(args)
    items = P.all_items()
    h, lays, det = load_model(args, [items[i] for i in sorted({x["item"] for x in todo})])
    check_guides(paths, [x["item"] for x in todo])
    rt = runtime_now(h)
    change = runtime_change((read(paths.a0 / "a0_gen.json") or {}).get("runtime"), rt)
    if change and args.scale == "full" and not runtime_change_allowed():
        raise SetupError(f"this runtime differs from the one A0 ran on: {json.dumps(change)}. Resume on the GPU and "
                         f"runtime version A0 ran on (an L4), or set {ALLOW_RUNTIME_CHANGE}=1 (ALLOW_RUNTIME_CHANGE in the "
                         "notebook) to go on with the change recorded in every row")
    ctx = {"code_sha256": reg["code_sha256"], "registered": reg["registered"], "test_flags": test_flags(args),
           "scale": args.scale, "device": args.device, "a0_sha256": a0["sha256"], "runtime": rt,
           "runtime_change": change or None}
    write(out / f"gen_setup_{int(time.time())}_{os.getpid()}.json",
          {"utc": utc(), "determinism": det, "registration": reg["problems"], "items": sorted({x["item"] for x in todo}),
           **{k: v for k, v in ctx.items() if k != "code_sha256"}})
    streak: list[str] = []      # the items of the failures in a row
    failed_now: list[str] = []
    for run in todo:
        rid = run["run_id"]
        attempt = used.get(rid, 0) + 1
        append_jsonl(paths.ledger(), {"run_id": rid, "stage": "test", "status": "started", "attempt": attempt,
                                      "utc": utc(), "pid": os.getpid()})
        try:
            row, _ = RB.generate_run(h, run, lays[run["item"]], items[run["item"]], out, paths.guides)
            append_jsonl(paths.ledger(), {**row, **ctx, "attempt": attempt})
            print(f"{rid:>28s} {row['gen_seconds']:5.1f}s calls {row['hook_calls']:2d} peak "
                  f"{row['peak_before_norm']}", flush=True)
            streak = []
        except Exception as exc:  # recorded per run; the pass goes on
            append_jsonl(paths.ledger(), {"run_id": rid, "stage": "test", "status": "failed", "attempt": attempt,
                                          "error": repr(exc), "traceback": traceback.format_exc(), **ctx})
            print(f"{rid:>28s} attempt {attempt} FAILED: {exc!r}", flush=True)
            failed_now.append(rid)
            streak.append(run["item"])
        finally:
            free_cuda(args.device)
        if len(set(streak)) >= 2:
            print(f"C1 gen: {len(streak)} runs failed in a row, from {streak[0]} to {streak[-1]}: the error likely sticks "
                  f"to this process or the GPU, so this process stops and a fresh one goes on (exit {EXIT_STICKY})",
                  flush=True)
            sys.exit(EXIT_STICKY)
    left = attempts_used(out)
    again = [rid for rid in failed_now if left.get(rid, 0) < R.MAX_ATTEMPTS]
    if again:
        print(f"C1 gen: {len(again)} failed runs have attempts left; a fresh process tries them (exit {EXIT_AGAIN})",
              flush=True)
        sys.exit(EXIT_AGAIN)


# setup

def runs_to_generate(paths: Paths, args) -> int:
    have = {p.stem for p in paths.test.glob("*.wav")} if paths.test.exists() else set()
    return sum(r["run_id"] not in have for r in P.test_runs(args.scale))


def space_problem(paths: Paths, args) -> str | None:
    paths.root.mkdir(parents=True, exist_ok=True)
    need = runs_to_generate(paths, args) * BYTES_PER_RUN + SPACE_MARGIN
    du = shutil.disk_usage(paths.root)
    return None if du.free >= need else f"{du.free / 1e9:.1f} GB free, {need / 1e9:.1f} GB needed"


def setup_check(args, paths: Paths) -> None:
    """Main env. Every problem stops the full scale; the tiny scale records them and goes on."""
    import make_items_c1 as M

    rec = {"utc": utc(), "scale": args.scale, "registration": REG.registration_problems()}
    checks = register_checks()
    rec["register_checks"] = checks
    rec["packages"] = package_problems(REG.pins()["main"])
    got = M.render_all(paths.guides)
    plan_reg = (read(REG.ADDENDUM) or {}).get("plan", {})
    want = {"sha256": plan_reg.get("guides_sha256") or {}, "soundfont_sha256": plan_reg.get("soundfont_sha256")}
    rec["guides"] = M.guide_mismatches(got, want) if want["sha256"] else ["addendum 4 has no guide hashes"]
    rec["soundfont"] = {"sha256": got["soundfont_sha256"], "fluidsynth": got["fluidsynth"]}
    rec["seeds"] = P.seed_problems()
    if P.read_json(P.SEEDS_JSON) != json.loads(json.dumps(P.seed_table())):
        rec["seeds"].append("seeds_c1.json differs from plan_c1.seed_table()")
    if plan_reg and plan_reg.get("seed_table_sha256") != sha256(P.SEEDS_JSON):
        rec["seeds"].append("seeds_c1.json differs from addendum 4")
    reg_files = sorted(set(REG.runtime_files()) | {str(p.relative_to(ROOT)) for p in REG.REG_DIR.glob("addendum_*.json")})
    st = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *reg_files], capture_output=True, text=True)
    rec["git_uncommitted"] = None if st.returncode else [x[3:] for x in st.stdout.splitlines() if x.strip()]
    rec["space"] = space_problem(paths, args)
    problems = [f"registration: {p}" for p in rec["registration"]]
    problems += [f"{k} --check failed" for k, v in checks.items() if v["returncode"] != 0]
    problems += [f"guide {x}" for x in rec["guides"]]
    problems += [f"package {x}" for x in rec["packages"]]
    problems += [f"seeds: {x}" for x in rec["seeds"]]
    if rec["git_uncommitted"] is None or rec["git_uncommitted"]:
        problems.append(f"registered files are not committed: {rec['git_uncommitted']}")
    problems += [f"disk space: {rec['space']}"] if rec["space"] else []
    rec["problems"] = problems
    keep_session_copy(paths, "setup_check", rec)
    print(json.dumps({k: rec[k] for k in ("problems", "soundfont")}, indent=1), flush=True)
    if problems and args.scale == "full":
        raise SetupError("; ".join(problems))
    if problems:
        print(f"{args.scale} scale: setup problems recorded, the chain goes on", flush=True)


def setup_model(args, paths: Paths) -> None:
    """ACE env: sdpa, alignment heads, determinism flags, ACE-Step commit, weights revision and sha256, and ACE-Step's
    own lyric keys of all 98 items."""
    from dacapo import ace

    refuse_full_scale_tests(args, needs_cuda=True)
    rec = {"utc": utc(), "epoch": time.time(), "scale": args.scale, "pid": os.getpid()}
    try:
        h, lays, det = load_model(args, list(P.all_items().values()))
        rec.update(load="ok", determinism=det, runtime=ace.runtime_facts(h), lyric_keys="equal to tok_check_c1.json")
    except SetupError as exc:
        rec["load"] = f"failed: {exc}"
        det = None
    rec["runtime_fingerprint"] = runtime_fingerprint(rec.get("runtime"))
    rec["packages"] = package_problems(REG.pins()["ace"])
    reg2 = json.loads(REG.REG2.read_text())
    ckpt = Path(os.environ.get("ACESTEP_CHECKPOINTS_DIR") or Path(os.environ["ACESTEP_REPO"]) / "checkpoints")
    rec["weights_sha256"] = {w: sha256(ckpt / w) if (ckpt / w).exists() else None for w in reg2["ace"]["weights_sha256"]}
    meta = ckpt / ".cache" / "huggingface" / "download" / "acestep-v15-turbo" / "model.safetensors.metadata"
    rec["hf_revision"] = meta.read_text().splitlines()[0].strip() if meta.exists() else None
    try:
        rec["ace_commit"] = subprocess.run(["git", "-C", os.environ["ACESTEP_REPO"], "rev-parse", "HEAD"], capture_output=True,
                                           text=True, check=True).stdout.strip()
    except Exception:
        rec["ace_commit"] = None
    d = det or {}
    rec["checks"] = {"model_and_lyric_keys": rec["load"] == "ok",
                     "weights": rec["weights_sha256"] == reg2["ace"]["weights_sha256"],
                     "hf_revision": rec["hf_revision"] == reg2["ace"]["hf_revision"],
                     "ace_commit": rec["ace_commit"] == reg2["ace"]["commit"],
                     "determinism": bool(d.get("deterministic_algorithms") and d.get("cudnn_deterministic")
                                         and not d.get("cudnn_benchmark") and d.get("vae_decode_chunk_env") == "256"),
                     "packages": not rec["packages"]}
    rec["problems"] = [k for k, v in rec["checks"].items() if not v]
    keep_session_copy(paths, "setup_model", rec)
    print(json.dumps({"checks": rec["checks"], "load": rec["load"]}, indent=1), flush=True)
    if rec["problems"] and args.scale == "full":
        raise SetupError(f"model facts: {rec['problems']}")


# A0

def a0_checks(runs: list[dict], rep: dict, tens: dict, scale: str) -> None:
    """The hard checks that need the A0 tensors, written into rep (pure apart from torch.equal)."""
    import torch

    ok = {n: r.get("status") == "ok" and r.get("peak_before_norm") is not None for n, r in rep["runs"].items()}
    rep["hard"].update({f"{n}_ok": v for n, v in ok.items()})
    good = [r for r in rep["runs"].values() if r.get("status") == "ok"]
    hooked, plain = [r for r in good if r["hooked"]], [r for r in good if not r["hooked"]]
    rep["hard"]["hook_calls_40"] = bool(hooked) and all(r["hook_calls"] == P.HOOK_CALLS and r["max_key"] < r["n_lyr"]
                                                        for r in hooked) and all(r["hook_calls"] == 0 for r in plain)
    rep["hard"]["guide_latents_equal"] = len(good) == len(runs) and len({r["guide_latent_sha256"] for r in good}) == 1

    def equal(a, b):
        return a in tens and b in tens and bool(torch.equal(tens[a]["pred"], tens[b]["pred"]))
    rep["bit_identity"]["beta0_vs_no_hook"] = equal("s_b0", "beta0")
    rep["hard"]["beta0_bit_identical"] = rep["bit_identity"]["beta0_vs_no_hook"]
    if "s_b0" in tens and "beta0" in tens:
        rep["soft"]["beta0_max_abs_diff"] = float((tens["s_b0"]["pred"].float() - tens["beta0"]["pred"].float()).abs().max())
    if scale == "full":
        rep["hard"]["s_star_changes_output"] = "s_star" in tens and "s_b0" in tens and not equal("s_star", "s_b0")


def a0_gen(args, paths: Paths) -> None:
    """ACE env. The A0 generations and every check that needs their tensors -> a0/a0_gen.json."""
    refuse_full_scale_tests(args, needs_cuda=True)
    out = paths.a0
    out.mkdir(parents=True, exist_ok=True)
    runs = P.a0_runs(args.scale)
    rep: dict = {"utc": utc(), "scale": args.scale, "hard": {}, "soft": {}, "runs": {}, "bit_identity": {}}
    it = P.all_items()[P.A0_ITEM[args.scale]]
    try:
        reg = registration(args)
        check_guides(paths, [it["id"]])
        h, lays, det = load_model(args, [it])
        rep["hard"]["setup"] = True
        rep["determinism"] = det
        rep["runtime"] = runtime_now(h)
    except Exception as exc:
        rep["hard"]["setup"] = False
        rep["error"] = repr(exc)
        write(out / "a0_gen.json", rep)
        print("C1 A0 gen: SETUP FAILED", repr(exc), flush=True)
        sys.exit(EXIT_A0)
    lay = lays[it["id"]]
    tens: dict[str, dict] = {}
    for run in runs:
        try:
            row, t = RB.generate_run(h, run, lay, it, out, paths.guides)
            tens[run["name"]] = t
            rep["runs"][run["name"]] = {**row, "name": run["name"], "why": run["why"], "score": run["score"],
                                        "registered": reg["registered"]}
            print(f"A0 {run['name']:>8s} {row['gen_seconds']:5.1f}s calls {row['hook_calls']}", flush=True)
        except Exception as exc:
            rep["runs"][run["name"]] = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
            print(f"A0 {run['name']:>8s} FAILED {exc!r}", flush=True)
    a0_checks(runs, rep, tens, args.scale)
    if args.scale == "full" and not rep["bit_identity"]["beta0_vs_no_hook"] and "s_b0" in tens:
        again = {**next(r for r in runs if r["name"] == "s_b0"), "run_id": "a0_s_b0_repeat", "name": "s_b0_repeat"}
        try:  # tells nondeterminism from the hook; the check has failed either way
            _, t = RB.generate_run(h, again, lay, it, out, paths.guides)
            import torch
            rep["soft"]["no_hook_determinism_repeat"] = bool(torch.equal(tens["s_b0"]["pred"], t["pred"]))
        except Exception as exc:
            rep["soft"]["no_hook_determinism_repeat"] = f"failed: {exc!r}"
    rep["code_sha256"] = reg["code_sha256"]
    rep["registration"] = reg["problems"]
    write(out / "a0_gen.json", rep)
    print(json.dumps({"hard": rep["hard"], "bit_identity": rep["bit_identity"], "soft": rep["soft"]}, indent=1), flush=True)


def scorer(args):
    from score_c1 import C1Scorer
    try:
        return C1Scorer(args.device, test_whisper=args.test_whisper)
    except RuntimeError as exc:
        raise SetupError(str(exc)) from exc


def others_of(item_id: str) -> list[dict]:
    return [o for o in P.item_set(item_id) if o["id"] != item_id]


def a0_score(args, paths: Paths) -> None:
    """Main env: the registered scoring path on the A0 clips that A0 scores, with its timing."""
    refuse_full_scale_tests(args, needs_cuda=True)
    gen_rep = read(paths.a0 / "a0_gen.json")
    if gen_rep is None:
        raise SetupError("no a0_gen.json: run a0 --part gen first")
    sc = scorer(args)
    it = P.all_items()[P.A0_ITEM[args.scale]]
    rows = {}
    for name, g in gen_rep["runs"].items():
        if g.get("status") != "ok" or not g.get("score"):
            continue
        t0 = time.time()
        try:
            row = sc.measure(paths.a0 / f"{g['run_id']}.wav", it, others_of(it["id"]), P.guide_path(paths.guides, it["id"]))
        except Exception as exc:
            row = {"error": repr(exc), "traceback": traceback.format_exc()}
        row.update(run_id=g["run_id"], seconds_total=round(time.time() - t0, 2), wav_sha256=g["wav_sha256"])
        rows[name] = row
        print(f"A0 score {name:>8s} {row['seconds_total']:5.1f}s " + (f"FAILED {row['error']}" if "error" in row else
                                                                    f"words {len(row['whisper_chunks'])}"), flush=True)
    write(paths.a0 / "a0_scores.json", {"utc": utc(), "whisper": sc.base.whisper, "whisper_revision": sc.base.whisper_revision,
                                        "ctc_revision": sc.ctc_revision, "test_flags": sc.test_flags, "rows": rows})


def a0_projection(runs: dict, rows: dict, n_runs: int, n_blocks: int) -> dict:
    """Recorded, never applied: no trim in Run C. Session hours from A0's medians (the first generation of the
    process left out, as B2's A0 did) and from Run B's measured Stage T cost."""
    names = [n for n, r in runs.items() if r.get("status") == "ok"]
    gen_s = [runs[n]["gen_seconds"] for n in names[1:]] or [runs[n]["gen_seconds"] for n in names]
    score_s = [r["seconds_total"] for r in rows.values() if "error" not in r]
    out = {"n_runs": n_runs, "n_blocks": n_blocks,
           "from_run_b": round(P.projected_hours(P.STAGE_T_SECONDS_B2, n_runs, n_blocks), 2)}
    if gen_s and score_s:
        sec = float(np.median(gen_s)) + float(np.median(score_s))
        out.update(seconds_per_run_a0=round(sec, 2), from_a0=round(P.projected_hours(sec, n_runs, n_blocks), 2))
    return out


def a0_report(args, paths: Paths) -> None:
    """Main env: the A0 verdict and the time projection."""
    import score_c1 as S

    gen_rep, sc = read(paths.a0 / "a0_gen.json"), read(paths.a0 / "a0_scores.json")
    if gen_rep is None or sc is None:
        raise SetupError("A0 generation or scoring did not run")
    hard = dict(gen_rep["hard"])
    rows = sc["rows"]
    want = {n for n, r in gen_rep["runs"].items() if r.get("status") == "ok" and r.get("score")}
    scored_names = {n for n, r in gen_rep["runs"].items() if r.get("score")}
    hard["scorer"] = bool(rows) and set(rows) == want == scored_names and all("error" not in r for r in rows.values())
    if args.scale == "full":
        hard["scorer_registered"] = hard["scorer"] and all(
            S.CS.registered_reference(r) and r.get("ctc_revision") == S.CTC_REVISION and r.get("ctc_stem_text") is not None
            for r in rows.values())
    n_runs = len(P.test_runs(args.scale))
    proj = a0_projection(gen_rep["runs"], rows, n_runs, len(P.blocks(args.scale, args.block_items)))
    rep = {"utc": utc(), "scale": args.scale, "pass": all(hard.values()), "recorded": True, "hard": hard,
           "soft": gen_rep["soft"], "bit_identity": gen_rep["bit_identity"], "projection": proj,
           "code_sha256": gen_rep.get("code_sha256"), "a0_gen_sha256": sha256(paths.a0 / "a0_gen.json"),
           "a0_scores_sha256": sha256(paths.a0 / "a0_scores.json"),
           "test_flags": sorted(set(test_flags(args)) | set(sc["test_flags"]))}
    write(paths.a0_report, rep)
    print(json.dumps({k: rep[k] for k in ("pass", "hard", "bit_identity", "projection")}, indent=1), flush=True)
    print("C1 A0:", "PASS" if rep["pass"] else "FAIL", flush=True)
    if not rep["pass"]:
        sys.exit(EXIT_A0)


# scoring (main env)

def score_current(old: dict | None, g: dict, want_whisper: str, test: bool) -> bool:
    import score_c1 as S
    return bool(old and "error" not in old and old.get("wav_sha256") == g["wav_sha256"] and old.get("whisper") == want_whisper
                and (test or old.get("whisper_revision") == R.WHISPER_REVISION)
                and old.get("ctc_revision") == S.CTC_REVISION and old.get("ctc_stem_text") is not None)


def score(args, paths: Paths) -> None:
    refuse_full_scale_tests(args, needs_cuda=True)
    a0_state(paths)
    plan = P.test_runs(args.scale, plan_items(args))
    gens = generated_of(paths.test, [x["run_id"] for x in plan])
    last = {r["run_id"]: r for r in read_jsonl(paths.scores())}
    want = args.test_whisper or R.WHISPER
    todo = [run for run in plan if gens.get(run["run_id"], {}).get("status") == "ok"
            and not score_current(last.get(run["run_id"]), gens[run["run_id"]], want, bool(args.test_whisper))]
    if not todo:
        print("C1 score: every generated run" + (" of this block" if plan_items(args) else "") + " is scored", flush=True)
        return
    sc = scorer(args)
    items = P.all_items()
    for run in todo:
        g, it = gens[run["run_id"]], items[run["item"]]
        try:
            row = sc.measure(paths.test / f"{run['run_id']}.wav", it, others_of(it["id"]), P.guide_path(paths.guides, it["id"]))
        except Exception as exc:  # one bad clip must not stop the stage; it counts as unscored
            row = {"error": repr(exc), "traceback": traceback.format_exc(), "whisper": sc.base.whisper,
                   "whisper_revision": sc.base.whisper_revision, "test_flags": list(sc.test_flags)}
        row.update(run_id=run["run_id"], stage="test", wav_sha256=g["wav_sha256"], scale=args.scale)
        append_jsonl(paths.scores(), row)
        print(f"{run['run_id']:>28s} " + (f"SCORE FAILED: {row['error']}" if "error" in row else
                                          f"vocals {row['stem_share']['vocals']:.3f} NPA_v {row['npa_v']:.2f} "
                                          f"words {len(row['whisper_chunks'])}"), flush=True)


# decision (main env)

def clips_of(plan: list[dict], gens: dict, scores: dict) -> tuple[dict, dict]:
    """(usable score rows, per-clip values) of the planned runs; a score of another file is dropped, and a run that
    is missing, failed or unscored counts as missing."""
    import score_c1 as S

    items = P.all_items()
    used, clips = {}, {}
    for run in plan:
        g, s = gens.get(run["run_id"]), scores.get(run["run_id"])
        if g is None or g.get("status") != "ok":
            clips[run["run_id"]] = dict(S.MISSING)
            continue
        if s is not None and s.get("wav_sha256") != g["wav_sha256"]:
            s = None
        if s is not None:
            used[run["run_id"]] = s
        clips[run["run_id"]] = {**S.clip(s, g.get("peak_before_norm"), items[run["item"]]), "item": run["item"],
                                "draw": run["draw"], "config": run["config"]}
    return used, clips


def b2_outcomes() -> dict | None:
    import decide_c1 as D
    try:
        return D.b2_item_outcomes(json.loads(P.B2_RESULT.read_text()))
    except (OSError, ValueError, KeyError):
        return None


def runtime_record(paths: Paths, plan: list[dict], gens: dict) -> dict:
    """The runtime A0 ran on, and the generated runs that recorded a different one (allowed only with
    DACAPO_ALLOW_RUNTIME_CHANGE=1). Reported; a split (item, draw) still fails P0 through its guide latents."""
    planned = {r["run_id"] for r in plan}
    changed = {g["run_id"]: g["runtime_change"] for g in gens.values()
               if g.get("status") == "ok" and g["run_id"] in planned and g.get("runtime_change")}
    distinct = []
    for c in changed.values():
        if c not in distinct:
            distinct.append(c)
    return {"a0": (read(paths.a0 / "a0_gen.json") or {}).get("runtime"), "runs_on_another_runtime": len(changed),
            "run_ids": sorted(changed)[:20], "changes": distinct}


def decide(args, paths: Paths) -> None:
    import decide_c1 as D
    from dacapo.provenance import write_result

    t0 = time.time()
    a0 = a0_state(paths, need=False)
    flags = set(test_flags(args)) | set((a0["report"] or {}).get("test_flags", []))
    plan = P.test_runs(args.scale)
    gens = generated(paths.test)
    scores = {r["run_id"]: r for r in read_jsonl(paths.scores())}
    used, clips = clips_of(plan, gens, scores)
    planned = {r["run_id"] for r in plan}
    for g in gens.values():
        if g.get("status") == "ok" and g["run_id"] in planned:
            flags |= set(g.get("test_flags") or [])
    for s in used.values():
        flags |= set(s.get("test_flags") or [])
    plan_reg = (read(REG.ADDENDUM) or {}).get("plan", {})
    chk = read(paths.setup / "setup_check.json") or {}
    at_decision = register_checks()
    st = {"registration": REG.registration_problems(),
          "checks_failed": failed_checks(chk.get("register_checks"), "setup") + failed_checks(at_decision, "decision"),
          "registered_code": REG.registered_runtime() if REG.ADDENDUM.exists() else {},
          "registered_guides": plan_reg.get("guides_sha256") or {},
          "setup_hashes_ok": bool(chk) and bool(plan_reg) and chk.get("guides") == [] and chk.get("seeds") == []
          and chk.get("packages") == [] and (chk.get("soundfont") or {}).get("sha256") == plan_reg.get("soundfont_sha256"),
          "a0": a0["report"], "a0_sha256": a0["sha256"], "plan": plan, "gen": gens, "scores": used}
    p0 = D.integrity(st)
    p0_ok = all(c["held"] for c in p0.values())
    runtime = runtime_record(paths, plan, gens)
    b2 = b2_outcomes()
    analysis = D.analyse(plan, clips, b2)
    verdict, failed = D.verdict(p0_ok, analysis["rule"], sorted(flags))
    reasons = failed + [f"P0 {k}: {v['detail']}" for k, v in p0.items() if not v["held"]]
    items = P.all_items()
    clips_out = []
    for run in plan:
        g = gens.get(run["run_id"], {})
        clips_out.append({"run_id": run["run_id"], "item": run["item"], "draw": run["draw"], "config": run["config"],
                          "labels": run["labels"], "status": g.get("status", "missing"), "lyrics": items[run["item"]]["lyrics"],
                          "whisper_text": used.get(run["run_id"], {}).get("whisper_text"),
                          "ctc_text": used.get(run["run_id"], {}).get("ctc_text"),
                          "ctc_stem_text": used.get(run["run_id"], {}).get("ctc_stem_text"), **clips.get(run["run_id"], {})})
    result = {"verdict": verdict, "reasons": reasons, "labels": analysis["labels"], "p0": p0, "p0_held": p0_ok,
              "rule": analysis["rule"], "test_flags": sorted(flags), "scale": args.scale, "a0": a0["report"],
              "counts": {"planned": len(plan), "generated": sum(gens.get(r["run_id"], {}).get("status") == "ok" for r in plan),
                         "scored": len([s for s in used.values() if "error" not in s])},
              "arms": analysis["arms"], "contrasts": analysis["contrasts"], "pooled_with_b2": analysis["pooled_with_b2"],
              "bootstrap": analysis["bootstrap"], "demo": analysis["demo"], "runtime": runtime,
              "registration_checks_at_decision": {k: v["returncode"] for k, v in at_decision.items()}, "clips": clips_out}
    cfg = {"experiment": "c1", "registration": str(REG.ADDENDUM.relative_to(ROOT)), "scale": args.scale,
           "thresholds": {"j": D.J_MIN, "r1": "bootstrap p2_5 of Delta_J > 0", "size_bar_label": D.SIZE_BAR, "joint": 0.5,
                          "strict": 0.75, "max_missing_share": D.MAX_MISSING_SHARE, "n_boot": D.N_BOOT,
                          "boot_seed": D.BOOT_SEED}, "test_flags": sorted(flags)}
    paths.res.mkdir(parents=True, exist_ok=True)
    write_result(paths.res / "c1_result.json", cfg, result, t0)
    short = {"verdict": verdict, "reasons": reasons, "labels": analysis["labels"],
             "J": {a: v["J"] for a, v in analysis["arms"].items()}, "rule": analysis["rule"],
             "pooled_with_b2": analysis["pooled_with_b2"], "runtime_changes": runtime["runs_on_another_runtime"]}
    write(paths.res / "c1_summary.json", short)
    print(json.dumps(short, indent=1, default=str), flush=True)
    print("C1:", verdict.upper(), "|", "; ".join(reasons) or "all conditions held", flush=True)


def package(args, paths: Paths) -> None:
    import package_c1

    pkg = Path(args.package) if args.package else paths.root / "package"
    key = Path(args.key) if args.key else paths.root / "c1_blind_key.json"
    pack = Path(args.pack) if args.pack else key.parent / "c1_blind_pack.zip"
    res = package_c1.package(paths, pkg, key, pack, P.all_items())
    print(json.dumps(res, indent=1), flush=True)


def main():
    die_with_parent()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "a0", "gen", "score", "decide", "package"])
    ap.add_argument("--part", default="", help="setup: check | model; a0: gen | score | report")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scale", choices=["full", "tiny"], default="full")
    ap.add_argument("--root", default="")
    ap.add_argument("--guides", default=str(P.GUIDES_DIR))
    ap.add_argument("--items", default="", help="gen, score: a comma list of test items (one block)")
    ap.add_argument("--block-items", type=int, default=P.BLOCK_ITEMS, help="a0 report: block size for the projection")
    ap.add_argument("--test-whisper", default=None, help="tiny scale only: score with this Whisper model")
    ap.add_argument("--package", default="", help="package: the folder to fill")
    ap.add_argument("--key", default="", help="package: where the blind-pack key goes (never inside the package)")
    ap.add_argument("--pack", default="", help="package: where the blind pack goes (never inside the package)")
    args = ap.parse_args()
    if args.scale == "full" and args.test_whisper:
        print("C1: the registered full scale takes no test Whisper", flush=True)
        sys.exit(2)
    paths = Paths(args.root or str(HERE / ("outputs_c1" if args.scale == "full" else f"outputs_c1_{args.scale}")), args.guides)
    cmds = {("setup", "check"): setup_check, ("setup", "model"): setup_model, ("a0", "gen"): a0_gen,
            ("a0", "score"): a0_score, ("a0", "report"): a0_report}
    fn = cmds.get((args.cmd, args.part)) or {"gen": gen, "score": score, "decide": decide, "package": package}.get(args.cmd)
    if fn is None:
        ap.error(f"{args.cmd} needs --part")
    try:
        fn(args, paths)
    except SetupError as exc:
        print(f"C1 {args.cmd}{' ' + args.part if args.part else ''}: SETUP FAILED: {exc}", flush=True)
        sys.exit(EXIT_SETUP)


if __name__ == "__main__":
    main()
