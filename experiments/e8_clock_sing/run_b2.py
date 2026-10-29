"""B2 of Run B, the lyric clock with a melody guide, as addendum 3 registers it.

  python run_b2.py setup --part check                 main env: hashes, B1 gate, soundfont, guides, seeds, space
  python run_b2.py setup --part model --device cuda   ACE env: model facts and ACE-Step's own lyric keys
  python run_b2.py a0 --part gen --device cuda        ACE env: the A0 generations and their checks
  python run_b2.py a0 --part score --device cuda      main env: the registered scorer on the A0 clips
  python run_b2.py a0 --part report                   main env: pass or stop, the fallback, the timing trim
  python run_b2.py gen --stage dev|test --device cuda     ACE env: WAV, latents and a ledger row per run
  python run_b2.py score --stage dev|test --device cuda   main env: one score row per generated clip
  python run_b2.py select | decide | package          main env
  --root DIR      run state (default outputs_b2[_tiny] next to this file); --guides DIR (default guides/)
  --b1-out DIR    setup: B1's outputs, which the space check counts; --package, --key, --pack: package paths
  --scale tiny    1-2 runs per stage, any device, a test Whisper allowed; every verdict is "test only"

Exit codes: 0 done, 2 a setup problem (nothing generated or scored), 3 an A0 hard failure.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "e6_clock"))
import plan_b2 as P  # noqa: E402
import readout as R  # noqa: E402
import register_b2 as REG  # noqa: E402
from run_b1 import append_jsonl, read_jsonl, sha256  # noqa: E402

GATE_RES = ROOT / "results" / "gpu_run2"
SECONDS_KEEP = 2
BYTES_PER_RUN = 3_300_000          # a 15 s 48 kHz stereo PCM_16 WAV, pred and target latents, with headroom
SPACE_MARGIN = 1_000_000_000


class SetupError(RuntimeError):
    """A problem that affects every run (exit 2 before any generation or scoring)."""


class Paths:
    def __init__(self, root: str, guides: str):
        self.root, self.guides = Path(root), Path(guides)
        self.setup, self.a0, self.dev, self.test = (self.root / n for n in ("setup", "a0", "dev", "test"))
        self.res = self.root / "results"
        self.a0_report = self.a0 / "a0_report.json"
        self.selection = self.res / "b2_selection.json"

    def out(self, stage: str) -> Path:
        return {"dev": self.dev, "test": self.test, "a0": self.a0}[stage]

    def ledger(self, stage: str) -> Path:
        return self.out(stage) / "ledger.jsonl"

    def scores(self, stage: str) -> Path:
        return self.root / f"scores_{stage}.jsonl"


def read(p: Path):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, p)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tensor_sha256(x) -> str:
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


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
    return {"problems": problems, "registered": not problems,
            "code_sha256": REG.hashes(REG.runtime_files())}


def a0_state(paths: Paths, args, need: bool = True) -> dict:
    """The A0 report that generation follows: its trim, its fallback and its sha256."""
    rep = read(paths.a0_report)
    if rep is None or not rep.get("pass"):
        if need:
            raise SetupError("no passing A0 report: run the a0 stage first")
        return {"trim": dict(P.NO_TRIM), "fallback": False, "sha256": None, "report": rep}
    return {"trim": {k: bool(rep["trim"][k]) for k in P.NO_TRIM}, "fallback": bool(rep["fallback"]),
            "sha256": sha256(paths.a0_report), "report": rep}


# plans

def selection_head(paths: Paths) -> dict | None:
    rows = read_jsonl(paths.ledger("test"))
    return rows[0] if rows and "selection_sha256" in rows[0] and "run_id" not in rows[0] else None


def stage_plan(stage: str, paths: Paths, args, a0: dict) -> list[dict]:
    if stage == "dev":
        runs = P.dev_runs(args.scale, a0["trim"])
    else:
        sel = read(paths.selection)
        if sel is None or not sel.get("stage_t"):
            raise SetupError("no selection that allows Stage T")
        runs = P.test_runs(P.parse(sel["s_star"]), P.parse(sel["b_star"]), args.scale, a0["trim"])
    return [P.with_fallback(r) for r in runs] if a0["fallback"] else runs


# generation (ACE env)

def files_match(out: Path, row: dict) -> bool:
    rid = row["run_id"]
    f = [out / f"{rid}.wav", out / f"{rid}.pred.npy", out / f"{rid}.target.npy"]
    return all(p.exists() for p in f) and [sha256(p) for p in f] == [row.get("wav_sha256"), row.get("pred_sha256"),
                                                                     row.get("target_sha256")]


def generated(out: Path) -> dict[str, dict]:
    """run_id -> its last ledger row; an ok row counts only while its files still hash to it."""
    last: dict[str, dict] = {}
    for row in read_jsonl(out / "ledger.jsonl"):
        if "run_id" in row:
            last[row["run_id"]] = row
    for rid, row in last.items():
        if row.get("status") == "ok" and not files_match(out, row):
            last[rid] = {**row, "status": "files_missing"}
    return last


def failed_attempts(out: Path) -> dict[str, int]:
    n: dict[str, int] = {}
    for r in read_jsonl(out / "ledger.jsonl"):
        if r.get("status") == "failed":
            n[r["run_id"]] = n.get(r["run_id"], 0) + 1
    return n


def load_model(args, items: list[dict]):
    """setup_determinism, the model, the alignment heads, and ACE-Step's own lyric keys for these items,
    which must equal tok_check_b2.json."""
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
            raise SetupError(f"{it['id']}: ACE-Step's lyric keys differ from tok_check_b2.json")
        lays[it["id"]] = lay
    return h, lays, det


def generate_run(h, run: dict, lay: dict, it: dict, out: Path, guides: Path, pre=None) -> tuple[dict, dict]:
    """One run: ace.generate inside CrossAttnBias when the run is hooked. Raises on a technical problem.
    Returns the row fields and the tensors A0 compares."""
    from dacapo import ace
    from dacapo.attn_bias import CrossAttnBias

    t0 = time.time()
    bias = None
    if run["hooked"]:
        sp = P.spans(it, lay["keys"], run["shuffle"])
        bias = CrossAttnBias(h.model.decoder, {int(L): hs for L, hs in run["heads"].items()}, sp, run["beta"], lay["n_lyr"])
    src = P.guide_path(guides, run["item"], run["guide"]) if run["guide"] else None
    wav = out / f"{run['run_id']}.wav"
    with bias if bias is not None else contextlib.nullcontext():
        r = ace.generate(h, wav, run["caption"], run["duration"], run["dit_seed"], lyrics=run["lyrics"], steps=P.STEPS,
                         shift=P.SHIFT, task=run["task"], src=None if src is None else str(src),
                         cover_strength=run["cover_strength"], cover_noise=0.0, bpm=run["bpm"], keyscale=run["keyscale"],
                         vocal_language="en", extra={"dcw_enabled": False}, pre=pre, keep_extra=True,
                         rng_seed=run["rng_seed"], out_subtype=R.WAV_SUBTYPE)
    ex = r["extra"]
    for k in ("pred_latents", "target_latents", "lyric_token_idss", "encoder_attention_mask"):
        if ex.get(k) is None:
            raise RuntimeError(f"ACE-Step returned no {k}: is ACESTEP_SAVE_MEMORY set?")
    if r["n_denoiser_calls"] != P.STEPS or r.get("dcw_enabled") is not False:
        raise RuntimeError(f"{r['n_denoiser_calls']} denoiser calls (expected {P.STEPS}), DCW {r.get('dcw_enabled')}")
    if [int(i) for i in list(ex["lyric_token_idss"][0])] != lay["ids"]:
        raise RuntimeError("the lyric ids of the run differ from the ids the windows were built on")
    if not bool((ex["encoder_attention_mask"] == 1).all()):
        raise RuntimeError("encoder attention mask is not all ones")
    pred, target = out / f"{run['run_id']}.pred.npy", out / f"{run['run_id']}.target.npy"
    np.save(pred, ex["pred_latents"][0].float().cpu().numpy())
    np.save(target, ex["target_latents"][0].float().cpu().numpy())
    rec = bias.record() if bias is not None else {"hook_calls": 0, "lq_lk": [], "max_key": None, "cells": None,
                                                  "mask_in": None}
    row = {k: run[k] for k in ("run_id", "stage", "item", "draw", "config", "labels", "guide", "k", "clock", "head_set",
                               "heads", "beta", "shuffle", "hooked", "fallback", "task", "cover_strength", "dit_seed",
                               "rng_seed")}
    row.update(status="ok", hook_calls=rec["hook_calls"], lq_lk=rec["lq_lk"], max_key=rec["max_key"], n_lyr=lay["n_lyr"],
               cells=rec["cells"], mask_in=rec["mask_in"], peak_before_norm=r.get("peak_before_norm"),
               n_denoiser_calls=r["n_denoiser_calls"], dcw_enabled=r.get("dcw_enabled"), frames=r["frames"],
               instruction=r.get("instruction"), decode_chunk=r.get("decode_chunk"),
               cpu_decode_lines=r.get("cpu_decode_lines"), oom_lines=r.get("oom_lines"),
               determinism_warnings=r.get("determinism_warnings"), gen_seconds=r["seconds"],
               total_seconds=round(time.time() - t0, 2), wav_sha256=sha256(wav), pred_sha256=sha256(pred),
               target_sha256=sha256(target), guide_latent_sha256=tensor_sha256(ex["target_latents"]) if run["guide"] else None,
               guide_sha256=None if src is None else sha256(src))
    tensors = {"pred": ex["pred_latents"].detach().clone(), "target": ex["target_latents"].detach().clone()}
    del ex
    return row, tensors


def gen(args, paths: Paths) -> None:
    refuse_full_scale_tests(args, needs_cuda=True)
    stage = args.stage
    out = paths.out(stage)
    out.mkdir(parents=True, exist_ok=True)
    a0 = a0_state(paths, args)
    plan = stage_plan(stage, paths, args, a0)
    head = None
    if stage == "test":
        head = selection_head(paths)
        if head is None or head["selection_sha256"] != sha256(paths.selection):
            raise SetupError("Stage T needs the selection's sha256 as the first line of its ledger (run select)")
    todo = [x for x in plan if not args.only or x["run_id"] in args.only.split(",")]
    done = {rid for rid, row in generated(out).items() if row.get("status") == "ok"}
    fails = failed_attempts(out)
    todo = [x for x in todo if x["run_id"] not in done and fails.get(x["run_id"], 0) < R.MAX_ATTEMPTS]
    if not todo:
        print(f"B2 gen {stage}: every planned run is generated or has used its attempts", flush=True)
        return
    reg = registration(args)
    items = P.all_items()
    h, lays, det = load_model(args, [items[i] for i in sorted({x["item"] for x in plan})])
    for gid in sorted({(x["item"], x["guide"]) for x in plan if x["guide"]}):
        want = P.read_json(P.GUIDES_JSON)["sha256"][gid[0]][gid[1]]
        if sha256(P.guide_path(paths.guides, *gid)) != want:
            raise SetupError(f"guide {gid} differs from guides_b2.json: run setup")
    ctx = {"code_sha256": reg["code_sha256"], "registered": reg["registered"], "test_flags": test_flags(args),
           "scale": args.scale, "device": args.device, "a0_sha256": a0["sha256"],
           "selection_sha256": None if head is None else head["selection_sha256"]}
    write(out / f"gen_setup_{int(time.time())}.json", {"utc": utc(), "determinism": det, "registration": reg["problems"],
                                                       **{k: v for k, v in ctx.items() if k != "code_sha256"}})
    ledger = paths.ledger(stage)
    for run in todo:
        for attempt in range(fails.get(run["run_id"], 0) + 1, R.MAX_ATTEMPTS + 1):
            try:
                row, _ = generate_run(h, run, lays[run["item"]], items[run["item"]], out, paths.guides)
                append_jsonl(ledger, {**row, **ctx, "attempt": attempt})
                print(f"{run['run_id']:>28s} {row['gen_seconds']:5.1f}s calls {row['hook_calls']:2d} Lq/Lk {row['lq_lk']} "
                      f"peak {row['peak_before_norm']}", flush=True)
                break
            except Exception as exc:  # recorded per run; the stage goes on
                append_jsonl(ledger, {"run_id": run["run_id"], "stage": stage, "status": "failed", "attempt": attempt,
                                      "error": repr(exc), "traceback": traceback.format_exc(), **ctx})
                print(f"{run['run_id']:>28s} attempt {attempt} FAILED: {exc!r}", flush=True)
            finally:
                if args.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()


# setup

def setup_check(args, paths: Paths) -> None:
    """Main env. Every problem stops the full scale; the tiny scale records them and goes on."""
    import make_items_b2 as M

    rec = {"utc": utc(), "scale": args.scale}
    rec["registration"] = REG.registration_problems()
    checks = {}
    for name, cmd in (("addendum_2", [str(ROOT / "experiments" / "e5_motif" / "register.py"), "--check"]),
                      ("b1_code_addendum", [str(ROOT / "experiments" / "e6_clock" / "register_b1code.py"), "--check"]),
                      ("addendum_3", [str(HERE / "register_b2.py"), "--check"])):
        p = subprocess.run([sys.executable, *cmd], capture_output=True, text=True, cwd=ROOT)
        checks[name] = {"returncode": p.returncode, "output": (p.stdout + p.stderr).strip()[-2000:]}
    rec["register_checks"] = checks
    g = subprocess.run([sys.executable, str(ROOT / "experiments" / "e6_clock" / "readout.py"), "check-gate"],
                       capture_output=True, text=True, env={**os.environ, "E6_RES": str(GATE_RES)})
    rec["b1_gate"] = {"ok": g.returncode == 0, "output": (g.stdout + g.stderr).strip()}
    got = M.render_all(paths.guides)
    add = read(REG.ADDENDUM) or {}
    want = {"sha256": add.get("plan", {}).get("guides_sha256") or {}, "soundfont_sha256": add.get("plan", {}).get("soundfont_sha256")}
    rec["guides"] = M.guide_mismatches(got, want) if want["sha256"] else ["addendum 3 has no guide hashes"]
    rec["soundfont"] = {"sha256": got["soundfont_sha256"], "fluidsynth": got["fluidsynth"]}  # the guide sha256 decide
    rec["seeds"] = P.seed_problems()
    if P.read_json(P.SEEDS_JSON) != json.loads(json.dumps(P.seed_table())):
        rec["seeds"].append("seeds_b2.json differs from plan_b2.seed_table()")
    reg_files = sorted(set(REG.runtime_files()) | {str(p.relative_to(ROOT)) for p in (REG.REG_DIR).glob("addendum_*.json")})
    st = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *reg_files], capture_output=True, text=True)
    rec["git_uncommitted"] = None if st.returncode else [x[3:] for x in st.stdout.splitlines() if x.strip()]
    rec["space"] = space_problem(paths, args)
    problems = [f"registration: {p}" for p in rec["registration"]]
    problems += [f"{k} --check failed" for k, v in checks.items() if v["returncode"] != 0]
    problems += [f"guide {x}" for x in rec["guides"]]
    problems += [f"seeds: {x}" for x in rec["seeds"]]
    if rec["git_uncommitted"] is None or rec["git_uncommitted"]:
        problems.append(f"registered files are not committed: {rec['git_uncommitted']}")
    problems += [f"disk space: {rec['space']}"] if rec["space"] else []
    rec["problems"] = problems
    rec["b1_note"] = None if rec["b1_gate"]["ok"] else "readout.py check-gate refuses B1; B2 goes on"
    write(paths.setup / "setup_check.json", rec)
    print(json.dumps({k: rec[k] for k in ("problems", "b1_gate", "soundfont")}, indent=1), flush=True)
    if problems and args.scale == "full":
        raise SetupError("; ".join(problems))
    if problems:
        print(f"{args.scale} scale: setup problems recorded, the chain goes on", flush=True)


def runs_to_generate(paths: Paths, args) -> dict[str, int]:
    """Planned runs without a WAV yet, per stage: B1 (in --b1-out; all of them if it is not given), Stage D
    after A0's trim, and Stage T from the selection once it exists, else at its largest."""
    import run_b1

    def left(planned: list[dict], out: Path | None) -> int:
        have = {p.stem for p in out.glob("*.wav")} if out is not None and out.exists() else set()
        return sum(r["run_id"] not in have for r in planned)
    a0 = a0_state(paths, args, need=False)
    sel = read(paths.selection)
    if sel is None:  # a selection with all 8 arms
        n_test = len(P.test_runs(P.parse("clarinet_k5_A8"), P.parse("clarinet_k4_none"), args.scale, a0["trim"]))
    else:
        n_test = left(stage_plan("test", paths, args, a0), paths.test) if sel.get("stage_t") else 0
    b1_out = Path(args.b1_out) if getattr(args, "b1_out", "") else None
    return {"b1": left(run_b1.plan(args.scale), b1_out), "dev": left(stage_plan("dev", paths, args, a0), paths.dev),
            "test": n_test}


def space_problem(paths: Paths, args) -> str | None:
    """Free space where the run state lives, for every run still to generate."""
    paths.root.mkdir(parents=True, exist_ok=True)
    need = sum(runs_to_generate(paths, args).values()) * BYTES_PER_RUN + SPACE_MARGIN
    du = shutil.disk_usage(paths.root)
    return None if du.free >= need else f"{du.free / 1e9:.1f} GB free, {need / 1e9:.1f} GB needed"


def setup_model(args, paths: Paths) -> None:
    """ACE env: what actually runs. sdpa, alignment heads, determinism flags, ACE-Step commit, weights revision
    and sha256, and ACE-Step's own lyric keys of all 36 items."""
    from dacapo import ace

    refuse_full_scale_tests(args, needs_cuda=True)
    rec = {"utc": utc(), "scale": args.scale}
    try:
        h, lays, det = load_model(args, list(P.all_items().values()))
        rec.update(load="ok", determinism=det, runtime=ace.runtime_facts(h), lyric_keys="equal to tok_check_b2.json")
    except SetupError as exc:
        rec["load"] = f"failed: {exc}"
        det = None
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
                                         and not d.get("cudnn_benchmark") and d.get("vae_decode_chunk_env") == "256")}
    rec["problems"] = [k for k, v in rec["checks"].items() if not v]
    write(paths.setup / "setup_model.json", rec)
    print(json.dumps({"checks": rec["checks"], "load": rec["load"]}, indent=1), flush=True)
    if rec["problems"] and args.scale == "full":
        raise SetupError(f"model facts: {rec['problems']}")


# A0

def a0_gen(args, paths: Paths) -> None:
    """ACE env. The A0 generations and every check that needs their tensors -> a0/a0_gen.json."""
    import torch

    refuse_full_scale_tests(args, needs_cuda=True)
    out = paths.a0
    out.mkdir(parents=True, exist_ok=True)
    runs = P.a0_runs(args.scale)
    rep: dict = {"utc": utc(), "scale": args.scale, "hard": {}, "soft": {}, "runs": {}, "bit_identity": {}}
    try:
        reg = registration(args)
        it = P.all_items()["item00"]
        h, lays, det = load_model(args, [it])
        rep["hard"]["setup"] = True
        rep["determinism"] = det
    except Exception as exc:
        rep["hard"]["setup"] = False
        rep["error"] = repr(exc)
        write(out / "a0_gen.json", rep)
        print("B2 A0 gen: SETUP FAILED", repr(exc), flush=True)
        sys.exit(3)
    lay = lays["item00"]
    n_lyr, K = lay["n_lyr"], 4
    tens: dict[str, dict] = {}
    for run in runs:
        enc: dict = {}

        def pre(step, t, kw, enc=enc):  # reads the encoder states, changes nothing
            if step in (0, K):
                enc[step] = kw["encoder_hidden_states"].detach().clone()
            return kw
        try:
            row, t = generate_run(h, run, lay, it, out, paths.guides, pre=pre)
            t["enc"] = enc
            tens[run["name"]] = t
            rep["runs"][run["name"]] = {**row, "name": run["name"], "why": run["why"], "registered": reg["registered"]}
            print(f"A0 {run['name']:>18s} {row['gen_seconds']:5.1f}s calls {row['hook_calls']}", flush=True)
        except Exception as exc:
            rep["runs"][run["name"]] = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
            print(f"A0 {run['name']:>18s} FAILED {exc!r}", flush=True)
    ok = {n: r.get("status") == "ok" and r.get("peak_before_norm") is not None for n, r in rep["runs"].items()}
    rep["hard"].update({f"{n}_ok": v for n, v in ok.items()})
    hooked = [r for r in rep["runs"].values() if r.get("status") == "ok" and r["hooked"]]
    plain = [r for r in rep["runs"].values() if r.get("status") == "ok" and not r["hooked"]]
    rep["hard"]["hook_calls_40"] = bool(hooked) and all(r["hook_calls"] == P.HOOK_CALLS and r["max_key"] < r["n_lyr"]
                                                        for r in hooked) and all(r["hook_calls"] == 0 for r in plain)

    def equal(a, b, what="pred"):
        return a in tens and b in tens and bool(torch.equal(tens[a][what], tens[b][what]))

    def maxdiff(a, b):
        return float((tens[a]["pred"].float() - tens[b]["pred"].float()).abs().max()) if a in tens and b in tens else None
    if "cover_nohook" in tens:
        e0, e4 = tens["cover_nohook"]["enc"].get(0), tens["cover_nohook"]["enc"].get(K)
        rep["hard"]["lyric_rows_equal_across_switch"] = e0 is not None and e4 is not None and bool(
            torch.equal(e0[:, :n_lyr], e4[:, :n_lyr]))
        rep["soft"]["whole_encoder_states_equal_across_switch"] = e0 is not None and e4 is not None and tuple(
            e0.shape) == tuple(e4.shape) and bool(torch.equal(e0, e4))
        rep["soft"]["lk_cover_non_cover"] = None if e0 is None or e4 is None else [int(e0.shape[1]), int(e4.shape[1])]
    else:
        rep["hard"]["lyric_rows_equal_across_switch"] = False
    # one guide latent for every cover run of the draw, whatever its k (P0 groups k3, k4 and k5 together)
    covers = [r["name"] for r in runs if r["guide"] and r["name"] in tens]
    rep["hard"]["guide_latents_equal"] = len(covers) >= 2 and len({rep["runs"][n]["guide_latent_sha256"] for n in covers}) == 1
    rep["soft"]["guide_latent_ks"] = sorted({rep["runs"][n]["k"] for n in covers})
    rep["bit_identity"]["cover_k4"] = equal("cover_nohook", "cover_A0")
    rep["soft"]["cover_beta0_max_abs_diff"] = maxdiff("cover_nohook", "cover_A0")
    if args.scale == "full":
        rep["bit_identity"]["text2music"] = equal("t2m_nohook", "t2m_A0")
        rep["soft"]["t2m_beta0_max_abs_diff"] = maxdiff("t2m_nohook", "t2m_A0")
        rep["soft"]["t2m_determinism_repeat"] = equal("t2m_nohook", "t2m_nohook_repeat")
        rep["hard"]["beta2_changes_output"] = "cover_A2" in tens and not equal("cover_nohook", "cover_A2")
        if not rep["bit_identity"]["cover_k4"] and "cover_nohook" in tens:  # tell nondeterminism from the hook
            run = next(r for r in runs if r["name"] == "cover_nohook")
            again = {**run, "run_id": "a0_cover_nohook_repeat", "name": "cover_nohook_repeat"}
            try:
                row, t = generate_run(h, again, lay, it, out, paths.guides)
                tens["cover_nohook_repeat"] = t
                rep["soft"]["cover_determinism_repeat"] = equal("cover_nohook", "cover_nohook_repeat")
            except Exception as exc:
                rep["soft"]["cover_determinism_repeat"] = f"failed: {exc!r}"
    rep["code_sha256"] = reg["code_sha256"]
    rep["registration"] = reg["problems"]
    write(out / "a0_gen.json", rep)
    print(json.dumps({"hard": rep["hard"], "bit_identity": rep["bit_identity"], "soft": rep["soft"]}, indent=1), flush=True)


def scorer(args):
    from score_b2 import B2Scorer
    try:
        return B2Scorer(args.device, test_whisper=args.test_whisper)
    except RuntimeError as exc:
        raise SetupError(str(exc)) from exc


def a0_score(args, paths: Paths) -> None:
    """Main env: the registered scoring path on every A0 clip, with its timing."""
    refuse_full_scale_tests(args, needs_cuda=True)
    gen_rep = read(paths.a0 / "a0_gen.json")
    if gen_rep is None:
        raise SetupError("no a0_gen.json: run a0 --part gen first")
    sc = scorer(args)
    items = P.all_items()
    it = items["item00"]
    others = [o for o in P.dev_items() if o["id"] != "item00"]
    rows = {}
    for name, g in gen_rep["runs"].items():
        if g.get("status") != "ok":
            continue
        t0 = time.time()
        try:
            row = sc.measure(paths.a0 / f"{g['run_id']}.wav", it, others, P.guide_path(paths.guides, "item00", g["guide"] or "oohs"))
        except Exception as exc:
            row = {"error": repr(exc), "traceback": traceback.format_exc()}
        row.update(run_id=g["run_id"], seconds_total=round(time.time() - t0, 2), wav_sha256=g["wav_sha256"])
        rows[name] = row
        print(f"A0 score {name:>18s} {row['seconds_total']:5.1f}s " + (f"FAILED {row['error']}" if "error" in row else
                                                                     f"words {len(row['whisper_chunks'])}"), flush=True)
    write(paths.a0 / "a0_scores.json", {"utc": utc(), "whisper": sc.base.whisper, "whisper_revision": sc.base.whisper_revision,
                                        "ctc_revision": sc.ctc_revision, "test_flags": sc.test_flags, "rows": rows})


def a0_timing(runs: dict, gen_ok: list[str], rows: dict) -> dict:
    """Seconds per run by kind (guided, text2music): the median generation seconds of the kind's A0 runs, leaving
    out the first run of the process (it warms up the device), plus the median scoring seconds of its clips, plus
    the registered loop allowance. A kind that A0 did not run (the tiny scale has no text2music run) takes the
    other kind's medians; with no timing at all, every run counts as infinitely long, so both trim steps apply."""
    med = {}
    for kind in ("guided", "text2music"):
        names = [n for n in gen_ok if bool(runs[n]["guide"]) == (kind == "guided")]
        gen_s = [runs[n]["gen_seconds"] for n in names if n != gen_ok[0]] or [runs[n]["gen_seconds"] for n in names]
        score_s = [rows[n]["seconds_total"] for n in names if n in rows and "error" not in rows[n]]
        if gen_s and score_s:
            med[kind] = {"gen_seconds_median": float(np.median(gen_s)), "score_seconds_median": float(np.median(score_s)),
                         "n_gen": len(gen_s), "n_scored": len(score_s)}
    for kind, other in (("guided", "text2music"), ("text2music", "guided")):
        if kind not in med and other in med:
            med[kind] = {**med[other], "from": other}
    sec = {k: (P.seconds_per_run(med[k]["gen_seconds_median"], med[k]["score_seconds_median"], k) if k in med
               else float("inf")) for k in ("guided", "text2music")}
    return {"medians": med, "seconds_per_run": {k: round(v, 2) for k, v in sec.items()}}


def a0_report(args, paths: Paths) -> None:
    """Main env: the A0 verdict, the fallback for B2, the B1 note and the registered timing trim."""
    import score_b2 as S

    gen_rep, sc = read(paths.a0 / "a0_gen.json"), read(paths.a0 / "a0_scores.json")
    if gen_rep is None or sc is None:
        raise SetupError("A0 generation or scoring did not run")
    hard = dict(gen_rep["hard"])
    rows = sc["rows"]
    gen_ok = [n for n, r in gen_rep["runs"].items() if r.get("status") == "ok"]
    hard["scorer"] = bool(rows) and set(rows) == set(gen_ok) and all("error" not in r for r in rows.values())
    if args.scale == "full":
        hard["scorer_registered"] = hard["scorer"] and all(S.CS.registered_reference(r) and r.get("ctc_revision") == S.CTC_REVISION
                                                           for r in rows.values())
    bit = gen_rep["bit_identity"]
    fallback = not all(bit.values())
    timing = a0_timing(gen_rep["runs"], gen_ok, rows)
    trim = P.timing_trim(timing["seconds_per_run"])
    note = None
    if fallback:
        note = (f"A0 found the hook at beta 0 not bit-identical to no hook on this device ({bit}); the registered "
                "fallback applies to every unclocked B2 run, and B1 keeps its registered no-hook arm")
    rep = {"utc": utc(), "scale": args.scale, "pass": all(hard.values()), "recorded": True, "hard": hard,
           "soft": gen_rep["soft"], "bit_identity": bit, "fallback": fallback, "b1_note": note, "timing": timing,
           "trim": trim, "code_sha256": gen_rep.get("code_sha256"), "a0_gen_sha256": sha256(paths.a0 / "a0_gen.json"),
           "a0_scores_sha256": sha256(paths.a0 / "a0_scores.json"), "test_flags": sorted(set(test_flags(args)) | set(sc["test_flags"]))}
    write(paths.a0_report, rep)
    print(json.dumps({k: rep[k] for k in ("pass", "hard", "bit_identity", "fallback", "timing", "trim")}, indent=1), flush=True)
    print("B2 A0:", "PASS" if rep["pass"] else "FAIL", flush=True)
    if not rep["pass"]:
        sys.exit(3)


# scoring (main env)

def item_set(item_id: str) -> list[dict]:
    """The set an item belongs to (the 12 dev items or the 24 test items): its other melodies give the chance
    level of pitch accuracy. The tiny scale's Stage T item is a dev item."""
    dev = P.dev_items()
    return dev if item_id in {it["id"] for it in dev} else P.test_items()


def score(args, paths: Paths) -> None:
    refuse_full_scale_tests(args, needs_cuda=True)
    stage = args.stage
    a0 = a0_state(paths, args)
    plan = stage_plan(stage, paths, args, a0)
    gens = generated(paths.out(stage))
    path = paths.scores(stage)
    last = {r["run_id"]: r for r in read_jsonl(path)}
    want = args.test_whisper or R.WHISPER
    import score_b2 as S

    def current(old: dict | None, g: dict) -> bool:
        return bool(old and "error" not in old and old.get("wav_sha256") == g["wav_sha256"] and old.get("whisper") == want
                    and (args.test_whisper or old.get("whisper_revision") == R.WHISPER_REVISION)
                    and old.get("ctc_revision") == S.CTC_REVISION)
    todo = [run for run in plan if gens.get(run["run_id"], {}).get("status") == "ok"
            and not current(last.get(run["run_id"]), gens[run["run_id"]])]
    if not todo:
        print(f"B2 score {stage}: every generated run is scored", flush=True)
        return
    sc = scorer(args)
    items = P.all_items()
    head = selection_head(paths) if stage == "test" else None
    for run in todo:
        g, it = gens[run["run_id"]], items[run["item"]]
        others = [o for o in item_set(it["id"]) if o["id"] != it["id"]]
        try:
            row = sc.measure(paths.out(stage) / f"{run['run_id']}.wav", it, others,
                             P.guide_path(paths.guides, it["id"], run["guide"] or "oohs"))
        except Exception as exc:  # one bad clip must not stop the stage; it counts as unscored
            row = {"error": repr(exc), "traceback": traceback.format_exc(), "whisper": sc.base.whisper,
                   "whisper_revision": sc.base.whisper_revision, "test_flags": list(sc.test_flags)}
        row.update(run_id=run["run_id"], stage=stage, wav_sha256=g["wav_sha256"], scale=args.scale,
                   selection_sha256=None if head is None else head["selection_sha256"])
        append_jsonl(path, row)
        print(f"{run['run_id']:>28s} " + (f"SCORE FAILED: {row['error']}" if "error" in row else
                                          f"vocals {row['stem_share']['vocals']:.3f} NPA_v {row['npa_v']:.2f} "
                                          f"words {len(row['whisper_chunks'])}"), flush=True)


# selection and decision (main env)

def stage_clips(stage: str, plan: list[dict], paths: Paths) -> tuple[dict, dict, dict]:
    """(gen rows, score rows, per-clip values) of a stage's planned runs; a score of another file is dropped."""
    import score_b2 as S

    gens = generated(paths.out(stage))
    scores = {r["run_id"]: r for r in read_jsonl(paths.scores(stage))}
    items = P.all_items()
    clips, used = {}, {}
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
    return gens, used, clips


def selection_now(paths: Paths, args, a0: dict) -> tuple[dict, dict]:
    import select_b2 as SEL

    plan = stage_plan("dev", paths, args, a0)
    _, _, clips = stage_clips("dev", plan, paths)
    cfg_ids = list(dict.fromkeys(r["config"] for r in plan))
    configs = [P.parse(c) for c in cfg_ids]
    items = sorted({r["item"] for r in plan})
    draws = sorted({r["draw"] for r in plan})
    vals = [{**c, "item": r["item"], "draw": r["draw"], "config": r["config"]} for r, c in
            ((r, clips[r["run_id"]]) for r in plan)]
    tab = SEL.table(vals, configs, items, draws)
    sel = SEL.select(tab, waive_gate=args.scale == "tiny")
    ledgers = {str(p.relative_to(paths.root)): SEL.sha256_file(p) for p in (paths.ledger("dev"), paths.scores("dev"))}
    code = REG.hashes(REG.CODE)
    return SEL.selection_doc(tab, sel, configs, ledgers, code, args.scale, a0["trim"]), sel


def select(args, paths: Paths) -> None:
    import select_b2 as SEL

    a0 = a0_state(paths, args)
    doc, sel = selection_now(paths, args, a0)
    new = SEL.doc_bytes(doc)
    new_sha = hashlib.sha256(new).hexdigest()
    run_rows = [r for r in read_jsonl(paths.ledger("test")) if "run_id" in r]
    head = selection_head(paths)
    if paths.selection.exists() and paths.selection.read_bytes() == new and head and head["selection_sha256"] == new_sha:
        print(f"B2 select: unchanged (S* {doc['s_star']}, B* {doc['b_star']})", flush=True)
        return
    if run_rows:  # select is locked once Stage T has a row
        raise SetupError("Stage T has rows: the selection and its ledger line are fixed, and the Stage D rows now give "
                         "another selection or the line is missing")
    paths.res.mkdir(parents=True, exist_ok=True)
    paths.test.mkdir(parents=True, exist_ok=True)
    paths.selection.write_bytes(new)
    head = {"selection_sha256": new_sha, "s_star": doc["s_star"], "b_star": doc["b_star"], "stage_t": doc["stage_t"],
            "utc": utc()}
    paths.ledger("test").write_text(json.dumps(head) + "\n")  # only the line of an earlier selection is replaced
    print(f"B2 select: S* {doc['s_star']}, B* {doc['b_star']}, Stage T {'runs' if doc['stage_t'] else 'does not run'}"
          + (" (tiny scale: gate eligibility waived)" if doc["gate_waived"] else ""), flush=True)


def decide(args, paths: Paths) -> None:
    import decide_b2 as D
    import select_b2 as SEL
    from dacapo.provenance import write_result

    t0 = time.time()
    a0 = a0_state(paths, args, need=False)
    sel_doc = read(paths.selection)
    flags = set(test_flags(args)) | set((a0["report"] or {}).get("test_flags", []))
    reg_problems = REG.registration_problems()
    # the selection, recomputed from the Stage D rows
    dev_plan = stage_plan("dev", paths, args, a0)
    dev_gens, dev_scores, _ = stage_clips("dev", dev_plan, paths)
    recomputed, sel = selection_now(paths, args, a0) if a0["report"] else (None, None)
    keys = ("s_star", "b_star", "stage_t", "configs", "trim", "dev_table")
    same = bool(sel_doc and recomputed and all(json.dumps(sel_doc.get(k), sort_keys=True) == json.dumps(recomputed.get(k), sort_keys=True)
                                               for k in keys))
    stage_t = bool(sel_doc and sel_doc.get("stage_t"))
    test_plan, test_gens, test_scores, test_vals = [], {}, {}, {}
    if stage_t:
        test_plan = stage_plan("test", paths, args, a0)
        test_gens, test_scores, test_vals = stage_clips("test", test_plan, paths)
    ledger_rows = read_jsonl(paths.ledger("test"))
    head = selection_head(paths)
    file_sha = sha256(paths.selection) if paths.selection.exists() else None
    planned = {r["run_id"] for r in dev_plan} | {r["run_id"] for r in test_plan}
    gen_rows = [g for g in list(dev_gens.values()) + list(test_gens.values()) if g.get("status") == "ok" and g["run_id"] in planned]
    for g in gen_rows:
        flags |= set(g.get("test_flags") or [])
    for s in test_scores.values():
        flags |= set(s.get("test_flags") or [])
    b1rec = json.loads(REG.REG_B1.read_text())["cpu_test"]
    plan_reg = (read(REG.ADDENDUM) or {}).get("plan", {})
    chk = read(paths.setup / "setup_check.json") or {}
    st = {"registration": reg_problems, "registered_code": REG.registered_runtime() if REG.ADDENDUM.exists() else {},
          "registered_guides": plan_reg.get("guides_sha256") or {},
          "setup_hashes_ok": bool(chk) and chk.get("guides") == [] and bool(plan_reg)
          and (chk.get("soundfont") or {}).get("sha256") == plan_reg.get("soundfont_sha256"),
          "gen_rows": gen_rows,
          "selection": {"recomputed_equal": same, "file_sha256": file_sha,
                        "ledger_head": None if head is None else head["selection_sha256"],
                        "row_hashes": [r.get("selection_sha256") for r in ledger_rows if "run_id" in r]},
          "cpu_test": {"passed": b1rec["passed"], "sha256_ok": sha256(ROOT / b1rec["path"]) == b1rec["sha256"]},
          "a0": a0["report"], "dev_plan": dev_plan, "dev_gen": dev_gens, "dev_scores": dev_scores,
          "test_plan": test_plan, "test_gen": test_gens, "test_scores": test_scores}
    p0 = D.integrity(st)
    a0_bad = [g["run_id"] for g in gen_rows if g.get("a0_sha256") != a0["sha256"]]
    a0_code = bool(a0["report"]) and a0["report"].get("code_sha256") == st["registered_code"]
    p0["a0"]["held"] = p0["a0"]["held"] and not a0_bad and a0_code
    p0["a0"]["detail"].update(rows_following_another_a0=a0_bad[:5], ran_on_the_registered_code=a0_code)
    if not stage_t:
        p0["selection"]["held"] = same
        p0.pop("complete")
    p0_ok = all(c["held"] for c in p0.values())
    analysis = None
    if stage_t:
        s_star = P.parse(sel_doc["s_star"])
        clips = {rid: v for rid, v in test_vals.items()}
        analysis = D.analyse(test_plan, clips, s_star, sel_doc.get("dev_futility"))
    verdict, failed = D.verdict(p0_ok, analysis["rule"] if analysis else None, sorted(flags), stage_t)
    labels = analysis["labels"] if analysis else ({"clock_silences_the_voice": True} if sel_doc and not stage_t else {})
    reasons = failed + [f"P0 {k}: {v['detail']}" for k, v in p0.items() if not v["held"]]
    items = P.all_items()
    clips_out = []
    for run in test_plan:
        g = test_gens.get(run["run_id"], {})
        clips_out.append({"run_id": run["run_id"], "item": run["item"], "draw": run["draw"], "config": run["config"],
                          "labels": run["labels"], "status": g.get("status", "missing"), "lyrics": items[run["item"]]["lyrics"],
                          "whisper_text": test_scores.get(run["run_id"], {}).get("whisper_text"),
                          "ctc_text": test_scores.get(run["run_id"], {}).get("ctc_text"), **test_vals.get(run["run_id"], {})})
    result = {"verdict": verdict, "reasons": reasons, "labels": labels, "p0": p0, "p0_held": p0_ok,
              "rule": analysis["rule"] if analysis else None, "test_flags": sorted(flags), "scale": args.scale,
              "selection": {k: (sel_doc or {}).get(k) for k in ("s_star", "b_star", "stage_t", "label", "eligible",
                                                                "gate_waived", "dev_futility", "configs", "trim")},
              "selection_recomputed": None if sel is None else {k: sel.get(k) for k in ("s_star", "b_star", "stage_t")},
              "dev_table": (sel_doc or {}).get("dev_table"), "a0": a0["report"],
              "counts": {"stage_t_planned": len(test_plan),
                         "stage_t_generated": sum(test_gens.get(r["run_id"], {}).get("status") == "ok" for r in test_plan),
                         "stage_t_scored": len(test_scores)},
              "arms": analysis["arms"] if analysis else None, "contrasts": analysis["contrasts"] if analysis else None,
              "primary_comparator": analysis["primary_comparator"] if analysis else None,
              "demo": analysis["demo"] if analysis else [], "reel": analysis["reel"] if analysis else None,
              "clips": clips_out}
    cfg = {"experiment": "b2", "registration": str(REG.ADDENDUM.relative_to(ROOT)), "scale": args.scale,
           "thresholds": {"delta_j": D.DELTA_J_MIN, "j": D.J_MIN, "joint": 0.5, "strict": 0.75,
                          "max_missing_share": D.MAX_MISSING_SHARE, "n_boot": D.N_BOOT, "boot_seed": D.BOOT_SEED},
           "test_flags": sorted(flags)}
    paths.res.mkdir(parents=True, exist_ok=True)
    write_result(paths.res / "b2_result.json", cfg, result, t0)
    short = {"verdict": verdict, "reasons": reasons, "labels": labels,
             "J": {a: v["J"] for a, v in (analysis or {}).get("arms", {}).items()}, "rule": result["rule"]}
    write(paths.res / "b2_summary.json", short)
    print(json.dumps(short, indent=1, default=str), flush=True)
    print("B2:", verdict.upper(), "|", "; ".join(reasons) or "all conditions held", flush=True)


def package(args, paths: Paths) -> None:
    import package_b2

    pkg = Path(args.package) if args.package else paths.root / "package"
    key = Path(args.key) if args.key else paths.root / "b2_blind_key.json"
    pack = Path(args.pack) if args.pack else key.parent / "b2_blind_pack.zip"
    res = package_b2.package(paths, pkg, key, pack, P.all_items())
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "a0", "gen", "score", "select", "decide", "package"])
    ap.add_argument("--part", default="", help="setup: check | model; a0: gen | score | report")
    ap.add_argument("--stage", choices=["dev", "test"], default="dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scale", choices=["full", "tiny"], default="full")
    ap.add_argument("--root", default="")
    ap.add_argument("--guides", default=str(P.GUIDES_DIR))
    ap.add_argument("--only", default="", help="gen: comma list of run ids")
    ap.add_argument("--test-whisper", default=None, help="tiny scale only: score with this Whisper model")
    ap.add_argument("--b1-out", default="", help="setup: B1's output folder, so the space check counts its runs")
    ap.add_argument("--package", default="", help="package: the folder to fill")
    ap.add_argument("--key", default="", help="package: where the blind-pack key goes (never inside the package)")
    ap.add_argument("--pack", default="", help="package: where the blind pack goes (never inside the package)")
    args = ap.parse_args()
    if args.scale == "full" and args.test_whisper:
        print("B2: the registered full scale takes no test Whisper", flush=True)
        sys.exit(2)
    paths = Paths(args.root or str(HERE / ("outputs_b2" if args.scale == "full" else f"outputs_b2_{args.scale}")), args.guides)
    cmds = {("setup", "check"): setup_check, ("setup", "model"): setup_model, ("a0", "gen"): a0_gen,
            ("a0", "score"): a0_score, ("a0", "report"): a0_report}
    fn = cmds.get((args.cmd, args.part)) or {"gen": gen, "score": score, "select": select, "decide": decide,
                                              "package": package}.get(args.cmd)
    if fn is None:
        ap.error(f"{args.cmd} needs --part")
    try:
        fn(args, paths)
    except SetupError as exc:
        print(f"B2 {args.cmd}{' ' + args.part if args.part else ''}: SETUP FAILED: {exc}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
