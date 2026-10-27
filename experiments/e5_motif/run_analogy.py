"""E5 v2 runner (ACE-Step environment): ACE-Step repaint fills the silent gap after three context units.

  python run_analogy.py --phase p0b  --device cuda [--attempt N] [--new-pass]
  python run_analogy.py --phase main --device cuda [--attempt N] [--new-pass]

The runner only generates; it never imports a scorer or a tracker. Every job of the phase is listed in
design.json, written once per attempt, and every attempt of a job is logged in ledger.jsonl (append-only,
flushed and fsynced). A completed job is never generated again, except when its audio is lost before the
quality stage has frozen; no outcome of it was seen, so it is generated again with the same seeds.

A pass tries every job that is not done, has fewer than 3 attempts, and was not yet tried in this pass.
A generation that raises ends the process (exit 75, so a retry never shares a process with ACE-Step's
timed-out service thread); run_all starts it again in the same pass. --new-pass starts an automatic resume
pass, at most two per phase. A start whose process died before its outcome was logged (runtime reset,
kill, a crash outside the generation call) is marked interrupted and not counted as an attempt.
Exit codes: 0 every job done; 4 jobs remain; 75 a generation raised; 2 refused before generating;
5 a finished generation broke a hard per-run check (the attempt is void); 6 halted after five generations
in a row raised (a broken environment, not bad luck); 7 the runtime or addendum differs from the one A0
checked (A0 runs again). Any other exit is a crash. Only 0 and 4 lead to the quality stage.
"""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import v2_config as C  # noqa: E402
from dacapo import motif_integrity as I  # noqa: E402

EXIT_DONE, EXIT_REFUSED, EXIT_PENDING, EXIT_VOID, EXIT_HALT, EXIT_A0_STALE, EXIT_RETRY = 0, 2, 4, 5, 6, 7, 75


def refuse(msg: str) -> int:
    print("E5 v2 runner refused:", msg, flush=True)
    return EXIT_REFUSED


def append(ledger: Path, event: dict) -> None:
    with open(ledger, "a") as fh:
        fh.write(json.dumps(event) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def check_run(job: dict, res: dict) -> list[str]:
    """Hard per-run checks: a failure means the code or configuration is wrong, not bad luck."""
    bad = []
    if not (res["instruction"] == res["instruction_logged"] == job["instruction"]):
        bad.append(f"instruction {res['instruction']!r} / logged {res['instruction_logged']!r}")
    if res["dcw_enabled"] is not False or res["dcw_active_lines"]:
        bad.append("dcw")
    if res["frames"] != C.expected_frames(job["source_samples"]):
        bad.append(f"frames {res['frames']}")
    if res["duration_meta"] != C.expected_duration_meta(job["source_samples"]):
        bad.append(f"duration meta {res['duration_meta']}")
    if res["repaint_cache_used"]:
        bad.append("repaint cache")
    if (res["dit_seed"], res["rng_seed"]) != (job["dit_seed"], job["rng_seed"]):
        bad.append("seeds")
    return bad


def ace_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", os.environ["ACESTEP_REPO"], "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def hf_revision() -> str | None:
    meta = Path(os.environ.get("ACESTEP_CHECKPOINTS_DIR", "")) / ".cache" / "huggingface" / "download" / \
        "acestep-v15-turbo" / "model.safetensors.metadata"
    return meta.read_text().splitlines()[0].strip() if meta.exists() else None


def device_name(device: str) -> str:
    if device != "cuda":
        return "cpu"
    import torch
    return torch.cuda.get_device_name(0)


def build_design(phase: str, attempt: int, scale: str, cell: str | None) -> dict:
    """The phase's full design, deterministic given the committed files (written once per attempt)."""
    e5 = C.e5_dir()
    man_path = C.SCALES[scale][f"{phase}_manifest"]
    man = json.loads((e5 / man_path).read_text())
    st = C.addendum_status()
    excluded = C.k0_excluded_cells(scale)
    jobs = C.jobs(phase, man, scale, cell=cell, excluded_cells=excluded if phase == "p0b" else ())
    srcs = sorted({j["src"] for j in jobs})
    d = {"phase": phase, "scale": scale, "attempt": attempt, "tag": C.SCALES[scale]["tags"][phase], "manifest": man_path,
         "manifest_sha256": C.sha256_file(e5 / man_path), "sources_sha256": {s: I.audio_sha256(e5 / s) for s in srcs},
         "addendum_sha256": st["addendum_sha256"], "seed_table_sha256": C.sha256_file(e5 / "registration" / "e5v2_seeds.json"),
         "ci_method": "melody", "k1_frac": C.K1_FRAC, "q1_frac": C.Q1_FRAC, "q2_frac": C.Q2_FRAC,
         "min_pass_per_cell": C.min_pass_per_cell(len(C.SCALES[scale][f"{phase}_draws"])),
         "repaint": C.REPAINT, "gen": C.GEN, "ace_commit": C.ACE_COMMIT, "hf_revision": C.HF_REVISION, "jobs": jobs}
    if phase == "p0b":
        d.update(arm_order=[c for c in C.CELLS if c in C.SCALES[scale]["p0b_cells"]], excluded_arms=list(excluded),
                 ruler_program=None)
    else:
        i, L, cap = C.cell_parts(cell)
        sel = C.read_json(C.selection_path()) or {}
        p0b_attempt = sel.get("attempt", 1)
        d.update(config={"cell": cell, "instruction": C.INSTRUCTIONS[i], "length": L, "caption": C.CAPTIONS[cap]},
                 ruler_program=C.PROGRAMS[cap], arm_order=["base"],
                 pilot_dir=C.phase_dir("p0b", p0b_attempt).name)
    return d


def preflight(args) -> tuple[int | None, dict | None]:
    """Refusals that come before any generation (exit 2, or 7 when A0 must run again); returns the design."""
    scale = C.scale()
    void = C.read_json(C.void_path())
    if args.attempt == 1 and void and void.get("attempt") == 1 and (void["phase"] == args.phase or args.phase == "main"):
        return refuse("attempt 1 is void; attempt 2 runs after a fix recorded in an amended addendum"), None
    if args.attempt not in (1, 2):
        return refuse("E5 v2 has at most two attempts"), None
    st = C.addendum_status()
    if args.attempt == 2:
        ok, why = C.attempt2_allowed(void, st)
        if not ok:
            return refuse(f"attempt 2: {why}"), None
    if C.terminal_path().exists():
        return refuse(f"terminal outcome exists: {C.terminal_path().read_text()}"), None
    if C.quality_path(args.phase, args.attempt).exists():
        return refuse("the quality stage has frozen this phase; nothing may be generated"), None
    if not st["ok"]:
        return refuse(f"code differs from addendum 2: {st['changed']}"), None
    committed = C.read_json(C.e5_dir() / "registration" / "e5v2_seeds.json")
    if committed != json.loads(json.dumps(C.seed_table())):
        return refuse("seeds differ from registration/e5v2_seeds.json"), None
    if scale == "full":
        if args.device != "cuda":
            return refuse("the registered run uses CUDA"), None
        if C.GPU_REQUIRED not in device_name("cuda"):
            return refuse(f"GPU is {device_name('cuda')}, the registration pins an NVIDIA {C.GPU_REQUIRED}"), None
        if ace_head() != C.ACE_COMMIT:
            return refuse(f"ACE-Step HEAD {ace_head()} is not {C.ACE_COMMIT}"), None
        if hf_revision() != C.HF_REVISION:
            return refuse(f"weights revision {hf_revision()} is not {C.HF_REVISION}"), None
    a0 = C.read_json(C.a0_path())
    if not a0 or not a0.get("go") or a0.get("scale") != scale:
        return refuse("A0 checks have not passed at this scale"), None
    if a0.get("addendum_sha256") != st["addendum_sha256"]:
        print("A0 passed under another addendum 2; A0 must run again", flush=True)
        return EXIT_A0_STALE, None
    cell = None
    if args.phase == "main":
        sel = C.read_json(C.selection_path()) or {}
        if sel.get("decision") == "selected":
            cell = sel["cell"]
        elif scale != "full" and args.cell:
            cell = args.cell
            print(f"{scale} scale: Phase 0b selected no cell ({sel.get('decision')}); running {cell} to exercise the chain")
        else:
            return refuse(f"Phase 0b decision is {sel.get('decision')!r}, not 'selected'"), None
    elif args.cell:
        return refuse("--cell applies to the main phase only"), None
    if args.cell and scale == "full":
        return refuse("--cell is for test scales only"), None
    if scale == "full":
        need = [C.PROGRAMS["flute"]] + ([C.PROGRAMS[C.cell_parts(cell)[2]]] if cell else [])
        failed = [p for p in need if not C.registered_k0(p)["passed"]]
        if failed:
            return refuse(f"the registered K0 has not passed for GM program(s) {failed}"), None
    e5 = C.e5_dir()
    man = json.loads((e5 / C.SCALES[scale][f"{args.phase}_manifest"]).read_text())
    missing = [s for j in C.jobs(args.phase, man, scale, cell=cell or C.CELLS[0]) for s in [j["src"]] if not (e5 / s).exists()]
    if missing:
        return refuse(f"{len(missing)} sources are not rendered (run the setup stage), e.g. {missing[0]}"), None
    design = build_design(args.phase, args.attempt, scale, cell)
    items = {it["id"]: it for it in man["items"]}
    wrong = [j["src"] for j in design["jobs"]
             if design["sources_sha256"][j["src"]] != items[j["id"]]["source_sha256"][j["length"]][j["cond"]]]
    if wrong:
        return refuse(f"{len(set(wrong))} sources differ from the sha256 in the manifest, e.g. {wrong[0]}"), None
    od = C.phase_dir(args.phase, args.attempt)
    old = C.read_json(od / "design.json")
    if old is not None and old != json.loads(json.dumps(design, default=str)):
        return refuse(f"{od / 'design.json'} differs from the design computed now"), None
    return None, design


def part_name(name: str) -> str:
    stem, ext = os.path.splitext(name)
    return f"{stem}.part{ext}"


def repair(od: Path, design: dict, ledger: Path, events: list[dict], pass_no: int, verify: bool = True) -> list[dict]:
    """Before the quality freeze: mark a start whose process died as interrupted, finish or drop half-written
    files, and mark completed jobs whose audio is gone or changed as lost, so they are generated again with
    the same seeds. verify: hash every finished file (the first process of a run_all call); otherwise
    compare sizes only (a relaunch in the same runtime). The quality stage hashes every file again."""
    for e in I.open_starts(events):
        append(ledger, {"event": "interrupted", **{f: e[f] for f in I.JOB_FIELDS}, "pass": e.get("pass", 1),
                        "utc": C.utc()})
    st, _ = I.ledger_state(I.read_ledger(ledger))
    jobs = {I.audio_name(j): j for j in design["jobs"]}
    orphans = od / "_orphans"
    for part in od.glob(f"*.part{I.AUDIO_EXT}"):
        name = part.name.replace(f".part{I.AUDIO_EXT}", I.AUDIO_EXT)
        j = jobs.get(name)
        done = st.get(I.job_key(j), {}).get("done", []) if j else []
        if done and not (od / name).exists() and I.sha256_file(part) == done[-1]["wav_sha256"]:
            os.replace(part, od / name)
        else:
            part.unlink()
    for f in od.glob(f"*{I.AUDIO_EXT}"):
        j = jobs.get(f.name)
        done = st.get(I.job_key(j), {}).get("done", []) if j else []
        if not done:
            orphans.mkdir(exist_ok=True)
            dst = orphans / f"{f.stem}.{int(time.time())}{I.AUDIO_EXT}"
            os.replace(f, dst)
            append(ledger, {"event": "orphan", "file": f.name, "moved_to": str(dst.relative_to(od)),
                            "pass": pass_no, "utc": C.utc(), **({k: j[k] for k in I.JOB_FIELDS} if j else {})})

    def same(f: Path, done: dict) -> bool:
        if not verify and done.get("wav_bytes") is not None:
            return f.stat().st_size == done["wav_bytes"]
        return I.sha256_file(f) == done["wav_sha256"]

    for name, j in jobs.items():
        done = st.get(I.job_key(j), {}).get("done", [])
        if not done:
            continue
        f = od / name
        reason = None if f.exists() and same(f, done[-1]) else \
            ("audio missing" if not f.exists() else "audio differs from the ledger")
        if reason:
            if f.exists():
                orphans.mkdir(exist_ok=True)
                os.replace(f, orphans / f"{f.stem}.{int(time.time())}{I.AUDIO_EXT}")
            append(ledger, {"event": "lost", "reason": reason, "pass": pass_no, "utc": C.utc(),
                            **{k: j[k] for k in I.JOB_FIELDS}})
    return I.read_ledger(ledger)


def local_tmp(od: Path) -> Path:
    """One scratch folder per phase folder on the local disk, emptied by every process that uses it, so a
    killed process leaves at most one generation behind."""
    tag = hashlib.sha256(str(od.resolve()).encode()).hexdigest()[:12]
    tmp = Path(tempfile.gettempdir()) / f"e5v2_{tag}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    return tmp


def space_shortfall(od: Path, n_todo: int, phase: str) -> str | None:
    """Refuse before generating when the disk that holds the run state (Google Drive on Colab) cannot take
    the remaining jobs; at Phase 0b the confirmatory grid is counted too. A disk that reports no size is
    not judged."""
    later = len(C.jobs("main", json.loads((C.e5_dir() / C.SCALES[C.scale()]["main_manifest"]).read_text()),
                       C.scale(), cell=C.CELLS[0])) if phase == "p0b" else 0
    need = (n_todo + later) * C.BYTES_PER_JOB + C.SPACE_MARGIN
    du = shutil.disk_usage(od)
    if du.total == 0 or du.free >= need:
        return None
    return f"{du.free / 1e9:.1f} GB free where the run state lives, {need / 1e9:.1f} GB needed"


def comparable(facts: dict) -> dict:
    return json.loads(json.dumps({k: v for k, v in facts.items() if k != "setup"}, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["p0b", "main"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--new-pass", action="store_true")
    ap.add_argument("--verify-audio", action="store_true", help="hash every finished file before generating")
    ap.add_argument("--cell", default="", help="test scales only: the main-phase cell when Phase 0b selected none")
    args = ap.parse_args()
    # cuBLAS reads this when CUDA starts, and preflight's device check starts CUDA
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rc, design = preflight(args)
    if rc is not None:
        return rc
    od = C.phase_dir(args.phase, args.attempt)
    od.mkdir(parents=True, exist_ok=True)
    if not (od / "design.json").exists():
        C.write_json(od / "design.json", design)
        print(f"E5 v2 {args.phase} attempt {args.attempt} starts: design.json sha256 {C.sha256_file(od / 'design.json')}",
              flush=True)
    ledger = od / "ledger.jsonl"
    events = I.read_ledger(ledger)
    cur = I.current_pass(events)
    _, n_resume = I.ledger_state(events)
    if cur == 0:
        append(ledger, {"event": "pass", "pass": 1, "utc": C.utc()})
        cur = 1
    elif args.new_pass:
        if n_resume >= C.MAX_RESUME_PASSES:
            return refuse(f"{C.MAX_RESUME_PASSES} automatic resume passes were already made")
        cur += 1
        append(ledger, {"event": "resume", "pass": cur, "utc": C.utc()})
        append(ledger, {"event": "pass", "pass": cur, "utc": C.utc()})
    else:
        append(ledger, {"event": "restart", "pass": cur, "utc": C.utc()})
    events = repair(od, design, ledger, I.read_ledger(ledger), cur, verify=args.verify_audio)
    errs = I.trailing_errors(events)
    if len(errs) >= C.HALT_AFTER_ERRORS:
        append(ledger, {"event": "halt", "pass": cur, "utc": C.utc(),
                        "jobs": [{**{f: e[f] for f in I.JOB_FIELDS}, "pass": e.get("pass", 1)} for e in errs],
                        "errors": sorted({e.get("error", "")[:300] for e in errs})})
        print(f"generation halted: {len(errs)} generations in a row raised, e.g. {errs[-1].get('error', '')[:300]}",
              flush=True)
        return EXIT_HALT
    st, _ = I.ledger_state(events)
    dev = device_name(args.device)
    prior = [e.get("device") for e in events if e.get("event") == "start" and e.get("device")]
    if prior and prior[0] != dev:
        return refuse(f"this phase started on {prior[0]}, now {dev}")

    def runnable(j):
        s = st.get(I.job_key(j), {})
        return not s.get("done") and s.get("starts", 0) < C.MAX_STARTS and not s.get("starts_by_pass", {}).get(cur, 0)

    todo = [j for j in design["jobs"] if runnable(j)]
    if todo:
        short = space_shortfall(od, len(todo), args.phase)
        if short:
            return refuse(short)
        from dacapo import ace
        det = ace.setup_determinism()
        if os.environ.get("ACESTEP_VAE_DECODE_CHUNK_SIZE") != "256":
            return refuse("ACESTEP_VAE_DECODE_CHUNK_SIZE is not 256")
        h = ace.load(args.device)
        facts = ace.runtime_facts(h)
        a0_facts = comparable((C.read_json(C.a0_path()) or {}).get("facts") or {})
        if comparable(facts) != a0_facts:
            diff = sorted(k for k in set(a0_facts) | set(comparable(facts)) if a0_facts.get(k) != comparable(facts).get(k))
            print(f"the runtime differs from the one A0 checked ({diff}); A0 must run again", flush=True)
            return EXIT_A0_STALE
        C.write_json(od / f"provenance_pass{cur}_{int(time.time())}.json", {"facts": facts, "determinism": det,
                                                                           "ace_head": ace_head(), "hf_revision": hf_revision()})
        tmp = local_tmp(od)
        out = tmp / f"out{I.AUDIO_EXT}"
        try:
            for j in todo:
                key = {f: j[f] for f in I.JOB_FIELDS}
                append(ledger, {"event": "start", **key, "pass": cur, "device": dev, "utc": C.utc()})
                g = dict(C.GEN)
                t0 = time.time()
                try:
                    res = ace.generate(h, out, j["caption"], g.pop("duration"), j["dit_seed"], steps=g.pop("steps"),
                                       shift=g.pop("shift"), src=str(C.e5_dir() / j["src"]),
                                       extra={**C.REPAINT, "instruction": j["instruction"]}, rng_seed=j["rng_seed"],
                                       out_subtype=C.AUDIO_SUBTYPE, **g)
                except BaseException as e:  # noqa: BLE001  (ACE-Step's TimeoutError included)
                    append(ledger, {"event": "error", **key, "pass": cur, "error": repr(e)[:1000], "utc": C.utc()})
                    sys.stdout.flush()
                    shutil.rmtree(tmp, ignore_errors=True)
                    os._exit(EXIT_RETRY)
                facts_run = {k: v for k, v in res.items() if k not in ("trace", "captured", "extra")}
                bad = check_run(j, res)
                name = I.audio_name(j)
                if bad:
                    shutil.copyfile(out, od / f"{Path(name).stem}.check_failed")
                    append(ledger, {"event": "check_failed", **key, "pass": cur, "failures": bad, "facts": facts_run,
                                    "utc": C.utc()})
                    print(f"HARD CHECK FAILED {name}: {bad}", flush=True)
                    return EXIT_VOID
                part = od / part_name(name)
                shutil.copyfile(out, part)
                with open(part, "rb") as fh:
                    os.fsync(fh.fileno())
                append(ledger, {"event": "done", **key, "pass": cur, "wav_sha256": I.sha256_file(part),
                                "wav_bytes": part.stat().st_size, "seconds": round(time.time() - t0, 2), "device": dev,
                                "attn_implementation": facts.get("attn_implementation"), "facts": facts_run, "utc": C.utc()})
                os.replace(part, od / name)
                print(f"{name:>45s} {res['seconds']:6.1f}s", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    st, _ = I.ledger_state(I.read_ledger(ledger))
    pending = [j for j in design["jobs"] if not st.get(I.job_key(j), {}).get("done")]
    print(f"phase {args.phase} attempt {args.attempt} pass {cur}: {len(design['jobs']) - len(pending)} done, "
          f"{len(pending)} pending", flush=True)
    return EXIT_DONE if not pending else EXIT_PENDING


if __name__ == "__main__":
    sys.exit(main())
