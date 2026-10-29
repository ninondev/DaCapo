"""B1 of Run B: does the lyric clock move sung words onto their written notes? Text2music on the 12 E4
lines, no melody guide. The plan and the rule are registered in experiments/e5_motif/registration/
addendum_2.json ("e6_readout", "B1, fixed now"); this code, its tests, the CPU test and the details it
fills in are registered in addendum_2_b1code.json.

  python run_b1.py plan                      the 180 planned runs, in the order item, draw, arm
  python run_b1.py gen --device cuda         ACE env: a WAV, pred_latents and a ledger row per run
  python run_b1.py score --device cuda       main env: vocal gate and Whisper word chunks
  python run_b1.py decide                    the registered rule -> b1_result.json
  --scale tiny                               two runs with their own seeds (tag runB1-tiny), any device, test
                                             flags allowed; the verdict is "test only"

All arms of a draw share (dit_seed, rng_seed) = (63000 + 10 item + r, + 900000): no hook (beta 0);
beta 1, 2 and 4 on the seven alignment heads; beta 2 on registered random set r for draw r. The hooks
add beta wherever a query frame's centre lies in a word's window (written onset to the next onset, the
last word to its note end) and the key is one of that word's tokens, in layers 2-6 at all 8 steps.

The rule, at beta 2, with seconds and reductions rounded to 1e-6 before every comparison:
  1  a reduction of at least 30% in at least 9 of the 12 items
  2  recall and vocal-gate rate over the arm's 36 clips each no more than 0.10 below beta 0
  3  the random arm's median reduction below half of beta 2's
Beta 1 and 4 are dose-response only. A clip's error is the median |Whisper onset - written onset| over
its matched words (evaluable clips only); an item's error is the median over its evaluable draws, and
reduction = 1 - error(arm) / error(beta 0). An item with no evaluable draw in the arm or at beta 0 is not
improved: it fails the 30% count and enters the medians with reduction 0. Integrity can block a verdict
("technical") but never create one.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
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
import readout as R  # noqa: E402

REG2 = ROOT / "experiments" / "e5_motif" / "registration" / "addendum_2.json"
REG_B1 = ROOT / "experiments" / "e5_motif" / "registration" / "addendum_2_b1code.json"
B1_CODE = ("dacapo/attn_bias.py", "experiments/e6_clock/clock_score.py", "experiments/e6_clock/run_b1.py")
E4_MANIFEST = ROOT / "experiments" / "e4_sing" / "manifest.json"
GATE_RES = ROOT / "results" / "gpu_run2"   # Run A's E6 gate
OUT = HERE / "outputs_b1"
RES = HERE / "results_b1"

SEED_BASE, N_DRAWS = 63000, 3
ARMS = (("nohook", 0.0, None), ("align1", 1.0, "align"), ("align2", 2.0, "align"), ("align4", 4.0, "align"),
        ("random2", 2.0, "random"))
PRIMARY, RANDOM_ARM, BASE = "align2", "random2", "nohook"
DOSE = ("align1", "align4")
TINY = ("item00_r0_nohook", "item00_r0_align2")
TINY_TAG = "runB1-tiny"      # the tiny scale's own seeds, so it never generates a registered B1 run
REDUCTION_MIN, ITEMS_MIN = 0.30, 9
GUARD = 0.10                 # recall and gate rate may fall at most this much below beta 0
RANDOM_FRACTION = 0.5        # random arm's median reduction must stay below this share of beta 2's
MAX_INCOMPLETE = 9           # 5% of 180 planned runs
HOOK_CALLS = 40              # 5 layers x 8 steps
DIGITS = R.DIGITS


class SetupError(RuntimeError):
    """A problem that affects every run (exit before any generation or scoring)."""


def rnd(x: float) -> float:
    return round(float(x), DIGITS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# plan

def manifest() -> list[dict]:
    return json.loads(E4_MANIFEST.read_text())["items"]


def plan(scale: str = "full") -> list[dict]:
    sets = R.random_head_sets()[:N_DRAWS]
    out = []
    for i, it in enumerate(manifest()):
        for r in range(N_DRAWS):
            seed = SEED_BASE + 10 * i + r
            for arm, beta, kind in ARMS:
                heads = {"align": R.ALIGN, "random": sets[r]}.get(kind)
                out.append({"run_id": f"{it['id']}_r{r}_{arm}", "item": it["id"], "draw": r, "arm": arm, "beta": beta,
                            "head_set": kind if kind != "random" else f"random {r}",
                            "heads": None if heads is None else {str(L): list(hs) for L, hs in heads.items()},
                            "dit_seed": seed, "rng_seed": seed + R.RNG_OFFSET, "lyrics": it["lyrics"],
                            "caption": R.CAPTION_F, "duration": float(it["duration"]), "bpm": int(it["melody"]["bpm"]),
                            "keyscale": it["key"]})
    assert len(out) == 180 and len({x["run_id"] for x in out}) == 180
    if scale != "tiny":
        return out
    from dacapo.seeds import run_seeds
    tiny = [x for x in out if x["run_id"] in TINY]
    for x in tiny:
        x["dit_seed"], x["rng_seed"] = run_seeds(TINY_TAG, x["item"], x["draw"])
    return tiny


def written_onsets(it: dict) -> list[float]:
    spb = 60.0 / float(it["melody"]["bpm"])
    return [rnd(n["start"] * spb) for n in it["melody"]["notes"]]


# integrity

def code_state() -> dict:
    """The eight E6 files against addendum 2 and the B1 code against the B1 code addendum."""
    e6, reg_e6 = R.code_hashes(), R.registered_hashes(REG2)
    b1 = {f: sha256(ROOT / f) for f in B1_CODE}
    try:
        reg = json.loads(REG_B1.read_text())
        reg_b1 = {f: reg["code_sha256"].get(f) for f in B1_CODE}
        reg_add2 = reg.get("addendum_2_sha256")
    except (OSError, KeyError, ValueError):
        reg_b1, reg_add2 = None, None
    problems = []
    if reg_e6 is None or e6 != reg_e6:
        problems.append("the E6 files differ from addendum 2: "
                        f"{sorted(f for f in e6 if reg_e6 is None or e6[f] != reg_e6.get(f))}")
    if reg_b1 is None:
        problems.append(f"no B1 code addendum ({REG_B1.relative_to(ROOT)})")
    elif b1 != reg_b1:
        problems.append(f"the B1 code differs from the B1 code addendum: {sorted(f for f in b1 if b1[f] != reg_b1[f])}")
    if reg_add2 is not None and sha256(REG2) != reg_add2:
        problems.append("addendum_2.json differs from the copy the B1 code addendum names")
    return {"e6_sha256": e6, "b1_sha256": b1, "registered": not problems, "problems": problems}


def gate_check() -> tuple[bool, str]:
    """readout.py check-gate on Run A's gate (E6_RES=results/gpu_run2)."""
    r = subprocess.run([sys.executable, str(HERE / "readout.py"), "check-gate"], capture_output=True, text=True,
                       env={**os.environ, "E6_RES": str(GATE_RES)})
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def test_flags(args) -> list[str]:
    flags = []
    if args.scale != "full":
        flags.append(f"scale {args.scale}")
    if getattr(args, "unregistered", False):
        flags.append("--unregistered")
    if getattr(args, "test_whisper", None):
        flags.append(f"--test-whisper {args.test_whisper}")
    return flags


def refuse_full_scale_tests(args, needs_cuda: bool) -> None:
    if args.scale == "full" and (test_flags(args) or (needs_cuda and args.device != "cuda")):
        raise SetupError("the registered full scale runs on CUDA without test flags")


# ledgers (append-only JSON lines)

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # a line cut short by a crash
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell():
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((json.dumps(row) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def generated(out: Path) -> dict[str, dict]:
    """run_id -> its last ledger row (ok rows only count when the WAV and latents still hash to it)."""
    last: dict[str, dict] = {}
    for row in read_jsonl(out / "ledger.jsonl"):
        last[row["run_id"]] = row
    for rid, row in last.items():
        if row.get("status") == "ok" and not files_match(out, row):
            last[rid] = {**row, "status": "files_missing"}
    return last


def files_match(out: Path, row: dict) -> bool:
    wav, lat = out / f"{row['run_id']}.wav", out / f"{row['run_id']}.latents.npy"
    return wav.exists() and lat.exists() and sha256(wav) == row.get("wav_sha256") and sha256(lat) == row.get("latents_sha256")


def failed_attempts(out: Path) -> dict[str, int]:
    n: dict[str, int] = {}
    for r in read_jsonl(out / "ledger.jsonl"):
        if r.get("status") == "failed":
            n[r["run_id"]] = n.get(r["run_id"], 0) + 1
    return n


# gen (ACE env)

def run_one(h, run: dict, lay: dict, spans: list, out: Path, ctx: dict) -> dict:
    """One planned run. Raises on any technical problem; the caller records the attempt."""
    import torch

    from dacapo import ace
    from dacapo.attn_bias import CrossAttnBias

    t0 = time.time()
    bias = None
    if run["heads"] is not None:
        bias = CrossAttnBias(h.model.decoder, {int(L): hs for L, hs in run["heads"].items()}, spans, run["beta"],
                             lay["n_lyr"])
    wav = out / f"{run['run_id']}.wav"
    with bias if bias is not None else contextlib.nullcontext():
        r = ace.generate(h, wav, run["caption"], run["duration"], run["dit_seed"], lyrics=run["lyrics"], steps=R.STEPS,
                         shift=R.SHIFT, bpm=run["bpm"], keyscale=run["keyscale"], vocal_language="en",
                         extra={"dcw_enabled": False}, keep_extra=True, rng_seed=run["rng_seed"],
                         out_subtype=R.WAV_SUBTYPE)
    ex = r["extra"]
    if ex.get("pred_latents") is None or ex.get("lyric_token_idss") is None:
        raise RuntimeError("ACE-Step returned no pred_latents or lyric ids: is ACESTEP_SAVE_MEMORY set?")
    if r["n_denoiser_calls"] != R.STEPS or r.get("dcw_enabled") is not False:
        raise RuntimeError(f"{r['n_denoiser_calls']} denoiser calls (expected {R.STEPS}), DCW {r.get('dcw_enabled')}")
    if [int(i) for i in list(ex["lyric_token_idss"][0])] != lay["ids"]:
        raise RuntimeError("the lyric ids of the run differ from the ids the windows were built on")
    if not bool((ex["encoder_attention_mask"] == 1).all()):
        raise RuntimeError("encoder attention mask is not all ones")
    lat = out / f"{run['run_id']}.latents.npy"
    np.save(lat, ex["pred_latents"][0].float().cpu().numpy())
    rec = bias.record() if bias is not None else {"hook_calls": 0, "lq_lk": [], "max_key": None, "cells": None,
                                                  "mask_in": None}
    row = {k: run[k] for k in ("run_id", "item", "draw", "arm", "beta", "head_set", "heads", "dit_seed", "rng_seed")}
    row.update(status="ok", hook_calls=rec["hook_calls"], lq_lk=rec["lq_lk"], max_key=rec["max_key"], n_lyr=lay["n_lyr"],
               cells=rec["cells"], mask_in=rec["mask_in"], peak_before_norm=r.get("peak_before_norm"),
               n_denoiser_calls=r["n_denoiser_calls"], dcw_enabled=r.get("dcw_enabled"), frames=r["frames"],
               decode_chunk=r.get("decode_chunk"),
               cpu_decode_lines=r.get("cpu_decode_lines"), oom_lines=r.get("oom_lines"),
               determinism_warnings=r.get("determinism_warnings"), gen_seconds=r["seconds"],
               total_seconds=round(time.time() - t0, 2), wav_sha256=sha256(wav), latents_sha256=sha256(lat), **ctx)
    del ex
    if ctx["device"] == "cuda":
        torch.cuda.empty_cache()
    return row


def gen(args) -> None:
    refuse_full_scale_tests(args, needs_cuda=True)
    flags = test_flags(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [x for x in plan(args.scale) if not args.only or x["run_id"] in args.only.split(",")]
    done = {rid for rid, row in generated(out).items() if row.get("status") == "ok"}
    fails = failed_attempts(out)
    todo = [x for x in todo if x["run_id"] not in done and fails.get(x["run_id"], 0) < R.MAX_ATTEMPTS]
    if not todo:
        print("B1 gen: every planned run is generated or has used its attempts", flush=True)
        return
    from dacapo import ace
    from dacapo.attn_bias import lyric_keys, word_spans
    from acestep.core.scoring.dit_alignment import MusicStampsAligner

    state = code_state()
    if not state["registered"] and "--unregistered" not in flags:
        raise SetupError("; ".join(state["problems"]))
    gate_ok, gate_msg = gate_check()
    if not gate_ok:
        raise SetupError(f"readout.py check-gate refuses B1: {gate_msg}")
    det = ace.setup_determinism()
    h = ace.load(args.device)
    if {int(k): list(v) for k, v in h.custom_layers_config.items()} != R.ALIGN:
        raise SetupError(f"unexpected alignment heads {h.custom_layers_config}")
    aligner = MusicStampsAligner(h.text_tokenizer)
    items = {it["id"]: it for it in manifest()}
    lays, spans = {}, {}
    for iid, it in items.items():
        lays[iid] = lyric_keys(h, aligner, it["lyrics"])
        spans[iid] = word_spans(it["melody"]["notes"], it["melody"]["bpm"], lays[iid]["keys"])
    ctx = {"code_sha256": {**state["e6_sha256"], **state["b1_sha256"]}, "registered": state["registered"],
           "gate_ok": gate_ok, "test_flags": flags, "scale": args.scale, "device": args.device}
    (out / "b1_setup.json").write_text(json.dumps({"runtime": ace.runtime_facts(h), "determinism": det, "gate": gate_msg,
                                                    "code": state, **R.git_state(), "layouts": lays,
                                                    "spans": {k: [[list(a), b, c] for a, b, c in v] for k, v in spans.items()},
                                                    "test_flags": flags, "scale": args.scale}, indent=1))
    ledger = out / "ledger.jsonl"
    for run in todo:
        for attempt in range(fails.get(run["run_id"], 0) + 1, R.MAX_ATTEMPTS + 1):
            try:
                row = run_one(h, run, lays[run["item"]], spans[run["item"]], out, ctx)
                append_jsonl(ledger, {**row, "attempt": attempt})
                print(f"{run['run_id']:>22s} {row['gen_seconds']:5.1f}s calls {row['hook_calls']:2d} "
                      f"Lq/Lk {row['lq_lk']} peak {row['peak_before_norm']}", flush=True)
                break
            except Exception as exc:  # recorded per run; the stage goes on so that score and decide still run
                append_jsonl(ledger, {"run_id": run["run_id"], "status": "failed", "attempt": attempt, "error": repr(exc),
                                      "traceback": traceback.format_exc(), **ctx})
                print(f"{run['run_id']:>22s} attempt {attempt} FAILED: {exc!r}", flush=True)
                if args.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()


# score (main env)

def score(args) -> None:
    import clock_score as CS

    refuse_full_scale_tests(args, needs_cuda=True)
    out, res = Path(args.out), Path(args.res)
    path = res / "b1_scores.jsonl"
    last = {r["run_id"]: r for r in read_jsonl(path)}
    gens = generated(out)
    want = args.test_whisper or CS.WHISPER

    def current(old: dict | None, g: dict) -> bool:
        return bool(old and "error" not in old and old.get("wav_sha256") == g["wav_sha256"] and old.get("whisper") == want
                    and (args.test_whisper or old.get("whisper_revision") == CS.WHISPER_REVISION))
    todo = [run for run in plan(args.scale) if gens.get(run["run_id"], {}).get("status") == "ok"
            and not current(last.get(run["run_id"]), gens[run["run_id"]])]
    if not todo:
        print("B1 score: every generated run is scored", flush=True)
        return
    try:
        sc = CS.Scorer(args.device, test_whisper=args.test_whisper)
    except RuntimeError as exc:
        raise SetupError(str(exc)) from exc
    for run in todo:
        g = gens[run["run_id"]]
        try:
            row, _ = sc.measure(out / f"{run['run_id']}.wav")
        except Exception as exc:  # one bad clip must not stop the stage; decide counts it as unscored
            row = {"error": repr(exc), "traceback": traceback.format_exc(), "whisper": sc.whisper,
                   "whisper_revision": sc.whisper_revision, "test_flags": list(sc.test_flags)}
        row.update(run_id=run["run_id"], wav_sha256=g["wav_sha256"], scale=args.scale)
        append_jsonl(path, row)
        print(f"{run['run_id']:>22s} " + (f"SCORE FAILED: {row['error']}" if "error" in row else
                                          f"vocals {row['stem_share']['vocals']:.3f} words {len(row['whisper_chunks'])}"),
              flush=True)


# decide (numpy only)

def clip_rows(scale: str, gens: dict, scores: dict) -> list[dict]:
    import clock_score as CS
    from dacapo.words import lyric_words

    items = {it["id"]: it for it in manifest()}
    rows = []
    for run in plan(scale):
        it, g, s = items[run["item"]], gens.get(run["run_id"]), scores.get(run["run_id"])
        base = {k: run[k] for k in ("run_id", "item", "draw", "arm", "beta", "head_set")}
        if g is None or g.get("status") != "ok":
            status = "missing" if g is None else ("failed" if g.get("status") == "failed" else g.get("status"))
            rows.append({**base, "status": status, **CS.clip(None, None, lyric_words(it["lyrics"]), written_onsets(it))})
            continue
        if s is not None and s.get("wav_sha256") != g["wav_sha256"]:
            s = None  # a score of another file: unscored
        c = CS.clip(s, g.get("peak_before_norm"), lyric_words(it["lyrics"]), written_onsets(it))
        rows.append({**base, "status": "ok", **c, "peak_before_norm": g.get("peak_before_norm"),
                     "hook_calls": g.get("hook_calls"), "lq_lk": g.get("lq_lk"), "max_key": g.get("max_key"),
                     "n_lyr": g.get("n_lyr"), "code_sha256": g.get("code_sha256"), "registered": g.get("registered"),
                     "gen_test_flags": g.get("test_flags", []),
                     "score_test_flags": [] if s is None else s.get("test_flags", []),
                     "whisper": None if s is None else s.get("whisper"),
                     "whisper_revision": None if s is None else s.get("whisper_revision"),
                     "whisper_text": None if s is None else s.get("whisper_text")})
    return rows


def item_table(rows: list[dict]) -> dict:
    """{arm: {item: {"error": median over evaluable draws or None, "n_evaluable": ...}}}"""
    out: dict = {}
    for r in rows:
        cell = out.setdefault(r["arm"], {}).setdefault(r["item"], {"errors": []})
        if r["evaluable"]:
            cell["errors"].append(r["err_median"])
    for arm in out.values():
        for cell in arm.values():
            cell["n_evaluable"] = len(cell["errors"])
            cell["error"] = rnd(np.median(cell["errors"])) if cell["errors"] else None
    return out


def reductions(table: dict, arm: str, items: list[str]) -> list[dict]:
    """Per item: reduction = 1 - error(arm) / error(beta 0); not improved (reduction 0) without an error
    in either arm, or when the beta 0 error is 0."""
    out = []
    for i in items:
        e, e0 = table.get(arm, {}).get(i, {}).get("error"), table.get(BASE, {}).get(i, {}).get("error")
        if e is None or e0 is None or e0 == 0:
            out.append({"item": i, "reduction": 0.0, "not_improved": True, "improved": False, "error": e, "error_b0": e0})
        else:
            red = rnd(1 - e / e0)
            out.append({"item": i, "reduction": red, "not_improved": False, "improved": red >= REDUCTION_MIN,
                        "error": e, "error_b0": e0})
    return out


def arm_summary(rows: list[dict], table: dict, arm: str, items: list[str]) -> dict:
    """Recall and gate rate are means over the arm's planned clips (a missing, failed or unscored clip
    counts as recall 0 and gate failed)."""
    clips = [r for r in rows if r["arm"] == arm]
    s = {"n_planned": len(clips), "recall": rnd(np.mean([r["recall"] for r in clips])) if clips else None,
         "gate_rate": rnd(np.mean([r["vocal"] for r in clips])) if clips else None,
         "n_evaluable": sum(r["evaluable"] for r in clips),
         "item_errors": {i: table.get(arm, {}).get(i, {}).get("error") for i in items}}
    if arm != BASE:
        red = reductions(table, arm, items)
        s["items"] = red
        s["n_improved"] = sum(x["improved"] for x in red)
        s["median_reduction"] = rnd(np.median([x["reduction"] for x in red])) if red else None
        s["n_not_improved_undefined"] = sum(x["not_improved"] for x in red)
    return s


def rule(arms: dict) -> dict:
    p, b, rd = arms[PRIMARY], arms[BASE], arms[RANDOM_ARM]
    c1 = p["n_improved"] >= ITEMS_MIN
    d_recall, d_gate = rnd(p["recall"] - b["recall"]), rnd(p["gate_rate"] - b["gate_rate"])
    c2 = d_recall >= -GUARD and d_gate >= -GUARD
    half = rnd(RANDOM_FRACTION * p["median_reduction"])
    c3 = rd["median_reduction"] < half
    return {"c1_items_improved": {"held": c1, "n_improved": p["n_improved"], "needed": ITEMS_MIN},
            "c2_recall_and_gate": {"held": c2, "recall_change": d_recall, "gate_rate_change": d_gate, "max_drop": GUARD},
            "c3_random_heads": {"held": c3, "random_median_reduction": rd["median_reduction"],
                                "half_of_primary": half, "primary_median_reduction": p["median_reduction"]}}


def integrity(rows: list[dict], state: dict, gate: tuple[bool, str]) -> list[str]:
    out = list(state["problems"])
    reg = {**(R.registered_hashes(REG2) or {}), **state["b1_sha256"]} if state["registered"] else None
    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r["run_id"] for r in ok if reg is None or r.get("code_sha256") != reg or not r.get("registered")]
    if bad:
        out.append(f"{len(bad)} runs generated with code other than the registered code: {bad[:5]}")
    if not gate[0]:
        out.append(f"readout.py check-gate refuses B1: {gate[1]}")
    wrong = [r["run_id"] for r in rows if r["scored"] and not r.get("registered_reference")]
    if wrong:
        out.append(f"{len(wrong)} clips scored with another reference than {R.WHISPER}@{R.WHISPER_REVISION[:7]}")
    nopeak = [r["run_id"] for r in ok if r.get("peak_before_norm") is None]
    if nopeak:
        out.append(f"pre-normalisation peak missing in {len(nopeak)} clips")
    incomplete = sum(not r["scored"] for r in rows)
    if incomplete > MAX_INCOMPLETE:
        out.append(f"incomplete: {incomplete} of {len(rows)} planned runs missing, failed or unscored")
    calls = [r["run_id"] for r in ok if r["hook_calls"] != (0 if r["arm"] == BASE else HOOK_CALLS)
             or (r["arm"] != BASE and not (r.get("max_key") is not None and r["max_key"] < r["n_lyr"]))]
    if calls:
        out.append(f"{len(calls)} runs with the wrong number of hook calls or a biased key outside the lyric: {calls[:5]}")
    return out


def bootstrap(table: dict, items: list[str]) -> dict:
    """Descriptive 95% intervals over items (10,000 resamples, seed 0): median reduction and items improved."""
    rng = np.random.default_rng(R.BOOT_SEED)
    per = {arm: np.array([x["reduction"] for x in reductions(table, arm, items)]) for arm, _, _ in ARMS if arm != BASE}
    imp = {arm: np.array([x["improved"] for x in reductions(table, arm, items)]) for arm, _, _ in ARMS if arm != BASE}
    idx = rng.integers(0, len(items), size=(R.N_BOOT, len(items)))
    out = {}
    for arm in per:
        med = np.median(per[arm][idx], axis=1)
        n = imp[arm][idx].sum(axis=1)
        out[arm] = {"median_reduction": [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
                    "n_improved": [float(np.percentile(n, 2.5)), float(np.percentile(n, 97.5))]}
    return out


def exploratory(rows: list[dict], table: dict, items: list[str]) -> dict:
    """The parts of addendum 2's exploratory list that apply to B1, per arm."""
    out = {}
    for arm, _, _ in ARMS:
        ev = [r for r in rows if r["arm"] == arm and r["evaluable"]]
        out[arm] = {"n_evaluable": len(ev),
                    "within_025_of_written": rnd(np.mean([r["within_025"] for r in ev])) if ev else None,
                    "err_signed_median": rnd(np.median([r["err_signed_median"] for r in ev])) if ev else None,
                    "err_median_excl_first_word": rnd(np.median([r["err_median_excl_first"] for r in ev
                                                                 if r["err_median_excl_first"] is not None])) if ev else None}
    return {"per_arm": out, "ci95_items": bootstrap(table, items) if len(items) > 1 else None,
            "not_applicable": ["re-noising seeds 43 and 44, the read-out at each sampling step, single heads, the rank "
                               "among random sets: B1 measures sung onsets with Whisper and reads no attention",
                               "MMS forced alignment: B1's registered reference is Whisper alone",
                               "results by set and voice: B1 has one set (the E4 lines) and one voice"]}


def decide(args) -> None:
    from dacapo.provenance import write_result

    t0 = time.time()
    out, res = Path(args.out), Path(args.res)
    gens = generated(out)
    scores = {r["run_id"]: r for r in read_jsonl(res / "b1_scores.jsonl")}
    rows = clip_rows(args.scale, gens, scores)
    items = sorted({r["item"] for r in rows})
    table = item_table(rows)
    arms = {arm: arm_summary(rows, table, arm, items) for arm, _, _ in ARMS if any(r["arm"] == arm for r in rows)}
    state, gate = code_state(), gate_check()
    flags = sorted(set(test_flags(args)) | {f for r in rows for f in r.get("gen_test_flags", []) + r.get("score_test_flags", [])})
    problems = integrity(rows, state, gate)
    cond = rule(arms) if all(a in arms for a in (PRIMARY, BASE, RANDOM_ARM)) else None
    held = cond is not None and all(c["held"] for c in cond.values())
    if flags:
        verdict = "test only"
    elif problems or cond is None:
        verdict = "technical"
    else:
        verdict = "supported" if held else "not supported"
    reasons = [f"test run ({', '.join(flags)})"] if flags else []
    reasons += problems
    if cond is not None:
        reasons += [f"{k} failed" for k, c in cond.items() if not c["held"]]
    labels = []
    if cond is not None and verdict in ("supported", "not supported"):
        labels.append("words moved toward their notes" if cond["c1_items_improved"]["held"] else "timing not moved in 9 items")
        labels.append("recall and voice kept" if cond["c2_recall_and_gate"]["held"] else "recall or voice lost")
        labels.append("head-specific" if cond["c3_random_heads"]["held"] else "random heads move timing as well")
    dose = {arm: {k: arms[arm][k] for k in ("n_improved", "median_reduction", "recall", "gate_rate", "n_evaluable")}
            for arm in DOSE + (PRIMARY,) if arm in arms}
    result = {"verdict": verdict, "reasons": reasons, "labels": labels, "conditions": cond, "integrity_problems": problems,
              "gate": gate[1], "code": state, "test_flags": flags, "scale": args.scale,
              "counts": {"n_planned": len(rows), "n_generated": sum(r["status"] == "ok" for r in rows),
                         "n_failed": sum(r["status"] == "failed" for r in rows),
                         "n_missing": sum(r["status"] not in ("ok", "failed") for r in rows),
                         "n_scored": sum(r["scored"] for r in rows)},
              "arms": arms, "dose_response": dose, "exploratory": exploratory(rows, table, items), "clips": rows}
    cfg = {"experiment": "b1", "heads": {str(k): v for k, v in R.ALIGN.items()},
           "random_sets": [{str(k): v for k, v in s.items()} for s in R.random_head_sets()[:N_DRAWS]],
           "arms": [list(a) for a in ARMS], "seed": "63000+10*item+r", "rng_offset": R.RNG_OFFSET, "steps": R.STEPS,
           "shift": R.SHIFT, "caption": R.CAPTION_F, "whisper": R.WHISPER, "whisper_revision": R.WHISPER_REVISION,
           "thresholds": {"reduction": REDUCTION_MIN, "items": ITEMS_MIN, "guard": GUARD, "random_fraction": RANDOM_FRACTION,
                          "max_incomplete": MAX_INCOMPLETE, "hook_calls": HOOK_CALLS, "min_matched": R.MIN_MATCHED,
                          "digits": DIGITS}, "test_flags": flags, "scale": args.scale}
    res.mkdir(parents=True, exist_ok=True)
    write_result(res / "b1_result.json", cfg, result, t0)
    print(json.dumps({"verdict": verdict, "reasons": reasons, "labels": labels, "counts": result["counts"],
                      "dose_response": dose}, indent=1))
    print("B1:", verdict.upper(), "|", "; ".join(reasons) or "all conditions held", flush=True)


def show_plan(args) -> None:
    runs = plan(args.scale)
    print(json.dumps([{k: x[k] for k in ("run_id", "arm", "beta", "head_set", "dit_seed", "rng_seed")} for x in runs], indent=1))
    print(f"{len(runs)} runs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "gen", "score", "decide"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scale", choices=["full", "tiny"], default="full")
    ap.add_argument("--out", default=str(OUT), help="WAVs, latents and the generation ledger")
    ap.add_argument("--res", default=str(RES), help="the score ledger and b1_result.json")
    ap.add_argument("--only", default="", help="gen: comma list of run ids")
    ap.add_argument("--unregistered", action="store_true", help="tiny scale only: gen with code the addendum does not list")
    ap.add_argument("--test-whisper", default=None, help="tiny scale only: score with this Whisper model")
    args = ap.parse_args()
    try:
        {"plan": show_plan, "gen": gen, "score": score, "decide": decide}[args.cmd](args)
    except SetupError as exc:
        print(f"B1 {args.cmd}: SETUP FAILED: {exc}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
