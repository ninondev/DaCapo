"""E5 v2 integrity checks (addendum 2, S10), the generation ledger and draw states.

Each check returns {"name", "ok", "detail"}. A failure found before any hypothesis accuracy exists voids
the attempt; attempt 1 may then be followed by one attempt 2, and a void attempt 2 is strike two.

Files written by run_analogy before and during generation:
  outputs_<phase>/design.json   phase, attempt, manifest + sha256, source sha256, addendum sha256,
                                ci_method, ruler_program, arm_order, jobs[{id, split, cond, arm, seed,
                                dit_seed, rng_seed, src}]
  outputs_<phase>/ledger.jsonl  one event per line:
      pass n          a generation pass begins (pass 1, then at most two automatic resume passes)
      resume          an automatic resume pass begins (counted against the cap of two)
      restart         a process restarted inside the current pass (relaunch or runtime reset)
      start | done | error          one generation attempt of a job (done carries wav_sha256, the sha256
                      of the audio file, and wav_bytes)
      interrupted     a start whose process died before done or error (runtime reset, kill, a crash
                      outside the generation call): no outcome was seen, so the attempt is not counted
      halt            five generations in a row raised: a technical stop; their attempts are not counted
      lost            before the quality freeze: a completed job's audio is gone or changed; the job is
                      generated again with the same seeds, since no outcome of it was seen
      orphan          before the quality freeze: an audio file without a done event was moved aside
      check_failed    a finished generation broke a hard per-run check (configuration, not luck)
E5 v2 audio is stored as 16-bit FLAC (lossless; half the size of WAV on Drive).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ALLOWED_FORMATS = {"WAV", "FLAC"}
ALLOWED_SUBTYPES = {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
AUDIO_EXT = ".flac"
SR = 48000
PEAK_TARGET_DBFS, PEAK_TOL_DB = -1.0, 0.5
SILENT_PEAK_DBFS = -120.0          # 20*log10(1e-6): ACE-Step skips normalisation below this peak
NULL_CTX_MAX_DBFS, CTX_MIN_DBFS, D_MIN_DBFS = -90.0, -40.0, -40.0
JOB_FIELDS = ("id", "cond", "arm", "seed", "dit_seed", "rng_seed")


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_NATIVE = {"PCM_16": "int16", "PCM_24": "int32", "PCM_32": "int32", "FLOAT": "float32", "DOUBLE": "float64"}


def audio_sha256(p: str | Path) -> str:
    """sha256 of the decoded samples in their stored precision, with rate, channels and subtype. Rendered
    sources are compared by this hash: a FLOAT wav carries a PEAK chunk with a timestamp, so the same
    samples written twice give different file bytes."""
    import numpy as np
    import soundfile as sf

    info = sf.info(str(p))
    x, sr = sf.read(str(p), dtype=_NATIVE.get(info.subtype, "float64"), always_2d=True)
    h = hashlib.sha256(f"{sr}|{info.channels}|{len(x)}|{info.subtype}|".encode())
    h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def job_key(j: dict) -> tuple[str, str, str, str]:
    return (j["id"], j["cond"], j.get("arm", "base"), j["seed"])


def audio_name(j: dict) -> str:
    return f"{j['id']}__{j['cond']}__{j.get('arm', 'base')}__{j['seed']}{AUDIO_EXT}"


def _res(name, ok, detail=None):
    return {"name": name, "ok": bool(ok), "detail": detail}


# Ledger

def read_ledger(path: str | Path) -> list[dict]:
    """Events in file order. A torn last line (the process died while writing it) is skipped."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def ledger_state(events: list[dict]) -> tuple[dict, int]:
    """key -> {'done': [done events since the last lost], 'starts': counted starts since the last lost,
    'starts_by_pass': {pass: n}, 'errors': n, 'lost': n, 'check_failed': n, 'not_counted': n}; plus the
    number of automatic resume passes. An interrupted start and the starts a halt lists are not counted."""
    st: dict = defaultdict(lambda: {"done": [], "starts": 0, "starts_by_pass": defaultdict(int), "errors": 0,
                                    "lost": 0, "check_failed": 0, "not_counted": 0})

    def uncount(s: dict, p) -> None:
        if s["starts_by_pass"].get(p, 0) > 0:
            s["starts"] -= 1
            s["starts_by_pass"][p] -= 1
            s["not_counted"] += 1

    n_resume = 0
    for e in events:
        ev = e.get("event")
        if ev == "resume":
            n_resume += 1
            continue
        if ev == "halt":
            for j in e.get("jobs", []):
                uncount(st[job_key(j)], j.get("pass", 1))
            continue
        if ev not in ("start", "done", "error", "lost", "check_failed", "interrupted"):
            continue
        s = st[job_key(e)]
        if ev == "done":
            s["done"].append(e)
        elif ev == "start":
            s["starts"] += 1
            s["starts_by_pass"][e.get("pass", 1)] += 1
        elif ev == "error":
            s["errors"] += 1
        elif ev == "check_failed":
            s["check_failed"] += 1
        elif ev == "interrupted":
            uncount(s, e.get("pass", 1))
        elif ev == "lost":
            s["lost"] += 1
            s["done"], s["starts"], s["starts_by_pass"] = [], 0, defaultdict(int)
    return dict(st), n_resume


def open_starts(events: list[dict]) -> list[dict]:
    """Start events with no done, error, check_failed or interrupted event after them for the same job:
    the process that made them died while generating."""
    last: dict = {}
    for e in events:
        ev = e.get("event")
        if ev == "start":
            last[job_key(e)] = e
        elif ev in ("done", "error", "check_failed", "interrupted", "lost"):
            last.pop(job_key(e), None)
    return list(last.values())


def trailing_errors(events: list[dict]) -> list[dict]:
    """Error events since the last completed generation or halt."""
    out: list[dict] = []
    for e in events:
        if e.get("event") in ("done", "halt"):
            out = []
        elif e.get("event") == "error":
            out.append(e)
    return out


def current_pass(events: list[dict]) -> int:
    return max((int(e["pass"]) for e in events if e.get("event") == "pass"), default=0)


def draw_status(design: dict, events: list[dict], outdir: Path) -> dict:
    """'ok' (completed, wav present) | 'lost' (completed, wav gone: a gate failure once the quality
    stage has frozen) | 'missing' (never completed: no outcome was seen, so it may be generated)."""
    st, _ = ledger_state(events)
    out = {}
    for j in design["jobs"]:
        k = job_key(j)
        done = st.get(k, {}).get("done", [])
        if not done:
            out[k] = "missing"
        else:
            out[k] = "ok" if (outdir / audio_name(j)).exists() else "lost"
    return out


# Checks

def check_lossless(infos: dict) -> dict:
    """1. Every scored file is lossless (WAV or FLAC, PCM or float) at 48 kHz. infos: key -> {format,
    subtype, samplerate, channels}."""
    bad = {"|".join(k): v for k, v in infos.items()
           if v["format"] not in ALLOWED_FORMATS or v["subtype"] not in ALLOWED_SUBTYPES or v["samplerate"] != SR}
    return _res("lossless_audio", not bad, bad or None)


def check_peaks(rows: list[dict]) -> dict:
    """2. ACE-Step's peak normalisation ran: the channel peak (max over all channels) is -1 +- 0.5 dBFS,
    unless the output is silent (peak below 1e-6, where ACE-Step skips normalisation)."""
    bad = {"|".join(job_key(r)): r["quality"]["peak_dbfs"]
           for r in rows if r.get("status") == "ok"
           and not (abs(r["quality"]["peak_dbfs"] - PEAK_TARGET_DBFS) <= PEAK_TOL_DB
                    or r["quality"]["peak_dbfs"] <= SILENT_PEAK_DBFS)}
    return _res("peak_normalised", not bad, bad or None)


def source_levels(path: str | Path) -> dict:
    import soundfile as sf

    from dacapo.motif_features import W_CTX, W_D, _seg, rms_dbfs
    x, sr = sf.read(str(path), always_2d=True)
    m = x.mean(axis=1)
    return {"ctx_dbfs": rms_dbfs(_seg(m, sr, *W_CTX)), "d_dbfs": rms_dbfs(_seg(m, sr, *W_D))}


def check_sources(design: dict, levels: dict) -> dict:
    """3. Null contexts are digital silence (< -90 dBFS), non-null contexts sound (> -40 dBFS), and the
    D bar sounds in every source (> -40 dBFS). levels: src path -> source_levels()."""
    bad = {}
    for j in design["jobs"]:
        lv = levels[j["src"]]
        ok_ctx = lv["ctx_dbfs"] < NULL_CTX_MAX_DBFS if j["cond"] == "null" else lv["ctx_dbfs"] > CTX_MIN_DBFS
        if not ok_ctx or lv["d_dbfs"] <= D_MIN_DBFS:
            bad[j["src"]] = lv
    return _res("source_levels", not bad, bad or None)


def check_seeds(design: dict, events: list[dict], committed: dict | None = None) -> dict:
    """4. The noise design is the declared one: with ci_method 'melody' a (dit_seed, rng_seed) pair never
    appears in two melodies or twice among a melody's draws; every generation used the seeds listed for
    it in design.json; with a committed seed table, design.json uses exactly its pairs."""
    pairs_by_mel: dict = defaultdict(set)
    seen: dict = {}
    problems = []
    for j in design["jobs"]:
        p = (j["dit_seed"], j["rng_seed"])
        pairs_by_mel[j["id"]].add((j["seed"], p))
        if design.get("ci_method", "melody") == "melody" and p in seen and seen[p] != j["id"]:
            problems.append(f"seed pair {p} shared by {seen[p]} and {j['id']}")
        seen.setdefault(p, j["id"])
        if committed is not None:
            want = committed.get(j["id"])
            d = int(str(j["seed"]).lstrip("d"))
            if want is None or d >= len(want) or tuple(want[d]) != p:
                problems.append(f"{'|'.join(job_key(j))}: seeds differ from the committed table")
    for mid, s in pairs_by_mel.items():
        by_draw = defaultdict(set)
        for lab, p in s:
            by_draw[lab].add(p)
        if any(len(v) != 1 for v in by_draw.values()):
            problems.append(f"{mid}: one draw label has several seed pairs")
        if len({next(iter(v)) for v in by_draw.values()}) != len(by_draw):
            problems.append(f"{mid}: two draws share a seed pair")
    want = {job_key(j): (j["dit_seed"], j["rng_seed"]) for j in design["jobs"]}
    for e in events:
        if e.get("event") in ("start", "done") and job_key(e) in want:
            if (e.get("dit_seed"), e.get("rng_seed")) != want[job_key(e)]:
                problems.append(f"{'|'.join(job_key(e))}: ledger seeds differ from design")
    return _res("seeds_as_designed", not problems, problems[:50] or None)


def check_disjoint(design_main: dict, design_pilot: dict | None) -> dict:
    """5. Confirmatory melodies and seed values are disjoint from Phase 0b."""
    if design_pilot is None:
        return _res("disjoint_from_pilot", False, "Phase 0b design.json missing")
    mel = {j["id"] for j in design_main["jobs"]} & {j["id"] for j in design_pilot["jobs"]}
    sv = lambda d: {s for j in d["jobs"] for s in (j["dit_seed"], j["rng_seed"])}
    seeds = sv(design_main) & sv(design_pilot)
    return _res("disjoint_from_pilot", not mel and not seeds,
                {"melodies": sorted(mel), "seed_values": sorted(seeds)} if mel or seeds else None)


def check_files_vs_design(design: dict, outdir: Path, source_sha: dict) -> dict:
    """6. The scored files are exactly the planned ones, and every source still has the sha256 recorded
    in design.json before generation (a resume on a runtime that rendered the sources differently is
    caught here)."""
    planned = {audio_name(j) for j in design["jobs"]}
    extra = sorted(p.name for p in outdir.iterdir() if p.suffix in (".wav", ".flac") and p.name not in planned)
    changed = sorted(s for s, h in design["sources_sha256"].items() if source_sha.get(s) != h)
    return _res("files_match_design", not extra and not changed,
                {"unplanned_wavs": extra, "changed_sources": changed} if extra or changed else None)


def check_ruler(gate_file: dict | None, design: dict) -> dict:
    """7. tau comes from a passed ruler gate (K0) on the same manifest and timbre (GM program)."""
    if gate_file is None:
        return _res("ruler_gate", False, "ruler gate file missing")
    s = gate_file["results"]["summary"]
    ok = bool(s.get("passed")) and s.get("manifest_sha256") == design["manifest_sha256"] \
        and s.get("program") == design["ruler_program"]
    return _res("ruler_gate", ok, None if ok else {"passed": s.get("passed"), "manifest_sha256": s.get("manifest_sha256"),
                                                  "program": s.get("program")})


def check_wav_hashes(quality_rows: list[dict], current: dict) -> dict:
    """8. (score stage) Each wav still has the sha256 recorded by the quality stage."""
    bad = sorted("|".join(job_key(r)) for r in quality_rows
                 if r.get("status") == "ok" and current.get(job_key(r)) != r["wav_sha256"])
    return _res("wav_unchanged_since_quality", not bad, bad[:50] or None)


def check_ledger(design: dict, events: list[dict], wav_sha: dict) -> dict:
    """9. At most one completed attempt per draw since its last 'lost' event (a draw is generated again
    only when its audio was lost before the quality freeze), no hard per-run check failed, and the wav
    on disk is the one the ledger recorded last."""
    st, _ = ledger_state(events)
    problems = []
    for j in design["jobs"]:
        k = job_key(j)
        s = st.get(k, {})
        done = s.get("done", [])
        if len(done) > 1:
            problems.append(f"{'|'.join(k)}: {len(done)} completed attempts")
        if s.get("check_failed"):
            problems.append(f"{'|'.join(k)}: a hard per-run check failed")
        if done and k in wav_sha and wav_sha[k] != done[-1]["wav_sha256"]:
            problems.append(f"{'|'.join(k)}: wav differs from the ledger")
    return _res("ledger_consistent", not problems, problems[:50] or None)


def code_sha256(files, root: Path) -> dict:
    """Repo-relative path -> sha256 of the file under root (None if it is missing)."""
    return {f: (sha256_file(root / f) if (root / f).exists() else None) for f in files}


def check_registration(addendum: dict | None, current_code_sha: dict, design: dict, addendum_sha: str | None,
                       events: list[dict], current_data_sha: dict | None = None) -> dict:
    """10. The code and data that run are the registered ones (code_sha256 and data_sha256 of addendum 2);
    design.json names this addendum; the addendum is older than the first generation of the phase."""
    if addendum is None:
        return _res("registered_code", False, "addendum_2.json missing")
    reg = addendum.get("code_sha256") or {}
    diff = sorted(f for f in reg if current_code_sha.get(f) != reg[f])
    diff += sorted(f for f in current_code_sha if f not in reg)
    reg_data = addendum.get("data_sha256") or {}
    data_now = current_data_sha or {}
    diff += sorted(f for f in reg_data if data_now.get(f) != reg_data[f])
    starts = sorted(e["utc"] for e in events if e.get("event") == "start")
    late = bool(starts) and not str(addendum.get("addendum_utc", "9999")) < starts[0]
    ok = bool(reg) and not diff and design.get("addendum_sha256") == addendum_sha and not late
    return _res("registered_code", ok, None if ok else {
        "changed_files": diff, "no_hashes_registered": not reg,
        "design_names_this_addendum": design.get("addendum_sha256") == addendum_sha,
        "addendum_after_first_generation": late})


def summarize_checks(results: list[dict]) -> dict:
    return {"ok": all(r["ok"] for r in results), "checks": results}
