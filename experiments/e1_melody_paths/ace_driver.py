"""E1 driver: run known-melody stimuli through ACE-Step 1.5 pathways and save the outputs.

Needs the official ACE-Step-1.5 repo importable (it is not a dependency of this package).
Scoring is separate (score.py, calibrated ruler) and imports no ACE-Step code.

  ACESTEP_REPO=/path/to/ACE-Step-1.5 ACESTEP_CHECKPOINTS_DIR=/path/to/ckpt \
  python ace_driver.py --manifest stimuli/manifest.json --out outputs/

Pathways (per stimulus):
  vae_rt      VAE encode -> decode, no DiT                 (ceiling of the acoustic latent)
  codes_rt    tokenize -> 5 Hz FSQ codes -> DiT render     (what the LM planner can express)
  codes_reorch same codes, caption asks for other instruments (does melody survive re-orchestration?)
  nofsq       raw 25 Hz VAE latents as context -> DiT      (no semantic bottleneck)
  nocodes     text2music, same caption, no audio context   (chance baseline: melody unknown)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, os.environ["ACESTEP_REPO"])

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

PATHWAYS = ["vae_rt", "codes_rt", "codes_reorch", "nofsq", "nocodes"]


def init(device: str, dtype: str) -> AceStepHandler:
    h = AceStepHandler()
    status, ok = h.initialize_service(project_root=os.environ["ACESTEP_REPO"], config_path="acestep-v15-turbo",
                                      device=device, use_mlx_dit=False)
    if not ok:
        raise RuntimeError(status)
    if device == "cpu" and dtype == "bf16":  # CPUs with AMX/AVX512-BF16: halves RAM, faster matmuls
        h.model = h.model.to(torch.bfloat16)
        h.text_encoder = h.text_encoder.to(torch.bfloat16)
        h.silence_latent = h.silence_latent.to(torch.bfloat16)
        h.dtype = torch.bfloat16
    return h


def codes_of(h: AceStepHandler, wav: str) -> list[int]:
    s = h.convert_src_audio_to_codes(wav)
    return [int(x) for x in s.replace("<|audio_code_", " ").replace("|>", " ").split()]


def vae_roundtrip(h: AceStepHandler, wav: str, out: Path) -> None:
    audio = h.process_src_audio(wav)
    with torch.inference_mode():
        lat = h._encode_audio_to_latents(audio)  # [T, 64]
        dec = h.vae.decode(lat.T.unsqueeze(0).to(h.vae.dtype)).sample[0]
    sf.write(out, dec.float().cpu().numpy().T, 48000)


def generate(h, out: Path, caption: str, duration: float, seed: int, steps: int, task: str,
             src: str | None = None, codes: str = "", shift: float = 1.0) -> dict:
    params = GenerationParams(task_type=task, caption=caption, lyrics="[Instrumental]", instrumental=True,
                              duration=duration, inference_steps=steps, seed=seed, thinking=False, shift=shift,
                              use_cot_caption=False, use_cot_language=False, use_cot_metas=False,
                              src_audio=src, audio_codes=codes, audio_cover_strength=1.0)
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
    t0 = time.time()
    r = generate_music(h, None, params, cfg, save_dir=str(out.parent / "_tmp"))
    if not r.success:
        raise RuntimeError(r.error)
    Path(r.audios[0]["path"]).replace(out)
    return {"seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shift", type=float, default=1.0, help="timestep shift; GenerationParams default is 1.0, docs recommend 3.0 for turbo")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps (GPU dtypes are chosen by ACE-Step)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="CPU only")
    ap.add_argument("--pathways", default=",".join(PATHWAYS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    man_dir = Path(args.manifest).resolve().parent
    stimuli = json.loads(Path(args.manifest).read_text())["stimuli"]
    if args.limit:
        stimuli = stimuli[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "driver_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"runs": []}
    done = {(r["id"], r["pathway"]) for r in log["runs"]}

    t0 = time.time()
    h = init(args.device, args.dtype)
    log["init_seconds"] = round(time.time() - t0, 1)
    log["settings"] = {"steps": args.steps, "seed": args.seed, "shift": args.shift, "dtype": args.dtype, "model": "acestep-v15-turbo",
                       "audio_cover_strength": 1.0, "device": args.device, "torch": torch.__version__}

    for s in stimuli:
        wav = str(man_dir / s["wav"])  # absolute paths pass through unchanged
        dur = float(s["duration"])
        if "codes" not in s:
            s["codes"] = codes_of(h, wav)
        codes_str = "".join(f"<|audio_code_{c}|>" for c in s["codes"])
        for p in args.pathways.split(","):
            if (s["id"], p) in done:
                continue
            dst = out / f"{s['id']}__{p}.wav"
            t1 = time.time()
            if p == "vae_rt":
                vae_roundtrip(h, wav, dst)
            elif p == "codes_rt":
                generate(h, dst, s["caption"], dur, args.seed, args.steps, "text2music", codes=codes_str, shift=args.shift)
            elif p == "codes_reorch":
                generate(h, dst, s["reorch_caption"], dur, args.seed, args.steps, "text2music", codes=codes_str, shift=args.shift)
            elif p == "nofsq":
                generate(h, dst, s["caption"], dur, args.seed, args.steps, "cover-nofsq", src=wav, shift=args.shift)
            elif p == "nocodes":
                generate(h, dst, s["caption"], dur, args.seed, args.steps, "text2music", shift=args.shift)
            log["runs"].append({"id": s["id"], "pathway": p, "wav": dst.name, "seconds": round(time.time() - t1, 1)})
            log_path.write_text(json.dumps(log, indent=1))
            print(f"{s['id']:>16s} {p:>12s} {time.time() - t1:6.1f}s", flush=True)
        (out / "codes.json").write_text(json.dumps({x["id"]: x.get("codes") for x in stimuli}))


if __name__ == "__main__":
    main()
