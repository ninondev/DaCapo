"""E1b, ACE-Step env: encode probe clips to 25 Hz VAE latents and 5 Hz FSQ codes.

Also keeps the codes' quantized vectors. All goes to one .npz for analysis in the main env.

  ACESTEP_REPO=... python ace_codes.py --manifest probe_stimuli/manifest.json --out results/e1b_codes.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ["ACESTEP_REPO"])
from acestep.handler import AceStepHandler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    man = Path(args.manifest)
    items = json.loads(man.read_text())["items"]

    h = AceStepHandler()
    status, ok = h.initialize_service(project_root=os.environ["ACESTEP_REPO"], config_path="acestep-v15-turbo",
                                      device=args.device, use_mlx_dit=False)
    if not ok:
        raise RuntimeError(status)
    if args.device == "cpu":  # same bf16 conversion as ace_driver.py, so codes match E1's tokenizer precision
        h.model = h.model.to(torch.bfloat16)
        h.text_encoder = h.text_encoder.to(torch.bfloat16)
        h.silence_latent = h.silence_latent.to(torch.bfloat16)
        h.dtype = torch.bfloat16
        import gc; gc.collect()
    torch.manual_seed(0)
    data, t0 = {}, time.time()
    for k, it in enumerate(items):
        audio = h.process_src_audio(str(man.parent / it["wav"]))
        with torch.inference_mode():
            lat = h._encode_audio_to_latents(audio)  # [T25, 64]
            mask = torch.ones(1, lat.shape[0], dtype=torch.bool)
            quant, idx, _ = h.model.tokenize(lat.unsqueeze(0).to(h.dtype), h.silence_latent.to(h.dtype), mask)
        data[f"{it['id']}/latent"] = lat.float().cpu().numpy()
        data[f"{it['id']}/codes"] = idx.reshape(-1).cpu().numpy()
        data[f"{it['id']}/quant"] = quant[0].float().cpu().numpy()
        if it["variant"] == "orig" and it["timbre"] == "piano" and it["melody_id"] < 6:
            # encoder uses latent_dist.sample(), so a second encode gives the code-agreement ceiling from sampling noise alone
            with torch.inference_mode():
                lat2 = h._encode_audio_to_latents(audio)
                _, idx2, _ = h.model.tokenize(lat2.unsqueeze(0).to(h.dtype), h.silence_latent.to(h.dtype), mask)
            data[f"{it['id']}/codes_rep"] = idx2.reshape(-1).cpu().numpy()
        print(f"{it['id']:>24s} T25={lat.shape[0]} T5={idx.numel()} {time.time() - t0:6.1f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)


if __name__ == "__main__":
    main()
