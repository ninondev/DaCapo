"""VAE-only encode/decode with ACE-Step 1.5's Oobleck autoencoder (no DiT, ~0.35 GB).

Runs in the ACE-Step environment (needs diffusers). Preprocessing matches ACE-Step's
process_src_audio: stereo, 48 kHz, clamp to [-1, 1]. Encoding uses the posterior mean
(deterministic). ACE-Step's cover path samples instead.

  python vae_io.py encode --manifest manifest.json --out results/latents.npz
  python vae_io.py decode --latents results/edits.npz --outdir edited/
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from diffusers import AutoencoderOobleck

CKPT = Path(os.environ.get("ACESTEP_CHECKPOINTS_DIR", "checkpoints")) / "vae"


def load_audio(path: Path) -> torch.Tensor:
    import torchaudio

    x, sr = sf.read(str(path), always_2d=True)
    a = torch.tensor(x.T, dtype=torch.float32)
    if a.shape[0] == 1:
        a = torch.cat([a, a])
    a = a[:2]
    if sr != 48000:
        a = torchaudio.transforms.Resample(sr, 48000)(a)
    return a.clamp(-1, 1)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["encode", "decode"])
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--latents")
    ap.add_argument("--outdir")
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderOobleck.from_pretrained(str(CKPT)).eval().to(dev)
    t0 = time.time()
    if args.mode == "encode":
        man = Path(args.manifest)
        items = json.loads(man.read_text())["items"]
        data = {}
        for it in items:
            for key in ("wav", "target_wav"):
                if key in it:
                    a = load_audio(man.parent / it[key])
                    z = vae.encode(a[None].to(dev)).latent_dist.mean[0].T  # [T, 64]
                    data[f"{it['id']}/{key}"] = z.float().cpu().numpy()
            print(f"{it['id']:>32s} {time.time() - t0:6.1f}s", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **data)
    else:
        z = np.load(args.latents)
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        for k in z.files:
            wav = vae.decode(torch.tensor(z[k].T[None], dtype=torch.float32, device=dev)).sample[0].float().cpu().numpy().T
            sf.write(str(out / f"{k.replace('/', '__')}.wav"), wav, 48000)
            print(f"{k:>48s} {time.time() - t0:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
