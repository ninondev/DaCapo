"""Seeds for E5 v2 runs and the global random-state pin.

No ACE-Step import, so the main environment, the tests and the seed-table writer can use it.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np

DERIVATION = ("d = sha256(f'{tag}|{item_id}|{draw}'.encode()).digest(); "
              "dit_seed = int.from_bytes(d[0:4], 'little') & 0x7FFFFFFF; "
              "rng_seed = int.from_bytes(d[4:8], 'little') & 0x7FFFFFFF, xor 1 if it equals dit_seed")


def run_seeds(tag: str, item_id: str, draw: int) -> tuple[int, int]:
    """(dit_seed, rng_seed) for one (item, draw). Every condition and cell of the item uses this pair,
    so conditions share their initial noise while items and draws do not."""
    d = hashlib.sha256(f"{tag}|{item_id}|{draw}".encode()).digest()
    a = int.from_bytes(d[:4], "little") & 0x7FFFFFFF
    b = int.from_bytes(d[4:8], "little") & 0x7FFFFFFF
    return a, (b ^ 1 if b == a else b)


def pin_global_rng(seed: int) -> None:
    """Seed every global generator an ACE-Step run can draw from: torch (CPU and every CUDA device),
    numpy and python `random`. The repaint source's VAE posterior sample uses torch's global generator.
    Use a seed different from the DiT seed: ACE-Step draws the initial noise from a fresh
    torch.Generator().manual_seed(dit_seed), and a global generator seeded with the same value repeats it."""
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
