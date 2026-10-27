"""Check `ace_lab.swap_conditioning` when runs have different lengths. No model weights, runs in seconds.

Regression test for the E4 swap crash: Voice-Oohs guides of 14.88 s and 14.94 s give 372 and 373
latent frames, and swapping one item's captured context_latents into the other's run failed at the
DiT's torch.cat([context_latents, hidden_states], -1). Needs the ACE-Step env ($ACESTEP_REPO) only
because ace_lab imports it.

  ACESTEP_REPO=... python check_swap_lengths.py
"""
from __future__ import annotations

import torch
from transformers.cache_utils import EncoderDecoderCache

from dacapo import ace as ace_lab


def kwargs(frames: int, tokens: int = 20, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    ctx = torch.cat([torch.randn(1, frames, 64, generator=g), torch.ones(1, frames, 64)], dim=-1)
    return {"hidden_states": torch.randn(1, frames, 64, generator=g), "timestep": torch.ones(1),
            "context_latents": ctx, "encoder_hidden_states": torch.randn(1, tokens, 16, generator=g),
            "encoder_attention_mask": torch.ones(1, tokens), "past_key_values": None}


def dit_concat(kw: dict) -> torch.Tensor:  # the line that raised in modeling_acestep_v15_turbo.py
    return torch.cat([kw["context_latents"], kw["hidden_states"]], dim=-1)


def main():
    sil = torch.randn(1, 750, 64, generator=torch.Generator().manual_seed(1))
    own = kwargs(372, seed=2)

    # equal length: tensors pass through untouched (runs that never crashed are unchanged)
    same = kwargs(372, seed=3)["context_latents"]
    kw = ace_lab.swap_conditioning(1, {"context_latents": same}, silence_latent=sil)(1, 0.9, own)
    assert kw["context_latents"] is same
    assert ace_lab.swap_conditioning(1, {"context_latents": same})(0, 1.0, own) is own  # before the switch

    # longer context (373 -> 372): crop
    longer = kwargs(373, seed=4)["context_latents"]
    kw = ace_lab.swap_conditioning(1, {"context_latents": longer})(3, 0.5, own)
    assert kw["context_latents"].shape == (1, 372, 128) and torch.equal(kw["context_latents"], longer[:, :372])
    assert dit_concat(kw).shape == (1, 372, 192)

    # shorter context (372 -> 373): pad with silence-latent frames, chunk mask carried over
    own373 = kwargs(373, seed=5)
    shorter = kwargs(372, seed=6)["context_latents"]
    pre = ace_lab.swap_conditioning(1, {"context_latents": shorter}, silence_latent=sil)
    kw = pre(1, 0.9, own373)
    c = kw["context_latents"]
    assert c.shape == (1, 373, 128) and torch.equal(c[:, :372], shorter)
    assert torch.equal(c[0, 372, :64], sil[0, 372]) and torch.equal(c[0, 372, 64:], torch.ones(64))
    assert pre(2, 0.8, own373)["context_latents"] is c  # fitted once, reused at later steps
    assert dit_concat(kw).shape == (1, 373, 192)
    try:
        ace_lab.swap_conditioning(1, {"context_latents": shorter})(1, 0.9, own373)
        raise AssertionError("padding without silence_latent must raise")
    except ValueError:
        pass

    # a large mismatch would shift the melody in time: refuse rather than crop
    try:
        ace_lab.swap_conditioning(1, {"context_latents": kwargs(360, seed=9)["context_latents"]}, silence_latent=sil)(1, 0.9, own)
        raise AssertionError("a 12-frame mismatch must raise")
    except ValueError:
        pass

    # lyrics swap: encoder states of another token length are not time-aligned and stay as captured
    enc = {k: v for k, v in kwargs(350, tokens=27, seed=7).items() if k.startswith("encoder_")}
    kw = ace_lab.swap_conditioning(2, enc)(2, 0.7, kwargs(350, tokens=20, seed=8))
    assert kw["encoder_hidden_states"] is enc["encoder_hidden_states"]
    assert isinstance(kw["past_key_values"], EncoderDecoderCache)  # fresh cross-attention cache at the switch
    print("swap_conditioning length checks passed")


if __name__ == "__main__":
    main()
