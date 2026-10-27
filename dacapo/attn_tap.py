"""Read the cross-attention of chosen ACE-Step DiT layers without changing the forward pass.

ACE-Step can return attention weights itself (decoder(..., output_attentions=True)), but that switches
every cross-attention layer to the eager kernel, which changes the velocity (relative L2 difference
0.016 on CPU bf16), so a run with it is a different run. These hooks recompute the probabilities
from the module's own inputs instead and return None, which leaves the output bit-identical.
No ACE-Step import: the hook only needs the module attributes of AceStepAttention.
"""
from __future__ import annotations

import time

import numpy as np
import torch


class CrossAttnTap:
    """Forward hooks on `decoder.layers[L].cross_attn` for L in `layers`.

    For every decoder call the hook on the first layer opens a new record; each hook stores
    softmax(q k^T * scaling + mask) for all heads as float16 on the CPU, shape [heads, queries, keys],
    with q = q_norm(q_proj(hidden_states)), k = k_norm(k_proj(encoder_hidden_states)) repeated over the
    GQA groups (query head h reads key head h // groups, as in transformers' repeat_kv).
    `keys` keeps a slice of the key axis (e.g. the lyric tokens) to save memory.
    """

    def __init__(self, decoder, layers=(2, 3, 4, 5, 6), keys: slice = slice(None)):
        self.decoder, self.layers, self.keys = decoder, tuple(layers), keys
        self.records: list[dict[int, torch.Tensor]] = []
        self.seconds = 0.0
        self._handles: list = []

    def __enter__(self) -> "CrossAttnTap":
        for L in self.layers:
            mod = self.decoder.layers[L].cross_attn
            self._handles.append(mod.register_forward_hook(self._hook(L), with_kwargs=True))
        return self

    def __exit__(self, *exc) -> bool:
        for hd in self._handles:
            hd.remove()
        self._handles = []
        return False

    def _hook(self, L: int):
        def hook(mod, args, kwargs, output):
            t0 = time.time()
            if L == self.layers[0]:
                self.records.append({})
            hs, ehs = kwargs["hidden_states"], kwargs["encoder_hidden_states"]
            mask = kwargs.get("attention_mask")
            b, lq, _ = hs.shape
            lk = ehs.shape[1]
            with torch.no_grad():
                q = mod.q_norm(mod.q_proj(hs).view(b, lq, -1, mod.head_dim)).transpose(1, 2)
                k = mod.k_norm(mod.k_proj(ehs).view(b, lk, -1, mod.head_dim)).transpose(1, 2)
                k = k.repeat_interleave(mod.num_key_value_groups, dim=1)
                logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * mod.scaling
                if mask is not None:
                    logits = logits + mask[..., :lk].float()
                p = torch.softmax(logits, dim=-1)[0, :, :, self.keys]
            self.records[-1][L] = p.to(torch.float16).cpu()
            self.seconds += time.time() - t0
        return hook

    def stacked(self, start: int, end: int) -> np.ndarray:
        """[calls, layers, heads, end-start, queries] float16: the key range [start, end) of every
        record, token-major like ACE-Step's alignment matrices."""
        return np.stack([np.stack([rec[L][:, :, start:end].transpose(-1, -2).numpy() for L in self.layers])
                         for rec in self.records])
