"""The lyric clock of Run B: a fixed bias on the pre-softmax cross-attention logits of chosen DiT heads.

For each word, beta is added to the logits from the query frames inside the word's window to the keys
of the word's lyric tokens, on the steered heads only. All other logits are unchanged. B1 is registered
in experiments/e5_motif/registration/addendum_2.json ("e6_readout", "B1, fixed now"), B2 in addendum 3.

The DiT builds its cross-attention mask on every decoder call, as an all-zero [1, 1, Lq, Lk] tensor on
the sdpa and eager paths (acestep/models/turbo/modeling_acestep_v15_turbo.py:1423-1432), and both kernels
add it to the scaled logits. A forward pre-hook on decoder.layers[L].cross_attn swaps in
mask + beta * template on the steered heads. The mask is rebuilt on every call, so the bias applies at
every step, also when the cross-attention keys and values come from the cache. Under flash_attention_2
the mask is None and nothing can be added, so the hook raises.

Enter CrossAttnBias around ace.generate, never through its taps= argument (taps must leave the forward
pass unchanged). At beta 0 the hook still runs and adds zeros. No ACE-Step import.
"""
from __future__ import annotations

import torch

from .words import group_words, lyric_words

HOP_S = 0.08     # DiT query q covers latent frames 2q and 2q+1 at 25 Hz (patch size 2)
OFFSET_S = 0.04  # query q is placed at its centre, 0.08 q + 0.04 s
DIGITS = 6       # seconds are rounded to 1e-6 before any comparison


def _get(note, name: str) -> float:
    return float(note[name] if isinstance(note, dict) else getattr(note, name))


def word_spans(notes, bpm: float, word_keys) -> list[tuple[tuple[int, ...], float, float]]:
    """B1's windows, one note per word: word j gets [onset_j, onset_(j+1)) and the last word
    [onset, note end), in seconds. notes: dicts or dacapo.symbolic.Note with start and dur in beats
    (dur is the sounding length, as stored in the manifests); word_keys: the absolute key positions
    of each word's lyric tokens. Returns (keys, start s, end s) per word."""
    if len(notes) != len(word_keys):
        raise ValueError(f"{len(notes)} notes for {len(word_keys)} words")
    spb = 60.0 / float(bpm)
    onsets = [round(_get(n, "start") * spb, DIGITS) for n in notes]
    end = round((_get(notes[-1], "start") + _get(notes[-1], "dur")) * spb, DIGITS)
    spans = []
    for keys, a, b in zip(word_keys, onsets, onsets[1:] + [end]):
        if not keys or b <= a:
            raise ValueError(f"word with keys {list(keys)}: window [{a}, {b}) is empty or has no keys")
        spans.append((tuple(int(k) for k in keys), a, b))
    return spans


def rotate_spans(spans, shift: int) -> list[tuple[tuple[int, ...], float, float]]:
    """Word j keeps its keys and takes the window of word (j + shift) mod n: the same heads and keys
    with the wrong timing."""
    n = len(spans)
    return [(tuple(spans[j][0]), spans[(j + shift) % n][1], spans[(j + shift) % n][2]) for j in range(n)]


def template(spans, lq: int, lk: int, hop_s: float = HOP_S, offset_s: float = OFFSET_S) -> torch.Tensor:
    """[lq, lk] float32, 1 where the centre of query frame q lies in a word's window and the key is one
    of that word's tokens, 0 elsewhere."""
    t = torch.zeros(lq, lk)
    centres = [round(hop_s * q + offset_s, DIGITS) for q in range(lq)]
    for keys, a, b in spans:
        rows = [q for q, c in enumerate(centres) if a <= c < b]
        if rows:
            t[torch.tensor(rows)[:, None], torch.tensor(keys)[None, :]] = 1.0
    return t


def lyric_keys(h, aligner, lyrics: str, language: str = "en") -> dict:
    """Where each lyric word sits among the encoder keys, from ACE-Step's own prompt format, tokenizer
    and lyric slice (as experiments/e6_clock/readout.py does). h: AceStepHandler; aligner:
    MusicStampsAligner(h.text_tokenizer). The lyric ids are keys [0, n_lyr) of the packed encoder
    states, so word token t sits at key s + t (s = the header length, 9 for English)."""
    ids = h.text_tokenizer(h._format_lyrics(lyrics, language), padding="longest", truncation=True, max_length=2048,
                           return_tensors="pt").input_ids
    raw, pure, s, e = h._extract_lyric_segment(ids, language)
    words = group_words(aligner._decode_tokens_incrementally(pure))
    if [w.norm for w in words] != lyric_words(lyrics):
        raise ValueError(f"token words {[w.norm for w in words]} != lyric words {lyric_words(lyrics)}")
    return {"ids": [int(i) for i in raw], "start": int(s), "end": int(e), "n_lyr": len(raw),
            "words": [w.norm for w in words], "keys": [[int(s) + t for t in w.tokens] for w in words]}


class CrossAttnBias:
    """Context manager: forward pre-hooks on decoder.layers[L].cross_attn for every layer L in `heads`.

    heads   {layer: [query heads]}: the alignment heads, a random set, all heads of a layer
    spans   one (keys, window start s, window end s) per word; keys are absolute encoder positions
    beta    the bias in logit units (0, 1, 2, 4 and 8 are exact in bf16)
    n_lyr   the number of lyric ids: keys [0, n_lyr) are the lyric block, and every biased key lies in it

    Each call raises unless the kernel is sdpa or eager and the mask is a [1, 1, Lq, >= Lk] tensor at
    batch 1, checks max key < n_lyr < Lk, and returns mask[..., :Lk] + beta * template on the steered
    heads as a [1, n_heads, Lq, Lk] tensor in the mask's dtype. One 0/1 template is built per
    (Lq, Lk, device, dtype) and reused. `shapes` keeps (layer, Lq, Lk) of every call in call order;
    record() gives what a run row stores.
    """

    def __init__(self, decoder, heads: dict, spans, beta: float, n_lyr: int, n_heads: int = 16,
                 hop_s: float = HOP_S, offset_s: float = OFFSET_S):
        self.decoder = decoder
        self.heads = {int(L): sorted({int(x) for x in hs}) for L, hs in heads.items()}
        self.spans = [(tuple(int(k) for k in keys), float(a), float(b)) for keys, a, b in spans]
        self.beta, self.n_lyr, self.n_heads = float(beta), int(n_lyr), int(n_heads)
        self.hop_s, self.offset_s = hop_s, offset_s
        if not self.heads or any(not hs or hs[0] < 0 or hs[-1] >= self.n_heads for hs in self.heads.values()):
            raise ValueError(f"head map {heads} must name heads in [0, {self.n_heads}) for each layer")
        if not self.spans or any(not keys for keys, _, _ in self.spans):
            raise ValueError("every span needs at least one key")
        self.max_key = max(max(keys) for keys, _, _ in self.spans)
        if min(min(keys) for keys, _, _ in self.spans) < 0 or self.max_key >= self.n_lyr:
            raise ValueError(f"biased keys must lie in the lyric block [0, {self.n_lyr}); largest is {self.max_key}")
        self.calls = 0
        self.shapes: list[tuple[int, int, int]] = []
        self.mask_in: tuple | None = None  # shape and dtype of the first incoming mask
        self._templates: dict = {}
        self._handles: list = []

    def __enter__(self) -> "CrossAttnBias":
        if self._handles:
            raise RuntimeError("CrossAttnBias is already entered")
        for L in self.heads:
            mod = self.decoder.layers[L].cross_attn
            self._handles.append(mod.register_forward_pre_hook(self._hook(L), with_kwargs=True))
        return self

    def __exit__(self, *exc) -> bool:
        for hd in self._handles:
            hd.remove()
        self._handles = []
        return False

    def cells(self, lq: int, lk: int) -> int:
        """Biased (query, key) cells per steered head at this shape."""
        return int(template(self.spans, lq, lk, self.hop_s, self.offset_s).sum())

    def _hook(self, L: int):
        steered = self.heads[L]

        def pre(mod, args, kwargs):
            impl = getattr(mod.config, "_attn_implementation", None)
            if impl not in ("sdpa", "eager"):
                raise RuntimeError(f"layer {L}: attention implementation {impl!r}; the bias needs sdpa or eager "
                                   "(flash_attention_2 passes no mask)")
            m, hs, ehs = kwargs.get("attention_mask"), kwargs.get("hidden_states"), kwargs.get("encoder_hidden_states")
            if hs is None or ehs is None:
                raise RuntimeError(f"layer {L}: cross-attention called without hidden_states and encoder_hidden_states keywords")
            lq, lk = int(hs.shape[1]), int(ehs.shape[1])
            if (not torch.is_tensor(m) or m.dim() != 4 or tuple(m.shape[:3]) != (1, 1, lq) or m.shape[3] < lk
                    or hs.shape[0] != 1 or ehs.shape[0] != 1):
                raise RuntimeError(f"layer {L}: expected a [1, 1, {lq}, >= {lk}] mask at batch 1, got "
                                   f"{None if m is None else tuple(m.shape)} at batch {hs.shape[0]}")
            if mod.config.num_attention_heads != self.n_heads:
                raise RuntimeError(f"layer {L}: {mod.config.num_attention_heads} heads, expected {self.n_heads}")
            if not self.max_key < self.n_lyr < lk:
                raise RuntimeError(f"layer {L}: need max key {self.max_key} < n_lyr {self.n_lyr} < Lk {lk}")
            key = (lq, lk, m.device, m.dtype)
            if key not in self._templates:
                self._templates[key] = template(self.spans, lq, lk, self.hop_s, self.offset_s).to(m.device, m.dtype)
            bias = torch.zeros(1, self.n_heads, lq, lk, dtype=m.dtype, device=m.device)
            bias[0, steered] = self.beta * self._templates[key]
            if self.mask_in is None:
                self.mask_in = (list(m.shape), str(m.dtype))
            self.calls += 1
            self.shapes.append((L, lq, lk))
            return args, {**kwargs, "attention_mask": m[..., :lk] + bias}
        return pre

    def record(self) -> dict:
        """What a run row stores: calls, the (Lq, Lk) pairs seen with their call counts, the largest
        biased key, n_lyr, and the biased cells per steered head at each pair."""
        pairs: dict[tuple[int, int], int] = {}
        for _, lq, lk in self.shapes:
            pairs[(lq, lk)] = pairs.get((lq, lk), 0) + 1
        return {"hook_calls": self.calls, "lq_lk": [[lq, lk, n] for (lq, lk), n in pairs.items()],
                "max_key": self.max_key, "n_lyr": self.n_lyr, "beta": self.beta,
                "heads": {str(L): hs for L, hs in self.heads.items()},
                "cells": {f"{lq}x{lk}": self.cells(lq, lk) for lq, lk in pairs},
                "mask_in": self.mask_in}
