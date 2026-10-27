"""Unit tests for dacapo.words and dacapo.attn_tap (no ACE-Step, no model weights)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dacapo.attn_tap import CrossAttnTap
from dacapo.words import group_words, lyric_words, match_words, norm_word

# ACE-Step's lyric token ids and incremental decodes of the 16 read-out lyrics (Qwen3-Embedding tokenizer,
# prompt from ConditioningTextMixin._format_lyrics, slice from _extract_lyric_segment); written by
# tests/data/make_e6_lyric_tokens.py
FIXTURE = json.loads((Path(__file__).parent / "data" / "e6_lyric_tokens.json").read_text())["lyrics"]


def test_fixture_covers_every_readout_lyric():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "e6_clock"))
    import readout
    assert {lid: x["text"] for lid, x in FIXTURE.items()} == readout.lyric_texts()


@pytest.mark.parametrize("lid", sorted(FIXTURE))
def test_group_words_matches_lyric_words_for_every_lyric(lid):
    fx = FIXTURE[lid]
    assert fx["start"] == 9 and fx["ids"][fx["end"]] == 151643
    assert [w.norm for w in group_words(fx["tokens"])] == lyric_words(fx["text"])


def test_multi_token_words():
    multi = {lid: [w.text for w in group_words(fx["tokens"]) if len(w.tokens) > 1] for lid, fx in FIXTURE.items()}
    assert multi["item10"] == ["glows"]
    assert multi["v0"] == ["Carry", "willows"] and multi["v1"] == ["Gentle", "Underneath"]
    assert multi["v2"] == ["Sailing", "seagulls", "Painting"] and multi["v3"] == ["Dancing", "Carried", "higher,"]
    assert sum(map(len, multi.values())) == 11
    w1 = {x.norm: x.tokens for x in group_words(FIXTURE["v1"]["tokens"])}
    assert w1["gentle"] == (9, 10, 11) and w1["underneath"] == (17, 18)


def test_punctuation_attaches_and_never_starts_a_word():
    w = group_words(['[', 'Verse', ']\n', 'Hello', ',', ' world', '!'])
    assert [(x.norm, x.tokens) for x in w] == [("hello", (3, 4)), ("world", (5, 6))]


def test_norm_and_match():
    assert norm_word(" Don’t!") == "don't"
    assert norm_word(" 7") == "seven" and norm_word("1,") == "one" and norm_word("20") == "twenty"
    assert norm_word("1999") == ""  # not a lyric word here; dropped, never mismatched silently into a word
    pairs = match_words("i walk down to the sea".split(), "i walked down to sea".split())
    assert pairs == [(0, 0), (2, 2), (3, 3), (5, 4)]
    assert match_words(lyric_words("[Verse]\nSeven seagulls"), [norm_word(" 7"), norm_word(" seagulls.")]) == [(0, 0), (1, 1)]


@pytest.mark.skipif(not os.environ.get("E6_TOKENIZER"), reason="set E6_TOKENIZER to ace_ckpt/Qwen3-Embedding-0.6B")
def test_fixture_matches_the_live_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["E6_TOKENIZER"])
    for fx in FIXTURE.values():
        ids = tok(f"# Languages\nen\n\n# Lyric\n{fx['text']}<|endoftext|>", padding="longest", truncation=True,
                  max_length=2048, return_tensors="pt").input_ids[0].tolist()
        assert ids == fx["ids"]


class _FakeCross(nn.Module):
    def __init__(self, d=32, heads=4, kv=2, hd=8):
        super().__init__()
        self.head_dim, self.num_key_value_groups, self.scaling = hd, heads // kv, hd ** -0.5
        self.q_proj, self.k_proj = nn.Linear(d, heads * hd, bias=False), nn.Linear(d, kv * hd, bias=False)
        self.v_proj, self.o_proj = nn.Linear(d, kv * hd, bias=False), nn.Linear(heads * hd, d, bias=False)
        self.q_norm, self.k_norm = nn.LayerNorm(hd), nn.LayerNorm(hd)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        b, lq, _ = hidden_states.shape
        lk = encoder_hidden_states.shape[1]
        q = self.q_norm(self.q_proj(hidden_states).view(b, lq, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(encoder_hidden_states).view(b, lk, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).view(b, lk, -1, self.head_dim).transpose(1, 2)
        # transformers repeat_kv: expand then reshape
        k = k[:, :, None].expand(b, k.shape[1], self.num_key_value_groups, lk, self.head_dim).reshape(b, -1, lk, self.head_dim)
        v = v[:, :, None].expand(b, v.shape[1], self.num_key_value_groups, lk, self.head_dim).reshape(b, -1, lk, self.head_dim)
        w = torch.softmax(q @ k.transpose(-1, -2) * self.scaling + attention_mask, dim=-1)
        self.last_weights = w
        return self.o_proj((w @ v).transpose(1, 2).reshape(b, lq, -1)), None


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = _FakeCross()


class _Decoder(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n)])

    def forward(self, x, e, mask):
        for layer in self.layers:
            x = x + layer.cross_attn(hidden_states=x, encoder_hidden_states=e, attention_mask=mask)[0]
        return x


def test_tap_matches_eager_weights_and_leaves_output_unchanged():
    torch.manual_seed(0)
    dec = _Decoder()
    x, e = torch.randn(1, 6, 32), torch.randn(1, 5, 32)
    mask = torch.zeros(1, 1, 6, 5)
    with torch.no_grad():
        ref = dec(x, e, mask)
        with CrossAttnTap(dec, layers=(1, 2), keys=slice(0, 3)) as tap:
            out = dec(x, e, mask)
            out2 = dec(x, e, mask)
    assert torch.equal(ref, out) and torch.equal(ref, out2)
    assert len(tap.records) == 2 and set(tap.records[0]) == {1, 2}
    for L in (1, 2):
        np.testing.assert_allclose(tap.records[0][L].float().numpy(), dec.layers[L].cross_attn.last_weights[0, :, :, :3].numpy(),
                                   atol=2e-3)
    assert tap.stacked(0, 3).shape == (2, 2, 4, 3, 6)
    with torch.no_grad():
        assert torch.equal(dec(x, e, mask), ref)  # hooks removed on exit
