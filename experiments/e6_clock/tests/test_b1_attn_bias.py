"""dacapo.attn_bias without ACE-Step: windows, templates, refusals, head maps (fake cross-attention modules).
The checks on ACE-Step's own classes are in test_b1_tiny_dit.py.

  python -m pytest experiments/e6_clock/tests
"""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from dacapo.attn_bias import CrossAttnBias, rotate_spans, template, word_spans
from dacapo.words import group_words

ROOT = Path(__file__).resolve().parents[3]
ITEMS = {it["id"]: it for it in json.loads((ROOT / "experiments" / "e4_sing" / "manifest.json").read_text())["items"]}
TOKENS = json.loads((ROOT / "tests" / "data" / "e6_lyric_tokens.json").read_text())["lyrics"]
ALIGN = {2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}
LQ = 175  # 14 s text2music


def layout(item_id: str):
    """keys per word and n_lyr from the recorded ACE-Step tokenisation of an E4 lyric."""
    fx = TOKENS[item_id]
    words = group_words(fx["tokens"])
    return [[fx["start"] + t for t in w.tokens] for w in words], len(fx["ids"])


def spans_of(item_id: str):
    keys, n_lyr = layout(item_id)
    it = ITEMS[item_id]
    return word_spans(it["melody"]["notes"], it["melody"]["bpm"], keys), n_lyr


class _Cfg:
    def __init__(self, impl="sdpa", heads=16):
        self._attn_implementation, self.num_attention_heads = impl, heads


class _Cross(nn.Module):
    """Stands in for AceStepAttention: keeps every mask it receives."""

    def __init__(self, impl="sdpa"):
        super().__init__()
        self.config = _Cfg(impl)
        self.seen = []

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        self.seen.append(attention_mask)
        return hidden_states, None


class _Layer(nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.cross_attn = _Cross(impl)


class _Decoder(nn.Module):
    def __init__(self, impl="sdpa", n=8):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(impl) for _ in range(n)])

    def forward(self, lq=LQ, lk=140, batch=1, mask="zeros", dtype=torch.bfloat16):
        hs, ehs = torch.zeros(batch, lq, 4, dtype=dtype), torch.zeros(batch, lk, 4, dtype=dtype)
        m = torch.zeros(batch, 1, lq, lk, dtype=dtype) if isinstance(mask, str) else mask
        for layer in self.layers:
            layer.cross_attn(hidden_states=hs, encoder_hidden_states=ehs, attention_mask=m)


def masks(dec, L):
    return dec.layers[L].cross_attn.seen


def test_item00_windows():
    spans, n_lyr = spans_of("item00")
    assert n_lyr == 26 and len(spans) == 12
    assert spans[0] == ((12,), 2.4, 3.0) and spans[1][1:] == (3.0, 3.6)
    assert spans[-1] == ((23,), 11.4, 11.94)  # last word: onset 19 beats, note end 19.9 beats
    assert [s[1] for s in spans] == [2.4, 3.0, 3.6, 4.2, 4.8, 6.0, 7.2, 7.8, 9.0, 10.2, 10.8, 11.4]
    assert [s[2] for s in spans[:-1]] == [s[1] for s in spans[1:]]


def test_item00_template_rows_and_keys():
    spans, n_lyr = spans_of("item00")
    t = template(spans, LQ, 140)
    centre = [round(0.08 * q + 0.04, 6) for q in range(LQ)]
    biased = [q for q in range(LQ) if t[q].any()]
    # frames before the first onset (2.4 s) and after the last note end (11.94 s) get nothing
    assert biased[0] == 30 and centre[29] < 2.4 <= centre[30]
    assert biased[-1] == 148 and centre[148] < 11.94 <= centre[149]
    assert biased == list(range(30, 149)) and not t[:30].any() and not t[149:].any()
    # each biased row holds exactly the keys of one word, and the rows follow the written onsets
    word_keys = [set(s[0]) for s in spans]
    for q in biased:
        cols = set(torch.nonzero(t[q]).flatten().tolist())
        j = next(j for j, (_, a, b) in enumerate(spans) if a <= centre[q] < b)
        assert cols == word_keys[j]
    assert set(torch.nonzero(t[30]).flatten().tolist()) == {12}  # the first word's key
    for (keys, a, _), q0 in zip(spans, [next(q for q in range(LQ) if centre[q] >= s[1]) for s in spans]):
        assert t[q0, list(keys)].all() and (q0 == 0 or not t[q0 - 1, list(keys)].any())
    # frame 37 has its centre exactly on the second onset (3.0 s) and belongs to the second word
    assert centre[37] == 3.0 and set(torch.nonzero(t[37]).flatten().tolist()) == {13}
    assert int(t.sum()) == 149 - 30 and t[:, n_lyr:].sum() == 0


def test_multi_token_word_gets_all_its_keys():
    spans, n_lyr = spans_of("item10")
    glows = [s for s in spans if len(s[0]) == 2]
    assert n_lyr == 28 and len(glows) == 1 and glows[0][0] == (14, 15)  # "The lamp glows"
    t = template(spans, LQ, 140)
    rows = [q for q in range(LQ) if t[q, 14]]
    assert rows and all(t[q, 15] for q in rows) and all(int(t[q].sum()) == 2 for q in rows)


def test_bias_reaches_only_the_steered_heads_and_cells():
    spans, n_lyr = spans_of("item00")
    dec = _Decoder()
    with CrossAttnBias(dec, ALIGN, spans, 2.0, n_lyr) as cb:
        dec()
    t = template(spans, LQ, 140).to(torch.bfloat16)
    for L in range(8):
        if L not in ALIGN:
            assert masks(dec, L)[0].shape == (1, 1, LQ, 140) and not masks(dec, L)[0].any()
            continue
        m = masks(dec, L)[0]
        assert m.shape == (1, 16, LQ, 140) and m.dtype == torch.bfloat16
        for h in range(16):
            assert torch.equal(m[0, h], 2.0 * t if h in ALIGN[L] else torch.zeros_like(t))
    assert cb.calls == 5 and cb.max_key == 23 and cb.record()["lq_lk"] == [[LQ, 140, 5]]
    assert cb.record()["cells"] == {f"{LQ}x140": 119}
    with torch.no_grad():
        dec()
    assert masks(dec, 2)[-1].shape == (1, 1, LQ, 140)  # hooks removed on exit


def test_beta_zero_adds_exact_zeros():
    spans, n_lyr = spans_of("item00")
    dec = _Decoder()
    with CrossAttnBias(dec, ALIGN, spans, 0.0, n_lyr) as cb:
        dec()
    assert cb.calls == 5 and all(not masks(dec, L)[0].any() and masks(dec, L)[0].shape[1] == 16 for L in ALIGN)


def test_template_cache_per_shape():
    spans, n_lyr = spans_of("item00")
    dec = _Decoder()
    with CrossAttnBias(dec, ALIGN, spans, 1.0, n_lyr) as cb:
        dec(lk=140)
        dec(lk=140)
        dec(lk=133)
    assert len(cb._templates) == 2 and cb.calls == 15
    a, b = cb._templates[(LQ, 140, torch.device("cpu"), torch.bfloat16)], cb._templates[(LQ, 133, torch.device("cpu"), torch.bfloat16)]
    assert torch.equal(a[:, :133], b) and not a[:, 133:].any()  # same lyric keys at the second Lk
    assert cb.record()["lq_lk"] == [[LQ, 140, 10], [LQ, 133, 5]]
    assert [s[2] for s in cb.shapes] == [140] * 10 + [133] * 5


def test_wider_mask_is_cut_to_lk():
    spans, n_lyr = spans_of("item00")
    dec = _Decoder()
    with CrossAttnBias(dec, {2: [6]}, spans, 4.0, n_lyr):
        dec(lk=140, mask=torch.zeros(1, 1, LQ, 150, dtype=torch.bfloat16))
    assert masks(dec, 2)[0].shape == (1, 16, LQ, 140)


@pytest.mark.parametrize("case", ["flash", "none", "batch2", "short_lk", "key_beyond_lyric", "heads"])
def test_refusals(case):
    spans, n_lyr = spans_of("item00")
    if case == "key_beyond_lyric":
        with pytest.raises(ValueError):
            CrossAttnBias(_Decoder(), ALIGN, spans + [((n_lyr,), 12.0, 13.0)], 2.0, n_lyr)
        return
    if case == "heads":
        for bad in ({}, {2: []}, {2: [16]}, {2: [-1]}):
            with pytest.raises(ValueError):
                CrossAttnBias(_Decoder(), bad, spans, 2.0, n_lyr)
        return
    dec = _Decoder(impl="flash_attention_2" if case == "flash" else "sdpa")
    kw = {"none": {"mask": None}, "batch2": {"batch": 2}, "short_lk": {"lk": n_lyr}}.get(case, {})
    with CrossAttnBias(dec, ALIGN, spans, 0.0, n_lyr):
        with pytest.raises(RuntimeError):
            dec(**kw)


def test_any_head_map_and_rotated_spans_bias_exactly_the_intended_cells():
    import sys
    sys.path.insert(0, str(ROOT / "experiments" / "e6_clock"))
    import readout
    spans, n_lyr = spans_of("item00")
    shuffled = rotate_spans(spans, len(spans) // 2)
    assert [s[0] for s in shuffled] == [s[0] for s in spans]
    assert [s[1:] for s in shuffled] == [spans[(j + 6) % 12][1:] for j in range(12)]
    for heads in (readout.random_head_sets()[0], {L: list(range(16)) for L in (2, 3, 4, 5, 6)}, {4: [0, 15]}):
        for sp in (spans, shuffled):
            dec = _Decoder()
            with CrossAttnBias(dec, heads, sp, 2.0, n_lyr):
                dec()
            want = template(sp, LQ, 140).to(torch.bfloat16)
            for L in range(8):
                m = masks(dec, L)[0]
                if L not in heads:
                    assert m.shape[1] == 1 and not m.any()
                    continue
                for h in range(16):
                    assert torch.equal(m[0, h], 2.0 * want if h in heads[L] else torch.zeros_like(want))
    # the rotated template has the same keys and nearly the same number of cells, in other rows
    a, b = template(spans, LQ, 140), template(shuffled, LQ, 140)
    assert not torch.equal(a, b) and set(torch.nonzero(a.sum(0)).flatten().tolist()) == set(torch.nonzero(b.sum(0)).flatten().tolist())
    assert abs(int(a.sum()) - int(b.sum())) <= len(spans)


def test_word_spans_checks():
    with pytest.raises(ValueError):
        word_spans([{"start": 4.0, "dur": 0.9}], 100, [[12], [13]])
    with pytest.raises(ValueError):
        word_spans([{"start": 4.0, "dur": 0.9}, {"start": 4.0, "dur": 0.9}], 100, [[12], [13]])
    with pytest.raises(ValueError):
        word_spans([{"start": 4.0, "dur": 0.9}], 100, [[]])


def test_lyric_keys_from_acestep_tokenisation():
    """lyric_keys with a stand-in handler that returns the recorded ids and token texts of item10."""
    from dacapo.attn_bias import lyric_keys
    fx = TOKENS["item10"]

    class Tok:
        def __call__(self, text, **kw):
            assert text == f"# Languages\nen\n\n# Lyric\n{fx['text']}<|endoftext|>"
            return type("Enc", (), {"input_ids": torch.tensor([fx["ids"]])})()

    class H:
        text_tokenizer = Tok()

        def _format_lyrics(self, lyrics, language):
            return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

        def _extract_lyric_segment(self, ids, language):
            raw = ids[0].tolist()
            return raw, raw[fx["start"]:fx["end"]], fx["start"], fx["end"]

    class Aligner:
        def _decode_tokens_incrementally(self, pure):
            assert pure == fx["ids"][fx["start"]:fx["end"]]
            return fx["tokens"]

    lay = lyric_keys(H(), Aligner(), fx["text"])
    assert lay["n_lyr"] == len(fx["ids"]) == 28 and lay["start"] == 9 and lay["ids"] == fx["ids"]
    assert lay["keys"][:4] == [[12], [13], [14, 15], [16]] and lay["words"][2] == "glows"
    assert lay["keys"] == layout("item10")[0]


def test_word_spans_takes_note_objects():
    pytest.importorskip("pretty_midi")  # dacapo.symbolic needs it; the ACE environment may not have it
    from dacapo.symbolic import Note
    notes = [Note(**n) for n in ITEMS["item00"]["melody"]["notes"]]
    keys, _ = layout("item00")
    assert word_spans(notes, 100.0, keys) == spans_of("item00")[0]
