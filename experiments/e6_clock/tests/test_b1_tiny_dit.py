"""dacapo.attn_bias on ACE-Step's own DiT classes at ca1e85f with small random weights (ACE env, no checkpoint;
skipped elsewhere): beta 0 leaves the output bit-identical over 3 cached steps, the bias reaches only the
steered head, and the hook runs once per steered layer and step.

  ACESTEP_REPO=... venv-ace/bin/python -m pytest experiments/e6_clock/tests
"""
import contextlib
import os
import sys
import types

import pytest

if not os.environ.get("ACESTEP_REPO"):
    pytest.skip("ACESTEP_REPO not set", allow_module_level=True)
sys.path.insert(0, os.environ["ACESTEP_REPO"])
M = pytest.importorskip("acestep.models.turbo.modeling_acestep_v15_turbo")

import torch  # noqa: E402
from acestep.models.turbo.configuration_acestep_v15 import AceStepConfig  # noqa: E402
from transformers.cache_utils import DynamicCache, EncoderDecoderCache  # noqa: E402

from dacapo.attn_bias import CrossAttnBias, template, word_spans  # noqa: E402

ALIGN = {2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}
T, LK, N_LYR = 350, 60, 26  # 14 s of 25 Hz latents -> Lq 175
NOTES = [{"start": s, "dur": d} for s, d in ((4, .9), (5, .9), (6, .9), (7, .9), (8, 1.8), (10, 1.8), (12, .9),
                                             (13, 1.8), (15, 1.8), (17, .9), (18, .9), (19, .9))]  # item00
SPANS = word_spans(NOTES, 100, [[12 + j] for j in range(12)])


def config(impl: str, layers: int = 8) -> AceStepConfig:
    # the turbo checkpoint's structure (16 query heads, 8 key/value heads, alternating sliding and full
    # layers) at hidden size 256
    c = AceStepConfig(hidden_size=256, intermediate_size=512, head_dim=16, num_hidden_layers=layers)
    c._attn_implementation = impl
    return c


def tiny(impl: str, dtype):
    torch.manual_seed(0)
    dec = M.AceStepDiTModel(config(impl)).eval().to(dtype)
    g = torch.Generator().manual_seed(1)
    inputs = {"hidden_states": torch.randn(1, T, 64, generator=g).to(dtype),
              "context_latents": torch.randn(1, T, 128, generator=g).to(dtype),
              "encoder_hidden_states": torch.randn(1, LK, 256, generator=g).to(dtype),
              "attention_mask": torch.ones(1, T, dtype=dtype), "encoder_attention_mask": torch.ones(1, LK, dtype=dtype)}
    return dec, inputs


def run(dec, inputs, bias=None, steps=3):
    """3 decoder calls sharing one cross-attention cache, as ACE-Step's sampler does."""
    pkv = EncoderDecoderCache(DynamicCache(), DynamicCache())
    outs = []
    with torch.no_grad(), (bias if bias is not None else contextlib.nullcontext()):
        for s in range(steps):
            t = torch.full((1,), 1.0 - s / steps, dtype=inputs["hidden_states"].dtype)
            o = dec(timestep=t, timestep_r=t, use_cache=True, past_key_values=pkv, **inputs)
            pkv = o[1]
            outs.append(o[0])
    return torch.stack(outs)


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_beta_zero_is_bit_identical_and_beta_two_is_not(impl, dtype):
    dec, inputs = tiny(impl, dtype)
    base = run(dec, inputs)
    zero = CrossAttnBias(dec, ALIGN, SPANS, 0.0, N_LYR)
    assert torch.equal(run(dec, inputs, zero), base)
    assert zero.calls == 15 and zero.record()["lq_lk"] == [[175, LK, 15]]  # 3 steps x 5 layers
    assert zero.mask_in == ([1, 1, 175, LK], str(dtype))
    two = CrossAttnBias(dec, ALIGN, SPANS, 2.0, N_LYR)
    changed = run(dec, inputs, two)
    assert two.calls == 15 and not torch.equal(changed, base)
    assert torch.equal(run(dec, inputs), base)  # hooks gone after exit


def test_flash_attention_is_refused():
    dec, inputs = tiny("sdpa", torch.float32)
    # what the hook reads from a cross-attention module whose model would run flash_attention_2
    flash = types.SimpleNamespace(_attn_implementation="flash_attention_2", num_attention_heads=16)
    for L in ALIGN:
        dec.layers[L].cross_attn.config = flash
    with pytest.raises(RuntimeError, match="flash_attention_2"):
        run(dec, inputs, CrossAttnBias(dec, ALIGN, SPANS, 0.0, N_LYR), steps=1)


def test_only_the_steered_head_changes_and_matches_a_manual_softmax():
    c = config("sdpa")
    torch.manual_seed(0)

    class Dec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
            self.layers[2].cross_attn = M.AceStepAttention(c, 2, is_cross_attention=True)

    dec = Dec().eval()
    mod = dec.layers[2].cross_attn
    hs, ehs = torch.randn(1, 175, 256), torch.randn(1, LK, 256)
    mask = torch.zeros(1, 1, 175, LK)
    pre_o = {}
    mod.o_proj.register_forward_pre_hook(lambda m, a: pre_o.__setitem__("x", a[0].clone()))

    def heads_out(bias=None):
        with torch.no_grad(), (bias if bias is not None else contextlib.nullcontext()):
            mod(hidden_states=hs, encoder_hidden_states=ehs, attention_mask=mask, past_key_value=None)
        return pre_o["x"].view(1, 175, 16, 16)

    base = heads_out()
    cb = CrossAttnBias(dec, {2: [6]}, SPANS, 2.0, N_LYR)
    steered = heads_out(cb)
    tpl = template(SPANS, 175, LK)
    rows = sorted({int(q) for q in torch.nonzero(tpl.sum(1)).flatten()})
    assert [h for h in range(16) if not torch.equal(base[:, :, h], steered[:, :, h])] == [6]  # partner head 7 unchanged
    changed_rows = sorted({int(q) for q in (base[0, :, 6] != steered[0, :, 6]).any(-1).nonzero().flatten()})
    assert changed_rows == rows == list(range(30, 149))
    with torch.no_grad():
        q = mod.q_norm(mod.q_proj(hs).view(1, 175, -1, 16)).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(ehs).view(1, LK, -1, 16)).transpose(1, 2).repeat_interleave(2, 1)
        v = mod.v_proj(ehs).view(1, LK, -1, 16).transpose(1, 2).repeat_interleave(2, 1)
        bias = torch.zeros(1, 16, 175, LK)
        bias[0, 6] = 2.0 * tpl
        ref = (torch.softmax(q @ k.transpose(-1, -2) * mod.scaling + bias, -1) @ v).transpose(1, 2)
    assert float((steered - ref).abs().max()) < 1e-6
    assert cb.calls == 1
