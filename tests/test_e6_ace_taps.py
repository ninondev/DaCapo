"""dacapo.ace.generate with taps and keep_extra, on a fake generate_music (ACE env, no weights; skipped elsewhere).

  ACESTEP_REPO=... venv-ace/bin/python -m pytest tests/test_e6_ace_taps.py
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

if not os.environ.get("ACESTEP_REPO"):
    pytest.skip("ACESTEP_REPO not set", allow_module_level=True)
sys.path.insert(0, os.environ["ACESTEP_REPO"])
pytest.importorskip("acestep")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from dacapo import ace  # noqa: E402
from acestep.core.generation.handler.generate_music import GenerateMusicMixin  # noqa: E402
from acestep.core.generation.handler.generate_music_request import GenerateMusicRequestMixin  # noqa: E402


class Dec:
    def forward(self, *a, **kw):  # class-level: StepHook deletes the instance attribute on exit
        return (torch.zeros_like(kw["hidden_states"]),)


class H(GenerateMusicRequestMixin, GenerateMusicMixin):
    def __init__(self):
        self.model = types.SimpleNamespace(decoder=Dec())

    def is_turbo_model(self):
        return True


class Tap:
    def __init__(self):
        self.state = []

    def __enter__(self):
        self.state.append("in")
        return self

    def __exit__(self, *exc):
        self.state.append("out")
        return False


def _fake(active: list, tap: Tap, extra: dict):
    def fake_generate_music(h, llm, params, cfg, save_dir=None):
        active.append(list(tap.state))  # the tap is entered while ACE-Step runs
        for _ in range(params.inference_steps):
            h.model.decoder.forward(hidden_states=torch.zeros(1, 350, 64), timestep=torch.tensor([1.0]))
        logger.info("[Normalization] Audio 0 BEFORE: Peak=0.6775, Target=-1.0dB")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        p = Path(save_dir) / "x.wav"
        sf.write(str(p), np.zeros((480, 2), dtype="float32"), 48000)
        return types.SimpleNamespace(success=True, error=None, audios=[{"path": str(p)}], extra_outputs=extra)
    return fake_generate_music


def test_taps_wrap_the_call_and_keep_extra_returns_the_tensors(tmp_path, monkeypatch):
    tap, active = Tap(), []
    extra = {k: torch.ones(1) for k in ace.EXTRA_KEYS} | {"time_costs": {}}
    monkeypatch.setattr(ace, "generate_music", _fake(active, tap, extra))
    r = ace.generate(H(), tmp_path / "o.wav", "a cappella", 14.0, 61000, lyrics="[Verse]\nla", taps=(tap,),
                     keep_extra=True, rng_seed=961000, extra={"dcw_enabled": False})
    assert active == [["in"]] and tap.state == ["in", "out"]
    assert set(r["extra"]) == set(ace.EXTRA_KEYS) and r["n_denoiser_calls"] == 8
    assert r["peak_before_norm"] == 0.6775 and r["dcw_enabled"] is False


def test_defaults_add_nothing(tmp_path, monkeypatch):
    tap, active = Tap(), []
    monkeypatch.setattr(ace, "generate_music", _fake(active, tap, {}))
    r = ace.generate(H(), tmp_path / "o.wav", "a cappella", 14.0, 61000, lyrics="[Verse]\nla")
    assert "extra" not in r and tap.state == [] and r["rng_seed"] is None
