"""dacapo.ace without a model (ACE-Step environment; skipped elsewhere). A fake generate_music runs in a child
thread like ACE-Step's service thread, draws from the global torch generator like the VAE posterior sample,
calls the DiT decoder once, logs ACE-Step's own messages through loguru and emits a determinism warning.

  ACESTEP_REPO=... PYTHONPATH=. <ace python> -m pytest tests/test_e5v2_generation.py
"""
from __future__ import annotations

import os
import re
import sys
import threading
import types
import warnings
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

from acestep.constants import DEFAULT_DIT_INSTRUCTION, TASK_INSTRUCTIONS  # noqa: E402
from acestep.core.generation.handler.generate_music import GenerateMusicMixin  # noqa: E402
from acestep.core.generation.handler.generate_music_request import GenerateMusicRequestMixin  # noqa: E402
from dacapo import ace  # noqa: E402

DATA = Path(__file__).parent / "data"
T2M = "Fill the audio semantic mask based on the given conditions:"
REP = "Repaint the mask area based on the given conditions:"


class Dec:
    def forward(self, *a, **kw):  # class-level: StepHook deletes the instance attribute on exit
        return (torch.zeros_like(kw["hidden_states"]),)


class H(GenerateMusicRequestMixin, GenerateMusicMixin):
    """Only what ace.generate touches; the two _resolve_* methods are ACE-Step's own."""

    def __init__(self):
        self.model = types.SimpleNamespace(decoder=Dec())

    def is_turbo_model(self):
        return True


SEEN: dict = {}


def fake_generate_music(h, llm, params, cfg, save_dir=None):
    out: dict = {}

    def service():
        out["draw"] = torch.randn(8)
        T = 450 if "18" in str(params.src_audio) else 510
        h.model.decoder.forward(hidden_states=torch.randn(1, T, 64, generator=torch.Generator().manual_seed(params.seed)),
                                timestep=torch.tensor([1.0]))
        dur = 18 if T == 450 else 20
        logger.info(f"text_prompt:\n# Instruction\n{params.instruction}\n\n# Caption\n{params.caption}\n\n# Metas\n"
                    f"- bpm: 120\n- timesignature: N/A\n- keyscale: N/A\n- duration: {dur} seconds\n<|endoftext|>\n")
        logger.info(f"[tiled_decode] chunk_size=256, offload_wav_to_cpu=True, latents_shape=torch.Size([1, 64, {T}])")
        if params.dcw_enabled is None:
            logger.info("[DCW] Active - mode=double, scaler=0.0500, high_scaler=0.0200, wavelet='haar'")
        logger.info("[Normalization] Audio 0 BEFORE: Peak=0.2515, Target=-1.0dB")
        warnings.warn("index_add_cuda_ does not have a deterministic implementation, but you set "
                      "'torch.use_deterministic_algorithms(True, warn_only=True)'")

    t = threading.Thread(target=service)
    t.start()
    t.join()
    SEEN.update(params=params, cfg=cfg, draw=out["draw"], save_dir=save_dir)
    x = torch.zeros(2, 480)
    x[0, 10], x[1, 20] = 0.5, -0.25
    audio = {"path": "", "tensor": x, "sample_rate": 48000}             # ACE-Step: normalised, [channels, samples]
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        p = Path(save_dir) / "x.wav"
        sf.write(str(p), np.zeros((480, 2), dtype="float32"), 48000)
        audio["path"] = str(p)
    return types.SimpleNamespace(success=True, error=None, audios=[audio], extra_outputs={})


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    monkeypatch.setattr(ace, "generate_music", fake_generate_music)


def run(tmp: Path, **kw):
    return ace.generate(H(), tmp / "o.wav", "solo flute melody, monophonic, no accompaniment", 18.0, 1001,
                        lyrics="[Instrumental]", bpm=120, **kw)


def src_file(tmp: Path, name="src_18.wav") -> Path:
    p = tmp / name
    sf.write(str(p), np.zeros((48000, 2), dtype="float32"), 48000)
    return p


def _blocks(text: str, marker: str) -> list[dict]:
    lines = text.split("\n")
    starts = [i for i, l in enumerate(lines) if marker in l]
    out = []
    for a, b in zip(starts, starts[1:] + [len(lines)]):
        msgs = [m.split(" - ", 1)[1] if " | " in m and " - " in m else m for m in "\n".join(lines[a:b]).split("\n2026-")]
        out.append(ace.parse_run_log(["\n".join(msgs)]))
    return out


def test_constants_match_ace():
    assert TASK_INSTRUCTIONS["repaint"] == REP and DEFAULT_DIT_INSTRUCTION == T2M
    assert ace.TASK_INSTRUCTIONS["repaint"] == REP


def test_repaint_gets_repaint_instruction(tmp_path):
    r = run(tmp_path, task="repaint", src=str(src_file(tmp_path)), extra={"repainting_start": 12.0, "repainting_end": 16.0})
    assert SEEN["params"].instruction == REP
    assert r["instruction"] == r["instruction_requested"] == r["instruction_logged"] == REP
    assert r["frames"] == 450 and r["duration_meta"] == 18 and r["decode_chunk"] == 256 and r["decode_frames"] == 450
    assert r["peak_before_norm"] == 0.2515 and r["n_prompts_logged"] == 1


def test_other_tasks_unchanged(tmp_path):
    for task in ("text2music", "cover", "cover-nofsq"):
        r = run(tmp_path, task=task, src=None if task == "text2music" else str(src_file(tmp_path)))
        assert SEEN["params"].instruction == T2M and r["instruction"] == T2M, task


def test_resolved_instruction_follows_ace(tmp_path):
    r = run(tmp_path, task="text2music", codes="<|audio_code_1|>")
    assert r["instruction_requested"] == T2M and r["instruction"] == TASK_INSTRUCTIONS["cover"]


def test_dcw_resolved(tmp_path):
    r = run(tmp_path, task="repaint", src=str(src_file(tmp_path)))
    assert r["dcw_requested"] is None and r["dcw_enabled"] is True and r["dcw_active_lines"] == 1
    r = run(tmp_path, task="repaint", src=str(src_file(tmp_path)), extra={"dcw_enabled": False})
    assert r["dcw_requested"] is False and r["dcw_enabled"] is False and r["dcw_active_lines"] == 0


def test_explicit_instruction_override(tmp_path):
    r = run(tmp_path, task="repaint", src=str(src_file(tmp_path)), extra={"instruction": T2M})
    assert r["instruction"] == r["instruction_logged"] == T2M


def test_rng_seed_pins_draw_in_service_thread(tmp_path):
    s = src_file(tmp_path)
    run(tmp_path, task="repaint", src=str(s), rng_seed=5)
    a = SEEN["draw"].clone()
    torch.randn(100)
    run(tmp_path, task="repaint", src=str(s), rng_seed=5)
    assert torch.equal(a, SEEN["draw"])
    run(tmp_path, task="repaint", src=str(s), rng_seed=6)
    assert not torch.equal(a, SEEN["draw"])
    run(tmp_path, task="repaint", src=str(s))                            # None: global stream untouched (v1)
    assert not torch.equal(a, SEEN["draw"])


def test_fixed_subtype_is_written_by_soundfile(tmp_path):
    """out_subtype: ACE-Step saves nothing; the normalised tensor is written as 16-bit FLAC, whatever
    torchaudio would have chosen."""
    r = ace.generate(H(), tmp_path / "o.flac", "solo flute melody, monophonic, no accompaniment", 18.0, 1001,
                     lyrics="[Instrumental]", bpm=120, task="repaint", src=str(src_file(tmp_path)), out_subtype="PCM_16")
    info = sf.info(str(tmp_path / "o.flac"))
    assert SEEN["save_dir"] is None and (info.format, info.subtype, info.samplerate, info.channels) == ("FLAC", "PCM_16", 48000, 2)
    y, _ = sf.read(str(tmp_path / "o.flac"), dtype="int16")
    assert y[10, 0] == 16384 and y[20, 1] == -8192 and r["frames"] == 450
    run(tmp_path, task="repaint", src=str(src_file(tmp_path)))
    assert SEEN["save_dir"] is not None                                  # default: ACE-Step's own saver (v1)


def test_sidecar_guard(tmp_path):
    s = src_file(tmp_path)
    s.with_suffix(".json").write_text("{}")
    with pytest.raises(RuntimeError, match="sidecar"):
        run(tmp_path, task="repaint", src=str(s))
    run(tmp_path, task="cover-nofsq", src=str(s))                        # cover never reads sidecars


def test_determinism_warnings_recorded(tmp_path):
    r = run(tmp_path, task="repaint", src=str(src_file(tmp_path)))
    assert len(r["determinism_warnings"]) == 1 and "index_add_cuda_" in r["determinism_warnings"][0]


def test_parse_real_pilot_log():
    """parse_run_log on an excerpt of a real CPU log (4 runs: as rendered, then cut to 18 s)."""
    got = _blocks((DATA / "ace_pilot_runs.log").read_text(), "[repaint_cache] No cached source latents")
    assert [g["instruction_logged"] for g in got] == [T2M, REP, T2M, REP]
    assert [g["duration_meta"] for g in got] == [20, 20, 18, 18]
    assert [g["decode_frames"] for g in got] == [510, 510, 450, 450]
    assert all(g["decode_chunk"] == 256 and g["dcw_active_lines"] == 0 and g["repaint_cache_used"] == 0 for g in got)
    assert [g["peak_before_norm"] for g in got] == [0.2515, 0.2199, 0.1996, 0.1861]


def test_parse_v1_colab_log():
    """Run 1 on the L4: the text2music instruction, an active-DCW log line per run, decode chunk 256."""
    text = (DATA / "ace_v1_colab_runs.log").read_text()
    got = _blocks(text, "[repaint_cache] No cached source latents")
    assert len(got) == 3 and all(g["instruction_logged"] == T2M and g["dcw_active_lines"] == 1 for g in got)
    assert REP not in text and sorted(set(int(x) for x in re.findall(r"chunk_size=(\d+)", text))) == [256]


def test_setup_determinism(monkeypatch):
    saved = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("ACESTEP_VAE_DECODE_CHUNK_SIZE", "512")
    try:
        st = ace.setup_determinism()
        assert st["cudnn_deterministic"] and not st["cudnn_benchmark"]
        assert st["deterministic_algorithms"] and st["deterministic_warn_only"]
        assert st["cublas_workspace_config"] == ":4096:8" and st["vae_decode_chunk_env"] == "256"
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[0], saved[1]
        torch.use_deterministic_algorithms(saved[2], warn_only=saved[3])


def test_pin_matches_fresh_generator_numbers():
    """Why rng_seed must differ from the DiT seed: the global CPU generator seeded with s gives the same
    numbers as ACE-Step's fresh torch.Generator().manual_seed(s)."""
    ace.pin_global_rng(1001)
    a = torch.randn(64 * 450)
    b = torch.randn(1, 450, 64, generator=torch.Generator().manual_seed(1001)).flatten()
    assert torch.equal(a, b)
