"""Keep the registered ASR contract at the model and waveform boundaries."""
from __future__ import annotations

import importlib.metadata
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from experiments.e7_melody_adapter import pack, registration, suno


@pytest.fixture
def model_boundary(monkeypatch):
    calls = []
    versions = {"demucs": "4.1.0", "swift-f0": "0.3.0", "basic-pitch": "0.4.0",
                "transformers": "4.57.6"}
    dtype = object()
    torch = types.ModuleType("torch")
    torch.float16 = dtype
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Model:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            return self

    separator, model, recognizer, executor = Model(), Model(), object(), object()
    processor = types.SimpleNamespace(tokenizer=object(), feature_extractor=object())

    def get_model(name):
        calls.append(("separator", name))
        return separator

    def load_model(*args, **kwargs):
        calls.append(("model", args, kwargs))
        return model

    def load_processor(*args, **kwargs):
        calls.append(("processor", args, kwargs))
        return processor

    def pipeline(*args, **kwargs):
        calls.append(("pipeline", args, kwargs))
        return recognizer

    def version(package):
        calls.append(("version", package))
        return versions[package]

    def pool(workers):
        calls.append(("pool", workers))
        return executor

    demucs = types.ModuleType("demucs")
    pretrained = types.ModuleType("demucs.pretrained")
    pretrained.get_model = get_model
    demucs.pretrained = pretrained
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForSpeechSeq2Seq = types.SimpleNamespace(from_pretrained=load_model)
    transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=load_processor)
    transformers.pipeline = pipeline
    for name, module in (("demucs", demucs), ("demucs.pretrained", pretrained),
                         ("transformers", transformers)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(suno.lib, "reference_pool", pool)
    return types.SimpleNamespace(calls=calls, versions=versions, dtype=dtype, torch=torch,
                                 model=model, processor=processor, recognizer=recognizer,
                                 executor=executor)


def test_registered_asr_model_revision_and_pipeline_settings_reach_runtime(model_boundary):
    boundary = model_boundary
    cfg = registration.build_config()
    scorer = suno.AudioScorer(cfg["models"], cfg["suno"], cfg["asr"])
    assert [call[1] for call in boundary.calls if call[0] == "version"] == [
        "demucs", "swift-f0", "basic-pitch", "transformers"]
    assert ("model", (cfg["models"]["asr"],), {
        "revision": cfg["models"]["asr_revision"], "torch_dtype": boundary.dtype,
        "low_cpu_mem_usage": True, "use_safetensors": True}) in boundary.calls
    assert ("processor", (cfg["models"]["asr"],), {
        "revision": cfg["models"]["asr_revision"]}) in boundary.calls
    assert ("pipeline", ("automatic-speech-recognition",), {
        "model": boundary.model, "tokenizer": boundary.processor.tokenizer,
        "feature_extractor": boundary.processor.feature_extractor,
        "torch_dtype": boundary.dtype, "chunk_length_s": 30, "stride_length_s": 5,
        "batch_size": 1, "device": 0}) in boundary.calls
    assert boundary.calls.count(("to", "cuda")) == 2
    assert ("pool", cfg["suno"]["search_workers"]) in boundary.calls
    assert scorer.asr is boundary.recognizer and scorer.executor is boundary.executor


@pytest.mark.parametrize("package", ("demucs", "swift-f0", "basic-pitch", "transformers"))
def test_runtime_version_mismatch_stops_before_model_loading(model_boundary, package):
    cfg = registration.build_config()
    model_boundary.versions[package] = "0.0.0"
    with pytest.raises(RuntimeError, match="Use registered " + package):
        suno.AudioScorer(cfg["models"], cfg["suno"], cfg["asr"])
    assert all(call[0] == "version" for call in model_boundary.calls)


def test_no_cuda_stops_without_a_cpu_model_fallback(model_boundary):
    cfg = registration.build_config()
    model_boundary.torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="CUDA is required"):
        suno.AudioScorer(cfg["models"], cfg["suno"], cfg["asr"])
    assert model_boundary.calls == []


def test_unsupported_waveform_dtype_stops_before_model_loading(model_boundary):
    cfg = registration.build_config()
    cfg["asr"]["input_dtype"] = "float64"
    with pytest.raises(ValueError, match="Unsupported ASR input preparation"):
        suno.AudioScorer(cfg["models"], cfg["suno"], cfg["asr"])
    assert all(call[0] == "version" for call in model_boundary.calls)


def test_one_sample_decodable_audio_reaches_separator_with_finite_normalization(monkeypatch, tmp_path):
    import torch

    source = tmp_path / "one-sample.wav"
    waveform = np.array([[.2, -.1]], dtype="float32")
    sf.write(source, waveform, 44100, subtype="FLOAT")
    work = tmp_path / "separation"
    work.mkdir()
    scorer = object.__new__(suno.AudioScorer)
    scorer.cfg = registration.build_config()["suno"]
    scorer.separator = types.SimpleNamespace(samplerate=44100, audio_channels=2, sources=["vocals"])
    applied = []

    def apply_model(model, normalized, **kwargs):
        assert torch.isfinite(normalized).all()
        assert normalized.shape == (1, 2, 1)
        applied.append(kwargs)
        # One identity source isolates decoding, normalization and restoration.
        return normalized[:, None]

    demucs = types.ModuleType("demucs")
    apply = types.ModuleType("demucs.apply")
    apply.apply_model = apply_model
    demucs.apply = apply
    monkeypatch.setitem(sys.modules, "demucs", demucs)
    monkeypatch.setitem(sys.modules, "demucs.apply", apply)
    vocal_path, voice = scorer.separate(source, work)
    vocals, sample_rate = sf.read(vocal_path, always_2d=True, dtype="float32")
    assert len(applied) == 1 and applied[0]["device"] == "cuda"
    assert sample_rate == 44100 and vocals.shape == waveform.shape
    assert np.isfinite(vocals).all()
    np.testing.assert_allclose(vocals, waveform, atol=1e-7)
    assert voice["duration_seconds"] == 1 / 44100


@pytest.mark.parametrize("sample_rate,f0_frames", ((16000, 0), (48000, 1)))
def test_short_waveform_keeps_registered_asr_preparation_and_pitch_search(
        monkeypatch, tmp_path, sample_rate, f0_frames):
    import librosa
    from dacapo import analysis

    cfg = registration.build_config()
    items = pack.build_manifest()["items"]
    waveform = np.column_stack((np.linspace(-.2, .6, 160), np.linspace(.4, -.1, 160))).astype("float32")
    vocal_path = tmp_path / "short.wav"
    sf.write(vocal_path, waveform, sample_rate, subtype="FLOAT")
    scorer = object.__new__(suno.AudioScorer)
    scorer.asr_config = cfg["asr"]
    scorer.executor = object()
    scorer.separate = lambda path, work: (vocal_path, {"present": False})
    monkeypatch.setattr(analysis, "f0_swift", lambda *args, **kwargs:
                        (np.arange(f0_frames) * .01, np.full(f0_frames, 440.)))
    monkeypatch.setattr(analysis, "notes_basic_pitch", lambda *args, **kwargs: [])
    monkeypatch.setattr(analysis, "skyline", lambda events: events)
    pitch_calls, asr_calls, resample_calls = [], [], []
    resample = librosa.resample

    def capture_resample(wave, **kwargs):
        resample_calls.append((wave.copy(), kwargs))
        return resample(wave, **kwargs)

    def score_pitch(own, other, times, f0, **kwargs):
        pitch_calls.append((own, other, times.copy(), f0.copy(), kwargs))
        return {"recorded": True}

    def transcribe(value, **kwargs):
        asr_calls.append((value, kwargs))
        return {"text": items[0]["lyrics"]}

    monkeypatch.setattr(librosa, "resample", capture_resample)
    monkeypatch.setattr(suno.lib, "score_against_references", score_pitch)
    scorer.asr = transcribe
    clip = {"path": vocal_path, "source": "suno", "item_id": items[0]["id"],
            "route": "R1", "take": "a", "owner_record": {}}
    row = scorer.score(clip, items, tmp_path / "work")

    mono = waveform.mean(axis=1)
    if sample_rate == 16000:
        assert resample_calls == []
        expected = mono
    else:
        assert len(resample_calls) == 1
        np.testing.assert_array_equal(resample_calls[0][0], mono)
        assert resample_calls[0][1] == {
            "orig_sr": sample_rate, "target_sr": 16000, "res_type": "soxr_hq",
            "fix": True, "scale": False}
        expected = resample(mono, orig_sr=sample_rate, target_sr=16000,
                            res_type="soxr_hq", fix=True, scale=False)
    assert len(asr_calls) == 1
    payload, arguments = asr_calls[0]
    assert payload["sampling_rate"] == 16000 and payload["array"].dtype == np.float32
    np.testing.assert_array_equal(payload["array"], expected)
    assert arguments == {"generate_kwargs": {"language": "english", "task": "transcribe"}}
    assert row["words"]["wr"] == 1.0 and row["pitch"] == {"recorded": True}
    assert len(pitch_calls) == 1 and len(pitch_calls[0][1]) == 11
    np.testing.assert_array_equal(pitch_calls[0][2], [0., .01])
    np.testing.assert_array_equal(pitch_calls[0][3], [440. if f0_frames else 0., 0.])
    assert pitch_calls[0][4]["executor"] is scorer.executor
