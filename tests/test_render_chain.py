"""Integration: render a known melody and check the analysis chain recovers it (skips without FluidSynth)."""
import shutil

import pytest

from dacapo.symbolic import melody_from_spec

pytestmark = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not installed")


def test_solo_flute_melody_is_recovered(tmp_path):
    from dacapo.analysis import chroma_argmax, f0
    from dacapo.metrics import frame_metrics, pitch_class_accuracy
    from dacapo.render import GM, render_melody

    try:
        m = melody_from_spec([(72, 1), (74, 1), (76, 1), (77, 1), (79, 2), (77, 1), (76, 1)], bpm=100)
        wav = render_melody(m, tmp_path / "x.wav", program=GM["flute"])
    except FileNotFoundError as e:  # no soundfont configured
        pytest.skip(str(e))
    rt, rf = m.f0_track()
    fm = frame_metrics(rt, rf, *f0(wav, "swift"))
    # SwiftF0 makes octave errors on high flute (observed RPA ~0.85, RCA ~0.99), so RPA gets a loose bound.
    assert fm["RCA"] > 0.95 and fm["RPA"] > 0.75
    assert pitch_class_accuracy(rt, rf, *chroma_argmax(wav))["pc_acc"] > 0.8
