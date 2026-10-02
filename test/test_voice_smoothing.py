"""実モデルを配布せず、固定波形と既知の周期列で任意の補正処理を検証する。"""

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("parselmouth", reason="CPU backend dependencies are not installed")

from coeirocore import voice_smoothing as smoothing
from coeirocore.voice_smoothing_common import _vowel_spans


def harmonic_wave(sampling_rate=44100, seconds=1.6):
    times = np.arange(round(sampling_rate * seconds)) / sampling_rate
    fundamental = 220.5
    harmonics = np.arange(1, 13)
    components = (
        0.06 / harmonics * np.cos(2 * np.pi * fundamental * times[:, None] * harmonics)
    )
    marks = np.arange(
        0.05 * sampling_rate,
        (seconds - 0.05) * sampling_rate,
        sampling_rate / fundamental,
    )
    return times, components, marks


@pytest.mark.parametrize("sampling_rate", [22050, 44100, 48000])
def test_stable_harmonics_are_preserved(sampling_rate):
    _, components, marks = harmonic_wave(sampling_rate)
    original = components.sum(axis=1)
    shaped = smoothing._smooth_magnitude(original, [marks], sampling_rate)
    result = smoothing._smooth_periods(shaped, [marks], sampling_rate)
    np.testing.assert_allclose(result, original, rtol=0, atol=1e-7)


def test_magnitude_correction_reduces_known_modulation():
    times, components, marks = harmonic_wave()
    clean = components.sum(axis=1)
    changed_components = components.copy()
    changed_components[:, 3:] *= np.exp(0.3 * np.sin(2 * np.pi * 30 * times))[:, None]
    noisy = changed_components.sum(axis=1)
    result = smoothing._smooth_magnitude(noisy, [marks], 44100)
    keep = (times > 0.3) & (times < 1.3)
    assert np.linalg.norm((result - clean)[keep]) < np.linalg.norm(
        (noisy - clean)[keep]
    )


def test_fft_batches_do_not_change_the_result():
    times, components, marks = harmonic_wave(seconds=2)
    components[:, 3:] *= np.exp(0.3 * np.sin(2 * np.pi * 30 * times))[:, None]
    original = components.sum(axis=1)
    with patch.object(smoothing, "_FRAME_BATCH", 10000):
        reference = smoothing._smooth_magnitude(original, [marks], 44100)
    with patch.object(smoothing, "_FRAME_BATCH", 17):
        batched = smoothing._smooth_magnitude(original, [marks], 44100)
    np.testing.assert_allclose(batched, reference, rtol=0, atol=1e-12)


def test_vowel_spans_preserve_prosody_markers_and_use_model_hop_length():
    assert _vowel_spans(
        ["^", "a", "[", "a", "]", "s", "u", "$"],
        [1, 2, 1, 2, 1, 2, 2, 1],
        4,
    ) == [(4, 28), (36, 44)]


@pytest.mark.parametrize("tokens", [["^", "a", "$"], ["s", "_", "s"]])
def test_no_target_does_not_run_pitch_analysis(tokens):
    wave = np.zeros(3 * 512, dtype=np.float32)
    with patch.object(
        smoothing, "_pulse_marks", side_effect=AssertionError("not needed")
    ):
        result = smoothing.smooth_voice(
            wave, tokens, [1, 1, 1], sampling_rate=44100, hop_length=512
        )
    assert result is wave


def test_silence_is_not_an_analysis_error():
    wave = np.zeros(50 * 512, dtype=np.float32)
    result = smoothing.smooth_voice(
        wave, ["a"], [50], sampling_rate=44100, hop_length=512
    )
    assert result is wave


def test_bulk_pulse_read_preserves_praat_sample_positions():
    _, components, _ = harmonic_wave(seconds=0.3)
    wave = components.sum(axis=1)
    sound = smoothing.parselmouth.Sound(wave, sampling_frequency=44100)
    pitch = sound.to_pitch_cc(
        time_step=0.005,
        pitch_floor=smoothing._PITCH_FLOOR,
        pitch_ceiling=smoothing._PITCH_CEILING,
    )
    points = smoothing.call([sound, pitch], "To PointProcess (cc)")
    expected = np.array(
        [
            smoothing.call(points, "Get time from index", index)
            for index in range(1, smoothing.call(points, "Get number of points") + 1)
        ]
    )
    assert len(expected) > 0
    np.testing.assert_array_equal(
        smoothing._pulse_marks(wave, 44100), expected * 44100 - 0.5
    )


def test_smoothing_is_finite_deterministic_and_does_not_modify_input():
    times = np.arange(80 * 512) / 44100
    wave = (
        0.1 * np.cos(2 * np.pi * 220.5 * times)
        + 0.03
        * np.exp(0.2 * np.sin(2 * np.pi * 30 * times))
        * np.cos(2 * np.pi * 661.5 * times)
    ).astype(np.float32)
    original = wave.copy()
    first = smoothing.smooth_voice(
        wave, ["a"], [80], sampling_rate=44100, hop_length=512
    )
    second = smoothing.smooth_voice(
        wave, ["a"], [80], sampling_rate=44100, hop_length=512
    )
    assert first.dtype == wave.dtype and first.shape == wave.shape
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert np.array_equal(wave, original)
    assert np.isclose(
        np.sum(first.astype(float) ** 2), np.sum(original.astype(float) ** 2), rtol=1e-7
    )


def test_misaligned_durations_are_not_silently_ignored():
    with pytest.raises(ValueError, match="do not align"):
        smoothing.smooth_voice(
            np.zeros(512, dtype=np.float32),
            ["a"],
            [2],
            sampling_rate=44100,
            hop_length=512,
        )


def test_analysis_failure_is_not_a_silent_bypass():
    wave = np.zeros(50 * 512, dtype=np.float32)
    with (
        patch.object(
            smoothing, "_pulse_marks", side_effect=RuntimeError("analysis failed")
        ),
        pytest.raises(RuntimeError, match="analysis failed"),
    ):
        smoothing.smooth_voice(wave, ["a"], [50], sampling_rate=44100, hop_length=512)
