from unittest.mock import patch

import numpy as np
import pytest

from coeirocore.waveform import (
    detect_non_silent_range,
    resample_waveform,
    trim_silence,
)


def test_trim_matches_existing_rms_boundary_semantics() -> None:
    wave = np.zeros(4096, dtype=np.float32)
    wave[1536:2560] = 1.0

    detected_range = detect_non_silent_range(wave, top_db=30)

    assert np.array_equal(detected_range, np.asarray([1024, 3584]))
    assert np.array_equal(trim_silence(wave, top_db=30), wave[1024:3584])


def test_trim_keeps_uniform_silence_for_legacy_compatibility() -> None:
    wave = np.zeros(4096, dtype=np.float32)

    assert np.array_equal(detect_non_silent_range(wave), np.asarray([0, 4096]))
    assert np.array_equal(trim_silence(wave), wave)


def test_default_resampler_preserves_resampy_configuration() -> None:
    wave = np.ones(32, dtype=np.float32)
    expected = np.arange(16, dtype=np.float32)

    with patch("resampy.resample", return_value=expected) as resample:
        result = resample_waveform(wave, 44100, 22050)

    assert result is expected
    resample.assert_called_once_with(
        wave,
        44100,
        22050,
        filter="kaiser_fast",
        parallel=True,
    )


def test_soxr_vhq_preserves_legacy_output_length() -> None:
    wave = np.sin(np.linspace(0, 20, 44101, dtype=np.float32))

    result = resample_waveform(wave, 44100, 48000, resampler="soxr-vhq")

    assert result.dtype == np.float32
    assert result.size == int(wave.size * 48000 / 44100)
    assert np.isfinite(result).all()


def test_resampler_rejects_unknown_implementation() -> None:
    with pytest.raises(ValueError, match="unsupported resampler"):
        resample_waveform(np.ones(8, dtype=np.float32), 44100, 48000, resampler="x")
