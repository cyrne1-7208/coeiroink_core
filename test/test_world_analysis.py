# WORLDによるF0・スペクトル包絡・非周期性指標の解析と音高抑揚合成のテスト

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from coeirocore import coeiro_manager
from coeirocore.coeiro_manager import AudioManager


def _synthetic_voiced_wave(sampling_rate: int, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    time_axis = np.arange(int(sampling_rate * seconds)) / sampling_rate
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.7 * time_axis)
    phase = 2 * np.pi * np.cumsum(f0) / sampling_rate
    wave = sum(np.sin(k * phase) / k for k in range(1, 6))
    return 0.2 * wave + 1e-3 * rng.standard_normal(time_axis.shape)


def test_f0_only_analysis_matches_world_without_spectral_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling_rate = 16000
    time_axis = np.arange(4800, dtype=np.float64) / sampling_rate
    wave = np.sin(2 * np.pi * 200 * time_axis)
    expected, _, _ = AudioManager.get_world(wave, sampling_rate)

    def unexpected_spectral_analysis(*args: object, **kwargs: object) -> None:
        pytest.fail("F0-only analysis must not compute spectral parameters")

    world = coeiro_manager.load_pyworld()
    monkeypatch.setattr(world, "cheaptrick", unexpected_spectral_analysis)
    monkeypatch.setattr(world, "d4c", unexpected_spectral_analysis)

    np.testing.assert_array_equal(
        AudioManager.get_world_f0(wave, sampling_rate), expected
    )


def test_pitch_intonation_preserves_unvoiced_f0_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, np.ndarray] = {}

    def fake_synthesize(
        f0: np.ndarray,
        _spectral_envelope: object,
        _aperiodicity: object,
        _sampling_rate: int,
    ) -> np.ndarray:
        captured["f0"] = np.asarray(f0).copy()
        return np.zeros(32, dtype=np.float64)

    fake_world = MagicMock()
    fake_world.synthesize.side_effect = fake_synthesize
    monkeypatch.setattr(
        AudioManager,
        "get_world",
        staticmethod(
            lambda _wave, _rate: (np.array([0.0, 100.0, 200.0, 0.0]), None, None)
        ),
    )
    monkeypatch.setattr(coeiro_manager, "load_pyworld", lambda: fake_world)

    result = AudioManager.pitch_intonation(
        np.zeros(32, dtype=np.float32),
        16000,
        pitch_scale=0.0,
        intonation_scale=0.5,
    )

    np.testing.assert_array_equal(captured["f0"], [0.0, 125.0, 175.0, 0.0])
    assert result.dtype == np.float32


def test_get_world_matches_serial_pyworld_analysis() -> None:
    sampling_rate = 44100
    wave = _synthetic_voiced_wave(sampling_rate, 1.5)
    world = coeiro_manager.load_pyworld()
    f0, time_axis = world.harvest(wave, sampling_rate)
    f0 = world.stonemask(wave, f0, time_axis, sampling_rate)
    expected_sp = world.cheaptrick(wave, f0, time_axis, sampling_rate)
    expected_ap = world.d4c(wave, f0, time_axis, sampling_rate)

    actual_f0, actual_sp, actual_ap = AudioManager.get_world(wave, sampling_rate)

    assert np.array_equal(actual_f0, f0)
    assert np.array_equal(actual_sp, expected_sp)
    assert np.array_equal(actual_ap, expected_ap)
    assert (actual_sp.dtype, actual_sp.shape) == (expected_sp.dtype, expected_sp.shape)
    assert (actual_ap.dtype, actual_ap.shape) == (expected_ap.dtype, expected_ap.shape)


def test_get_world_propagates_cheaptrick_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_cheaptrick(*args: object, **kwargs: object) -> None:
        raise RuntimeError("cheaptrick failed")

    world = coeiro_manager.load_pyworld()
    monkeypatch.setattr(world, "cheaptrick", failing_cheaptrick)

    wave = _synthetic_voiced_wave(16000, 0.3)
    with pytest.raises(RuntimeError, match="cheaptrick failed"):
        AudioManager.get_world(wave, 16000)
