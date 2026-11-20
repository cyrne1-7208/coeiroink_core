# 音声補正のCPU/GPU共通処理である母音区間抽出と入力検証を、parselmouthなしで検証するテスト

from __future__ import annotations

import numpy as np
import pytest

from coeirocore.voice_smoothing_common import _prepare_spans, _vowel_spans


def test_vowel_spans_preserve_prosody_markers_and_use_model_hop_length() -> None:
    assert _vowel_spans(
        ["^", "a", "[", "a", "]", "s", "u", "$"],
        [1, 2, 1, 2, 1, 2, 2, 1],
        4,
    ) == [(4, 28), (36, 44)]


@pytest.mark.parametrize(
    ("wave", "tokens", "durations", "sampling_rate", "hop_length", "message"),
    [
        (np.zeros(4, dtype=np.float32), ["a"], [1], 0, 4, "must be positive"),
        (np.zeros(4, dtype=np.float32), ["a"], [1], 44100, 0, "must be positive"),
        (np.zeros((2, 2), dtype=np.float32), ["a"], [1], 44100, 4, "mono floating"),
        (np.zeros(4, dtype=np.int16), ["a"], [1], 44100, 4, "mono floating"),
        (np.zeros(4, dtype=np.float32), ["a"], [-1], 44100, 4, "valid token durations"),
        (
            np.zeros(4, dtype=np.float32),
            ["a", "i"],
            [1],
            44100,
            4,
            "valid token durations",
        ),
        (np.zeros(4, dtype=np.float32), ["a"], [2], 44100, 4, "do not align"),
        (
            np.array([0.0, np.nan, 0.0, 0.0], dtype=np.float32),
            ["a"],
            [1],
            44100,
            4,
            "finite waveform",
        ),
    ],
)
def test_prepare_spans_rejects_invalid_inputs(
    wave: np.ndarray,
    tokens: list[str],
    durations: list[int],
    sampling_rate: int,
    hop_length: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare_spans(wave, tokens, durations, sampling_rate, hop_length)
