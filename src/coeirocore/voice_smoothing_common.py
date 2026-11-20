# 音声補正でCPU/GPU実装が共有する設定値、入力検証、対象区間の抽出

from collections.abc import Sequence

import numpy as np

# CPUとGPUの実装で一致させる、音声補正のアルゴリズムの設定値
_PITCH_FLOOR = 65.0
_PITCH_CEILING = 550.0
_PITCH_TIME_STEP = 0.005
_ONSET_PROTECTION = 0.100
_MAGNITUDE_SIGMA = 0.015
_PERIOD_SIGMA = 0.025


# 連続する同じ母音の区間をモデルの音響フレームからサンプル範囲へ変換する。
def _vowel_spans(
    tokens: Sequence[str], duration_frames: Sequence[int], hop_length: int
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    previous = None
    for token, duration in zip(tokens, duration_frames, strict=True):
        end = position + duration * hop_length
        if token in ("[", "]") and previous is not None:
            spans[-1] = (spans[-1][0], end)
        elif token in ("a", "i", "u", "e", "o"):
            if token == previous:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((position, end))
            previous = token
        else:
            previous = None
        position = end
    return spans


# CPUとGPUで共通の入力条件を検証し、補正する区間を返す。
def _prepare_spans(
    wave: np.ndarray,
    tokens: Sequence[str],
    duration_frames: Sequence[int],
    sampling_rate: int,
    hop_length: int,
) -> list[tuple[int, int]]:
    if sampling_rate <= 0 or hop_length <= 0:
        raise ValueError("sampling rate and hop length must be positive")
    if wave.ndim != 1 or not np.issubdtype(wave.dtype, np.floating):
        raise ValueError("voice smoothing requires a mono floating-point waveform")
    if len(tokens) != len(duration_frames) or any(
        duration < 0 for duration in duration_frames
    ):
        raise ValueError("voice smoothing requires valid token durations")
    if sum(duration_frames) * hop_length != len(wave):
        raise ValueError("token durations do not align with the waveform")
    if not np.isfinite(wave).all():
        raise ValueError("voice smoothing requires a finite waveform")
    return _vowel_spans(tokens, duration_frames, hop_length)
