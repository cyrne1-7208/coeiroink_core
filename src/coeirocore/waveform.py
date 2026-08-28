"""COEIROINKの波形トリムと出力リサンプリングを提供する。"""

from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np

Resampler = Literal["resampy", "soxr-vhq"]
SUPPORTED_RESAMPLERS: tuple[Resampler, ...] = ("resampy", "soxr-vhq")


def normalize_resampler(value: str) -> Resampler:
    """公開設定値を検証し、対応するリサンプラー名へ絞り込む。"""

    if value not in SUPPORTED_RESAMPLERS:
        supported = ", ".join(SUPPORTED_RESAMPLERS)
        raise ValueError(f"unsupported resampler: {value!r} ({supported})")
    return cast(Resampler, value)


@lru_cache(maxsize=1)
def _load_soxr() -> Any:
    """明示的に選択された場合だけsoxrを読み込む。"""

    try:
        import soxr
    except ModuleNotFoundError as error:
        if error.name != "soxr":
            raise
        raise RuntimeError(
            "soxr-vhq requires the optional 'soxr' extra; "
            "install it with `uv sync --extra <backend> --extra soxr`"
        ) from error
    return soxr


def ensure_resampler_available(resampler: str) -> Resampler:
    """サーバー起動時に明示選択された実装を検証し、暗黙の代替動作を防ぐ。"""

    normalized = normalize_resampler(resampler)
    if normalized == "soxr-vhq":
        _load_soxr()
    return normalized


def detect_non_silent_range(
    wave: np.ndarray,
    *,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """librosa.effects.trimと同じRMS判定で非無音区間を返す。"""

    samples = np.asarray(wave)
    if samples.ndim < 1:
        raise ValueError("wave must have at least one dimension")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")

    padding = [(0, 0)] * samples.ndim
    padding[-1] = (frame_length // 2, frame_length // 2)
    padded = np.pad(samples, padding, mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=frame_length,
        axis=-1,
    )[..., ::hop_length, :]
    power = np.mean(np.square(frames, dtype=np.float32), axis=-1)
    rms = np.sqrt(power)

    magnitude = np.abs(rms)
    reference = np.max(magnitude)
    power = np.square(magnitude, out=magnitude)
    decibels = 10.0 * np.log10(np.maximum(1e-10, power))
    decibels -= 10.0 * np.log10(np.maximum(1e-10, reference**2))
    if decibels.ndim > 1:
        decibels = np.max(decibels, axis=tuple(range(decibels.ndim - 1)))

    non_silent = np.flatnonzero(decibels > -top_db)
    if non_silent.size == 0:
        return np.asarray([0, 0], dtype=np.int64)
    start = int(non_silent[0] * hop_length)
    end = min(samples.shape[-1], int((non_silent[-1] + 1) * hop_length))
    return np.asarray([start, end], dtype=np.int64)


def trim_silence(wave: np.ndarray, *, top_db: float = 30.0) -> np.ndarray:
    """非無音区間だけを入力配列の最終軸から切り出す。"""

    detected_range = detect_non_silent_range(wave, top_db=top_db)
    return np.asarray(wave)[..., detected_range[0] : detected_range[1]]


def resample_waveform(
    wave: np.ndarray,
    sampling_rate: int,
    output_sampling_rate: int,
    *,
    resampler: str = "resampy",
) -> np.ndarray:
    """選択された実装でモノラル波形を変換し、従来と同じサンプル数を返す。"""

    samples = np.asarray(wave)
    if samples.ndim != 1:
        raise ValueError("resampling requires a one-dimensional waveform")
    if sampling_rate <= 0 or output_sampling_rate <= 0:
        raise ValueError("sampling rates must be positive")

    normalized = normalize_resampler(resampler)
    if normalized == "resampy":
        import resampy

        return resampy.resample(
            samples,
            sampling_rate,
            output_sampling_rate,
            filter="kaiser_fast",
            parallel=True,
        )

    converted = np.asarray(
        _load_soxr().resample(
            samples,
            sampling_rate,
            output_sampling_rate,
            quality="VHQ",
        )
    )
    # soxrは比率によって端数を切り上げるため、従来APIが返す長さへ末尾だけを揃える。
    expected_samples = int(samples.size * output_sampling_rate / sampling_rate)
    if converted.size < expected_samples:
        raise RuntimeError(
            "soxr returned fewer samples than the expected output length"
        )
    return converted[:expected_samples]


__all__ = [
    "SUPPORTED_RESAMPLERS",
    "detect_non_silent_range",
    "ensure_resampler_available",
    "normalize_resampler",
    "resample_waveform",
    "trim_silence",
]
