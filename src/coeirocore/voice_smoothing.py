"""母音内の音色と周期間隔の細かな揺れを弱める、任意のCPU後処理。"""

from collections.abc import Sequence

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import interpolate, ndimage

from .voice_smoothing_common import _prepare_spans

_PITCH_FLOOR = 65.0
_PITCH_CEILING = 550.0
_ONSET_PROTECTION = 0.100
_FADE_SECONDS = 0.025
_MAGNITUDE_SIGMA = 0.015
_MAGNITUDE_STRENGTH = 0.85
_MAGNITUDE_LIMIT = 4 * np.log(10) / 20
_PERIOD_SIGMA = 0.025
_PERIOD_STRENGTH = 0.5
# 周期ごとの位相座標は入力長と無関係で、長音でもFFT用配列を一定サイズに抑える。
_PHASE = np.arange(2048) / 1024 - 1
_WINDOW = 0.5 + 0.5 * np.cos(np.pi * _PHASE)
_FRAME_BATCH = 256


def _pulse_marks(wave: np.ndarray, sampling_rate: int) -> np.ndarray:
    sound = parselmouth.Sound(wave, sampling_frequency=sampling_rate)
    pitch = sound.to_pitch_cc(
        time_step=0.005, pitch_floor=_PITCH_FLOOR, pitch_ceiling=_PITCH_CEILING
    )
    points = call([sound, pitch], "To PointProcess (cc)")
    if call(points, "Get number of points") == 0:
        # 無声音だけのPointProcessは空のMatrixへ変換できない。
        return np.empty(0)
    # コマンドを周期ごとに往復させず、検出済みの時刻を一括で取得する。
    times = call(points, "To Matrix").values[0]
    # Praatの先頭サンプル時刻は半サンプルなので、NumPyの座標へ戻す。
    return times * sampling_rate - 0.5


def _voiced_groups(
    marks: np.ndarray, spans: Sequence[tuple[int, int]], sampling_rate: int
) -> list[np.ndarray]:
    groups = []
    for left, right in spans:
        inside = marks[(marks >= left) & (marks < right)]
        if len(inside) < 3:
            continue
        intervals = np.diff(inside)
        bad = (intervals > sampling_rate / _PITCH_FLOOR) | (
            intervals < sampling_rate / _PITCH_CEILING
        )
        ratio = np.maximum(
            intervals[1:] / intervals[:-1], intervals[:-1] / intervals[1:]
        )
        bad[1:] |= ratio > 1.5
        # 周期の取りこぼしや無声音をまたいで平滑化しない。
        for group in np.split(inside, np.flatnonzero(bad) + 1):
            group = group[group >= left + _ONSET_PROTECTION * sampling_rate]
            if len(group) >= 9:
                groups.append(group)
    return groups


def _fade(times: np.ndarray, left: float, right: float) -> np.ndarray:
    ramp = np.clip(
        np.minimum((times - left) / _FADE_SECONDS, (right - times) / _FADE_SECONDS),
        0,
        1,
    )
    return 0.5 - 0.5 * np.cos(np.pi * ramp)


def _smooth_magnitude(
    wave: np.ndarray, groups: Sequence[np.ndarray], sampling_rate: int
) -> np.ndarray:
    result = wave.copy()
    spline = interpolate.CubicSpline(np.arange(len(wave)), wave)
    for marks in groups:
        centers = marks[1:-1]
        periods = np.diff(marks)
        left_period, right_period = periods[:-1], periods[1:]
        # 区間内の周期長の中央値を使い、秒単位の時間幅を周期数へ換算する。
        sigma = _MAGNITUDE_SIGMA * sampling_rate / np.median(periods)
        radius = int(4 * sigma + 0.5)
        first, last = int(np.ceil(marks[0])), int(np.floor(marks[-1]))
        numerator, denominator = np.zeros(last - first), np.zeros(last - first)
        for start in range(0, len(centers), _FRAME_BATCH):
            stop = min(start + _FRAME_BATCH, len(centers))
            # 前後のGaussian半径も読むことで、分割位置が波形へ現れるのを防ぐ。
            low, high = max(0, start - radius), min(len(centers), stop + radius)
            coordinates = centers[low:high, None] + _PHASE * np.where(
                _PHASE < 0,
                left_period[low:high, None],
                right_period[low:high, None],
            )
            frames = spline(coordinates) * _WINDOW
            spectra = np.fft.rfft(frames, axis=1)
            log_magnitude = np.log(np.maximum(np.abs(spectra), 1e-12))
            trend = ndimage.gaussian_filter1d(log_magnitude, sigma, axis=0)
            selection = slice(start - low, stop - low)
            change = np.clip(
                trend[selection] - log_magnitude[selection],
                -_MAGNITUDE_LIMIT,
                _MAGNITUDE_LIMIT,
            )
            adjusted = spectra[selection] * np.exp(_MAGNITUDE_STRENGTH * change)
            reconstructed = np.fft.irfft(adjusted, n=len(_PHASE), axis=1)
            original = frames[selection]
            # 窓内の音量を保ち、音色補正に音量の平滑化を混ぜない。
            reconstructed *= np.sqrt(
                np.sum(original * original, axis=1)
                / np.maximum(np.sum(reconstructed * reconstructed, axis=1), 1e-24)
            )[:, None]
            delta = reconstructed - original
            # 同じ位相格子上の補間係数はバッチで解き、周期ごとの補間器生成を避ける。
            coefficients = interpolate.CubicSpline(_PHASE, delta, axis=1).c
            for local, index in enumerate(range(start, stop)):
                indices = np.arange(
                    int(np.ceil(marks[index])), int(np.floor(marks[index + 2]))
                )
                offset = indices - centers[index]
                phase = offset / np.where(
                    offset < 0, left_period[index], right_period[index]
                )
                window = 0.5 + 0.5 * np.cos(np.pi * phase)
                interval = np.clip(
                    np.searchsorted(_PHASE, phase, side="right") - 1,
                    0,
                    len(_PHASE) - 2,
                )
                offset = phase - _PHASE[interval]
                c = coefficients[:, interval, local]
                changed = ((c[0] * offset + c[1]) * offset + c[2]) * offset + c[3]
                numerator[indices - first] += window * changed
                denominator[indices - first] += window**2
        correction = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        indices = np.arange(int(np.ceil(marks[1])), int(np.floor(marks[-2])))
        mask = _fade(
            indices / sampling_rate, marks[1] / sampling_rate, marks[-2] / sampling_rate
        )
        result[indices] += mask * correction[indices - first]
    return result


def _smooth_periods(
    wave: np.ndarray, groups: Sequence[np.ndarray], sampling_rate: int
) -> np.ndarray:
    result = wave.copy()
    spline = interpolate.CubicSpline(np.arange(len(wave)), wave)
    for marks in groups:
        intervals = np.diff(marks)
        log_period = np.log(intervals)
        trend = ndimage.gaussian_filter1d(
            log_period, _PERIOD_SIGMA * sampling_rate / np.median(intervals)
        )
        target = np.exp(log_period + _PERIOD_STRENGTH * (trend - log_period))
        target = np.clip(target, intervals * 0.8, intervals * 1.2)
        # 周期数と区間の両端を維持し、後続の音素や休止の時刻をずらさない。
        target *= intervals.sum() / target.sum()
        destination = np.r_[marks[0], marks[0] + np.cumsum(target)]
        indices = np.arange(int(np.ceil(marks[0])), int(np.floor(marks[-1])))
        coordinates = interpolate.PchipInterpolator(destination, marks)(indices)
        mask = _fade(
            indices / sampling_rate, marks[0] / sampling_rate, marks[-1] / sampling_rate
        )
        result[indices] += mask * (spline(coordinates) - wave[indices])
    return result


def smooth_voice(
    wave: np.ndarray,
    tokens: Sequence[str],
    duration_frames: Sequence[int],
    *,
    sampling_rate: int,
    hop_length: int,
) -> np.ndarray:
    """未トリムの波形へ固定設定の補正を適用し、長さと平均音量を維持する。"""

    spans = _prepare_spans(
        wave,
        tokens,
        duration_frames,
        sampling_rate,
        hop_length,
    )
    if not any(
        right - left > _ONSET_PROTECTION * sampling_rate for left, right in spans
    ):
        return wave
    original = np.asarray(wave, dtype=np.float64)
    groups = _voiced_groups(_pulse_marks(original, sampling_rate), spans, sampling_rate)
    if not groups:
        # 無音・無声音や短母音に補正対象がないことは、解析エラーではない。
        return wave
    shaped = _smooth_magnitude(original, groups, sampling_rate)
    result = _smooth_periods(shaped, groups, sampling_rate)
    original_energy, result_energy = (
        np.sum(original * original),
        np.sum(result * result),
    )
    if not np.isfinite(result_energy) or result_energy <= 0:
        raise RuntimeError("voice smoothing produced an invalid waveform")
    # APIのvolumeScaleとは別に、補正前の平均音量を保つ。
    result *= np.sqrt(original_energy / result_energy)
    return result.astype(wave.dtype, copy=False)
