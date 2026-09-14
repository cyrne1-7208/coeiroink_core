"""周期検出から波形補正までを、選択されたCUDA/OpenCLデバイスで実行する。"""

import re
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .devices import DeviceBackend, DeviceSelection
from .voice_smoothing_common import _prepare_spans

_KERNEL_DIR = Path(__file__).with_name("voice_smoothing_kernels")
_FRAME_BATCH = 256
_FFT_SIZE = 2048


def _source() -> str:
    # 入力文・波形長・モデル名をコンパイル条件に含めない。
    return "\n".join(
        (_KERNEL_DIR / name).read_text(encoding="utf-8")
        for name in ("portable.h", "pitch.cl", "pulses.cl", "waveform.cl")
    )


class _OpenCL:
    def __init__(self, selection: DeviceSelection):
        import pyopencl as cl
        import pyopencl.array as arrays

        self.arrays = arrays
        platform = cl.get_platforms()[selection.platform_index]
        device = platform.get_devices()[selection.device_index]
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        self.program = cl.Program(self.context, _source()).build()
        self.kernels = {
            kernel.function_name: kernel for kernel in self.program.all_kernels()
        }

    @contextmanager
    def execution(self):
        try:
            yield
        finally:
            # 例外時にも処理中のバッファを残したまま次の合成へ進まない。
            self.queue.finish()

    def empty(self, shape, dtype=np.float32):
        return self.arrays.empty(self.queue, shape, dtype)

    def upload(self, data):
        return self.arrays.to_device(self.queue, np.asarray(data))

    def run(self, name, size, *args, local=None):
        self.kernels[name](
            self.queue,
            size,
            local,
            *(
                value.data if isinstance(value, self.arrays.Array) else value
                for value in args
            ),
        )


class _CUDA:
    def __init__(self, selection: DeviceSelection):
        import cupy as cp

        self.cp = cp
        self.device = cp.cuda.Device(selection.device_index)
        # 補正専用のプールを使い、終了時に解放してモデル保持用VRAMを圧迫しない。
        self.pool = cp.cuda.MemoryPool()
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            source = _source()
            self.module = cp.RawModule(code=source, options=("--std=c++11",))
            names = re.findall(r"KERNEL void (\w+)\(", source)
            self.kernels = {name: self.module.get_function(name) for name in names}

    @contextmanager
    def execution(self):
        with self.device, self.stream, self.cp.cuda.using_allocator(self.pool.malloc):
            try:
                yield
            finally:
                self.stream.synchronize()
                self.pool.free_all_blocks()

    def empty(self, shape, dtype=np.float32):
        return self.cp.empty(shape, dtype)

    def upload(self, data):
        return self.cp.asarray(data)

    def run(self, name, size, *args, local=None):
        if local is not None:
            block = tuple(int(value) for value in local)
        elif len(size) == 2:
            block = (128, 1)
        else:
            block = (min(int(size[0]), 256),)
        grid = tuple(
            (int(length) + width - 1) // width
            for length, width in zip(size, block, strict=True)
        )
        self.kernels[name](grid, block, args)


class GpuVoiceSmoother:
    """GPU資源はManagerごとに保持する。呼び出しは既存の推論ロックで直列化する。"""

    def __init__(self, selection: DeviceSelection):
        if selection.backend is DeviceBackend.CUDA:
            self.runtime = _CUDA(selection)
        elif selection.backend is DeviceBackend.OPENCL:
            self.runtime = _OpenCL(selection)
        else:
            raise ValueError("GPU voice smoothing requires CUDA or OpenCL")

    def __call__(
        self,
        wave: np.ndarray,
        tokens: Sequence[str],
        duration_frames: Sequence[int],
        *,
        sampling_rate: int,
        hop_length: int,
    ) -> np.ndarray:
        spans = _prepare_spans(wave, tokens, duration_frames, sampling_rate, hop_length)
        if not any(right - left > 0.1 * sampling_rate for left, right in spans):
            return wave
        # 既存のNumPy波形境界を維持する。波形解析はGPUへ移し、モデル推論のAPIは変更しない。
        with self.runtime.execution():
            result = self._smooth(wave, spans, sampling_rate)
        if not np.isfinite(result).all():
            raise RuntimeError("GPU voice smoothing produced an invalid waveform")
        return result.astype(wave.dtype, copy=False)

    def _slopes(self, wave, n, rows=1):
        edges = self.runtime.empty((rows, 2))
        output = self.runtime.empty((rows, n))
        self.runtime.run(
            "spline_edges", (rows,), wave, np.int32(n), np.int32(rows), edges
        )
        self.runtime.run(
            "spline_slopes", (n, rows), wave, np.int32(n), np.int32(rows), edges, output
        )
        return output

    def _weights(self, sigma):
        radius = int(4 * sigma + 0.5)
        values = self.runtime.empty(2 * radius + 1)
        self.runtime.run(
            "gaussian_weights", (1,), np.float32(sigma), np.int32(radius), values
        )
        return values, np.int32(radius)

    def _pulses(self, wave, sampling_rate):
        n = wave.size
        sr = sampling_rate
        frames = int(np.floor((n / sr - 2 / 65) / 0.005)) + 1
        first = 0.5 * (n - (frames - 1) * 0.005 * sr) - 0.5
        period = int(sr / 65)
        window = 2 * (period // 2 - 1)
        i, f = np.int32, np.float32
        empty, run = self.runtime.empty, self.runtime.run
        peak, correlations = empty(2), empty((frames, window + 1))
        intensity, statistics = empty(frames), empty((frames, 2))
        frequencies, scores = empty((frames, 15)), empty((frames, 15))
        trace = empty((frames, 15), np.int32)
        trace.fill(0)
        pitch, raster = empty(frames), empty(n)
        raster.fill(-1)
        run("global_peak", (256,), wave, i(n), peak, local=(256,))
        run(
            "frame_statistics",
            (frames,),
            wave,
            i(n),
            f(sr),
            f(first),
            i(frames),
            i(period),
            i(window),
            peak,
            statistics,
            intensity,
        )
        run(
            "correlations",
            (window + 1, frames),
            wave,
            i(n),
            f(sr),
            f(first),
            i(frames),
            i(window),
            statistics,
            correlations,
        )
        run(
            "candidates",
            (frames,),
            correlations,
            intensity,
            i(frames),
            i(window),
            f(sr),
            frequencies,
            scores,
        )
        run("pitch_path", (1,), frequencies, scores, i(frames), trace, pitch)
        run(
            "pulse_track_parallel",
            (256,),
            wave,
            i(n),
            f(sr),
            f(first),
            pitch,
            i(frames),
            peak,
            raster,
            local=(256,),
        )
        return raster, frames

    def _smooth(self, source, spans, sr):
        i, f = np.int32, np.float32
        empty, run = self.runtime.empty, self.runtime.run
        wave = self.runtime.upload(np.asarray(source, np.float32))
        n = wave.size
        raster, pitch_frames = self._pulses(wave, sr)
        span_data = self.runtime.upload(np.asarray(spans, np.int32))
        # 1群は9周期以上。音素由来の区間は重ならず、群数は5msピッチフレーム数より少ない。
        marks = empty(n)
        groups = empty((pitch_frames + len(spans), 4), np.int32)
        state = empty(2, np.int32)
        run(
            "collect_groups",
            (1,),
            raster,
            i(n),
            span_data,
            i(len(spans)),
            f(sr),
            marks,
            groups,
            state,
        )
        group_count, used = state.get()
        if not group_count:
            return source
        medians, scratch = empty(group_count), empty(used)
        run(
            "group_medians",
            (int(group_count),),
            marks,
            groups,
            i(group_count),
            scratch,
            medians,
        )
        # CPUへ戻すのは起動範囲用の小さなメタデータだけ。F0・周期列・音色の解析はGPU内で完結する。
        descriptors, periods = groups[:group_count].get(), medians.get()
        wave_slopes = self._slopes(wave, n)
        numerator, denominator = empty(n), empty(n)
        numerator.fill(0)
        denominator.fill(0)
        shaped = wave.copy()
        # 最大550Hzと15msのGaussian半径から前後33窓を確保し、入力長に比例するFFTバッファを作らない。
        context_frames = _FRAME_BATCH + 2 * 33
        original = empty((context_frames, _FFT_SIZE))
        spectra = empty((context_frames, _FFT_SIZE // 2 + 1), np.complex64)
        logs = empty((context_frames, _FFT_SIZE // 2 + 1))
        adjusted = empty((_FRAME_BATCH, _FFT_SIZE // 2 + 1), np.complex64)
        delta = empty((_FRAME_BATCH, _FFT_SIZE))
        for (first, count, left, right), period in zip(
            descriptors, periods, strict=True
        ):
            weights, radius = self._weights(0.015 * sr / float(period))
            for start in range(0, count - 2, _FRAME_BATCH):
                stop = min(start + _FRAME_BATCH, count - 2)
                low, high = max(0, start - radius), min(count - 2, stop + radius)
                frames, selected = int(high - low), int(stop - start)
                run(
                    "phase_spectra",
                    (frames * 256,),
                    wave,
                    wave_slopes,
                    i(n),
                    marks,
                    i(first + low),
                    i(frames),
                    original,
                    spectra,
                    local=(256,),
                )
                run("log_magnitudes", (frames * 1025,), spectra, i(frames * 1025), logs)
                run(
                    "magnitude_trend",
                    (1025, selected),
                    spectra,
                    i(frames),
                    i(start - low),
                    i(selected),
                    logs,
                    weights,
                    radius,
                    adjusted,
                )
                run(
                    "phase_delta",
                    (selected * 256,),
                    adjusted,
                    original,
                    i(start - low),
                    i(selected),
                    delta,
                    local=(256,),
                )
                delta_slopes = self._slopes(delta, _FFT_SIZE, selected)
                for parity in (0, 1):
                    run(
                        "overlap_add",
                        (int(np.ceil(2 * sr / 65)) + 2, (selected + 1) // 2),
                        delta,
                        delta_slopes,
                        marks,
                        i(first + start),
                        i(selected),
                        i(parity),
                        numerator,
                        denominator,
                    )
            run(
                "apply_magnitude",
                (int(right - left),),
                wave,
                shaped,
                numerator,
                denominator,
                marks,
                i(first),
                i(count),
                f(sr),
            )
        shape_slopes = self._slopes(shaped, n)
        result = shaped.copy()
        for (first, count, left, right), period in zip(
            descriptors, periods, strict=True
        ):
            logs, targets = empty(count - 1), empty(count - 1)
            destination, inverse = empty(count), empty(count)
            weights, radius = self._weights(0.025 * sr / float(period))
            run("log_periods", (int(count - 1),), marks, i(first), i(count), logs)
            run(
                "target_periods",
                (int(count - 1),),
                logs,
                i(count),
                weights,
                radius,
                targets,
            )
            run(
                "period_positions",
                (1,),
                marks,
                i(first),
                i(count),
                targets,
                destination,
            )
            run(
                "inverse_slopes",
                (int(count),),
                marks,
                i(first),
                i(count),
                destination,
                inverse,
            )
            run(
                "apply_periods",
                (int(right - left),),
                shaped,
                shape_slopes,
                i(n),
                marks,
                i(first),
                i(count),
                destination,
                inverse,
                f(sr),
                result,
            )
        blocks = (n + 1023) // 1024
        parts, scale = empty((blocks, 2)), empty(1)
        run("wave_energy", (blocks * 256,), wave, result, i(n), parts, local=(256,))
        run("energy_scale", (1,), parts, i(blocks), scale)
        run("normalize_wave", (n,), result, i(n), scale)
        return result.get()
