"""波形トリム・リサンプリング・CPUスレッド上限の伝播を検証する。"""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
import textwrap
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
import resampy
import torch

from coeirocore import coeiro_manager, waveform
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


def test_trim_squares_long_waveform_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wave = np.zeros(1_100_000, dtype=np.float32)
    wave[500_000:600_000] = 1.0
    original_square = np.square
    chunk_sizes: list[int] = []

    def track_square(values: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        chunk_sizes.append(values.size)
        return original_square(values, *args, **kwargs)

    monkeypatch.setattr(waveform.np, "square", track_square)
    detected = detect_non_silent_range(wave)

    assert detected[0] < detected[1]
    assert len(chunk_sizes) > 1
    assert max(chunk_sizes) * np.dtype(np.float32).itemsize <= 8 * 1024 * 1024


def test_default_resampler_preserves_resampy_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wave = np.ones(32, dtype=np.float32)
    expected = np.arange(16, dtype=np.float32)
    resample = MagicMock(return_value=expected)
    monkeypatch.setattr(resampy, "resample", resample)

    result = resample_waveform(wave, 44100, 22050)

    assert result is expected
    resample.assert_called_once_with(
        wave,
        44100,
        22050,
        filter="kaiser_fast",
        parallel=True,
    )


def test_cpu_thread_limit_is_available_during_first_numba_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(waveform, "_resampy_thread_limit", None)
    fake_numba = ModuleType("numba")
    fake_numba.get_num_threads = MagicMock(return_value=128)
    fake_numba.set_num_threads = MagicMock()

    def import_numba(name: str, *args: object, **kwargs: object) -> ModuleType:
        assert name == "numba"
        assert os.environ["NUMBA_NUM_THREADS"] == "2"
        return fake_numba

    monkeypatch.delitem(sys.modules, "numba", raising=False)
    monkeypatch.setenv("NUMBA_NUM_THREADS", "128")
    monkeypatch.setattr(torch, "set_num_threads", MagicMock())
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 4)
    monkeypatch.setattr(torch, "set_num_interop_threads", MagicMock())
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", import_numba)
        coeiro_manager._configure_cpu_threads(2)

    fake_numba.set_num_threads.assert_called_once_with(2)


def test_cpu_thread_limit_does_not_rewrite_numba_environment_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(waveform, "_resampy_thread_limit", None)
    fake_numba = ModuleType("numba")
    fake_numba.get_num_threads = MagicMock(return_value=8)
    fake_numba.set_num_threads = MagicMock()
    monkeypatch.setitem(sys.modules, "numba", fake_numba)
    monkeypatch.setenv("NUMBA_NUM_THREADS", "8")
    monkeypatch.setattr(torch, "set_num_threads", MagicMock())
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 2)

    coeiro_manager._configure_cpu_threads(2)

    assert os.environ["NUMBA_NUM_THREADS"] == "8"
    fake_numba.set_num_threads.assert_called_once_with(2)


def test_concurrent_resampling_with_numba_workqueue_does_not_abort() -> None:
    code = textwrap.dedent(
        """
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        import numba
        import numpy as np
        from coeirocore.coeiro_manager import _configure_cpu_threads
        from coeirocore.waveform import resample_waveform

        # Numbaを先に読み込んだ場合も、リクエストを処理するスレッドへ指定上限を引き継ぐ。
        _configure_cpu_threads(1)

        if sys.platform != "win32":
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        wave = np.sin(np.arange(220_500, dtype=np.float32) * 0.02)
        expected = resample_waveform(wave, 44100, 48000)
        barrier = Barrier(2)

        def convert(_):
            barrier.wait(timeout=10)
            result = resample_waveform(wave, 44100, 48000)
            assert numba.get_num_threads() == 1
            return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            for result in executor.map(convert, range(2)):
                assert np.array_equal(result, expected)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "NUMBA_THREADING_LAYER": "workqueue",
            "NUMBA_NUM_THREADS": "2",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_soxr_vhq_preserves_legacy_output_length() -> None:
    wave = np.sin(np.linspace(0, 20, 44101, dtype=np.float32))

    result = resample_waveform(wave, 44100, 48000, resampler="soxr-vhq")

    assert result.dtype == np.float32
    assert result.size == int(wave.size * 48000 / 44100)
    assert np.isfinite(result).all()


def test_resampler_rejects_unknown_implementation() -> None:
    with pytest.raises(ValueError, match="unsupported resampler"):
        resample_waveform(np.ones(8, dtype=np.float32), 44100, 48000, resampler="x")
