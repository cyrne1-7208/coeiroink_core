# モデル読み込み前のホスト・GPU・cgroupメモリ残量ガードのテスト

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import torch

from coeirocore import model_memory
from coeirocore.devices import DeviceBackend, DeviceSelection
from coeirocore.model_memory import _cuda_memory, model_load_memory_error

MIB = 1024 * 1024


def _selection(backend: DeviceBackend) -> DeviceSelection:
    return DeviceSelection(
        backend=backend,
        device_index=0,
        runtime_device=f"{backend.value}:0",
        platform="windows" if backend is DeviceBackend.DIRECTML else "linux",
        platform_index=0 if backend is DeviceBackend.OPENCL else None,
    )


def test_model_load_memory_guard_accepts_sufficient_host_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"0" * 1024)
    monkeypatch.setattr(model_memory, "_host_memory", lambda: (10**9, 10**9))

    error = model_load_memory_error(
        model_path=model_path,
        selection=_selection(DeviceBackend.CPU),
        generator_only=False,
        resident_device_bytes=0,
    )

    assert error is None


def test_model_load_memory_guard_rejects_insufficient_host_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"0" * 1024)
    monkeypatch.setattr(model_memory, "_host_memory", lambda: (10**9, 1))
    options = {
        "model_path": model_path,
        "selection": _selection(DeviceBackend.CPU),
        "generator_only": True,
        "resident_device_bytes": 0,
    }

    error = model_load_memory_error(**options)

    assert error is not None
    assert "host memory" in error
    assert model_load_memory_error(**options, reclaimable_bytes=512 * MIB) is None


@pytest.mark.parametrize("backend", [DeviceBackend.CUDA, DeviceBackend.OPENCL])
def test_model_load_memory_guard_checks_gpu_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: DeviceBackend,
) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"0" * 1024)
    monkeypatch.setattr(model_memory, "_host_memory", lambda: (10**9, 10**9))
    monkeypatch.setattr(model_memory, "_cuda_memory", lambda _selection: (10**9, 1))
    monkeypatch.setattr(model_memory, "_opencl_total_memory", lambda _selection: 10**9)
    options = {
        "model_path": model_path,
        "selection": _selection(backend),
        "generator_only": False,
        "resident_device_bytes": 10**9 - 1,
    }

    error = model_load_memory_error(**options)

    assert error is not None
    assert f"{backend.value} memory" in error
    assert model_load_memory_error(**options, reclaimable_bytes=512 * MIB) is None


@pytest.mark.parametrize(
    "backend", [DeviceBackend.CUDA, DeviceBackend.OPENCL, DeviceBackend.DIRECTML]
)
def test_gpu_eviction_credit_does_not_cover_host_shortage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: DeviceBackend,
) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"test checkpoint")
    monkeypatch.setattr(model_memory, "_host_memory", lambda: (10**9, 1))

    error = model_load_memory_error(
        model_path=model_path,
        selection=_selection(backend),
        generator_only=True,
        resident_device_bytes=10**9,
        reclaimable_bytes=10**9,
    )

    assert error is not None
    assert "host memory" in error


@pytest.mark.parametrize(
    "backend", [DeviceBackend.CPU, DeviceBackend.CUDA, DeviceBackend.OPENCL]
)
def test_eviction_credit_cannot_exceed_total_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: DeviceBackend,
) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"test checkpoint")
    host_total = 128 * MIB if backend is DeviceBackend.CPU else 1024 * MIB
    monkeypatch.setattr(model_memory, "_host_memory", lambda: (host_total, host_total))
    monkeypatch.setattr(model_memory, "_cuda_memory", lambda _selection: (128 * MIB, 1))
    monkeypatch.setattr(
        model_memory, "_opencl_total_memory", lambda _selection: 128 * MIB
    )

    error = model_load_memory_error(
        model_path=model_path,
        selection=_selection(backend),
        generator_only=True,
        resident_device_bytes=0,
        reclaimable_bytes=2048 * MIB,
    )

    assert error is not None


def test_cuda_reusable_memory_does_not_exceed_device_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection(DeviceBackend.CUDA)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _index: (900, 1000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _index: 500)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _index: 100)

    assert _cuda_memory(selection) == (1000, 1000)


@pytest.mark.parametrize(
    ("limit_name", "usage_name", "stat_keys"),
    [
        (
            "memory.max",
            "memory.current",
            ("inactive_file", "file_dirty", "file_writeback"),
        ),
        (
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            ("total_inactive_file", "total_dirty", "total_writeback"),
        ),
    ],
)
def test_cgroup_cache_allows_load_without_bypassing_parent_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    usage_name: str,
    stat_keys: tuple[str, str, str],
) -> None:
    root = tmp_path / "cgroup"
    child = root / "service"
    child.mkdir(parents=True)
    stats = "".join(
        f"{key} {value * MIB}\n"
        for key, value in zip(stat_keys, (300, 50, 50), strict=True)
    )
    for directory, limit, usage in ((root, 2048, 1800), (child, 1024, 900)):
        (directory / limit_name).write_text(str(limit * MIB), encoding="utf-8")
        (directory / usage_name).write_text(str(usage * MIB), encoding="utf-8")
        (directory / "memory.stat").write_text(stats, encoding="utf-8")
    model_path = tmp_path / "model.pth"
    with model_path.open("wb") as file:
        file.truncate(8 * MIB)

    monkeypatch.setattr(
        model_memory,
        "_process_cgroup_directories",
        lambda: ((root, child, limit_name, usage_name),),
    )
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=4096 * MIB, available=3072 * MIB),
    )
    options = {
        "model_path": model_path,
        "selection": _selection(DeviceBackend.CPU),
        "generator_only": True,
        "resident_device_bytes": 0,
    }

    assert model_load_memory_error(**options) is None
    # 祖先cgroupで他のプロセスがメモリを使った場合は、子に余裕があっても読み込まない。
    (root / usage_name).write_text(str(2000 * MIB), encoding="utf-8")
    error = model_load_memory_error(**options)
    assert error is not None
    assert "host memory" in error


@pytest.mark.parametrize(
    "stats",
    [None, "inactive_file 314572800\nfile_dirty 304087040\nfile_writeback 20971520\n"],
)
def test_missing_or_dirty_cache_does_not_relax_memory_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stats: str | None,
) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(str(1024 * MIB), encoding="utf-8")
    (root / "memory.current").write_text(str(900 * MIB), encoding="utf-8")
    if stats is not None:
        (root / "memory.stat").write_text(stats, encoding="utf-8")
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"model")

    monkeypatch.setattr(
        model_memory,
        "_process_cgroup_directories",
        lambda: ((root, root, "memory.max", "memory.current"),),
    )
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=4096 * MIB, available=3072 * MIB),
    )

    error = model_load_memory_error(
        model_path=model_path,
        selection=_selection(DeviceBackend.CPU),
        generator_only=True,
        resident_device_bytes=0,
    )

    assert error is not None
    assert "host memory" in error
