"""モデル追加前に利用可能メモリを確認するための軽量な判定処理を提供する。"""

from __future__ import annotations

import math
from pathlib import Path

import psutil
import torch

from .devices import DeviceBackend, DeviceSelection

_MIB = 1024 * 1024
_MIN_HOST_RESERVE = 256 * _MIB
_MIN_DEVICE_RESERVE = 256 * _MIB


def _read_memory_limit(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    # cgroup v1は制限なしを極端に大きな整数で表す。
    return None if parsed >= 1 << 60 else parsed


def _process_cgroup_directories() -> tuple[tuple[Path, Path, str, str], ...]:
    try:
        entries = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError:
        return ()

    directories: list[tuple[Path, Path, str, str]] = []
    for entry in entries:
        hierarchy, controllers, relative = entry.split(":", maxsplit=2)
        if hierarchy == "0" and not controllers:
            # cgroup v2では全コントローラーが統合階層に置かれる。
            root = Path("/sys/fs/cgroup")
            directories.append(
                (
                    root,
                    root / relative.lstrip("/"),
                    "memory.max",
                    "memory.current",
                )
            )
        elif "memory" in controllers.split(","):
            # cgroup v1ではmemoryコントローラーが独立した階層を持つ。
            root = Path("/sys/fs/cgroup/memory")
            directories.append(
                (
                    root,
                    root / relative.lstrip("/"),
                    "memory.limit_in_bytes",
                    "memory.usage_in_bytes",
                )
            )
    return tuple(directories)


def _cgroup_memory() -> tuple[int, int] | None:
    constraints: list[tuple[int, int]] = []
    for root, directory, limit_name, usage_name in _process_cgroup_directories():
        # 子cgroupより厳しい祖先の制限も実効上限になるため、ルートまで確認する。
        while directory == root or root in directory.parents:
            limit = _read_memory_limit(directory / limit_name)
            usage = _read_memory_limit(directory / usage_name)
            if limit is not None and usage is not None:
                constraints.append((limit, max(0, limit - usage)))
            if directory == root:
                break
            directory = directory.parent
    if not constraints:
        return None
    return min(limit for limit, _ in constraints), min(
        available for _, available in constraints
    )


def _host_memory() -> tuple[int, int]:
    memory = psutil.virtual_memory()
    total = int(memory.total)
    available = int(memory.available)
    cgroup = _cgroup_memory()
    if cgroup is not None:
        cgroup_total, cgroup_available = cgroup
        total = min(total, cgroup_total)
        available = min(available, cgroup_available)
    return total, available


def _cuda_memory(selection: DeviceSelection) -> tuple[int, int]:
    free, total = torch.cuda.mem_get_info(selection.runtime_device)
    reserved = torch.cuda.memory_reserved(selection.runtime_device)
    allocated = torch.cuda.memory_allocated(selection.runtime_device)
    # PyTorchが予約済みでも未使用の領域は次のモデルロードで再利用できる。
    reusable = max(0, int(reserved) - int(allocated))
    total = int(total)
    return total, min(total, int(free) + reusable)


def _opencl_total_memory(selection: DeviceSelection) -> int:
    import pyopencl as cl

    platforms = cl.get_platforms()
    assert selection.platform_index is not None
    device = platforms[selection.platform_index].get_devices()[selection.device_index]
    return int(device.global_mem_size)


def _format_mib(value: int) -> str:
    return f"{value / _MIB:.0f} MiB"


def model_load_memory_error(
    *,
    model_path: Path,
    selection: DeviceSelection,
    generator_only: bool,
    resident_device_bytes: int,
) -> str | None:
    """安全マージンを確保して次のチェックポイントをロードできない場合にのみ、その理由を返す。"""

    checkpoint_bytes = model_path.stat().st_size
    # 通常経路はモデル構築・state_dict・読み込み時の一時領域を併存させる。generator-onlyはmmapから必要な重みだけを複製する。
    host_load_bytes = math.ceil(checkpoint_bytes * (1.5 if generator_only else 3.0))
    host_total, host_available = _host_memory()
    host_reserve = max(_MIN_HOST_RESERVE, host_total // 20)
    host_required = host_load_bytes + host_reserve
    if host_available < host_required:
        return (
            f"host memory requires {_format_mib(host_required)} of headroom, "
            f"but only {_format_mib(host_available)} is available"
        )

    if selection.backend is DeviceBackend.CUDA:
        device_total, device_available = _cuda_memory(selection)
    elif selection.backend is DeviceBackend.OPENCL:
        device_total = _opencl_total_memory(selection)
        # OpenCL標準には空きデバイスメモリ取得APIがないため、総容量から本プロセスの推定常駐モデル量を引いて判定する。ドライバや他プロセスの使用量は直接取得できず、下の予約領域はそのリスクを抑えるための余裕である。
        device_available = max(0, device_total - resident_device_bytes)
    else:
        return None

    # デバイス側の倍率には、推論用重みのほかモデル移動時の一時領域とアロケーターの余裕を含める。
    device_load_bytes = math.ceil(checkpoint_bytes * (0.75 if generator_only else 2.0))
    device_reserve = max(_MIN_DEVICE_RESERVE, device_total // 10)
    device_required = device_load_bytes + device_reserve
    if device_available < device_required:
        return (
            f"{selection.backend.value} memory requires "
            f"{_format_mib(device_required)} of headroom, but only "
            f"{_format_mib(device_available)} is available"
        )
    return None


__all__ = ["model_load_memory_error"]
