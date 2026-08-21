"""OpenCL PrivateUse1バックエンドをPyTorch 2.10へ接続する実行時バインディング。"""

import pyopencl as cl
import torch

from .pt_ocl import (
    impl_empty_cache,
    impl_is_bad_fork,
    impl_seed_all,
    impl_synchronize_device,
)

__all__ = [
    "_is_in_bad_fork",
    "current_device",
    "device",
    "device_count",
    "empty_cache",
    "is_available",
    "manual_seed_all",
    "synchronize",
]


def _device_index(device: int | str | torch.device | None) -> int:
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "ocl":
            raise ValueError(f"Expected an ocl device, but got: {device}")
        return current_device() if device.index is None else device.index
    raise TypeError(f"Expected an ocl device, an integer, or None, but got: {device!r}")


def device_count() -> int:
    # pytorch_dlprimのPrivateUse1 Hooksはdevice_countを実装していないため、Coreと同じOpenCL列挙順を使う。
    return sum(len(platform.get_devices()) for platform in cl.get_platforms())


def is_available() -> bool:
    return device_count() > 0


def current_device() -> int:
    return int(torch._C._accelerator_getDeviceIndex())


class device:
    """現在のOpenCLデバイスを一時的に切り替えるコンテキストマネージャー。"""

    def __init__(self, device: int | str | torch.device | None) -> None:
        self.idx = _device_index(device)
        self.prev_idx = -1

    def __enter__(self) -> None:
        self.prev_idx = int(torch._C._accelerator_exchangeDevice(self.idx))

    def __exit__(self, *exc_info: object) -> bool:
        torch._C._accelerator_maybeExchangeDevice(self.prev_idx)
        return False


def synchronize(device: int | str | torch.device | None = None) -> None:
    impl_synchronize_device(-1 if device is None else _device_index(device))


def manual_seed_all(seed: int) -> None:
    impl_seed_all(int(seed))


def _is_in_bad_fork() -> bool:
    return bool(impl_is_bad_fork())


def empty_cache() -> None:
    impl_empty_cache()


class _OCL:
    """`torch._register_device_module`が要求するCUDA互換の最小デバイスAPI。"""

    device = device
    device_count = staticmethod(device_count)
    is_available = staticmethod(is_available)
    current_device = staticmethod(current_device)
    synchronize = staticmethod(synchronize)
    manual_seed_all = staticmethod(manual_seed_all)
    _is_in_bad_fork = staticmethod(_is_in_bad_fork)
    empty_cache = staticmethod(empty_cache)


# PrivateUse1を利用者向けの`ocl`名へ変更し、`torch.ocl`からデバイス操作APIを参照可能にする。
torch.utils.rename_privateuse1_backend("ocl")
torch._register_device_module("ocl", _OCL)
