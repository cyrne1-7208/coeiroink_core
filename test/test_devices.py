from types import SimpleNamespace
from typing import ClassVar

import pytest

from coeirocore.devices import (
    BackendImportError,
    BackendUnavailableError,
    DeviceBackend,
    DeviceIndexError,
    DevicePlatformError,
    DeviceResolver,
    get_supported_device_capabilities,
    normalize_backend,
    normalize_platform,
    resolve_device,
)


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    @staticmethod
    def device(value):
        return f"torch:{value}"


class FakeDirectML:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 2

    @staticmethod
    def device(index=0):
        return f"directml:{index}"


class FakeOpenCLPlatform:
    def __init__(self, names):
        self._devices = [SimpleNamespace(name=name) for name in names]

    def get_devices(self):
        return self._devices


class FakeOpenCL:
    platforms: ClassVar = [
        FakeOpenCLPlatform([b"AMD GPU"]),
        FakeOpenCLPlatform(["CPU device", "Intel GPU"]),
    ]

    @classmethod
    def get_platforms(cls):
        return cls.platforms


class FakePytorchOpenCL:
    pass


def test_backend_names_are_normalized_without_importing_backend_modules():
    assert normalize_backend(" CPU ") is DeviceBackend.CPU
    assert normalize_backend("CUDA") is DeviceBackend.CUDA
    assert normalize_backend("direct_ml") is DeviceBackend.DIRECTML
    assert normalize_backend("open-cl") is DeviceBackend.OPENCL
    assert normalize_platform("win32") == "windows"
    assert normalize_platform("Linux") == "linux"

    with pytest.raises(ValueError, match="unsupported backend"):
        normalize_backend("rocm")


def test_cpu_and_cuda_resolution_uses_injected_torch_module():
    resolver = DeviceResolver(modules={"torch": FakeTorch}, platform_name="linux")

    cpu = resolver.resolve("cpu")
    cuda = resolver.resolve("cuda", device_index=1)

    assert cpu.backend is DeviceBackend.CPU
    assert cpu.runtime_device == "torch:cpu"
    assert cuda.backend is DeviceBackend.CUDA
    assert cuda.device_index == 1
    assert cuda.runtime_device == "torch:cuda:1"

    with pytest.raises(DeviceIndexError, match="out of range"):
        resolver.resolve("cuda", device_index=2)

    with pytest.raises(DeviceIndexError, match="CPU backend"):
        resolver.resolve("cpu", device_index=1)


def test_cuda_unavailability_is_explicit():
    unavailable_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda value: value,
    )
    resolver = DeviceResolver(
        modules={"torch": unavailable_torch},
        platform_name="windows",
    )

    with pytest.raises(BackendUnavailableError, match="CUDA backend is not available"):
        resolver.resolve("cuda")


def test_directml_requires_windows_and_validates_device_index():
    resolver = DeviceResolver(
        modules={"torch_directml": FakeDirectML},
        platform_name="Windows",
    )

    selection = resolver.resolve("directml", device_index=1)
    assert selection.runtime_device == "directml:1"

    with pytest.raises(DeviceIndexError, match="out of range"):
        resolver.resolve("directml", device_index=2)

    linux_resolver = DeviceResolver(
        modules={"torch_directml": FakeDirectML},
        platform_name="linux",
    )
    with pytest.raises(DevicePlatformError, match="requires platform windows"):
        linux_resolver.resolve("directml")


def test_opencl_validates_platform_and_device_indices():
    resolver = DeviceResolver(
        modules={
            "pytorch_ocl": FakePytorchOpenCL,
            "pyopencl": FakeOpenCL,
            "torch": FakeTorch,
        },
        platform_name="linux",
    )

    selection = resolver.resolve("opencl", platform_index=1, device_index=1)
    assert selection.runtime_device == "torch:ocl:2"
    assert selection.platform_index == 1

    with pytest.raises(DeviceIndexError, match="platform_index"):
        resolver.resolve("opencl", platform_index=2)
    with pytest.raises(DeviceIndexError, match="device_index"):
        resolver.resolve("opencl", platform_index=0, device_index=1)

    windows_resolver = DeviceResolver(
        modules={
            "pytorch_ocl": FakePytorchOpenCL,
            "pyopencl": FakeOpenCL,
            "torch": FakeTorch,
        },
        platform_name="windows",
    )
    with pytest.raises(DevicePlatformError, match="requires platform linux"):
        windows_resolver.resolve("opencl")


def test_optional_import_is_lazy_and_missing_module_is_explicit():
    calls = []

    def importer(name):
        calls.append(name)
        if name == "torch":
            return FakeTorch
        raise ModuleNotFoundError(name=name)

    resolver = DeviceResolver(module_importer=importer, platform_name="linux")
    resolver.resolve("cpu")
    assert calls == ["torch"]

    windows_resolver = DeviceResolver(module_importer=importer, platform_name="windows")
    with pytest.raises(BackendUnavailableError, match="torch_directml"):
        windows_resolver.resolve("directml")

    def broken_importer(name):
        raise ModuleNotFoundError(name="required_dependency")

    with pytest.raises(BackendImportError, match="required_dependency"):
        DeviceResolver(module_importer=broken_importer, platform_name="linux").resolve(
            "cpu"
        )


def test_supported_capabilities_keep_backend_failures_visible():
    capabilities = get_supported_device_capabilities(
        modules={
            "torch": FakeTorch,
            "pyopencl": FakeOpenCL,
            "pytorch_ocl": FakePytorchOpenCL,
            "torch_directml": FakeDirectML,
        },
        platform_name="linux",
    )

    assert capabilities["cpu"].available
    assert capabilities["cuda"].device_count == 2
    assert capabilities["opencl"].device_count == 3
    assert capabilities["opencl"].platform_count == 2
    assert capabilities["opencl"].devices[0].name == "AMD GPU"
    assert not capabilities["directml"].available
    assert "requires platform windows" in capabilities["directml"].reason


def test_module_level_resolve_api_accepts_a_resolver():
    resolver = DeviceResolver(modules={"torch": FakeTorch}, platform_name="linux")
    selection = resolve_device("cuda", device_index=0, resolver=resolver)
    assert selection.runtime_device == "torch:cuda:0"

    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_device("cpu", resolver=resolver, platform_name="linux")
