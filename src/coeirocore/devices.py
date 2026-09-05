"""音声合成バックエンドのデバイス選択境界。

モデル実行部分からデバイス検出とプラットフォーム判定を分離する。
バックエンド固有のモジュールは遅延読込し、CPU利用時にDirectMLやOpenCLを必須にしない。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from operator import index as as_index
from typing import Any


class DeviceBackend(str, Enum):
    """Coreが選択できる実行バックエンド。"""

    CPU = "cpu"
    CUDA = "cuda"
    DIRECTML = "directml"
    OPENCL = "opencl"


class DeviceError(RuntimeError):
    """デバイス選択に関する基底エラー。"""


class DeviceConfigurationError(DeviceError, ValueError):
    """バックエンド名、OS、または番号の指定が不正な場合のエラー。"""


class DevicePlatformError(DeviceConfigurationError):
    """バックエンドと実行OSの組み合わせが対応していない場合のエラー。"""


class DeviceIndexError(DeviceConfigurationError):
    """デバイスまたはOpenCLプラットフォームの番号が不正な場合のエラー。"""


class BackendUnavailableError(DeviceError):
    """バックエンドが未導入、未検出、または利用不能な場合のエラー。"""


class BackendImportError(BackendUnavailableError):
    """バックエンドモジュールまたはその依存モジュールの読み込みエラー。"""


class BackendCapabilityError(BackendUnavailableError):
    """バックエンドに必要な検出APIが存在しない場合のエラー。"""


_BACKEND_ALIASES = {
    "cpu": DeviceBackend.CPU,
    "cuda": DeviceBackend.CUDA,
    "directml": DeviceBackend.DIRECTML,
    "direct_ml": DeviceBackend.DIRECTML,
    "direct-ml": DeviceBackend.DIRECTML,
    "opencl": DeviceBackend.OPENCL,
    "open_cl": DeviceBackend.OPENCL,
    "open-cl": DeviceBackend.OPENCL,
}

_MODULE_NAMES = {
    DeviceBackend.CPU: "torch",
    DeviceBackend.CUDA: "torch",
    DeviceBackend.DIRECTML: "torch_directml",
    DeviceBackend.OPENCL: "pytorch_ocl",
}

_PLATFORM_ALIASES = {
    "linux": "linux",
    "linux2": "linux",
    "gnu/linux": "linux",
    "windows": "windows",
    "win32": "windows",
    "win": "windows",
    "nt": "windows",
}


def normalize_backend(value: str | DeviceBackend) -> DeviceBackend:
    """バックエンド名を小文字の正規化値へ変換する。"""

    if isinstance(value, DeviceBackend):
        return value
    if not isinstance(value, str):
        raise TypeError("backend must be a string or DeviceBackend")

    normalized = value.strip().lower()
    try:
        return _BACKEND_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(backend.value for backend in DeviceBackend)
        raise ValueError(
            f"unsupported backend {value!r}; expected one of: {supported}"
        ) from exc


def normalize_platform(value: str | None = None) -> str:
    """OS名を検証用の正規化文字列へ変換する。"""

    raw_value = sys.platform if value is None else value
    if not isinstance(raw_value, str):
        raise TypeError("platform must be a string or None")

    normalized = raw_value.strip().lower()
    return _PLATFORM_ALIASES.get(normalized, normalized)


def validate_platform(
    backend: str | DeviceBackend,
    platform: str | None = None,
) -> str:
    """指定したバックエンドを実行できるOSか検証する。"""

    normalized_backend = normalize_backend(backend)
    normalized_platform = normalize_platform(platform)

    required_platforms = {
        DeviceBackend.DIRECTML: {"windows"},
        DeviceBackend.OPENCL: {"linux"},
    }.get(normalized_backend)
    if required_platforms is not None and normalized_platform not in required_platforms:
        allowed = ", ".join(sorted(required_platforms))
        raise DevicePlatformError(
            f"backend {normalized_backend.value!r} requires platform {allowed}; "
            f"current platform is {normalized_platform!r}"
        )

    if normalized_backend is DeviceBackend.CUDA and normalized_platform not in {
        "linux",
        "windows",
    }:
        raise DevicePlatformError(
            f"CUDA backend is supported on Linux and Windows, not {normalized_platform!r}"
        )

    return normalized_platform


def _normalize_index(value: int, name: str) -> int:
    """Python整数とnumpy整数を受け付け、負数やboolは拒否する。"""

    if isinstance(value, bool):
        raise DeviceIndexError(f"{name} must be a non-negative integer")
    try:
        normalized = as_index(value)
    except TypeError as exc:
        raise DeviceIndexError(f"{name} must be a non-negative integer") from exc
    if normalized < 0:
        raise DeviceIndexError(f"{name} must be a non-negative integer")
    return normalized


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """検証済みのデバイス設定と、Coreが実行時に利用するデバイス実体を保持する。"""

    backend: DeviceBackend
    device_index: int
    runtime_device: Any
    platform: str
    platform_index: int | None = None

    @property
    def device(self) -> Any:
        """後続コード向けの短い別名。"""

        return self.runtime_device


@dataclass(frozen=True, slots=True)
class DeviceDescriptor:
    """利用可能性一覧に含めるデバイスの軽量情報。"""

    backend: DeviceBackend
    device_index: int
    platform_index: int | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class DeviceCapability:
    """バックエンドの利用可能性と検出結果。"""

    backend: DeviceBackend
    available: bool
    devices: tuple[DeviceDescriptor, ...] = ()
    device_count: int | None = None
    platform_count: int | None = None
    reason: str | None = None

    @property
    def supported(self) -> bool:
        """`available`をAPI表示向けに表す別名。"""

        return self.available

    @property
    def device_indices(self) -> tuple[int, ...]:
        return tuple(device.device_index for device in self.devices)

    @property
    def error(self) -> str | None:
        """失敗理由を`error`名で取得する互換用プロパティ。"""

        return self.reason


ModuleImporter = Callable[[str], Any]


class DeviceResolver:
    """実行環境を検査し、明示されたバックエンドのデバイスを解決する。

    `modules`または`module_overrides`にモックを渡せるため、GPUドライバのない環境でも検出処理をテストできる。上書きされていないバックエンドモジュールは、必要になるまで遅延importされる。
    """

    def __init__(
        self,
        *,
        modules: Mapping[str | DeviceBackend, Any] | None = None,
        module_overrides: Mapping[str | DeviceBackend, Any] | None = None,
        module_importer: ModuleImporter | None = None,
        platform_name: str | None = None,
        platform: str | None = None,
    ) -> None:
        if modules is not None and module_overrides is not None:
            raise ValueError("pass either modules or module_overrides, not both")
        if platform_name is not None and platform is not None:
            raise ValueError("pass either platform_name or platform, not both")

        overrides = modules if modules is not None else module_overrides
        self._module_overrides = dict(overrides or {})
        self._module_importer = module_importer or import_module
        self.platform = normalize_platform(
            platform_name if platform_name is not None else platform
        )

    def resolve(
        self,
        backend: str | DeviceBackend,
        device_index: int = 0,
        platform_index: int = 0,
    ) -> DeviceSelection:
        """バックエンド、デバイス番号、OpenCLプラットフォーム番号を解決する。"""

        normalized_backend = normalize_backend(backend)
        normalized_device_index = _normalize_index(device_index, "device_index")
        normalized_platform_index = _normalize_index(platform_index, "platform_index")
        validate_platform(normalized_backend, self.platform)

        if normalized_backend is DeviceBackend.CPU:
            return self._resolve_cpu(normalized_device_index)
        if normalized_backend is DeviceBackend.CUDA:
            return self._resolve_cuda(normalized_device_index)
        if normalized_backend is DeviceBackend.DIRECTML:
            return self._resolve_directml(normalized_device_index)
        return self._resolve_opencl(
            normalized_device_index,
            normalized_platform_index,
        )

    def is_available(self, backend: str | DeviceBackend) -> bool:
        """指定バックエンドの検出結果だけを返す。詳細はcapabilityに残る。"""

        return self.get_supported_device_capabilities()[
            normalize_backend(backend)
        ].available

    def get_supported_device_capabilities(
        self,
    ) -> dict[DeviceBackend, DeviceCapability]:
        """4種類のバックエンドを個別に検査する。

        未導入バックエンドはCPUへ置き換えず、`available=False`と理由を返す。
        予期しない例外は検出処理の外へ伝播させ、重大な障害を隠さない。
        """

        capabilities: dict[DeviceBackend, DeviceCapability] = {}
        for backend in DeviceBackend:
            try:
                validate_platform(backend, self.platform)
                capabilities[backend] = self._probe_backend(backend)
            except DeviceError as exc:
                capabilities[backend] = DeviceCapability(
                    backend=backend,
                    available=False,
                    reason=str(exc),
                )
        return capabilities

    def _module_override(self, backend: DeviceBackend) -> tuple[bool, Any]:
        module_name = _MODULE_NAMES[backend]
        for key in (module_name, backend.value, backend):
            if key in self._module_overrides:
                return True, self._module_overrides[key]
        return False, None

    def _load_module(self, backend: DeviceBackend) -> Any:
        module_name = _MODULE_NAMES[backend]
        return self._load_named_module(module_name, backend, use_backend_alias=True)

    def _load_named_module(
        self,
        module_name: str,
        backend: DeviceBackend,
        *,
        use_backend_alias: bool = False,
    ) -> Any:
        """未導入モジュールと、導入済みモジュール内の依存関係エラーを区別して読み込む。"""

        override_keys: tuple[str | DeviceBackend, ...] = (module_name,)
        if use_backend_alias:
            override_keys += (backend.value, backend)
        overridden = False
        module = None
        for key in override_keys:
            if key in self._module_overrides:
                overridden = True
                module = self._module_overrides[key]
                break
        if overridden:
            if module is None:
                raise BackendUnavailableError(
                    f"{backend.value} backend module {module_name!r} is unavailable"
                )
            return module

        try:
            return self._module_importer(module_name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or module_name
            if missing_name == module_name:
                raise BackendUnavailableError(
                    f"{backend.value} backend requires optional module {module_name!r}"
                ) from exc
            raise BackendImportError(
                f"{backend.value} backend module {module_name!r} could not import "
                f"dependency {missing_name!r}"
            ) from exc
        except ImportError as exc:
            raise BackendImportError(
                f"{backend.value} backend module {module_name!r} could not be imported: {exc}"
            ) from exc

    @staticmethod
    def _required_callable(
        module: Any, name: str, backend: DeviceBackend
    ) -> Callable[..., Any]:
        value = getattr(module, name, None)
        if not callable(value):
            raise BackendCapabilityError(
                f"{backend.value} backend does not provide callable {name!r}"
            )
        return value

    @staticmethod
    def _read_bool(module: Any, name: str, backend: DeviceBackend) -> bool:
        value = getattr(module, name, None)
        if callable(value):
            value = value()
        if not isinstance(value, bool):
            raise BackendCapabilityError(
                f"{backend.value} backend {name!r} must return bool"
            )
        return value

    @staticmethod
    def _read_optional_count(
        module: Any, name: str, backend: DeviceBackend
    ) -> int | None:
        value = getattr(module, name, None)
        if value is None:
            return None
        if callable(value):
            value = value()
        try:
            count = as_index(value)
        except TypeError as exc:
            raise BackendCapabilityError(
                f"{backend.value} backend {name!r} must return an integer"
            ) from exc
        if count < 0:
            raise BackendCapabilityError(
                f"{backend.value} backend {name!r} returned a negative count"
            )
        return count

    @staticmethod
    def _make_torch_device(
        torch_module: Any,
        value: str,
        backend: DeviceBackend,
    ) -> Any:
        factory = DeviceResolver._required_callable(torch_module, "device", backend)
        try:
            return factory(value)
        except Exception as exc:
            raise BackendCapabilityError(
                f"{backend.value} backend could not create torch device {value!r}: {exc}"
            ) from exc

    def _resolve_cpu(self, device_index: int) -> DeviceSelection:
        if device_index != 0:
            raise DeviceIndexError("CPU backend only has device_index 0")
        torch_module = self._load_module(DeviceBackend.CPU)
        runtime_device = self._make_torch_device(
            torch_module,
            "cpu",
            DeviceBackend.CPU,
        )
        return DeviceSelection(
            backend=DeviceBackend.CPU,
            device_index=0,
            runtime_device=runtime_device,
            platform=self.platform,
        )

    def _resolve_cuda(self, device_index: int) -> DeviceSelection:
        torch_module = self._load_module(DeviceBackend.CUDA)
        cuda_module = getattr(torch_module, "cuda", None)
        if cuda_module is None:
            raise BackendCapabilityError("CUDA backend is missing torch.cuda")
        if not self._read_bool(cuda_module, "is_available", DeviceBackend.CUDA):
            raise BackendUnavailableError("CUDA backend is not available on this host")
        count = self._read_optional_count(
            cuda_module,
            "device_count",
            DeviceBackend.CUDA,
        )
        if count is None:
            raise BackendCapabilityError("CUDA backend does not provide device_count")
        if count == 0:
            raise BackendUnavailableError("CUDA backend reported zero devices")
        if device_index >= count:
            raise DeviceIndexError(
                f"CUDA device_index {device_index} is out of range; device_count={count}"
            )
        return DeviceSelection(
            backend=DeviceBackend.CUDA,
            device_index=device_index,
            runtime_device=self._make_torch_device(
                torch_module,
                f"cuda:{device_index}",
                DeviceBackend.CUDA,
            ),
            platform=self.platform,
        )

    def _resolve_directml(self, device_index: int) -> DeviceSelection:
        directml_module = self._load_module(DeviceBackend.DIRECTML)
        availability = getattr(directml_module, "is_available", None)
        if availability is not None and not self._read_bool(
            directml_module,
            "is_available",
            DeviceBackend.DIRECTML,
        ):
            raise BackendUnavailableError(
                "DirectML backend is not available on this host"
            )

        count = self._read_optional_count(
            directml_module,
            "device_count",
            DeviceBackend.DIRECTML,
        )
        if count == 0:
            raise BackendUnavailableError("DirectML backend reported zero devices")
        if count is not None and device_index >= count:
            raise DeviceIndexError(
                f"DirectML device_index {device_index} is out of range; device_count={count}"
            )

        factory = self._required_callable(
            directml_module,
            "device",
            DeviceBackend.DIRECTML,
        )
        try:
            runtime_device = factory(device_index)
        except Exception as exc:
            raise BackendUnavailableError(
                f"DirectML device {device_index} could not be created: {exc}"
            ) from exc

        return DeviceSelection(
            backend=DeviceBackend.DIRECTML,
            device_index=device_index,
            runtime_device=runtime_device,
            platform=self.platform,
        )

    def _opencl_platforms(self) -> list[Any]:
        # pytorch_oclのimportがPrivateUse1を`ocl`として登録し、pyopenclは上流バックエンドと同じ順序でplatform/deviceを列挙するためだけに使う。
        self._load_module(DeviceBackend.OPENCL)
        opencl_module = self._load_named_module(
            "pyopencl",
            DeviceBackend.OPENCL,
        )
        get_platforms = self._required_callable(
            opencl_module,
            "get_platforms",
            DeviceBackend.OPENCL,
        )
        try:
            platforms = list(get_platforms())
        except Exception as exc:
            raise BackendUnavailableError(
                f"OpenCL platform detection failed: {exc}"
            ) from exc
        if not platforms:
            raise BackendUnavailableError("OpenCL reported no platforms")
        return platforms

    @staticmethod
    def _opencl_devices(platform: Any) -> list[Any]:
        get_devices = getattr(platform, "get_devices", None)
        if not callable(get_devices):
            raise BackendCapabilityError(
                "OpenCL platform does not provide callable get_devices"
            )
        try:
            return list(get_devices())
        except Exception as exc:
            raise BackendUnavailableError(
                f"OpenCL device detection failed: {exc}"
            ) from exc

    def _resolve_opencl(
        self, device_index: int, platform_index: int
    ) -> DeviceSelection:
        platforms = self._opencl_platforms()
        if platform_index >= len(platforms):
            raise DeviceIndexError(
                f"OpenCL platform_index {platform_index} is out of range; "
                f"platform_count={len(platforms)}"
            )
        devices = self._opencl_devices(platforms[platform_index])
        if not devices:
            raise BackendUnavailableError(
                f"OpenCL platform_index {platform_index} reported no devices"
            )
        if device_index >= len(devices):
            raise DeviceIndexError(
                f"OpenCL device_index {device_index} is out of range; "
                f"device_count={len(devices)} on platform_index={platform_index}"
            )
        # PyTorchの`ocl:N`は一次元の通し番号だけを受け取るため、OpenCLのplatform/device指定を全プラットフォーム通しの番号へ変換する。
        flat_device_index = (
            sum(
                len(self._opencl_devices(platform))
                for platform in platforms[:platform_index]
            )
            + device_index
        )
        torch_module = self._load_named_module(
            "torch",
            DeviceBackend.OPENCL,
        )
        return DeviceSelection(
            backend=DeviceBackend.OPENCL,
            device_index=device_index,
            runtime_device=self._make_torch_device(
                torch_module,
                f"ocl:{flat_device_index}",
                DeviceBackend.OPENCL,
            ),
            platform=self.platform,
            platform_index=platform_index,
        )

    def _probe_cpu(self) -> DeviceCapability:
        backend = DeviceBackend.CPU
        self._resolve_cpu(0)
        return DeviceCapability(
            backend=backend,
            available=True,
            devices=(DeviceDescriptor(backend=backend, device_index=0, name="CPU"),),
            device_count=1,
        )

    def _probe_cuda(self) -> DeviceCapability:
        backend = DeviceBackend.CUDA
        torch_module = self._load_module(backend)
        cuda_module = getattr(torch_module, "cuda", None)
        if cuda_module is None:
            raise BackendCapabilityError("CUDA backend is missing torch.cuda")
        if not self._read_bool(cuda_module, "is_available", backend):
            raise BackendUnavailableError("CUDA backend is not available on this host")
        count = self._read_optional_count(cuda_module, "device_count", backend)
        if count is None:
            raise BackendCapabilityError("CUDA backend does not provide device_count")
        if count == 0:
            raise BackendUnavailableError("CUDA backend reported zero devices")
        return DeviceCapability(
            backend=backend,
            available=True,
            devices=tuple(
                DeviceDescriptor(backend=backend, device_index=index)
                for index in range(count)
            ),
            device_count=count,
        )

    def _probe_directml(self) -> DeviceCapability:
        backend = DeviceBackend.DIRECTML
        directml_module = self._load_module(backend)
        availability = getattr(directml_module, "is_available", None)
        if availability is not None and not self._read_bool(
            directml_module,
            "is_available",
            backend,
        ):
            raise BackendUnavailableError(
                "DirectML backend is not available on this host"
            )
        count = self._read_optional_count(directml_module, "device_count", backend)
        if count == 0:
            raise BackendUnavailableError("DirectML backend reported zero devices")
        self.resolve(backend, 0)
        known_devices = range(count) if count is not None else range(1)
        return DeviceCapability(
            backend=backend,
            available=True,
            devices=tuple(
                DeviceDescriptor(backend=backend, device_index=index)
                for index in known_devices
            ),
            device_count=count,
        )

    def _probe_opencl(self) -> DeviceCapability:
        backend = DeviceBackend.OPENCL
        platforms = self._opencl_platforms()
        devices: list[DeviceDescriptor] = []
        for current_platform_index, current_platform in enumerate(platforms):
            for current_device_index, current_device in enumerate(
                self._opencl_devices(current_platform)
            ):
                name = getattr(current_device, "name", None)
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                if name is not None and not isinstance(name, str):
                    name = str(name)
                devices.append(
                    DeviceDescriptor(
                        backend=backend,
                        device_index=current_device_index,
                        platform_index=current_platform_index,
                        name=name,
                    )
                )
        if not devices:
            raise BackendUnavailableError("OpenCL reported no devices")
        return DeviceCapability(
            backend=backend,
            available=True,
            devices=tuple(devices),
            device_count=len(devices),
            platform_count=len(platforms),
        )

    def _probe_backend(self, backend: DeviceBackend) -> DeviceCapability:
        """実際のデバイス生成まで試し、利用可能なバックエンドだけを公開情報へ含める。"""

        if backend is DeviceBackend.CPU:
            return self._probe_cpu()
        if backend is DeviceBackend.CUDA:
            return self._probe_cuda()
        if backend is DeviceBackend.DIRECTML:
            return self._probe_directml()
        if backend is DeviceBackend.OPENCL:
            return self._probe_opencl()
        raise DeviceConfigurationError(f"unsupported backend: {backend}")


def resolve_device(
    backend: str | DeviceBackend,
    device_index: int = 0,
    platform_index: int = 0,
    *,
    resolver: DeviceResolver | None = None,
    modules: Mapping[str | DeviceBackend, Any] | None = None,
    module_overrides: Mapping[str | DeviceBackend, Any] | None = None,
    module_importer: ModuleImporter | None = None,
    platform_name: str | None = None,
) -> DeviceSelection:
    """DeviceResolverを直接指定せずにバックエンドのデバイスを解決する簡易API。"""

    if resolver is not None and any(
        value is not None
        for value in (modules, module_overrides, module_importer, platform_name)
    ):
        raise ValueError("resolver cannot be combined with resolver configuration")
    active_resolver = resolver or DeviceResolver(
        modules=modules,
        module_overrides=module_overrides,
        module_importer=module_importer,
        platform_name=platform_name,
    )
    return active_resolver.resolve(
        backend,
        device_index=device_index,
        platform_index=platform_index,
    )


def get_supported_device_capabilities(
    *,
    resolver: DeviceResolver | None = None,
    modules: Mapping[str | DeviceBackend, Any] | None = None,
    module_overrides: Mapping[str | DeviceBackend, Any] | None = None,
    module_importer: ModuleImporter | None = None,
    platform_name: str | None = None,
) -> dict[DeviceBackend, DeviceCapability]:
    """DeviceResolverを直接指定せずに、全バックエンドの利用可能性を取得する簡易API。"""

    if resolver is not None and any(
        value is not None
        for value in (modules, module_overrides, module_importer, platform_name)
    ):
        raise ValueError("resolver cannot be combined with resolver configuration")
    active_resolver = resolver or DeviceResolver(
        modules=modules,
        module_overrides=module_overrides,
        module_importer=module_importer,
        platform_name=platform_name,
    )
    return active_resolver.get_supported_device_capabilities()


__all__ = [
    "BackendCapabilityError",
    "BackendImportError",
    "BackendUnavailableError",
    "DeviceBackend",
    "DeviceCapability",
    "DeviceConfigurationError",
    "DeviceDescriptor",
    "DeviceError",
    "DeviceIndexError",
    "DevicePlatformError",
    "DeviceResolver",
    "DeviceSelection",
    "get_supported_device_capabilities",
    "normalize_backend",
    "normalize_platform",
    "resolve_device",
    "validate_platform",
]
