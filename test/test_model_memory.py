from pathlib import Path
from unittest.mock import patch

from coeirocore.devices import DeviceBackend, DeviceSelection
from coeirocore.model_memory import _cuda_memory, model_load_memory_error


def _selection(backend: DeviceBackend) -> DeviceSelection:
    return DeviceSelection(
        backend=backend,
        device_index=0,
        runtime_device=f"{backend.value}:0",
        platform="linux",
        platform_index=0 if backend is DeviceBackend.OPENCL else None,
    )


def test_model_load_memory_guard_accepts_sufficient_host_memory(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"0" * 1024)

    with patch("coeirocore.model_memory._host_memory", return_value=(10**9, 10**9)):
        error = model_load_memory_error(
            model_path=model_path,
            selection=_selection(DeviceBackend.CPU),
            generator_only=False,
            resident_device_bytes=0,
        )

    assert error is None


def test_model_load_memory_guard_rejects_insufficient_host_memory(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"0" * 1024)

    with patch("coeirocore.model_memory._host_memory", return_value=(10**9, 1)):
        error = model_load_memory_error(
            model_path=model_path,
            selection=_selection(DeviceBackend.CPU),
            generator_only=True,
            resident_device_bytes=0,
        )

    assert error is not None
    assert "host memory" in error


def test_model_load_memory_guard_checks_cuda_headroom(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"0" * 1024)

    with (
        patch("coeirocore.model_memory._host_memory", return_value=(10**9, 10**9)),
        patch("coeirocore.model_memory._cuda_memory", return_value=(10**9, 1)),
    ):
        error = model_load_memory_error(
            model_path=model_path,
            selection=_selection(DeviceBackend.CUDA),
            generator_only=False,
            resident_device_bytes=0,
        )

    assert error is not None
    assert "cuda memory" in error


def test_cuda_reusable_memory_does_not_exceed_device_total() -> None:
    selection = _selection(DeviceBackend.CUDA)
    with (
        patch("torch.cuda.mem_get_info", return_value=(900, 1000)),
        patch("torch.cuda.memory_reserved", return_value=500),
        patch("torch.cuda.memory_allocated", return_value=100),
    ):
        assert _cuda_memory(selection) == (1000, 1000)
