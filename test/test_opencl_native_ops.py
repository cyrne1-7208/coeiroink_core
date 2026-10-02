"""OpenCLで実装したVITS向けATen演算のCPU比較テスト。"""

from __future__ import annotations

import copy

import pytest

from coeirocore.opencl import OpenCLVitsMode

torch = pytest.importorskip("torch")

try:
    import pytorch_ocl
except (ImportError, OSError, RuntimeError) as error:
    pytest.skip(f"pytorch_ocl is unavailable: {error}", allow_module_level=True)

try:
    _OCL_DEVICE_COUNT = int(pytorch_ocl.device_count())
except (ImportError, OSError, RuntimeError) as error:
    pytest.skip(f"OpenCL device is unavailable: {error}", allow_module_level=True)

if _OCL_DEVICE_COUNT < 1:
    pytest.skip("No OpenCL device is available", allow_module_level=True)


_OCL_DEVICE = torch.device("ocl:0")
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@pytest.fixture(autouse=True)
def _reject_unsupported_operator_warnings(recwarn):
    yield
    unsupported = [
        str(record.message)
        for record in recwarn
        if "not currently supported" in str(record.message).lower()
    ]
    if unsupported:
        pytest.fail(
            "OpenCL emitted an unsupported-operator warning:\n" + "\n".join(unsupported)
        )


def _to_ocl(value):
    if isinstance(value, torch.Tensor):
        return value.to(_OCL_DEVICE)
    if isinstance(value, tuple):
        return tuple(_to_ocl(item) for item in value)
    return value


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _assert_same(expected, actual):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        if expected.dtype == torch.bool or expected.dtype in _INTEGER_DTYPES:
            assert torch.equal(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return
    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)
    assert len(actual) == len(expected)
    for expected_item, actual_item in zip(expected, actual, strict=True):
        _assert_same(expected_item, actual_item)


def test_gather_last_dimension_matches_cpu():
    values = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], dtype=torch.float32
    )
    indices = torch.tensor([[3, 1, 1], [0, 2, 3]], dtype=torch.int64)

    expected = torch.gather(values, -1, indices)
    actual = torch.gather(_to_ocl(values), -1, _to_ocl(indices))

    _assert_same(expected, _to_cpu(actual))


def test_index_select_matches_cpu():
    values = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    indices = torch.tensor([4, 1, 1, 0], dtype=torch.int64)

    expected = torch.index_select(values, 0, indices)
    actual = torch.index_select(_to_ocl(values), 0, _to_ocl(indices))

    _assert_same(expected, _to_cpu(actual))


def test_axis_zero_integer_index_matches_cpu():
    values = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    indices = torch.tensor([3, 0, 3], dtype=torch.int64)

    expected = values[indices]
    actual = _to_ocl(values)[_to_ocl(indices)]

    _assert_same(expected, _to_cpu(actual))


def test_axis_zero_negative_integer_index_matches_cpu():
    values = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    indices = torch.tensor([-1, 0, -3], dtype=torch.int64)

    expected = values[indices]
    actual = _to_ocl(values)[_to_ocl(indices)]

    _assert_same(expected, _to_cpu(actual))


@pytest.mark.parametrize("operation", ["gather", "index_select", "index"])
def test_invalid_integer_indices_raise(operation):
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    invalid = torch.tensor([3], dtype=torch.int64)
    ocl_values = _to_ocl(values)
    ocl_invalid = _to_ocl(invalid)

    with pytest.raises(RuntimeError, match=r"index.*out of range"):
        if operation == "gather":
            torch.gather(ocl_values, 0, ocl_invalid.reshape(1, 1).expand(1, 4))
        elif operation == "index_select":
            torch.index_select(ocl_values, 0, ocl_invalid)
        else:
            ocl_values[ocl_invalid]


def test_axis_zero_boolean_index_matches_cpu():
    values = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    mask = torch.tensor([True, False, True, False, True])

    expected = values[mask]
    actual = _to_ocl(values)[_to_ocl(mask)]

    _assert_same(expected, _to_cpu(actual))


def test_boolean_index_put_without_accumulation_matches_cpu():
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask = torch.tensor([True, False, True])
    replacement = torch.tensor(
        [[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]],
        dtype=torch.float32,
    )

    expected = values.clone()
    expected.index_put_((mask,), replacement, accumulate=False)
    actual = _to_ocl(values).clone()
    actual.index_put_((_to_ocl(mask),), _to_ocl(replacement), accumulate=False)

    _assert_same(expected, _to_cpu(actual))


def test_boolean_nonzero_matches_cpu():
    mask = torch.tensor([[False, True, False], [True, True, False]])

    expected = torch.nonzero(mask)
    actual = torch.nonzero(_to_ocl(mask))

    _assert_same(expected, _to_cpu(actual))


def test_last_dimension_float32_cumsum_matches_cpu():
    values = torch.tensor(
        [[1.25, -2.0, 3.5, 4.0], [-1.0, 0.5, 2.25, -3.0]], dtype=torch.float32
    )

    expected = torch.cumsum(values, dim=-1)
    actual = torch.cumsum(_to_ocl(values), dim=-1)

    _assert_same(expected, _to_cpu(actual))


def test_float32_softplus_matches_cpu():
    values = torch.tensor(
        [-30.0, -1.0, 0.0, 19.0, 20.0, 21.0, 30.0], dtype=torch.float32
    )

    expected = torch.nn.functional.softplus(values, beta=1.0, threshold=20.0)
    actual = torch.nn.functional.softplus(_to_ocl(values), beta=1.0, threshold=20.0)

    _assert_same(expected, _to_cpu(actual))


def test_broadcast_boolean_masked_fill_matches_cpu():
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    mask = torch.tensor([[True], [False]])

    expected = values.clone()
    expected.masked_fill_(mask, -7.5)
    actual = _to_ocl(values).clone()
    actual.masked_fill_(_to_ocl(mask), -7.5)

    _assert_same(expected, _to_cpu(actual))


def test_tensor_clamp_matches_cpu():
    values = torch.tensor([[-3.0, 0.5, 8.0], [2.0, 4.0, 10.0]], dtype=torch.float32)
    minimum = torch.tensor([-1.0, 1.0, 3.0], dtype=torch.float32)
    maximum = torch.tensor([1.0, 2.0, 6.0], dtype=torch.float32)

    expected = torch.clamp(values, min=minimum, max=maximum)
    actual = torch.clamp(_to_ocl(values), min=_to_ocl(minimum), max=_to_ocl(maximum))

    _assert_same(expected, _to_cpu(actual))


def test_last_dimension_float32_max_matches_cpu():
    values = torch.tensor(
        [[-2.0, 4.0, 4.0, 1.0], [8.0, -1.0, 3.0, 2.0]], dtype=torch.float32
    )

    expected = torch.max(values, dim=-1)
    actual = torch.max(_to_ocl(values), dim=-1)

    _assert_same(expected, _to_cpu(actual))


def test_boolean_all_matches_cpu():
    values = torch.tensor([True, True, False, True])

    expected = torch.all(values)
    actual = torch.all(_to_ocl(values))

    _assert_same(expected, _to_cpu(actual))


def test_flip_matches_cpu():
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    expected = torch.flip(values, dims=[-1])
    actual = torch.flip(_to_ocl(values), dims=[-1])

    _assert_same(expected, _to_cpu(actual))


def test_weight_norm_interface_dim_zero_matches_cpu():
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [2.0, -2.5]],
            [[4.0, -3.0], [1.5, 2.5]],
        ],
        dtype=torch.float32,
    )
    scale = torch.tensor([[[1.0]], [[0.75]], [[1.25]]], dtype=torch.float32)

    expected = torch.ops.aten._weight_norm_interface(values, scale, 0)
    actual = torch.ops.aten._weight_norm_interface(_to_ocl(values), _to_ocl(scale), 0)

    _assert_same(expected, _to_cpu(actual))


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Conv1d(4, 6, 3, padding=1, groups=2),
        torch.nn.ConvTranspose1d(
            4,
            6,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=2,
        ),
    ],
    ids=("conv1d", "conv_transpose1d"),
)
def test_convolution_reuses_backend_across_dynamic_lengths(module):
    """同じOpenCL畳み込みを異なる時間長で実行しても計算結果を維持する。"""

    torch.manual_seed(4)
    cpu_module = copy.deepcopy(module)
    opencl_module = copy.deepcopy(module).to(_OCL_DEVICE)

    for length in (5, 11, 23):
        input_tensor = torch.randn(2, 4, length)
        expected = cpu_module(input_tensor)
        with OpenCLVitsMode():
            actual = opencl_module(input_tensor.to(_OCL_DEVICE))
        _assert_same(expected, _to_cpu(actual))
