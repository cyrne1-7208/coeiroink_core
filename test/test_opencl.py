import torch

from coeirocore.opencl import OpenCLVitsMode


def _run(module: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    with OpenCLVitsMode():
        return module(input_tensor)


def test_grouped_conv1d_matches_cpu_result():
    torch.manual_seed(1)
    module = torch.nn.Conv1d(4, 6, 3, padding=1, groups=2)
    input_tensor = torch.randn(2, 4, 9)

    expected = module(input_tensor)
    actual = _run(module, input_tensor)

    torch.testing.assert_close(actual, expected)


def test_grouped_conv_transpose1d_matches_cpu_result():
    torch.manual_seed(2)
    module = torch.nn.ConvTranspose1d(
        4,
        6,
        3,
        stride=2,
        padding=1,
        output_padding=1,
        groups=2,
    )
    input_tensor = torch.randn(2, 4, 5)

    expected = module(input_tensor)
    actual = _run(module, input_tensor)

    torch.testing.assert_close(actual, expected)


def test_string_padding_path_matches_cpu_result():
    torch.manual_seed(3)
    input_tensor = torch.randn(2, 4, 8)

    for padding in ("same", "valid"):
        module = torch.nn.Conv1d(4, 6, 3, padding=padding, groups=2)
        expected = module(input_tensor)
        actual = _run(module, input_tensor)
        torch.testing.assert_close(actual, expected)


def test_unrelated_operations_keep_their_normal_behavior():
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    with OpenCLVitsMode():
        actual = torch.sin(input_tensor) + 2

    torch.testing.assert_close(actual, torch.sin(input_tensor) + 2)
