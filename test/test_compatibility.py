import subprocess
import sys


def test_pyworld_compat_uses_stdlib_metadata_without_deprecation_warning():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import warnings; "
                "warnings.simplefilter('error', DeprecationWarning); "
                "from coeirocore.pyworld_compat import load_pyworld; "
                "world = load_pyworld(); "
                "assert callable(world.synthesize); "
                "assert world.__version__"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""


def test_torch_weight_norm_compat_uses_parametrization_api():
    import torch
    from coeirocore.coeiro_manager import _install_torch_weight_norm_compat

    _install_torch_weight_norm_compat()

    layer = torch.nn.Conv1d(1, 1, kernel_size=3)
    torch.nn.utils.weight_norm(layer)

    assert hasattr(layer, "parametrizations")
    assert "weight" in layer.parametrizations

    torch.nn.utils.remove_weight_norm(layer)
    assert not hasattr(layer, "parametrizations") or "weight" not in layer.parametrizations
