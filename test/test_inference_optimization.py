from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn.utils.weight_norm import WeightNorm

from coeirocore.devices import DeviceBackend
from coeirocore.espnet_inference import (
    _VitsPathDurationCalculator,
    optimize_espnet_for_inference,
)


def test_vits_duration_calculator_counts_path_without_token_loop() -> None:
    attention = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    durations, focus_rate = _VitsPathDurationCalculator()(attention)

    assert torch.equal(durations, torch.tensor([2, 1, 2]))
    assert focus_rate.item() == 1.0


def test_vits_optimization_freezes_weight_norm_without_changing_output() -> None:
    class FakeVits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            convolution = torch.nn.Conv1d(2, 2, kernel_size=3, padding=1)
            # ESPnet VITSが現在使う旧weight normを、非推奨ラッパー経由の警告なしで再現する。
            with pytest.warns(FutureWarning, match="weight_norm.*deprecated"):
                WeightNorm.apply(convolution, "weight", 0)
            self.generator = torch.nn.Sequential(convolution)

    vits = FakeVits().eval()
    input_tensor = torch.randn(1, 2, 16)
    before = vits.generator(input_tensor).detach().clone()
    text_to_speech = SimpleNamespace(
        model=SimpleNamespace(tts=vits),
        duration_calculator=object(),
    )

    with patch(
        "coeirocore.espnet_inference._vits_type",
        return_value=FakeVits,
    ):
        optimize_espnet_for_inference(text_to_speech)

    after = vits.generator(input_tensor).detach()
    convolution = vits.generator[0]
    assert not hasattr(convolution, "weight_g")
    assert not hasattr(convolution, "weight_v")
    assert torch.equal(before, after)
    assert isinstance(text_to_speech.duration_calculator, _VitsPathDurationCalculator)


def test_espnet_inference_runs_in_inference_mode() -> None:
    from coeirocore.coeiro_manager import EspnetModel

    class FakeTextToSpeech:
        def __call__(self, text):
            assert torch.is_inference_mode_enabled()
            return {"wav": torch.zeros(1)}

    model = EspnetModel.__new__(EspnetModel)
    model.tts_model = FakeTextToSpeech()
    model._supports_speed_control = False
    model.device_selection = SimpleNamespace(backend=DeviceBackend.CPU)

    result = model._run_inference("test", seed=0)

    assert torch.equal(result["wav"], torch.zeros(1))
