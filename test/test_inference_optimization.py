# ESPnet VITS推論向け最適化であるweight_norm固定とduration計算のテスト

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn.utils.weight_norm import WeightNorm

from coeirocore import espnet_inference
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


def test_vits_optimization_freezes_weight_norm_without_changing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            convolution = torch.nn.Conv1d(2, 2, kernel_size=3, padding=1)
            # ESPnet VITSが現在使う旧weight normを、非推奨ラッパー経由の警告なしで再現する。
            with pytest.warns(FutureWarning, match="weight_norm.*deprecated"):
                WeightNorm.apply(convolution, "weight", 0)
            self.generator = torch.nn.Sequential(convolution)
            self.discriminator = torch.nn.Linear(2, 2)

    vits = FakeVits().eval()
    input_tensor = torch.randn(1, 2, 16)
    before = vits.generator(input_tensor).detach().clone()
    text_to_speech = SimpleNamespace(
        model=SimpleNamespace(tts=vits),
        duration_calculator=object(),
    )
    monkeypatch.setattr(espnet_inference, "_vits_type", lambda: FakeVits)

    optimize_espnet_for_inference(text_to_speech)

    after = vits.generator(input_tensor).detach()
    convolution = vits.generator[0]
    assert not hasattr(convolution, "weight_g")
    assert not hasattr(convolution, "weight_v")
    assert torch.equal(before, after)
    assert isinstance(text_to_speech.duration_calculator, _VitsPathDurationCalculator)
    assert vits.discriminator is None
