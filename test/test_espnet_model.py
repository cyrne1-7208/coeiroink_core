# EspnetModelの推論モード・話速制御・正規化・常駐バイト数計測のテスト

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from coeirocore import coeiro_manager
from coeirocore.coeiro_manager import EspnetModel
from coeirocore.devices import DeviceBackend, resolve_device


def test_smoothing_text_encoding_uses_the_models_normalizer() -> None:
    model = object.__new__(EspnetModel)
    ids = np.array([1, 2, 3], dtype=np.int64)
    model.tts_model = SimpleNamespace(
        preprocess_fn=MagicMock(return_value={"text": ids})
    )
    model.token_id_converter = SimpleNamespace(
        ids2tokens=MagicMock(return_value=["^", "a", "$"])
    )

    actual_ids, tokens = model.encode_text_with_tokens("ｱ")

    assert actual_ids is ids
    assert tokens == ["^", "a", "$"]
    model.tts_model.preprocess_fn.assert_called_once_with("<dummy>", {"text": "ｱ"})
    model.token_id_converter.ids2tokens.assert_called_once_with(ids)


def test_espnet_model_uses_request_speed_without_duration_materialization() -> None:
    class FakeText2Speech:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, dict[str, float]]] = []

        def __call__(
            self, text: np.ndarray, decode_conf: dict[str, float]
        ) -> dict[str, torch.Tensor]:
            self.calls.append((text, decode_conf))
            return {"wav": torch.ones(4, dtype=torch.float32)}

    model = EspnetModel.__new__(EspnetModel)
    model._speed_control_alpha = 1.0
    model._supports_speed_control = True
    model.device_selection = resolve_device("cpu")
    model.tts_model = FakeText2Speech()

    first = model.make_voice(np.array([1, 2], dtype=np.int64))
    model.set_speed_control_alpha(0.8)
    second = model.make_voice(np.array([1, 2], dtype=np.int64))

    assert np.array_equal(first, np.ones(4, dtype=np.float32))
    assert np.array_equal(second, np.ones(4, dtype=np.float32))
    assert model.tts_model.calls[0][1] == {"alpha": 1.0}
    assert model.tts_model.calls[1][1] == {"alpha": 0.8}


def test_espnet_model_passes_runtime_device_as_string(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("token_list: ['<unk>']\n", encoding="utf-8")

    class FakeText2Speech:
        def __init__(self, *args: object, device: object, **kwargs: object) -> None:
            captured["device"] = device
            self.decode_conf = {"alpha": 1.0}
            self.train_args = SimpleNamespace(token_list=["<unk>"])

    class FakeTokenIDConverter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(coeiro_manager, "_load_text_to_speech", lambda: FakeText2Speech)
    monkeypatch.setattr(
        coeiro_manager, "_load_token_id_converter", lambda: FakeTokenIDConverter
    )

    EspnetModel(
        config_path=config_path,
        model_path=tmp_path / "model.pth",
        device=resolve_device("cpu"),
        generator_only=False,
    )

    assert captured["device"] == "cpu"


def test_espnet_inference_runs_in_inference_mode() -> None:
    class FakeTextToSpeech:
        def __call__(self, _text: str) -> dict[str, torch.Tensor]:
            assert torch.is_inference_mode_enabled()
            return {"wav": torch.zeros(1)}

    model = EspnetModel.__new__(EspnetModel)
    model.tts_model = FakeTextToSpeech()
    model._supports_speed_control = False
    model.device_selection = SimpleNamespace(backend=DeviceBackend.CPU)

    result = model._run_inference("test", seed=0)

    assert torch.equal(result["wav"], torch.zeros(1))


def test_espnet_model_resident_bytes_includes_unregistered_position_data() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.pe = torch.ones(3)

    model = EspnetModel.__new__(EspnetModel)
    model.tts_model = SimpleNamespace(model=FakeModel())

    assert model.resident_bytes == 5 * torch.tensor(1.0).element_size()

    # ESPnetは長い入力に合わせて未登録の位置埋め込みを拡張するため、読み込み時の値を固定しない。
    model.tts_model.model.pe = torch.ones(7)
    assert model.resident_bytes == 9 * torch.tensor(1.0).element_size()
