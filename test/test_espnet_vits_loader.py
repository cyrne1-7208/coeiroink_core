# VITSチェックポイントの推論用重み読み込み・学習用モジュール削除・推論設定合成のテスト

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coeirocore.espnet_vits_loader import (
    GeneratorOnlyText2Speech,
    _checkpoint_state,
    _load_inference_state,
    _prune_training_modules,
)


def test_load_inference_state_copies_checkpoint_storage(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "model.pth"
    # 全体チェックポイントに含まれる学習用の重みは、推論用モデルへ読み込まない。
    torch.save(
        {**source.state_dict(), "discriminator.weight": torch.ones(2)}, checkpoint
    )
    expected = {name: value.clone() for name, value in source.state_dict().items()}

    destination = torch.nn.Linear(3, 2)
    _load_inference_state(destination, checkpoint)
    checkpoint.unlink()

    for name, value in destination.state_dict().items():
        assert torch.equal(value, expected[name])


def test_load_inference_state_rejects_missing_weight(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save({}, checkpoint)

    with pytest.raises(RuntimeError, match="Missing key"):
        _load_inference_state(torch.nn.Linear(3, 2), checkpoint)


def test_checkpoint_state_unwraps_module_mapping_and_rejects_invalid_payload(
    tmp_path: Path,
) -> None:
    wrapped = tmp_path / "wrapped.pth"
    torch.save({"module": {"weight": torch.ones(2)}}, wrapped)
    state = _checkpoint_state(wrapped)
    assert torch.equal(state["weight"], torch.ones(2))

    invalid = tmp_path / "invalid.pth"
    torch.save({"weight": "not-a-tensor"}, invalid)
    with pytest.raises(RuntimeError, match="unsupported ESPnet checkpoint state"):
        _checkpoint_state(invalid)


def test_prune_training_modules_clears_training_only_components() -> None:
    generator = SimpleNamespace(posterior_encoder=object())
    tts = SimpleNamespace(
        generator=generator,
        discriminator=object(),
        generator_adv_loss=object(),
        discriminator_adv_loss=object(),
        feat_match_loss=object(),
        mel_loss=object(),
        kl_loss=object(),
        _cache=object(),
    )
    model = SimpleNamespace(
        tts=tts,
        feats_extract=object(),
        pitch_extract=object(),
        energy_extract=object(),
        normalize=object(),
        pitch_normalize=object(),
        energy_normalize=object(),
    )

    _prune_training_modules(model)

    assert generator.posterior_encoder is None
    assert tts.discriminator is None
    assert tts._cache is None
    assert model.feats_extract is None
    assert model.energy_normalize is None

    with pytest.raises(RuntimeError, match="posterior_encoder"):
        _prune_training_modules(
            SimpleNamespace(tts=SimpleNamespace(generator=object()))
        )


def test_generator_only_text2speech_merges_decode_conf_without_mutating_defaults() -> (
    None
):
    captured: dict[str, object] = {}

    class FakeModel:
        tts = object()

        def inference(self, **kwargs: object) -> dict[str, torch.Tensor]:
            captured.update(kwargs)
            return {"wav": torch.ones(4)}

    wrapper = GeneratorOnlyText2Speech(
        model=FakeModel(),
        train_args=argparse.Namespace(),
        preprocess_fn=lambda _id, data: {"text": np.array([1, 2], dtype=np.int64)},
        device="cpu",
        noise_scale=0.333,
        noise_scale_dur=0.333,
    )

    result = wrapper("あ", decode_conf={"alpha": 0.8})

    assert torch.equal(result["wav"], torch.ones(4))
    assert captured["alpha"] == 0.8
    assert captured["noise_scale"] == 0.333
    assert wrapper.decode_conf["alpha"] == 1.0
