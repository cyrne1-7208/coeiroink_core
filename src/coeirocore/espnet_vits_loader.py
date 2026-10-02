"""ESPnet VITSチェックポイントから推論に必要な重みだけを読み込む。"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .espnet_inference import _VitsPathDurationCalculator

_WRAPPER_TRAINING_MODULES = (
    "feats_extract",
    "pitch_extract",
    "energy_extract",
    "normalize",
    "pitch_normalize",
    "energy_normalize",
)
_VITS_TRAINING_MODULES = (
    "discriminator",
    "generator_adv_loss",
    "discriminator_adv_loss",
    "feat_match_loss",
    "mel_loss",
    "kl_loss",
)


class GeneratorOnlyText2Speech:
    """COEIROINKが使用するText2Speechの推論用インターフェースだけを保持する。"""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_args: argparse.Namespace,
        preprocess_fn: Callable[[str, dict[str, str]], dict[str, np.ndarray]],
        device: str,
        noise_scale: float,
        noise_scale_dur: float,
    ) -> None:
        self.model = model
        self.tts = model.tts
        self.train_args = train_args
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.decode_conf = {
            "use_teacher_forcing": False,
            "alpha": 1.0,
            "noise_scale": noise_scale,
            "noise_scale_dur": noise_scale_dur,
        }
        self.duration_calculator = _VitsPathDurationCalculator().eval()

    @torch.no_grad()
    def __call__(
        self,
        text: str | torch.Tensor | np.ndarray,
        decode_conf: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        from espnet2.torch_utils.device_funcs import to_device

        if isinstance(text, str):
            text = self.preprocess_fn("<dummy>", {"text": text})["text"]
        batch = to_device({"text": text}, self.device)
        config = self.decode_conf
        if decode_conf is not None:
            config = self.decode_conf | decode_conf
        output = self.model.inference(**batch, **config)
        attention = output.get("att_w")
        if attention is not None:
            duration, focus_rate = self.duration_calculator(attention)
            output.update(duration=duration, focus_rate=focus_rate)
        return output


def _read_train_args(config_path: Path) -> argparse.Namespace:
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"failed to read ESPnet config: {config_path}") from error
    if not isinstance(config, dict):
        raise TypeError(f"ESPnet config must be a YAML mapping: {config_path}")
    return argparse.Namespace(**config)


def _prune_training_modules(model: torch.nn.Module) -> None:
    tts = model.tts
    generator = tts.generator
    if not hasattr(generator, "posterior_encoder"):
        raise RuntimeError("unsupported VITS generator without posterior_encoder")

    for name in _WRAPPER_TRAINING_MODULES:
        setattr(model, name, None)
    for name in _VITS_TRAINING_MODULES:
        setattr(tts, name, None)
    generator.posterior_encoder = None
    tts._cache = None


def _restore_relative_position_encodings(model: torch.nn.Module) -> None:
    from espnet2.legacy.nets.pytorch_backend.transformer.embedding import (
        RelPositionalEncoding,
    )

    for module in model.modules():
        if not isinstance(module, RelPositionalEncoding):
            continue
        position = module.pe
        if not isinstance(position, torch.Tensor) or position.device.type != "meta":
            continue
        max_length = (position.size(1) + 1) // 2
        module.pe = None
        # buffer登録されていない相対位置埋め込みはto_empty後もmetaテンソルのまま残るため、設定済みの最大長でCPU上に再構築する。
        module.extend_pe(torch.empty((1, max_length), dtype=torch.float32))


def _move_relative_position_encodings(
    model: torch.nn.Module,
    device: str,
) -> None:
    from espnet2.legacy.nets.pytorch_backend.transformer.embedding import (
        RelPositionalEncoding,
    )

    for module in model.modules():
        if isinstance(module, RelPositionalEncoding):
            module.pe = module.pe.to(device=device, dtype=torch.float32)


def _checkpoint_state(model_path: Path) -> Mapping[str, torch.Tensor]:
    state: object = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if isinstance(state, Mapping) and isinstance(state.get("module"), Mapping):
        state = state["module"]
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise RuntimeError("unsupported ESPnet checkpoint state")
    return state


def _load_inference_state(model: torch.nn.Module, model_path: Path) -> None:
    checkpoint = _checkpoint_state(model_path)
    expected = model.state_dict()
    missing = set(expected).difference(checkpoint)
    if missing:
        names = ", ".join(sorted(missing)[:3])
        raise RuntimeError(f"VITS checkpoint is missing inference weights: {names}")

    selected = {name: checkpoint[name] for name in expected}
    mismatched = [
        name
        for name, destination in expected.items()
        if destination.shape != selected[name].shape
    ]
    if mismatched:
        names = ", ".join(sorted(mismatched)[:3])
        raise RuntimeError(
            f"VITS checkpoint has incompatible inference weights: {names}"
        )

    # assign=Falseでmmap領域からモデル側へ重みを複製し、推論中もWindowsでチェックポイントのファイルハンドルを保持しない。
    model.load_state_dict(selected, strict=True, assign=False)


def load_generator_only_text_to_speech(
    *,
    config_path: Path,
    model_path: Path,
    device: str,
    noise_scale: float,
    noise_scale_dur: float,
) -> GeneratorOnlyText2Speech:
    """VITSの構造だけをPyTorchのmetaデバイス上で構築し、推論用generatorの重みを厳密に読み込む。"""

    from espnet2.gan_tts.vits.vits import VITS
    from espnet2.tasks.tts import TTSTask

    train_args = _read_train_args(config_path)
    with torch.device("meta"):
        model = TTSTask.build_model(train_args)
    if not isinstance(model.tts, VITS):
        raise TypeError("generator-only loading supports ESPnet VITS models only")
    if getattr(model.tts, "use_gst", False):
        raise RuntimeError("generator-only loading does not support GST VITS models")

    _prune_training_modules(model)
    model.to_empty(device="cpu")
    _restore_relative_position_encodings(model)
    _load_inference_state(model, model_path)
    model.to(device=device, dtype=torch.float32).eval()
    _move_relative_position_encodings(model, device)

    preprocess_fn = TTSTask.build_preprocess_fn(train_args, False)
    if preprocess_fn is None:
        raise RuntimeError("ESPnet config does not provide a text preprocessor")
    return GeneratorOnlyText2Speech(
        model=model,
        train_args=train_args,
        preprocess_fn=preprocess_fn,
        device=device,
        noise_scale=noise_scale,
        noise_scale_dur=noise_scale_dur,
    )


__all__ = ["GeneratorOnlyText2Speech", "load_generator_only_text_to_speech"]
