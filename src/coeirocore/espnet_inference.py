# ESPnetの公開推論オブジェクトに、出力の互換性を維持した軽量化だけを適用する処理

from functools import lru_cache
from typing import Any

import torch


# VITSのone-hotアライメントをPythonループなしで列方向へ集計する継続長計算器
class _VitsPathDurationCalculator(torch.nn.Module):
    @torch.no_grad()
    def forward(self, attention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if attention.ndim != 2:
            raise ValueError("VITS attention must be a 2D tensor")
        durations = attention.sum(dim=0).to(dtype=torch.long)
        focus_rate = attention.max(dim=-1).values.mean()
        return durations, focus_rate


# モデル読み込み時までESPnetのVITS実装を読み込まない。
@lru_cache(maxsize=1)
def _vits_type() -> type:
    from espnet2.gan_tts.vits.vits import VITS

    return VITS


# 推論中に毎回再計算される旧weight normを現在の等価な重みへ固定する。
def _remove_legacy_weight_norm(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "weight_g") and hasattr(child, "weight_v"):
            torch.nn.utils.remove_weight_norm(child)


# 対応するVITSだけを最適化し、将来の別方式モデルはESPnet既定動作のままにする。
def optimize_espnet_for_inference(text_to_speech: Any) -> None:
    model = getattr(text_to_speech, "model", None)
    tts = getattr(model, "tts", None)
    if tts is None or not isinstance(tts, _vits_type()):
        return

    # 旧Text2Speech経路だけに適用し、継続長を直接返すgenerator-only経路には追加しない。
    if hasattr(text_to_speech, "duration_calculator"):
        text_to_speech.duration_calculator = _VitsPathDurationCalculator().eval()
    # VITS generatorは推論専用としてロードされるため、weight normの再パラメータ化を保持する必要がない。
    with torch.no_grad():
        _remove_legacy_weight_norm(tts.generator)

    # 音声合成ではteacher forcingを行わないため、学習とteacher forcingだけに使うモジュールを常駐させない。
    for name in (
        "discriminator",
        "generator_adv_loss",
        "discriminator_adv_loss",
        "feat_match_loss",
        "mel_loss",
        "kl_loss",
    ):
        if hasattr(tts, name):
            setattr(tts, name, None)
    if hasattr(tts.generator, "posterior_encoder"):
        tts.generator.posterior_encoder = None
    if hasattr(tts, "_cache"):
        tts._cache = None


__all__ = ["optimize_espnet_for_inference"]
