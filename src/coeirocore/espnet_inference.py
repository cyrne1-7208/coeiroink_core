"""ESPnetの公開推論オブジェクトへ、出力互換な軽量化だけを適用する。"""

from functools import lru_cache
from typing import Any

import torch


class _VitsPathDurationCalculator(torch.nn.Module):
    """VITSのone-hotアライメントを列方向へ集計し、Pythonループを除く。"""

    @torch.no_grad()
    def forward(self, attention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if attention.ndim != 2:
            raise ValueError("VITS attention must be a 2D tensor")
        durations = attention.sum(dim=0).to(dtype=torch.long)
        focus_rate = attention.max(dim=-1).values.mean()
        return durations, focus_rate


@lru_cache(maxsize=1)
def _vits_type() -> type:
    """モデル読込時までESPnetのVITS実装をimportしない。"""

    from espnet2.gan_tts.vits.vits import VITS

    return VITS


def _remove_legacy_weight_norm(module: torch.nn.Module) -> None:
    """推論中に毎回再計算される旧weight normを、現在の等価な重みへ固定する。"""

    for child in module.modules():
        if hasattr(child, "weight_g") and hasattr(child, "weight_v"):
            torch.nn.utils.remove_weight_norm(child)


def optimize_espnet_for_inference(text_to_speech: Any) -> None:
    """対応するVITSだけを最適化し、将来の別方式モデルはESPnet既定動作のままにする。"""

    model = getattr(text_to_speech, "model", None)
    tts = getattr(model, "tts", None)
    if tts is None or not isinstance(tts, _vits_type()):
        return

    text_to_speech.duration_calculator = _VitsPathDurationCalculator().eval()
    # VITS generatorは推論専用としてロードされるため、weight normの再パラメータ化を保持する必要がない。
    with torch.no_grad():
        _remove_legacy_weight_norm(tts.generator)


__all__ = ["optimize_espnet_for_inference"]
