# 実験的な高速化として、VITSのdecoderに通す範囲と実行の分割を切り替える処理

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from .espnet_inference import _vits_type

# 文末の無音を省く場合も、文末記号の開始からこのフレーム数まではdecoderに通し、トリムで残る範囲を変えない。
_TRAILING_SILENCE_MARGIN_FRAMES = 32
# 分割して処理する1回分のフレーム数と、分割位置の前後に重ねて処理するフレーム数
_CHUNK_FRAMES = 200
_CHUNK_OVERLAP_FRAMES = 32
# 短い入力では重なりの再計算が分割の利点を上回るため、このフレーム数までは分割しない。
_MAX_UNCHUNKED_FRAMES = 800


# VITS generatorのdecoderを包み、文末の無音の省略と分割実行を切り替えるモジュール
class VitsDecoder(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, *, chunked: bool) -> None:
        super().__init__()
        self.decoder = decoder
        self._chunked = chunked
        self._skip_trailing_silence = False
        self._frame_limit: int | None = None

    # 文末の無音をdecoderに通さない推論の範囲を返す。
    @contextmanager
    def skipping_trailing_silence(self) -> Iterator[None]:
        self._skip_trailing_silence = True
        try:
            yield
        finally:
            self._skip_trailing_silence = False
            self._frame_limit = None

    # ESPnetが音響フレームへの展開に使う継続長から、decoderに通すフレーム数を決める。
    def set_durations(self, durations: torch.Tensor) -> None:
        self._frame_limit = None
        if not self._skip_trailing_silence:
            return
        frames = durations.view(-1)
        total_frames = int(frames.sum())
        limit = total_frames - int(frames[-1]) + _TRAILING_SILENCE_MARGIN_FRAMES
        if limit < total_frames:
            self._frame_limit = limit

    def forward(self, z: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        total_frames = z.size(-1)
        if self._frame_limit is not None:
            z = z[..., : self._frame_limit]
        frames = z.size(-1)
        if self._chunked and frames > _MAX_UNCHUNKED_FRAMES:
            wave = self._decode_chunks(z, g)
        else:
            wave = self.decoder(z, g=g)
        if frames == total_frames:
            return wave
        # 省いた文末を無音で埋め、波形の長さとトークンの継続長の対応を保つ。
        samples_per_frame = wave.size(-1) // frames
        silence = wave.new_zeros(
            (*wave.shape[:-1], (total_frames - frames) * samples_per_frame)
        )
        return torch.cat((wave, silence), dim=-1)

    # 分割位置の前後を重ねて処理し、重なりを除いた範囲だけをつなげる。
    def _decode_chunks(self, z: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor:
        frames = z.size(-1)
        chunks: list[torch.Tensor] = []
        for start in range(0, frames, _CHUNK_FRAMES):
            stop = min(start + _CHUNK_FRAMES, frames)
            low = max(0, start - _CHUNK_OVERLAP_FRAMES)
            high = min(frames, stop + _CHUNK_OVERLAP_FRAMES)
            wave = self.decoder(z[..., low:high], g=g)
            samples_per_frame = wave.size(-1) // (high - low)
            begin = (start - low) * samples_per_frame
            chunks.append(wave[..., begin : begin + (stop - start) * samples_per_frame])
        return torch.cat(chunks, dim=-1)


# VITS generatorのdecoderを置き換え、ESPnetが求めた継続長を受け取れるようにする。
#
# VITS以外のモデルでは置き換えず、Noneを返す。
def install_vits_decoder(
    text_to_speech: Any, *, chunked: bool = False
) -> VitsDecoder | None:
    model = getattr(text_to_speech, "model", None)
    tts = getattr(model, "tts", None)
    if tts is None or not isinstance(tts, _vits_type()):
        return None
    generator = tts.generator
    decoder = VitsDecoder(generator.decoder, chunked=chunked)
    generator.decoder = decoder
    generate_path = generator._generate_path

    # 継続長からアライメントを作るESPnetの内部メソッドを包み、推論ごとの継続長をdecoderへ渡す。
    def generate_path_with_durations(
        dur: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        decoder.set_durations(dur)
        return generate_path(dur, mask)

    generator._generate_path = generate_path_with_durations
    return decoder


__all__ = ["VitsDecoder", "install_vits_decoder"]
