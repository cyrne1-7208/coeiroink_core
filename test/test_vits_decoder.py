# 実験的なVITS decoderの文末の無音の省略・分割実行・継続長の受け渡しのテスト

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from coeirocore import vits_decoder
from coeirocore.vits_decoder import VitsDecoder, install_vits_decoder

SAMPLES_PER_FRAME = 4
# 文末記号より前の8フレームに、省略されない余白を加えたフレーム数
KEPT_FRAMES = 8 + vits_decoder._TRAILING_SILENCE_MARGIN_FRAMES


# 隣接フレームを混ぜてから1フレームを4サンプルへ展開する、小さなdecoderの代替モジュール
class SmallDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mix = torch.nn.Conv1d(2, 2, kernel_size=3, padding=1)
        self.upsample = torch.nn.ConvTranspose1d(
            2, 1, kernel_size=SAMPLES_PER_FRAME, stride=SAMPLES_PER_FRAME
        )
        self.calls = 0

    def forward(self, z: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        self.calls += 1
        return self.upsample(self.mix(z))


def _durations(*frames: float) -> torch.Tensor:
    return torch.tensor([[frames]])


def test_skipped_trailing_silence_is_filled_with_silence() -> None:
    torch.manual_seed(0)
    small = SmallDecoder().eval()
    decoder = VitsDecoder(small, chunked=False)
    z = torch.randn(1, 2, KEPT_FRAMES + 16)

    with torch.no_grad():
        expected = small(z[..., :KEPT_FRAMES])
        with decoder.skipping_trailing_silence():
            decoder.set_durations(_durations(3, 5, KEPT_FRAMES + 8))
            actual = decoder(z)

    assert actual.shape == (1, 1, z.size(-1) * SAMPLES_PER_FRAME)
    assert torch.equal(actual[..., : KEPT_FRAMES * SAMPLES_PER_FRAME], expected)
    assert not actual[..., KEPT_FRAMES * SAMPLES_PER_FRAME :].any()


def test_trailing_silence_is_decoded_when_short_or_outside_the_scope() -> None:
    torch.manual_seed(1)
    small = SmallDecoder().eval()
    decoder = VitsDecoder(small, chunked=False)
    z = torch.randn(1, 2, KEPT_FRAMES + 16)

    with torch.no_grad():
        expected = small(z)
        with decoder.skipping_trailing_silence():
            # 文末記号の長さが余白より短い場合は、省略するフレームがない。
            decoder.set_durations(_durations(z.size(-1) - 8, 8))
            short_tail = decoder(z)
        decoder.set_durations(_durations(3, 5, KEPT_FRAMES + 8))
        outside = decoder(z)

    assert torch.equal(short_tail, expected)
    assert torch.equal(outside, expected)


@pytest.mark.parametrize(
    "frames",
    [vits_decoder._MAX_UNCHUNKED_FRAMES, vits_decoder._MAX_UNCHUNKED_FRAMES + 1],
)
def test_chunked_decoding_splits_only_long_inputs(frames: int) -> None:
    torch.manual_seed(2)
    small = SmallDecoder().eval()
    decoder = VitsDecoder(small, chunked=True)
    z = torch.randn(1, 2, frames)

    with torch.no_grad():
        expected = small(z)
        small.calls = 0
        actual = decoder(z)

    if frames > vits_decoder._MAX_UNCHUNKED_FRAMES:
        assert small.calls == math.ceil(frames / vits_decoder._CHUNK_FRAMES)
        torch.testing.assert_close(actual, expected)
    else:
        assert small.calls == 1
        assert torch.equal(actual, expected)


def test_install_vits_decoder_passes_durations_to_the_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGenerator:
        def __init__(self) -> None:
            self.decoder = SmallDecoder()

        def _generate_path(self, dur: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return mask * 2

    class FakeVits:
        def __init__(self) -> None:
            self.generator = FakeGenerator()

    monkeypatch.setattr(vits_decoder, "_vits_type", lambda: FakeVits)
    text_to_speech = SimpleNamespace(model=SimpleNamespace(tts=FakeVits()))
    generator = text_to_speech.model.tts.generator
    original = generator.decoder

    decoder = install_vits_decoder(text_to_speech, chunked=True)

    assert decoder is generator.decoder
    assert decoder.decoder is original
    assert decoder._chunked
    mask = torch.ones(1)
    with decoder.skipping_trailing_silence():
        path = generator._generate_path(_durations(3, 5, KEPT_FRAMES + 8), mask)
        assert decoder._frame_limit == KEPT_FRAMES
    assert torch.equal(path, mask * 2)
    assert decoder._frame_limit is None
    other_model = SimpleNamespace(model=SimpleNamespace(tts=object()))
    assert install_vits_decoder(other_model) is None
