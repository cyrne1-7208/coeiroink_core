# AudioManagerの推論・ポーズ調整・音声補正連携・パラメータ検証のテスト

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from coeirocore import coeiro_manager
from coeirocore.coeiro_manager import (
    AudioManager,
    InvalidSynthesisParameterError,
    ModelLoadError,
    PredictionResult,
    StyleNotFoundError,
    SynthesisError,
)

if TYPE_CHECKING:
    from conftest import SpeakerLayout


def test_predict_with_duration_returns_untrimmed_wave_and_frames(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()

    class DurationModel(fake_espnet_model_cls):
        def make_voice_with_duration(self, _text: np.ndarray) -> PredictionResult:
            return PredictionResult(
                wav=np.arange(12, dtype=np.float32),
                duration_frames=[2, 3, 7],
            )

    monkeypatch.setattr(coeiro_manager, "EspnetModel", DurationModel)
    monkeypatch.setattr(
        AudioManager,
        "trim",
        staticmethod(
            lambda _wave: pytest.fail("raw prediction must not trim"),
        ),
    )

    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    result = manager.predict_with_duration(
        ["^", "a", "$"],
        style_id=speaker_layout.style_id,
        speaker_uuid=speaker_layout.speaker_uuid,
    )

    assert result.duration_frames == [2, 3, 7]
    assert np.array_equal(result.wav, np.arange(12, dtype=np.float32))


@pytest.mark.parametrize("text", [["^", "a", "_", "i", "$"], "あ、い"])
def test_audio_manager_changes_only_internal_pause_when_requested(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    text: list[str] | str,
) -> None:
    speaker_layout.add_speaker()

    class PauseModel(fake_espnet_model_cls):
        def encode_text_with_tokens(self, _text: str) -> tuple[np.ndarray, list[str]]:
            tokens = ["^", "a", "_", "i", "$"]
            return self.tokens2ids(tokens), tokens

        def make_voice_with_duration(self, _text: np.ndarray) -> PredictionResult:
            return PredictionResult(
                wav=np.arange(80, dtype=np.float32),
                duration_frames=[1, 2, 2, 2, 1],
            )

    monkeypatch.setattr(coeiro_manager, "EspnetModel", PauseModel)
    monkeypatch.setattr(AudioManager, "trim", staticmethod(lambda wave: wave))

    manager = AudioManager(fs=100, speaker_info_dir=speaker_layout.root)
    wave = manager.synthesis(
        text,
        style_id=speaker_layout.style_id,
        pause_length=0.1,
        output_sampling_rate=100,
    )

    expected = np.concatenate(
        (
            np.arange(30, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
            np.arange(50, 80, dtype=np.float32),
        )
    )
    assert np.array_equal(wave, expected)


def test_audio_manager_rejects_pause_output_over_limit(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()

    class PauseLimitModel(fake_espnet_model_cls):
        def make_voice_with_duration(
            self, _text: np.ndarray, duration_frames: object = None
        ) -> PredictionResult:
            return PredictionResult(
                wav=np.ones(3, dtype=np.float32),
                duration_frames=[1, 1, 1],
            )

    monkeypatch.setattr(coeiro_manager, "EspnetModel", PauseLimitModel)

    manager = AudioManager(fs=1, speaker_info_dir=speaker_layout.root)
    with pytest.raises(
        InvalidSynthesisParameterError,
        match="controlled waveform must not exceed",
    ):
        manager.synthesis(
            ["^", "_", "$"],
            style_id=speaker_layout.style_id,
            pause_length=601.0,
            output_sampling_rate=1,
        )


@pytest.mark.parametrize("text", [["^", "a", "$"], "あ"])
def test_voice_smoothing_uses_one_prediction_and_leaves_raw_apis_unchanged(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    text: list[str] | str,
) -> None:
    speaker_layout.add_speaker()
    raw = np.full(3 * 512, 0.1, dtype=np.float32)
    events: list[str] = []

    class RecordingModel(fake_espnet_model_cls):
        def encode_text_with_tokens(self, _text: str) -> tuple[np.ndarray, list[str]]:
            return np.arange(3, dtype=np.int64), ["^", "a", "$"]

        def make_voice(self, _text: np.ndarray) -> np.ndarray:
            events.append("wave")
            return raw

        def make_voice_with_duration(self, _text: np.ndarray) -> PredictionResult:
            events.append("duration")
            return PredictionResult(raw, [1, 1, 1])

    def smooth(
        wave: np.ndarray,
        tokens: list[str],
        duration_frames: list[int],
        *,
        sampling_rate: int,
        hop_length: int,
    ) -> np.ndarray:
        assert tokens == ["^", "a", "$"]
        assert duration_frames == [1, 1, 1]
        assert sampling_rate == 44100
        assert hop_length == 512
        events.append("smooth")
        return wave * 2

    module = ModuleType("coeirocore.voice_smoothing")
    module.smooth_voice = smooth
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(coeiro_manager, "EspnetModel", RecordingModel)
    monkeypatch.setattr(AudioManager, "trim", staticmethod(lambda wave: wave))
    monkeypatch.setattr(
        coeiro_manager,
        "_replace_internal_pauses",
        lambda *args, **kwargs: pytest.fail("pause was not requested"),
    )

    manager = AudioManager(speaker_info_dir=speaker_layout.root, voice_smoothing=True)
    assert manager.voice_smoothing
    np.testing.assert_array_equal(
        manager.synthesis(text, speaker_layout.style_id, volume_scale=0.5), raw
    )
    assert events == ["duration", "smooth"]

    events.clear()
    assert manager.predict(text, speaker_layout.style_id) is raw
    assert manager.predict_with_duration(text, speaker_layout.style_id).wav is raw
    assert events == ["wave", "duration"]


def test_disabled_voice_smoothing_needs_no_optional_module(
    speaker_layout: SpeakerLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()
    monkeypatch.setitem(sys.modules, "coeirocore.voice_smoothing", None)

    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    assert not manager.voice_smoothing
    wave = np.ones(8, dtype=np.float32)
    assert manager.smooth_voice(wave, [], [], hop_length=512) is wave


def test_voice_smoothing_reports_the_original_failure(
    speaker_layout: SpeakerLayout,
) -> None:
    speaker_layout.add_speaker()
    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    failure = RuntimeError("pitch analysis failed")
    manager._voice_smoother = MagicMock(side_effect=failure)

    with pytest.raises(SynthesisError) as error:
        manager.smooth_voice(np.ones(512, dtype=np.float32), ["a"], [1], hop_length=512)

    assert error.value.__cause__ is failure


def test_predict_uses_composite_model_identity(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speakers(2)

    class IdentityModel(fake_espnet_model_cls):
        def make_voice(self, _text: np.ndarray) -> np.ndarray:
            value = 1 if speaker_layout.speaker_uuid in str(self.model_path) else 2
            return np.full(4, value, dtype=np.float32)

    monkeypatch.setattr(coeiro_manager, "EspnetModel", IdentityModel)

    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    first = manager.predict(
        ["^", "a", "$"],
        speaker_layout.style_id,
        speaker_uuid=speaker_layout.speaker_uuid,
    )
    second = manager.predict(
        ["^", "a", "$"],
        speaker_layout.style_id,
        speaker_uuid=speaker_layout.second_speaker_uuid,
    )

    assert np.array_equal(first, np.ones(4, dtype=np.float32))
    assert np.array_equal(second, np.full(4, 2, dtype=np.float32))


def test_reports_unknown_style(speaker_layout: SpeakerLayout) -> None:
    speaker_layout.add_speaker()
    manager = AudioManager(speaker_info_dir=speaker_layout.root)

    with pytest.raises(StyleNotFoundError, match="not installed"):
        manager.synthesis(["^", "a", "$"], style_id=0)


def test_wraps_corrupt_model_error(
    speaker_layout: SpeakerLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()
    monkeypatch.setattr(
        coeiro_manager,
        "EspnetModel",
        MagicMock(side_effect=RuntimeError("invalid checkpoint")),
    )

    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    with pytest.raises(ModelLoadError, match=str(speaker_layout.style_id)):
        manager.synthesis(["^", "a", "$"], style_id=speaker_layout.style_id)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("speed_scale", 0),
        ("volume_scale", -1),
        ("intonation_scale", float("inf")),
        ("pre_phoneme_length", -0.1),
        ("post_phoneme_length", float("nan")),
        ("output_sampling_rate", 0),
    ],
)
def test_rejects_invalid_synthesis_parameters(
    speaker_layout: SpeakerLayout,
    parameter: str,
    value: float,
) -> None:
    speaker_layout.add_speaker()
    manager = AudioManager(speaker_info_dir=speaker_layout.root)

    with pytest.raises(InvalidSynthesisParameterError):
        manager.synthesis(
            ["^", "a", "$"],
            style_id=speaker_layout.style_id,
            **{parameter: value},
        )


def test_trailing_silence_is_skipped_only_for_trimming_callers(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()
    ids = np.arange(3, dtype=np.int64)
    calls: list[tuple[str, bool]] = []
    scope_inputs: list[object] = []

    class ScopeModel(fake_espnet_model_cls):
        skipping = False

        def encode_text_with_tokens(self, _text: str) -> tuple[np.ndarray, list[str]]:
            return ids, ["^", "a", "$"]

        @contextmanager
        def skipping_trailing_silence(self, text: np.ndarray) -> Iterator[None]:
            scope_inputs.append(text)
            self.skipping = True
            try:
                yield
            finally:
                self.skipping = False

        def make_voice(self, _text: np.ndarray) -> np.ndarray:
            calls.append(("wave", self.skipping))
            return np.ones(4, dtype=np.float32)

        def make_voice_with_duration(self, _text: np.ndarray) -> PredictionResult:
            calls.append(("duration", self.skipping))
            return PredictionResult(np.ones(4, dtype=np.float32), [4])

    monkeypatch.setattr(coeiro_manager, "EspnetModel", ScopeModel)
    monkeypatch.setattr(AudioManager, "trim", staticmethod(lambda wave: wave))

    manager = AudioManager(
        speaker_info_dir=speaker_layout.root, skip_trailing_silence=True
    )
    tokens = ["^", "a", "$"]
    # 文字列は文末記号を判定できるよう、推論前にモデルの前処理でID列へ変換する。
    manager.synthesis("あ", speaker_layout.style_id)
    manager.synthesis(tokens, speaker_layout.style_id)
    # 未トリムの波形を返すAPIは、呼び出し元が指定した場合だけ文末の無音を省く。
    manager.predict(tokens, speaker_layout.style_id)
    manager.predict(tokens, speaker_layout.style_id, skip_trailing_silence=True)
    manager.predict_with_duration(tokens, speaker_layout.style_id)
    manager.predict_with_duration(
        tokens, speaker_layout.style_id, skip_trailing_silence=True
    )

    assert manager.skip_trailing_silence
    assert scope_inputs[0] is ids
    assert calls == [
        ("wave", True),
        ("wave", True),
        ("wave", False),
        ("wave", True),
        ("duration", False),
        ("duration", True),
    ]


def test_chunked_decoding_is_passed_to_model_loading(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()
    captured: list[object] = []

    class LoadingModel(fake_espnet_model_cls):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.append(kwargs["chunked_decoding"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(coeiro_manager, "EspnetModel", LoadingModel)

    for chunked_decoding in (False, True):
        manager = AudioManager(
            speaker_info_dir=speaker_layout.root, chunked_decoding=chunked_decoding
        )
        manager.initialize_speaker(speaker_layout.style_id)

    assert captured == [False, True]


@pytest.mark.parametrize("option", ["skip_trailing_silence", "chunked_decoding"])
def test_rejects_non_boolean_experimental_options(
    speaker_layout: SpeakerLayout,
    option: str,
) -> None:
    speaker_layout.add_speaker()

    with pytest.raises(TypeError, match=option):
        AudioManager(speaker_info_dir=speaker_layout.root, **{option: 1})
