import json
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from coeirocore.coeiro_manager import (
    AmbiguousStyleError,
    AudioManager,
    EspnetModel,
    InvalidSynthesisParameterError,
    MetaManager,
    ModelLoadError,
    PredictionResult,
    SpeakerInfoError,
    StyleNotFoundError,
)
from coeirocore.devices import resolve_device

SPEAKER_UUID = "00000000-0000-4000-8000-000000000001"
SPEAKER_UUID_2 = "00000000-0000-4000-8000-000000000002"
STYLE_ID = 1001


def create_old_mycoeiroink_fixture(
    speaker_info_dir: Path,
    *,
    folder_name: str = SPEAKER_UUID,
    speaker_uuid: str = SPEAKER_UUID,
    style_id: int = STYLE_ID,
) -> Path:
    speaker_dir = speaker_info_dir / folder_name
    model_dir = speaker_dir / "model" / str(style_id)
    model_dir.mkdir(parents=True)
    (speaker_dir / "metas.json").write_text(
        json.dumps(
            {
                "speakerName": "テスト話者",
                "speakerUuid": speaker_uuid,
                "styles": [{"styleName": "テスト", "styleId": style_id}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (model_dir / "config.yaml").write_text(
        "token_list: ['<blank>', '<unk>', '<sos/eos>']\n"
        "feats_extract_conf:\n"
        "  hop_length: 512\n"
        "tts: vits\n"
        "version: 0.10.3\n",
        encoding="utf-8",
    )
    (model_dir / "model.pth").write_bytes(b"test model placeholder")
    return speaker_dir


def test_discovers_old_mycoeiroink(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    speaker_dir = create_old_mycoeiroink_fixture(speaker_info_dir)

    manager = MetaManager(speaker_info_dir=speaker_info_dir)

    assert manager.get_metas_dict() == [
        {
            "name": "テスト話者",
            "speaker_uuid": SPEAKER_UUID,
            "styles": [{"name": "テスト", "id": STYLE_ID}],
            "version": "0.0.1",
        }
    ]
    model_path = manager.get_model_path(STYLE_ID)
    assert (
        model_path.config_path
        == (speaker_dir / "model" / str(STYLE_ID) / "config.yaml").resolve()
    )
    assert (
        model_path.model_path
        == (speaker_dir / "model" / str(STYLE_ID) / "model.pth").resolve()
    )
    assert model_path.hop_length == 512


def test_rejects_missing_metas(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    (speaker_info_dir / SPEAKER_UUID).mkdir(parents=True)

    with pytest.raises(SpeakerInfoError, match=r"metas\.json is missing"):
        MetaManager(speaker_info_dir=speaker_info_dir)


def test_rejects_missing_model(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    speaker_dir = create_old_mycoeiroink_fixture(speaker_info_dir)
    (speaker_dir / "model" / str(STYLE_ID) / "model.pth").unlink()

    with pytest.raises(SpeakerInfoError, match=r"exactly one \.pth model"):
        MetaManager(speaker_info_dir=speaker_info_dir)


def test_rejects_duplicate_style_id(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    speaker_dir = create_old_mycoeiroink_fixture(speaker_info_dir)
    metas_path = speaker_dir / "metas.json"
    meta = json.loads(metas_path.read_text(encoding="utf-8"))
    meta["styles"].append({"styleName": "重複", "styleId": STYLE_ID})
    metas_path.write_text(
        json.dumps(meta, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(SpeakerInfoError, match="Duplicate styleId"):
        MetaManager(speaker_info_dir=speaker_info_dir)


def test_allows_same_style_id_for_different_speakers(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)
    create_old_mycoeiroink_fixture(
        speaker_info_dir,
        folder_name="second-speaker",
        speaker_uuid=SPEAKER_UUID_2,
    )

    manager = MetaManager(speaker_info_dir=speaker_info_dir)

    assert len(manager.get_metas_dict()) == 2


def test_resolves_same_style_id_by_explicit_speaker_uuid(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    first_speaker_dir = create_old_mycoeiroink_fixture(speaker_info_dir)
    second_speaker_dir = create_old_mycoeiroink_fixture(
        speaker_info_dir,
        folder_name="second-speaker",
        speaker_uuid=SPEAKER_UUID_2,
    )
    manager = MetaManager(speaker_info_dir=speaker_info_dir)

    assert (
        manager.get_model_path(STYLE_ID, speaker_uuid=SPEAKER_UUID).model_path
        == (first_speaker_dir / "model" / str(STYLE_ID) / "model.pth").resolve()
    )
    assert (
        manager.get_model_path(STYLE_ID, speaker_uuid=SPEAKER_UUID_2).model_path
        == (second_speaker_dir / "model" / str(STYLE_ID) / "model.pth").resolve()
    )


def test_rejects_ambiguous_legacy_style_id(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)
    create_old_mycoeiroink_fixture(
        speaker_info_dir,
        folder_name="second-speaker",
        speaker_uuid=SPEAKER_UUID_2,
    )
    manager = MetaManager(speaker_info_dir=speaker_info_dir)

    with pytest.raises(AmbiguousStyleError, match="specify speaker_uuid"):
        manager.get_model_path(STYLE_ID)


def test_audio_manager_cache_key_includes_speaker_uuid(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)
    create_old_mycoeiroink_fixture(
        speaker_info_dir,
        folder_name="second-speaker",
        speaker_uuid=SPEAKER_UUID_2,
    )

    class FakeEspnetModel:
        instances: ClassVar[list] = []

        def __init__(self, *args, **kwargs):
            self.model_path = kwargs["model_path"]
            type(self).instances.append(self)

        def set_speed_control_alpha(self, value):
            pass

        def tokens2ids(self, tokens):
            return np.arange(len(tokens), dtype=np.int64)

        def make_voice(self, text):
            return np.ones(32, dtype=np.float32)

    with (
        patch("coeirocore.coeiro_manager.EspnetModel", FakeEspnetModel),
        patch.object(AudioManager, "trim", side_effect=lambda wave: wave),
    ):
        manager = AudioManager(speaker_info_dir=speaker_info_dir)
        manager.synthesis(["^", "a", "$"], style_id=STYLE_ID, speaker_uuid=SPEAKER_UUID)
        manager.synthesis(
            ["^", "i", "$"], style_id=STYLE_ID, speaker_uuid=SPEAKER_UUID_2
        )
        manager.synthesis(
            ["^", "u", "$"], style_id=STYLE_ID, speaker_uuid=SPEAKER_UUID_2
        )
        assert manager.is_speaker_initialized(STYLE_ID, speaker_uuid=SPEAKER_UUID_2)
        assert not manager.is_speaker_initialized(STYLE_ID, speaker_uuid=SPEAKER_UUID)

    assert len(FakeEspnetModel.instances) == 2
    assert (
        FakeEspnetModel.instances[0].model_path
        != FakeEspnetModel.instances[1].model_path
    )


def test_audio_manager_loads_lazily_and_reuses_model(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)

    class FakeEspnetModel:
        instances: ClassVar[list] = []

        def __init__(self, *args, **kwargs):
            self.speed_control_alpha = kwargs["speed_scale"]
            self.instances.append(self)

        def set_speed_control_alpha(self, value):
            self.speed_control_alpha = value

        def tokens2ids(self, tokens):
            return np.arange(len(tokens), dtype=np.int64)

        def make_voice(self, text):
            return np.ones(32, dtype=np.float32)

    with (
        patch("coeirocore.coeiro_manager.EspnetModel", FakeEspnetModel),
        patch.object(AudioManager, "trim", side_effect=lambda wave: wave),
    ):
        manager = AudioManager(speaker_info_dir=speaker_info_dir)
        assert FakeEspnetModel.instances == []

        manager.synthesis(["^", "a", "$"], style_id=STYLE_ID)
        manager.synthesis(["^", "i", "$"], style_id=STYLE_ID)
        assert len(FakeEspnetModel.instances) == 1

        manager.synthesis(["^", "u", "$"], style_id=STYLE_ID, speed_scale=1.25)
        assert len(FakeEspnetModel.instances) == 1
        assert FakeEspnetModel.instances[0].speed_control_alpha == 0.8
        with pytest.raises(InvalidSynthesisParameterError, match="speed_scale"):
            manager.initialize_speaker(STYLE_ID, speed_scale=0)


def test_audio_manager_loads_and_reuses_all_models_when_requested(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)
    create_old_mycoeiroink_fixture(
        speaker_info_dir,
        folder_name="second-speaker",
        speaker_uuid=SPEAKER_UUID_2,
    )

    class FakeEspnetModel:
        instances: ClassVar[list] = []

        def __init__(self, *args, **kwargs):
            self.model_path = kwargs["model_path"]
            self.instances.append(self)

        def set_speed_control_alpha(self, value):
            pass

        def tokens2ids(self, tokens):
            return np.arange(len(tokens), dtype=np.int64)

        def make_voice(self, text):
            return np.ones(32, dtype=np.float32)

    with (
        patch("coeirocore.coeiro_manager.EspnetModel", FakeEspnetModel),
        patch.object(AudioManager, "trim", side_effect=lambda wave: wave),
    ):
        manager = AudioManager(
            speaker_info_dir=speaker_info_dir,
            load_all_models=True,
        )

        assert len(FakeEspnetModel.instances) == 2
        assert manager.is_speaker_initialized(STYLE_ID, speaker_uuid=SPEAKER_UUID)
        assert manager.is_speaker_initialized(STYLE_ID, speaker_uuid=SPEAKER_UUID_2)

        manager.synthesis(["^", "a", "$"], style_id=STYLE_ID, speaker_uuid=SPEAKER_UUID)
        manager.synthesis(
            ["^", "i", "$"], style_id=STYLE_ID, speaker_uuid=SPEAKER_UUID_2
        )

    assert len(FakeEspnetModel.instances) == 2


def test_audio_manager_releases_model_before_forced_reload(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)

    class FakeEspnetModel:
        def __init__(self, *args, **kwargs):
            self.cycle = self

    with patch("coeirocore.coeiro_manager.EspnetModel", FakeEspnetModel):
        manager = AudioManager(speaker_info_dir=speaker_info_dir)
        manager.initialize_speaker(STYLE_ID)
        previous_model = weakref.ref(manager.current_speaker_model)

        manager.initialize_speaker(STYLE_ID, skip_reinit=False)

    assert previous_model() is None


def test_audio_manager_serializes_concurrent_inference(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)

    class TrackingEspnetModel:
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        def __init__(self, *args, **kwargs):
            pass

        def set_speed_control_alpha(self, value):
            pass

        def tokens2ids(self, tokens):
            return np.arange(len(tokens), dtype=np.int64)

        def make_voice(self, text):
            with self.state_lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            time.sleep(0.02)
            with self.state_lock:
                type(self).active -= 1
            return np.ones(32, dtype=np.float32)

    with (
        patch("coeirocore.coeiro_manager.EspnetModel", TrackingEspnetModel),
        patch.object(AudioManager, "trim", side_effect=lambda wave: wave),
    ):
        manager = AudioManager(speaker_info_dir=speaker_info_dir)
        with ThreadPoolExecutor(max_workers=4) as executor:
            waves = list(
                executor.map(
                    lambda _: manager.synthesis(["^", "a", "$"], style_id=STYLE_ID),
                    range(8),
                )
            )

    assert TrackingEspnetModel.max_active == 1
    assert all(wave.shape == (32,) for wave in waves)


def test_predict_with_duration_returns_untrimmed_wave_and_frames(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)

    class FakeEspnetModel:
        def __init__(self, *args, **kwargs):
            pass

        def tokens2ids(self, tokens):
            return np.arange(len(tokens), dtype=np.int64)

        def make_voice_with_duration(self, text):
            return PredictionResult(
                wav=np.arange(12, dtype=np.float32),
                duration_frames=[2, 3, 7],
            )

    with (
        patch("coeirocore.coeiro_manager.EspnetModel", FakeEspnetModel),
        patch.object(
            AudioManager,
            "trim",
            side_effect=AssertionError("raw prediction must not trim"),
        ),
    ):
        manager = AudioManager(speaker_info_dir=speaker_info_dir)
        result = manager.predict_with_duration(
            ["^", "a", "$"],
            style_id=STYLE_ID,
            speaker_uuid=SPEAKER_UUID,
        )

    assert result.duration_frames == [2, 3, 7]
    assert np.array_equal(result.wav, np.arange(12, dtype=np.float32))


def test_audio_manager_changes_only_internal_pause_when_requested(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)

    class FakeEspnetModel:
        def __init__(self, *args, **kwargs):
            pass

        def tokens2ids(self, tokens):
            return np.arange(len(tokens), dtype=np.int64)

        def make_voice_with_duration(self, text):
            return PredictionResult(
                wav=np.arange(80, dtype=np.float32),
                duration_frames=[1, 2, 2, 2, 1],
            )

    with (
        patch("coeirocore.coeiro_manager.EspnetModel", FakeEspnetModel),
        patch.object(AudioManager, "trim", side_effect=lambda wave: wave),
    ):
        manager = AudioManager(fs=100, speaker_info_dir=speaker_info_dir)
        wave = manager.synthesis(
            ["^", "a", "_", "i", "$"],
            style_id=STYLE_ID,
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


def test_audio_manager_rejects_pause_output_over_limit(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)

    class FakeEspnetModel:
        def __init__(self, *args, **kwargs):
            pass

        def tokens2ids(self, tokens):
            return np.arange(len(tokens), dtype=np.int64)

        def make_voice_with_duration(self, text, duration_frames=None):
            return PredictionResult(
                wav=np.ones(3, dtype=np.float32),
                duration_frames=[1, 1, 1],
            )

    with patch("coeirocore.coeiro_manager.EspnetModel", FakeEspnetModel):
        manager = AudioManager(fs=1, speaker_info_dir=speaker_info_dir)
        with pytest.raises(
            InvalidSynthesisParameterError,
            match="controlled waveform must not exceed",
        ):
            manager.synthesis(
                ["^", "_", "$"],
                style_id=STYLE_ID,
                pause_length=601.0,
                output_sampling_rate=1,
            )


def test_espnet_model_uses_request_speed_without_duration_materialization():
    class FakeText2Speech:
        def __init__(self):
            self.calls = []

        def __call__(self, text, decode_conf):
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


def test_espnet_model_passes_runtime_device_as_string(tmp_path: Path):
    captured = {}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("token_list: ['<unk>']\n", encoding="utf-8")

    class FakeText2Speech:
        def __init__(self, *args, device, **kwargs):
            captured["device"] = device
            self.decode_conf = {"alpha": 1.0}

    class FakeTokenIDConverter:
        def __init__(self, *args, **kwargs):
            pass

    with (
        patch(
            "coeirocore.coeiro_manager._load_text_to_speech",
            return_value=FakeText2Speech,
        ),
        patch(
            "coeirocore.coeiro_manager._load_token_id_converter",
            return_value=FakeTokenIDConverter,
        ),
    ):
        EspnetModel(
            config_path=config_path,
            model_path=tmp_path / "model.pth",
            device=resolve_device("cpu"),
        )

    assert captured["device"] == "cpu"


def test_pitch_intonation_preserves_unvoiced_f0_frames():
    captured = {}

    def fake_synthesize(f0, spectral_envelope, aperiodicity, sampling_rate):
        captured["f0"] = np.asarray(f0).copy()
        return np.zeros(32, dtype=np.float64)

    fake_world = MagicMock()
    fake_world.synthesize.side_effect = fake_synthesize

    with (
        patch.object(
            AudioManager,
            "get_world",
            return_value=(np.array([0.0, 100.0, 200.0, 0.0]), None, None),
        ),
        patch("coeirocore.coeiro_manager.load_pyworld", return_value=fake_world),
    ):
        result = AudioManager.pitch_intonation(
            np.zeros(32, dtype=np.float32),
            16000,
            pitch_scale=0.0,
            intonation_scale=0.5,
        )

    np.testing.assert_array_equal(captured["f0"], [0.0, 125.0, 175.0, 0.0])
    assert result.dtype == np.float32


def test_predict_uses_composite_model_identity(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)
    create_old_mycoeiroink_fixture(
        speaker_info_dir,
        folder_name="second-speaker",
        speaker_uuid=SPEAKER_UUID_2,
    )

    class FakeEspnetModel:
        def __init__(self, *args, **kwargs):
            self.model_path = kwargs["model_path"]

        def tokens2ids(self, tokens):
            return np.arange(len(tokens), dtype=np.int64)

        def make_voice(self, text):
            value = 1 if SPEAKER_UUID in str(self.model_path) else 2
            return np.full(4, value, dtype=np.float32)

    with patch("coeirocore.coeiro_manager.EspnetModel", FakeEspnetModel):
        manager = AudioManager(speaker_info_dir=speaker_info_dir)
        first = manager.predict(["^", "a", "$"], STYLE_ID, speaker_uuid=SPEAKER_UUID)
        second = manager.predict(["^", "a", "$"], STYLE_ID, speaker_uuid=SPEAKER_UUID_2)

    assert np.array_equal(first, np.ones(4, dtype=np.float32))
    assert np.array_equal(second, np.full(4, 2, dtype=np.float32))


def test_reports_unknown_style(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)
    manager = AudioManager(speaker_info_dir=speaker_info_dir)

    with pytest.raises(StyleNotFoundError, match="not installed"):
        manager.synthesis(["^", "a", "$"], style_id=0)


def test_wraps_corrupt_model_error(tmp_path: Path):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)

    with patch(
        "coeirocore.coeiro_manager.EspnetModel",
        side_effect=RuntimeError("invalid checkpoint"),
    ):
        manager = AudioManager(speaker_info_dir=speaker_info_dir)
        with pytest.raises(ModelLoadError, match=str(STYLE_ID)):
            manager.synthesis(["^", "a", "$"], style_id=STYLE_ID)


@pytest.mark.parametrize(
    "parameter, value",
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
    tmp_path: Path, parameter: str, value: float
):
    speaker_info_dir = tmp_path / "speaker_info"
    create_old_mycoeiroink_fixture(speaker_info_dir)
    manager = AudioManager(speaker_info_dir=speaker_info_dir)

    with pytest.raises(InvalidSynthesisParameterError):
        manager.synthesis(["^", "a", "$"], style_id=STYLE_ID, **{parameter: value})
