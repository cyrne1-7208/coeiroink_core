# AudioManagerのモデル遅延読み込み・LRU管理・メモリガード・排他制御のテスト

from __future__ import annotations

import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from coeirocore import coeiro_manager, model_memory
from coeirocore.coeiro_manager import (
    AudioManager,
    InvalidSynthesisParameterError,
    ModelLoadError,
)
from coeirocore.devices import DeviceBackend, DeviceSelection

if TYPE_CHECKING:
    from conftest import SpeakerLayout

MIB = 1024 * 1024


def _cyclic_model_cls(
    base_cls: type,
    refs: list[weakref.ReferenceType] | None = None,
) -> type:
    class CyclicModel(base_cls):
        track_instances: ClassVar[bool] = False

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.cycle = self
            if refs is not None:
                refs.append(weakref.ref(self))

    return CyclicModel


def test_audio_manager_cache_key_includes_speaker_uuid(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speakers(2)
    monkeypatch.setattr(coeiro_manager, "EspnetModel", fake_espnet_model_cls)
    monkeypatch.setattr(AudioManager, "trim", staticmethod(lambda wave: wave))

    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    manager.synthesis(
        ["^", "a", "$"],
        style_id=speaker_layout.style_id,
        speaker_uuid=speaker_layout.speaker_uuid,
    )
    manager.synthesis(
        ["^", "i", "$"],
        style_id=speaker_layout.style_id,
        speaker_uuid=speaker_layout.second_speaker_uuid,
    )
    manager.synthesis(
        ["^", "u", "$"],
        style_id=speaker_layout.style_id,
        speaker_uuid=speaker_layout.second_speaker_uuid,
    )

    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
    )
    assert not manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    assert len(fake_espnet_model_cls.instances) == 2
    assert (
        fake_espnet_model_cls.instances[0].model_path
        != fake_espnet_model_cls.instances[1].model_path
    )


def test_audio_manager_loads_lazily_and_reuses_model(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()
    monkeypatch.setattr(coeiro_manager, "EspnetModel", fake_espnet_model_cls)
    monkeypatch.setattr(AudioManager, "trim", staticmethod(lambda wave: wave))

    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    assert fake_espnet_model_cls.instances == []

    manager.synthesis(["^", "a", "$"], style_id=speaker_layout.style_id)
    manager.synthesis(["^", "i", "$"], style_id=speaker_layout.style_id)
    assert len(fake_espnet_model_cls.instances) == 1
    assert fake_espnet_model_cls.instances[0].generator_only is True

    manager.synthesis(
        ["^", "u", "$"], style_id=speaker_layout.style_id, speed_scale=1.25
    )
    assert len(fake_espnet_model_cls.instances) == 1
    assert fake_espnet_model_cls.instances[0].speed_control_alpha == 0.8
    with pytest.raises(InvalidSynthesisParameterError, match="speed_scale"):
        manager.initialize_speaker(speaker_layout.style_id, speed_scale=0)


def test_audio_manager_loads_and_reuses_all_models_when_requested(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speakers(2)
    monkeypatch.setattr(coeiro_manager, "EspnetModel", fake_espnet_model_cls)
    monkeypatch.setattr(AudioManager, "trim", staticmethod(lambda wave: wave))

    manager = AudioManager(
        speaker_info_dir=speaker_layout.root,
        max_loaded_models=None,
    )

    assert len(fake_espnet_model_cls.instances) == 2
    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
    )

    manager.synthesis(
        ["^", "a", "$"],
        style_id=speaker_layout.style_id,
        speaker_uuid=speaker_layout.speaker_uuid,
    )
    manager.synthesis(
        ["^", "i", "$"],
        style_id=speaker_layout.style_id,
        speaker_uuid=speaker_layout.second_speaker_uuid,
    )
    assert len(fake_espnet_model_cls.instances) == 2


def test_audio_manager_keeps_most_recently_used_models(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speakers(3)
    monkeypatch.setattr(coeiro_manager, "EspnetModel", fake_espnet_model_cls)

    manager = AudioManager(
        speaker_info_dir=speaker_layout.root,
        max_loaded_models=2,
    )
    manager.initialize_speaker(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    manager.initialize_speaker(
        speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
    )
    # 1番目を再利用して末尾へ移し、2番目を次の追い出し対象にする。
    manager.initialize_speaker(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    manager.initialize_speaker(
        speaker_layout.style_id, speaker_uuid=speaker_layout.third_speaker_uuid
    )

    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    assert not manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
    )
    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.third_speaker_uuid
    )


@pytest.mark.parametrize("max_loaded_models", [2, None])
def test_audio_manager_evicts_lru_model_under_memory_pressure(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    max_loaded_models: int | None,
) -> None:
    speaker_layout.add_speakers(2)

    class PressureModel(fake_espnet_model_cls):
        track_instances: ClassVar[bool] = False
        resident_bytes: ClassVar[int] = 256 * MIB
        resident_count: ClassVar[int] = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            type(self).resident_count += 1

        def __del__(self) -> None:
            type(self).resident_count -= 1

    monkeypatch.setattr(coeiro_manager, "EspnetModel", PressureModel)
    monkeypatch.setattr(
        model_memory,
        "_host_memory",
        lambda: (
            2048 * MIB,
            (512 - 256 * PressureModel.resident_count) * MIB,
        ),
    )

    manager = AudioManager(
        speaker_info_dir=speaker_layout.root,
        max_loaded_models=max_loaded_models,
    )
    if max_loaded_models is not None:
        manager.initialize_speaker(
            speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
        )
        manager.initialize_speaker(
            speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
        )

    assert not manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
    )
    if max_loaded_models is None:
        assert "Only 1 of 2 MYCOEIROINK models remain loaded" in caplog.text


@pytest.mark.parametrize("max_loaded_models", [1, None])
def test_audio_manager_checks_largest_model_at_startup_and_actual_model_on_switch(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    max_loaded_models: int | None,
) -> None:
    _, second = speaker_layout.add_speakers(2)
    with (second / "model" / str(speaker_layout.style_id) / "model.pth").open(
        "wb"
    ) as file:
        file.truncate(64 * MIB)

    host_memory = MagicMock(return_value=(2048 * MIB, 320 * MIB))
    monkeypatch.setattr(coeiro_manager, "EspnetModel", fake_espnet_model_cls)
    monkeypatch.setattr(model_memory, "_host_memory", host_memory)

    manager = AudioManager(
        speaker_info_dir=speaker_layout.root,
        max_loaded_models=max_loaded_models,
        preload_models=False,
    )
    host_memory.assert_not_called()
    assert len(fake_espnet_model_cls.instances) == 0

    with pytest.raises(ModelLoadError, match="host memory"):
        manager.prepare_models()
    assert len(fake_espnet_model_cls.instances) == 0

    host_memory.return_value = (2048 * MIB, 512 * MIB)
    manager.prepare_models()
    expected = 2 if max_loaded_models is None else 0
    assert len(fake_espnet_model_cls.instances) == expected
    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
    ) is (max_loaded_models is None)

    # 起動後は最大モデル分の空きがなくても、小さい対象モデルを読み込める。
    host_memory.return_value = (2048 * MIB, 320 * MIB)
    manager.initialize_speaker(
        speaker_layout.style_id,
        speaker_uuid=speaker_layout.speaker_uuid,
        skip_reinit=False,
    )
    assert len(fake_espnet_model_cls.instances) == expected + 1
    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )


@pytest.mark.parametrize("dangerous_skip_check", [False, True])
def test_all_model_startup_does_not_hide_a_failed_checkpoint(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    dangerous_skip_check: bool,
) -> None:
    speaker_layout.add_speakers(2)
    failure = RuntimeError("invalid checkpoint")

    class SecondFailsModel(fake_espnet_model_cls):
        constructions: ClassVar[int] = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            type(self).constructions += 1
            if type(self).constructions == 2:
                raise failure
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(coeiro_manager, "EspnetModel", SecondFailsModel)

    manager = AudioManager(
        speaker_info_dir=speaker_layout.root,
        max_loaded_models=None,
        preload_models=False,
        dangerous_skip_check=dangerous_skip_check,
    )
    with pytest.raises(ModelLoadError) as caught:
        manager.prepare_models()

    assert caught.value.__cause__ is failure
    assert not manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    assert not manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
    )


@pytest.mark.parametrize("max_loaded_models", [1, None])
def test_dangerous_override_warns_but_keeps_the_model_count_limit(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    max_loaded_models: int | None,
) -> None:
    speaker_layout.add_speakers(2)
    host_memory = MagicMock(return_value=(2048 * MIB, 1))
    monkeypatch.setattr(coeiro_manager, "EspnetModel", fake_espnet_model_cls)
    monkeypatch.setattr(model_memory, "_host_memory", host_memory)

    manager = AudioManager(
        speaker_info_dir=speaker_layout.root,
        max_loaded_models=max_loaded_models,
        dangerous_skip_check=True,
    )
    manager.initialize_speaker(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    manager.initialize_speaker(
        speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
    )

    host_memory.assert_called()
    assert "host memory" in caplog.text
    assert "dangerous_skip_check=True" in caplog.text
    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
    )
    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    ) is (max_loaded_models is None)


@pytest.mark.parametrize("max_loaded_models", [1, 2])
def test_audio_manager_keeps_loaded_model_when_new_model_cannot_fit(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    max_loaded_models: int,
) -> None:
    speaker_layout.add_speakers(2)

    class ResidentModel(fake_espnet_model_cls):
        resident_bytes: ClassVar[int] = 16 * MIB

    host_memory = MagicMock(return_value=(2048 * MIB, 1024 * MIB))
    monkeypatch.setattr(coeiro_manager, "EspnetModel", ResidentModel)
    monkeypatch.setattr(model_memory, "_host_memory", host_memory)

    manager = AudioManager(
        speaker_info_dir=speaker_layout.root,
        max_loaded_models=max_loaded_models,
    )
    before = manager.synthesis(
        ["^", "a", "$"],
        style_id=speaker_layout.style_id,
        speaker_uuid=speaker_layout.speaker_uuid,
    )
    host_memory.return_value = (2048 * MIB, 128 * MIB)

    with pytest.raises(ModelLoadError, match="host memory"):
        manager.initialize_speaker(
            speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
        )

    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    after = manager.synthesis(
        ["^", "a", "$"],
        style_id=speaker_layout.style_id,
        speaker_uuid=speaker_layout.speaker_uuid,
    )
    np.testing.assert_array_equal(before, after)
    assert len(ResidentModel.instances) == 1


@pytest.mark.parametrize("switch_speaker", [False, True])
def test_audio_manager_releases_model_before_forced_reload(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    switch_speaker: bool,
) -> None:
    speaker_layout.add_speakers(2 if switch_speaker else 1)
    monkeypatch.setattr(
        coeiro_manager, "EspnetModel", _cyclic_model_cls(fake_espnet_model_cls)
    )

    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    manager.initialize_speaker(
        speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
    )
    previous_model = weakref.ref(manager.current_speaker_model)
    target_uuid = (
        speaker_layout.second_speaker_uuid
        if switch_speaker
        else speaker_layout.speaker_uuid
    )

    manager.initialize_speaker(
        speaker_layout.style_id,
        speaker_uuid=target_uuid,
        skip_reinit=False,
    )

    assert previous_model() is None
    assert len(manager._loaded_models) == 1
    assert manager.is_speaker_initialized(
        speaker_layout.style_id, speaker_uuid=target_uuid
    )


def test_audio_manager_clears_opencl_cache_after_model_eviction(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()
    opencl = MagicMock()
    monkeypatch.setattr(coeiro_manager, "EspnetModel", fake_espnet_model_cls)
    monkeypatch.setattr(
        coeiro_manager,
        "resolve_device",
        lambda *args, **kwargs: DeviceSelection(
            backend=DeviceBackend.OPENCL,
            device_index=0,
            runtime_device="ocl:0",
            platform="linux",
            platform_index=0,
        ),
    )
    monkeypatch.setattr(
        coeiro_manager, "model_load_memory_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(torch, "ocl", opencl, raising=False)

    manager = AudioManager(device="opencl", speaker_info_dir=speaker_layout.root)
    manager.initialize_speaker(speaker_layout.style_id)
    manager._discard_all_models()

    opencl.empty_cache.assert_called_once_with()


def test_synthesis_releases_model_reference_before_postprocessing(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()
    model_refs: list[weakref.ReferenceType] = []
    manager: AudioManager

    def release_model(wave: np.ndarray) -> np.ndarray:
        manager._discard_all_models()
        assert model_refs[0]() is None
        return wave

    monkeypatch.setattr(
        coeiro_manager,
        "EspnetModel",
        _cyclic_model_cls(fake_espnet_model_cls, refs=model_refs),
    )
    monkeypatch.setattr(AudioManager, "trim", staticmethod(release_model))

    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    result = manager.synthesis(["^", "a", "$"], style_id=speaker_layout.style_id)

    assert result.shape == (32,)


def test_audio_manager_serializes_concurrent_inference(
    speaker_layout: SpeakerLayout,
    fake_espnet_model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_layout.add_speaker()

    class TrackingModel(fake_espnet_model_cls):
        active: ClassVar[int] = 0
        max_active: ClassVar[int] = 0
        state_lock: ClassVar[threading.Lock] = threading.Lock()

        def make_voice(self, _text: np.ndarray) -> np.ndarray:
            with self.state_lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            time.sleep(0.02)
            with self.state_lock:
                type(self).active -= 1
            return np.ones(32, dtype=np.float32)

    monkeypatch.setattr(coeiro_manager, "EspnetModel", TrackingModel)
    monkeypatch.setattr(AudioManager, "trim", staticmethod(lambda wave: wave))

    manager = AudioManager(speaker_info_dir=speaker_layout.root)
    with ThreadPoolExecutor(max_workers=4) as executor:
        waves = list(
            executor.map(
                lambda _: manager.synthesis(
                    ["^", "a", "$"], style_id=speaker_layout.style_id
                ),
                range(8),
            )
        )

    assert TrackingModel.max_active == 1
    assert all(wave.shape == (32,) for wave in waves)
