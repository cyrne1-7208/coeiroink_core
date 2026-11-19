"""Coreテストで共有する最小のフィクスチャとフェイクモデルを提供する。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

SPEAKER_UUID = "00000000-0000-4000-8000-000000000001"
SPEAKER_UUID_2 = "00000000-0000-4000-8000-000000000002"
SPEAKER_UUID_3 = "00000000-0000-4000-8000-000000000003"
STYLE_ID = 1001


@dataclass(frozen=True, slots=True)
class SpeakerLayout:
    """実モデルを使わずにMYCOEIROINK形式のspeaker_infoディレクトリを組み立てる。"""

    root: Path
    speaker_uuid: str = SPEAKER_UUID
    second_speaker_uuid: str = SPEAKER_UUID_2
    third_speaker_uuid: str = SPEAKER_UUID_3
    style_id: int = STYLE_ID

    def add_speaker(
        self,
        *,
        folder_name: str | None = None,
        speaker_uuid: str = SPEAKER_UUID,
        style_id: int = STYLE_ID,
    ) -> Path:
        speaker_dir = self.root / (folder_name or speaker_uuid)
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

    def add_speakers(self, count: int = 2) -> list[Path]:
        specs = (
            (self.speaker_uuid, self.speaker_uuid),
            ("second-speaker", self.second_speaker_uuid),
            ("third-speaker", self.third_speaker_uuid),
        )
        return [
            self.add_speaker(folder_name=folder, speaker_uuid=uuid)
            for folder, uuid in specs[:count]
        ]


@pytest.fixture
def speaker_layout(tmp_path: Path) -> SpeakerLayout:
    return SpeakerLayout(root=tmp_path / "speaker_info")


@pytest.fixture
def fake_espnet_model_cls() -> type:
    """テスト間で状態を共有しない最小のEspnetModel代替クラスを返す。"""

    class FakeEspnetModel:
        instances: ClassVar[list[FakeEspnetModel]] = []
        track_instances: ClassVar[bool] = True
        resident_bytes: ClassVar[int] = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.model_path = kwargs.get("model_path")
            self.speed_control_alpha = kwargs.get("speed_scale", 1.0)
            self.generator_only = kwargs.get("generator_only", True)
            if type(self).track_instances:
                type(self).instances.append(self)

        def set_speed_control_alpha(self, value: float) -> None:
            self.speed_control_alpha = value

        def tokens2ids(self, tokens: list[str]) -> np.ndarray:
            return np.arange(len(tokens), dtype=np.int64)

        def make_voice(self, _text: np.ndarray) -> np.ndarray:
            return np.ones(32, dtype=np.float32)

    return FakeEspnetModel
