# MetaManagerの話者探索・検証・重複styleId解決のテスト

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from coeirocore.coeiro_manager import (
    AmbiguousStyleError,
    MetaManager,
    SpeakerInfoError,
)

if TYPE_CHECKING:
    from conftest import SpeakerLayout


def test_discovers_old_mycoeiroink(speaker_layout: SpeakerLayout) -> None:
    speaker_dir = speaker_layout.add_speaker()

    manager = MetaManager(speaker_info_dir=speaker_layout.root)

    assert manager.get_metas_dict() == [
        {
            "name": "テスト話者",
            "speaker_uuid": speaker_layout.speaker_uuid,
            "styles": [{"name": "テスト", "id": speaker_layout.style_id}],
            "version": "0.0.1",
        }
    ]
    model_path = manager.get_model_path(speaker_layout.style_id)
    assert (
        model_path.config_path
        == (
            speaker_dir / "model" / str(speaker_layout.style_id) / "config.yaml"
        ).resolve()
    )
    assert (
        model_path.model_path
        == (
            speaker_dir / "model" / str(speaker_layout.style_id) / "model.pth"
        ).resolve()
    )
    assert model_path.hop_length == 512


def test_ignores_hidden_directories_and_macos_sidecars(
    speaker_layout: SpeakerLayout,
) -> None:
    speaker_layout.add_speaker()
    (speaker_layout.root / "__MACOSX").mkdir()
    (speaker_layout.root / ".hidden").mkdir()

    manager = MetaManager(speaker_info_dir=speaker_layout.root)

    assert [meta["speaker_uuid"] for meta in manager.get_metas_dict()] == [
        speaker_layout.speaker_uuid
    ]


def test_rejects_missing_speaker_info_directory(speaker_layout: SpeakerLayout) -> None:
    with pytest.raises(SpeakerInfoError, match="does not exist"):
        MetaManager(speaker_info_dir=speaker_layout.root / "missing")


def test_rejects_missing_metas(speaker_layout: SpeakerLayout) -> None:
    (speaker_layout.root / speaker_layout.speaker_uuid).mkdir(parents=True)

    with pytest.raises(SpeakerInfoError, match=r"metas\.json is missing"):
        MetaManager(speaker_info_dir=speaker_layout.root)


def test_rejects_missing_model(speaker_layout: SpeakerLayout) -> None:
    speaker_dir = speaker_layout.add_speaker()
    (speaker_dir / "model" / str(speaker_layout.style_id) / "model.pth").unlink()

    with pytest.raises(SpeakerInfoError, match=r"exactly one \.pth model"):
        MetaManager(speaker_info_dir=speaker_layout.root)


def test_rejects_duplicate_style_id(speaker_layout: SpeakerLayout) -> None:
    speaker_dir = speaker_layout.add_speaker()
    metas_path = speaker_dir / "metas.json"
    meta = json.loads(metas_path.read_text(encoding="utf-8"))
    meta["styles"].append({"styleName": "重複", "styleId": speaker_layout.style_id})
    metas_path.write_text(
        json.dumps(meta, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(SpeakerInfoError, match="Duplicate styleId"):
        MetaManager(speaker_info_dir=speaker_layout.root)


def test_rejects_duplicate_speaker_uuid_across_directories(
    speaker_layout: SpeakerLayout,
) -> None:
    speaker_layout.add_speaker(folder_name="first")
    speaker_layout.add_speaker(folder_name="second")

    with pytest.raises(SpeakerInfoError, match="Duplicate speakerUuid"):
        MetaManager(speaker_info_dir=speaker_layout.root)


def test_allows_same_style_id_for_different_speakers(
    speaker_layout: SpeakerLayout,
) -> None:
    speaker_layout.add_speakers(2)

    manager = MetaManager(speaker_info_dir=speaker_layout.root)

    assert len(manager.get_metas_dict()) == 2


def test_resolves_same_style_id_by_explicit_speaker_uuid(
    speaker_layout: SpeakerLayout,
) -> None:
    first_speaker_dir, second_speaker_dir = speaker_layout.add_speakers(2)
    manager = MetaManager(speaker_info_dir=speaker_layout.root)

    assert (
        manager.get_model_path(
            speaker_layout.style_id, speaker_uuid=speaker_layout.speaker_uuid
        ).model_path
        == (
            first_speaker_dir / "model" / str(speaker_layout.style_id) / "model.pth"
        ).resolve()
    )
    assert (
        manager.get_model_path(
            speaker_layout.style_id, speaker_uuid=speaker_layout.second_speaker_uuid
        ).model_path
        == (
            second_speaker_dir / "model" / str(speaker_layout.style_id) / "model.pth"
        ).resolve()
    )


def test_rejects_ambiguous_legacy_style_id(speaker_layout: SpeakerLayout) -> None:
    speaker_layout.add_speakers(2)
    manager = MetaManager(speaker_info_dir=speaker_layout.root)

    with pytest.raises(AmbiguousStyleError, match="specify speaker_uuid"):
        manager.get_model_path(speaker_layout.style_id)
