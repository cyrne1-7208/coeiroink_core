import os
from pathlib import Path

import pytest
import soundfile as sf

from coeirocore.coeiro_manager import AudioManager, EspnetModel


def test_generate_wav(tmp_path: Path):
    speaker_info_dir_value = os.getenv("COEIROINK_TEST_SPEAKER_INFO_DIR")
    style_id_value = os.getenv("COEIROINK_TEST_STYLE_ID")
    if speaker_info_dir_value is None or style_id_value is None:
        pytest.skip(
            "COEIROINK_TEST_SPEAKER_INFO_DIR and COEIROINK_TEST_STYLE_ID are not set"
        )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    text = "今日はいい天気ですね"
    tokens = EspnetModel.text2tokens(text)
    wav = AudioManager(speaker_info_dir=Path(speaker_info_dir_value)).synthesis(
        tokens,
        style_id=int(style_id_value),
    )
    output_path = output_dir / "output.wav"
    sf.write(output_path, wav, 44100, "PCM_16")
    assert sf.info(output_path).frames > 0
