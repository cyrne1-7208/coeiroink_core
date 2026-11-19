"""遅延importと非推奨警告なしの互換性境界をクリーンなサブプロセスで検証する。"""

from __future__ import annotations

import subprocess
import sys

import pyopenjtalk


def test_pyworld_compat_uses_stdlib_metadata_without_deprecation_warning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "dev",
            "-W",
            "error::DeprecationWarning",
            "-c",
            (
                "import sys; "
                "from coeirocore.pyworld_compat import load_pyworld; "
                "assert 'pyworld' not in sys.modules; "
                "world = load_pyworld(); "
                "assert callable(world.synthesize); "
                "assert world.__version__; "
                "assert 'pyworld' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""


def test_text_to_tokens_does_not_load_the_tts_inference_stack() -> None:
    # pyopenjtalk初回辞書展開時に発生する外部ライブラリの警告を、Coreの遅延import検証から分離する。
    pyopenjtalk.g2p("辞書初期化")
    subprocess.run(
        [
            sys.executable,
            "-X",
            "dev",
            "-W",
            "error::DeprecationWarning",
            "-c",
            (
                "import sys; "
                "from coeirocore.coeiro_manager import EspnetModel; "
                "assert EspnetModel.text2tokens('テスト'); "
                "assert 'espnet2.bin.tts_inference' not in sys.modules; "
                "assert 'pyworld' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
