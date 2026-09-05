import subprocess
import sys
import textwrap

import pyopenjtalk


def test_pyworld_compat_uses_stdlib_metadata_without_deprecation_warning():
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


def test_text_to_tokens_does_not_load_the_tts_inference_stack():
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


def test_tts_stack_imports_with_local_kaldiio_guard_and_rejects_kaldi_io():
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.metadata

                from kaldiio import UnsupportedKaldiDataIOError
                from espnet2.bin.tts_inference import Text2Speech
                from espnet2.train.dataset import kaldi_loader

                assert importlib.metadata.version("kaldiio").endswith("+coeiroink.guard1")
                assert Text2Speech.__name__ == "Text2Speech"
                try:
                    kaldi_loader("unused.scp")
                except UnsupportedKaldiDataIOError:
                    pass
                else:
                    raise AssertionError("Kaldi I/O must fail explicitly")
                """
            ),
        ],
        check=True,
    )
