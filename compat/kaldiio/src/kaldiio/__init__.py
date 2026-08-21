"""ESPnetのTTS importだけを成立させ、未提供のKaldiデータ入出力は明示的に拒否する。"""

__version__ = "2.18.1+coeiroink.guard1"


class UnsupportedKaldiDataIOError(RuntimeError):
    """COEIROINK Serverが提供しないKaldi形式入出力をESPnetが要求した場合の例外。"""


def __getattr__(name: str):
    # Pythonのimport・検査機構が問い合わせる特殊属性は通常の欠落属性として扱い、未提供APIの検出だけを専用例外にする。
    if name.startswith("__"):
        raise AttributeError(name)
    raise UnsupportedKaldiDataIOError(
        "ESPnet requested kaldiio API "
        f"{name!r}, but COEIROINK Server does not provide Kaldi ark/scp I/O"
    )
