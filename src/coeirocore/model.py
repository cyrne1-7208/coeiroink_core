from typing import List, Optional

from pydantic import BaseModel


class Mora(BaseModel):
    text: str
    consonant: Optional[str] = None
    consonant_length: Optional[float] = None
    vowel: str
    vowel_length: float
    pitch: float


class AccentPhrase(BaseModel):
    moras: List[Mora]
    accent: int
    pause_mora: Optional[Mora] = None
    is_interrogative: bool


class AudioQuery(BaseModel):
    """VOICEVOX互換のcamelCaseフィールドを保持する音声合成クエリです。"""

    accent_phrases: List[AccentPhrase]
    speedScale: float
    pitchScale: float
    intonationScale: float
    volumeScale: float
    prePhonemeLength: float
    postPhonemeLength: float
    outputSamplingRate: int
    outputStereo: bool
    kana: Optional[str] = None
