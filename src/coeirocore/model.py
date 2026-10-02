from pydantic import BaseModel


class Mora(BaseModel):
    text: str
    consonant: str | None = None
    consonant_length: float | None = None
    vowel: str
    vowel_length: float
    pitch: float


class AccentPhrase(BaseModel):
    moras: list[Mora]
    accent: int
    pause_mora: Mora | None = None
    is_interrogative: bool


class AudioQuery(BaseModel):
    accent_phrases: list[AccentPhrase]
    speedScale: float
    pitchScale: float
    intonationScale: float
    volumeScale: float
    prePhonemeLength: float
    postPhonemeLength: float
    outputSamplingRate: int
    outputStereo: bool
    kana: str | None = None
