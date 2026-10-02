from coeirocore.model import AccentPhrase, AudioQuery, Mora
from coeirocore.query_manager import query2tokens_prosody


def _mora(text: str, consonant: str | None, vowel: str) -> Mora:
    return Mora(
        text=text,
        consonant=consonant,
        consonant_length=0.05 if consonant is not None else None,
        vowel=vowel,
        vowel_length=0.1,
        pitch=5.0,
    )


def test_query_to_tokens_prosody():
    query = AudioQuery(
        accent_phrases=[
            AccentPhrase(
                moras=[
                    _mora("カ", "k", "a"),
                    _mora("キ", "k", "I"),
                    _mora("ク", "k", "u"),
                ],
                accent=2,
                is_interrogative=False,
            ),
            AccentPhrase(
                moras=[_mora("サ", "s", "a")],
                accent=1,
                is_interrogative=True,
            ),
        ],
        speedScale=1,
        pitchScale=0,
        intonationScale=1,
        volumeScale=1,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        outputSamplingRate=24000,
        outputStereo=False,
    )

    assert query2tokens_prosody(query) == [
        "^",
        "k",
        "a",
        "[",
        "k",
        "i",
        "]",
        "k",
        "u",
        "#",
        "s",
        "a",
        "?",
    ]


def test_empty_query_uses_end_token_without_exception_fallback():
    query = AudioQuery.model_construct(accent_phrases=[])

    assert query2tokens_prosody(query) == ["^", "$"]
