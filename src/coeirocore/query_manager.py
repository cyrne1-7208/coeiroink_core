from coeirocore.model import AudioQuery


def query2tokens_prosody(query: AudioQuery) -> list[str]:
    """VOICEVOX形式のアクセント句をESPnetの日本語プロソディ記号列へ変換する。"""

    tokens = ["^"]
    for i, accent_phrase in enumerate(query.accent_phrases):
        up_token_flag = False
        for j, mora in enumerate(accent_phrase.moras):
            if mora.consonant:
                tokens.append(mora.consonant.lower())
            if mora.vowel == "N":
                tokens.append(mora.vowel)
            else:
                tokens.append(mora.vowel.lower())
            # `]`はアクセント核直後の下降、`[`は句内で最初に生じる上昇を表す。
            if accent_phrase.accent == j + 1 and j + 1 != len(accent_phrase.moras):
                tokens.append("]")
            if accent_phrase.accent - 1 >= j + 1 and up_token_flag is False:
                tokens.append("[")
                up_token_flag = True
        if i + 1 != len(query.accent_phrases):
            # `_`は読点由来の休止、`#`は休止を伴わないアクセント句境界を表す。
            if accent_phrase.pause_mora:
                tokens.append("_")
            else:
                tokens.append("#")
    # 文末は疑問文なら`?`、それ以外は`$`で閉じる。
    if query.accent_phrases and query.accent_phrases[-1].is_interrogative:
        tokens.append("?")
    else:
        tokens.append("$")
    return tokens
