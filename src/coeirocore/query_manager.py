from typing import List

from coeirocore.model import AudioQuery


# ESPnetの音素列では、アクセント境界と休止を専用トークンで表します。


def query2tokens_prosody(query: AudioQuery) -> List[str]:
    tokens = ['^']
    for i, accent_phrase in enumerate(query.accent_phrases):
        up_token_flag = False
        for j, mora in enumerate(accent_phrase.moras):
            if mora.consonant:
                tokens.append(mora.consonant.lower())
            if mora.vowel == 'N':
                tokens.append(mora.vowel)
            else:
                tokens.append(mora.vowel.lower())
            # accentは1始まりで、Nだけ大文字を保ったままESPnetへ渡します。
            if accent_phrase.accent == j + 1 and j + 1 != len(accent_phrase.moras):
                tokens.append(']')
            if accent_phrase.accent - 1 >= j + 1 and up_token_flag is False:
                tokens.append('[')
                up_token_flag = True
        if i + 1 != len(query.accent_phrases):
            # 句間のpause_moraは実休止、ない場合はアクセント境界として区別します。
            if accent_phrase.pause_mora:
                tokens.append('_')
            else:
                tokens.append('#')
    if query.accent_phrases and query.accent_phrases[-1].is_interrogative:
        tokens.append('?')
    else:
        tokens.append('$')
    return tokens
