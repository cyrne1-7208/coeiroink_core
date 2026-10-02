# COEIROINK kaldiio guard

ESPnetはTTS推論でもKaldiデータ入出力用の`kaldiio`を必須依存として読み込みますが、COEIROINK Serverのモデル読込と音声合成では`ark`/`scp`機能を使用しません。

この独自実装は通常のTTS importだけを成立させ、Kaldiデータ入出力APIが要求された場合は明示的な例外を送出します。上流kaldiioのコードは含みません。
