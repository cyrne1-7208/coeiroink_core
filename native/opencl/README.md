# COEIROINK OpenCLバックエンド

CUDAを使えないLinux環境のGPUで、COEIROINK CoreのPyTorch推論を実行するためのバックエンドです。固定コミットの`pytorch_dlprim`と`dlprimitives`からビルドし、PyTorchのPrivateUse1デバイスを`ocl`として登録します。

## 動作仕様

- **OpenCL実行とフォールバック**：Coreが追加したVITS演算と`pytorch_dlprim`が対応する演算をOpenCLで実行します。未登録の演算は警告を出してCPUへフォールバックし、警告メッセージはそのまま表示します。
- **可変長でのカーネル再利用**：VITSの行列積と畳み込みでは、タイル構成を文章の長さに依存させません。畳み込みの時間長をカーネル実行時引数として渡すため、音声長が変わってもコンパイル済みカーネルを再利用できます。

## 要件と導入

- **ビルド要件**：OpenCL C++ヘッダー、ICDローダー、SQLite 3の開発用ヘッダーが必要です。
- **GPUドライバ**：wheelには含まれないため、使用するGPUに対応したドライバを別途インストールしてください。
- **導入手順**：インストールコマンドとuvのextra指定は[ルートREADME](../../README.md)を参照してください。

## カーネルキャッシュ

- **AMDとIntel GPU**：コンパイル済みカーネルを`$HOME/.dlprimitives/cache.db`へ保存し、同じデバイスとドライバの組み合わせで次回起動時に再利用します。
- **NVIDIA GPU**：ドライバ内蔵のキャッシュを利用します。

| 環境変数 | 説明 |
| --- | --- |
| `DLPRIM_CACHE_DIR` | キャッシュファイルの保存先ディレクトリを変更 |
| `DLPRIM_CACHE_DISABLE=1` | カーネルキャッシュを無効化 |
