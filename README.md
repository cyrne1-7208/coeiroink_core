# COEIROINK Core (Forked by Cyrne1)

Cyrne1によってフォークされたCOEIROINK Coreです。

## 動作環境

Python 3.12を使用します。

| バックエンド | OS（x64） | GPUの要件 |
| --- | --- | --- |
| CPU | Linux・Windows | 不要 |
| CUDA | Linux・Windows | CUDA 12.8対応のNVIDIAドライバ |
| OpenCL | Linux | OpenCL対応GPUとドライバ |
| DirectML | Windows | Windows 10 バージョン1709以降・DirectX 12対応GPUとドライバ |

## セットアップ

ソースから実行するには、[uv](https://docs.astral.sh/uv/)、[Git](https://git-scm.com/)、C/C++のビルド環境が必要です。Python 3.12は、インストールされていなければuvが取得します。

OpenCL版には追加で、OpenCL C++ヘッダー、ICDローダー、SQLite 3の開発用ヘッダーも必要です。

Coreのディレクトリで実行してください。

```bash
uv sync --locked --extra cpu
```

GPU版では、`cpu`を`cuda`、`opencl`、`directml`のいずれかに置き換えます。バックエンドは1つだけ選択してください。

Coreのディレクトリで、モデルを入れる`speaker_info`フォルダを作成してください。

```bash
mkdir speaker_info
```

MYCOEIROINKのZIPを展開し、モデルのフォルダを名前を変えずに入れてください。

```text
coeiroink_core/
└── speaker_info/
    └── 展開したモデルのフォルダ/
```

HTTP APIから利用する場合は、[coeiroink_engine](https://github.com/cyrne1-7208/coeiroink_engine)も同じ親ディレクトリに配置してください。

## モデルの保持数

`AudioManager`の`max_loaded_models`で指定します。既定では、最後に使った1モデルを保持します。

- 正の整数：最近使ったモデルを、指定した数まで保持します。
- `None`：起動時に全モデルを読み込みます。

空きメモリが足りなくなると、使用していない期間が長いモデルから解放します。Engineから使う場合は、`--max-loaded-models`で指定してください。

## 実験的な機能

`AudioManager(voice_smoothing=True, ...)`で、母音の音色や周期の細かな揺れを抑えます。既定では無効です。Engineから使う場合は、`--experimental voice-smoothing`を指定してください。

`synthesis`にのみ適用され、`predict`と`predict_with_duration`には適用されません。CPU・DirectMLではCPU、CUDA・OpenCLではGPUで処理します。

## テスト

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev pytest -q
```

## ライセンス

個別にライセンスが示されているものを除き、ソースコードはLGPL-3.0-onlyです。詳細は[LICENSE](./LICENSE)を参照してください。LGPLv3が参照するGPLv3本文は[licenses/GPL-3.0.txt](./licenses/GPL-3.0.txt)にあります。

同梱ソースと再ビルド方法は [licenses/SOURCES.md](./licenses/SOURCES.md) を参照してください。

## 謝辞

本プロジェクトは、[COEIROINK](https://coeiroink.com/)および[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[ESPnet](https://github.com/espnet/espnet)、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk)、[PyTorch](https://pytorch.org/)などのオープンソースソフトウェアを利用しています。各プロジェクトの開発者・貢献者に感謝します。
