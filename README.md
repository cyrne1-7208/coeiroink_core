# COEIROINK Core (Forked by Cyrne1)

Cyrne1によるCOEIROINK Coreのフォークです。音声合成、モデル管理、波形処理を担当します。HTTP APIサーバーとして使う場合は、同じ親ディレクトリに[coeiroink_engine](https://github.com/cyrne1-7208/coeiroink_engine)を配置してください。

## 動作環境

64bit環境のPython 3.14で動作します。[uv](https://docs.astral.sh/uv/)、[Git](https://git-scm.com/)、C/C++ビルド環境が必要です。Python 3.14が未導入の場合は`uv`が自動で取得します。

| 環境 | Linux x64 | Windows x64 |
| --- | --- | --- |
| CPU | `cpu` | `cpu` |
| NVIDIA GPU | `cuda` | `cuda` |
| GPU全般 | `opencl` | `directml` |

- **`cuda`**：CUDA 13.0対応のNVIDIAドライバが必要です。
- **`opencl`**：OpenCL対応GPUとドライバに加え、OpenCL C++ヘッダー、ICDローダー、SQLite 3開発用ヘッダーが必要です。詳細は[OpenCL README](./native/opencl/README.md)を参照してください。
- **`directml`**：Windows 10 バージョン1709以降、DirectX 12対応GPU、および[Microsoft Visual C++ x64再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe)が必要です。[torch-directml-backwards](https://github.com/cyrne1-7208/torch-directml-backwards)を使用するため、旧`torch-directml`とは併用しないでください。

## セットアップ

上の表からバックエンドを1つ選び、Coreのディレクトリで導入します。高品質リサンプラーの`libsoxr`を使う場合は`--extra soxr`も併せて指定してください。

```bash
uv sync --locked --extra cpu
mkdir speaker_info
```

`speaker_info`には、MYCOEIROINKのZIPを**展開したフォルダ**を名前を変えずに入れてください。

```text
coeiroink_core/
└── speaker_info/
    └── 展開したモデルのフォルダ/
```

## 主な設定

`AudioManager`の主な引数です。

| 引数 | 既定値 | 説明 |
| --- | --- | --- |
| `max_loaded_models` | `1` | LRU保持数の上限。`None`で起動時に全モデルを読込。Engineは`--max-loaded-models` |
| `dangerous_skip_check` | `False` | `True`で容量不足を警告だけにし、拒否と自動解放を省略。Engineは`--dangerous-skip-check` |
| `voice_smoothing` | `False` | `True`で`synthesis`時の母音の揺れを抑制。`predict`と`predict_with_duration`は対象外で、Engineは`--experimental voice-smoothing` |
| `skip_trailing_silence` | `False` | `True`で`synthesis`時に文末の無音のdecoder処理を省略。`predict`と`predict_with_duration`は呼び出しごとに指定し、Engineは`--experimental skip-trailing-silence` |
| `chunked_decoding` | `False` | `True`で約9秒を超える推論のdecoderを分割実行。Engineは`--experimental chunked-decoding` |

### メモリ管理

- 起動時は最大モデル、切替時は対象モデルのサイズで空きを確認し、不足時は古いモデルから解放します。
- 最大1個分の空き不足や読込失敗時は起動を中止し、`None`で同時保持できない場合のみ警告して続行します。
- `dangerous_skip_check=True`でも保持数の上限は維持され、実メモリ不足で異常終了する危険があります。

### 音声補正の実行先

`voice_smoothing=True`では、CPUとDirectMLはCPU、CUDAとOpenCLはGPUで処理します。

## テスト

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev pytest -q
uv run --locked --extra cpu --group dev ruff check src test
uv run --locked --extra cpu --group dev ruff format --check src test
```

`opencl`、`cuda`、`real_model`の各マーカーが付いたテストは、未対応環境では自動でスキップされます。手動で除外する場合は`-m "not opencl and not cuda and not real_model"`を指定してください。

## ライセンス

個別表記があるものを除き、ソースコードはLGPL-3.0-onlyです。

- **本体ライセンス**：[LICENSE](./LICENSE)、[GPL-3.0.txt](./licenses/GPL-3.0.txt)
- **ソースと再ビルド**：[licenses/SOURCES.md](./licenses/SOURCES.md)

## 謝辞

本プロジェクトは[COEIROINK](https://coeiroink.com/)と[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)を基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[ESPnet](https://github.com/espnet/espnet)、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk)、[PyTorch](https://pytorch.org/)などのOSSを利用しています。開発者と貢献者の皆様に感謝します。
