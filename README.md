# COEIROINK Core (Forked by Cyrne1)

Cyrne1によってフォークされたCOEIROINK Coreです。

MYCOEIROINKのVITSモデルを実行し、Engineへ波形と音素durationを提供します。GUIやHTTPサーバーは含みません。

## 対象環境

CoreとEngineを同じ親ディレクトリへ配置し、利用するバックエンドを1つ選択します。

| OS | バックエンド | uv extra | Python |
| --- | --- | --- | --- |
| Linux x64 | CPU | `cpu` | 3.12 |
| Linux x64 | CUDA | `cuda` | 3.12 |
| Linux x64 | OpenCL | `opencl` | 3.12 |
| Windows x64 | CPU | `cpu` | 3.12 |
| Windows x64 | CUDA | `cuda` | 3.12 |
| Windows x64 | DirectML | `directml` | 3.12 |

OpenCLの利用には、GPUベンダーのOpenCL ICD、OpenCLヘッダー、OpenCLローダー、SQLite 3の開発ヘッダーが必要です。

## セットアップ

依存関係は`pyproject.toml`で定義し、`uv.lock`で固定しています。LinuxまたはWindowsのCPU環境では次のコマンドを実行します。

```bash
uv sync --locked --extra cpu
```

CUDAまたはOpenCLでは`cpu`を`cuda`または`opencl`へ置き換えてください。Windows DirectMLでは次のコマンドを実行します。

```powershell
uv sync --python 3.12 --locked --extra directml
```

各extraは相互排他的です。`speaker_info`には展開したMYCOEIROINKモデルのフォルダを配置します。旧形式（`config.yaml`の`version: 0.10.3`）とCOEIROINK v2形式のモデルに対応し、`speakerUuid`と`styleId`の組でモデルを識別します。

## モデル保持とメモリ管理

Coreの`AudioManager(max_loaded_models=...)`（Engineでは`--max-loaded-models`）で同時保持モデル数を指定します。既定値は1で、数値指定時は直近に使用したモデルから順に最大指定数まで保持し、`None`（Engineでは`all`）では全モデルを起動時に読み込みます。次のモデルを安全に読み込める空きメモリがない場合は、指定値にかかわらずLRUモデルを解放するため、全モデルがメモリに収まらない環境では起動後の保持数が全件未満になることがあります。

実験的な`generator_only=True`では、VITSの学習専用モジュールへ実メモリを割り当てず推論に必要な重みだけを読み込むため、推論結果を変えない設計でモデルロード時のメモリ消費を抑えます。

## テスト

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev pytest -q
```

## ライセンス

LGPL-3.0-onlyです。詳細は[LICENSE](./LICENSE)を参照してください。LGPLv3が参照するGPLv3本文は[licenses/GPL-3.0.txt](./licenses/GPL-3.0.txt)に収録しています。

## 謝辞

本プロジェクトは、[COEIROINK](https://coeiroink.com/)および[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[ESPnet](https://github.com/espnet/espnet)、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk)、[PyTorch](https://pytorch.org/)などのオープンソースソフトウェアを利用しています。各プロジェクトの開発者・貢献者に感謝します。
