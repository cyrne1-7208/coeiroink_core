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

OpenCLの利用には、GPUベンダーのOpenCL ICD、OpenCLヘッダー、loader、SQLite 3の開発ヘッダーが必要です。

## セットアップ

依存関係は`pyproject.toml`で定義し、`uv.lock`で固定しています。LinuxまたはWindowsのCPU環境では次を実行します。

```bash
uv sync --locked --extra cpu
```

CUDAまたはOpenCLでは`cpu`を`cuda`または`opencl`へ置き換えてください。Windows DirectMLでは次を実行します。

```powershell
uv sync --python 3.12 --locked --extra directml
```

各extraは相互排他的です。`speaker_info`には展開したMYCOEIROINKモデルのフォルダを配置します。旧形式（`config.yaml`の`version: 0.10.3`）とCOEIROINK v2形式のモデルに対応し、`speakerUuid`と`styleId`の組でモデルを識別します。

## テスト

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev pytest -q
```

モデルは要求時にロードされ、正常にロードされたモデルはプロセス内に保持されます。明示的なモデル数上限は設けず、利用可能なメモリを上限とします。

## ライセンス

LGPL-3.0-onlyです。詳細は[LICENSE](./LICENSE)を参照してください。LGPLv3が参照するGPLv3本文は[licenses/GPL-3.0.txt](./licenses/GPL-3.0.txt)に収録しています。

## 謝辞

本プロジェクトは、[COEIROINK](https://coeiroink.com/)および[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[ESPnet](https://github.com/espnet/espnet)、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk)、[PyTorch](https://pytorch.org/)などのオープンソースソフトウェアを利用しています。各プロジェクトの開発者・貢献者に感謝します。
