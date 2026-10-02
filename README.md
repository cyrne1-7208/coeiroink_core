# COEIROINK Core (Forked by Cyrne1)

Cyrne1によってフォークされたCOEIROINK Coreです。

MYCOEIROINKのVITSモデルを読み込み、音声波形と音素の継続時間をEngineへ返します。GUIとHTTPサーバーは含みません。

## 対象環境

利用するバックエンドを1つ選択します。Engineと連携する場合は、CoreとEngineを同じ親ディレクトリに配置してください。

| OS | バックエンド | uv extra | Python |
| --- | --- | --- | --- |
| Linux x64 | CPU | `cpu` | 3.12 |
| Linux x64 | CUDA | `cuda` | 3.12 |
| Linux x64 | OpenCL | `opencl` | 3.12 |
| Windows x64 | CPU | `cpu` | 3.12 |
| Windows x64 | CUDA | `cuda` | 3.12 |
| Windows x64 | DirectML | `directml` | 3.12 |

## 事前に必要なもの

このリポジトリのソースから`uv sync`でセットアップする場合は、次のものが必要です。

- [uv](https://docs.astral.sh/uv/)
- [Git](https://git-scm.com/)
- C/C++のビルド環境（LinuxではGCC/G++、WindowsではMSVC Build Tools）
- インターネット接続（初回セットアップではPyPI、GitHub、PyTorchのパッケージ配布先に接続します）
- CUDA版：CUDA 12.8に対応するNVIDIAドライバ
- OpenCL版：GPUベンダーのOpenCLドライバ、OpenCL C++ヘッダー、ICDローダー、SQLite 3の開発用ヘッダー
- DirectML版：Windows 10 バージョン1709以降、DirectX 12対応GPU、最新のGPUドライバ

Pythonは3.12を使用します。インストールされていない場合は、`uv`がセットアップ時に取得します。

## セットアップ

`--extra`には、使用するバックエンドを指定します。複数のバックエンドを同時に指定することはできません。

依存関係は`pyproject.toml`で定義し、`uv.lock`で固定しています。LinuxまたはWindowsのCPU環境では次のコマンドを実行します。

```bash
uv sync --locked --extra cpu
```

CUDAまたはOpenCLを利用する場合は、`cpu`を`cuda`または`opencl`に置き換えてください。Windows DirectMLでは次のコマンドを実行します。

```powershell
uv sync --python 3.12 --locked --extra directml
```

`speaker_info/`ディレクトリを作成し、展開したMYCOEIROINKモデルのディレクトリを配置します。旧形式（`config.yaml`の`version: 0.10.3`）とCOEIROINK v2形式のモデルに対応し、`speakerUuid`と`styleId`の組でモデルを識別します。

## モデル保持とメモリ管理

Coreでは`AudioManager(max_loaded_models=...)`、Engineでは`--max-loaded-models`で、同時に保持するモデル数を指定します。既定値は1です。

- 数値を指定すると、最近使ったモデルから順にその数まで保持します。
- Coreで`None`、Engineで`all`を指定すると、起動時に全モデルを読み込みます。

空きメモリが足りない場合は、設定にかかわらず、最後に使ってから最も時間が経ったモデルを解放します。そのため、すべてのモデルがメモリに収まらない環境では、`all`を指定しても一部のモデルが解放されます。

モデルは既定で、VITSの推論に必要な重みだけを読み込みます。合成結果を変えずに、モデル読み込み時のメモリ使用量を抑えます。

## 実験的な音声補正

母音の音色や周期の細かな揺れを抑える処理を、`AudioManager(voice_smoothing=True, ...)`で有効にできます。既定では無効です。必要なライブラリは各バックエンド用のuv extraに含まれます。Engineから使う場合は、`--experimental voice-smoothing`を指定します。

補正は母音の冒頭を避け、音素区間と波形全体の平均音量を保ちます。`synthesis`にのみ適用され、`predict`と`predict_with_duration`は補正前の波形を返します。CPUとDirectMLではCPU上のParselmouthとSciPyを使い、CUDAとOpenCLでは選択したGPU上で処理します。

Engineは解析済みの音素列をCoreへ渡します。Coreへ文字列を直接渡す場合は、補正の有無にかかわらず、モデルが指定するESPnetのTTS前処理ライブラリも必要です。

CPUとDirectMLの周期検出には、[Parselmouth](https://github.com/YannickJadoul/Parselmouth)（`praat-parselmouth`、GPL-3.0-or-later）を利用します。追加で導入するライブラリにも、それぞれのライセンスが適用されます。

## テスト

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev pytest -q
```

## ライセンス

本リポジトリのソースコードは、個別にライセンスが示されているものを除き、LGPL-3.0-onlyです。詳細は[LICENSE](./LICENSE)を参照してください。LGPLv3が参照するGPLv3本文は[licenses/GPL-3.0.txt](./licenses/GPL-3.0.txt)に収録しています。

## 謝辞

本プロジェクトは、[COEIROINK](https://coeiroink.com/)および[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[VOICEVOX](https://github.com/VOICEVOX/voicevox)、[ESPnet](https://github.com/espnet/espnet)、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk)、[PyTorch](https://pytorch.org/)などのオープンソースソフトウェアを利用しています。各プロジェクトの開発者・貢献者に感謝します。
