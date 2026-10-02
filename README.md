# COEIROINK Core (Forked by Cyrne1)

Cyrne1によってフォークされたCOEIROINK Coreです。

## 対象環境

現在の公式検証対象はLinux x64 CPU、Python 3.12です。CoreとEngineを同じ親ディレクトリへ配置してください。

## セットアップ

Engine側のセットアップスクリプトが、CPU版PyTorch、Open JTalk辞書、固定したESPnet互換環境をまとめて構築します。

```bash
bash ../coeiroink_engine/build_util/setup_mycoeiroink_linux_cpu.bash ../coeiroink_engine/.venv .
```

`speaker_info`には、展開したMYCOEIROINKモデルのフォルダを配置します。旧形式（`config.yaml`の`version: 0.10.3`）と、COEIROINK v2形式のモデルを対象に、`speakerUuid`と`styleId`の組でモデルを識別します。

直接インストールする場合は、Coreの依存関係を導入した後に次を実行します。

```bash
python -m pip install --editable .
```

`requirements-espnet.txt`はモデル互換性のため固定した公開ESPnetコミットを指定しています。`requirements-pyopenjtalk.txt`はOpen JTalk辞書を提供します。

## テスト

```bash
PYTHONPATH=src python -m pytest -q
```

モデルは要求時にロードされ、正常にロードされたモデルはプロセス内に保持されます。明示的なモデル数上限は設けず、CPU環境の利用可能メモリを自然な上限とします。

## ライセンス

LGPL v3です。詳細は[LICENSE](./LICENSE)を参照してください。

## 謝辞

本プロジェクトは、[shirowanisan/coeiroink_core](https://github.com/shirowanisan/coeiroink_core)の公開ソースを基盤に、[ESPnet](https://github.com/espnet/espnet)、[pyopenjtalk](https://github.com/r9y9/pyopen_jtalk)、[PyTorch](https://pytorch.org/)などのオープンソースソフトウェアを利用しています。各プロジェクトの開発者・貢献者に感謝します。
