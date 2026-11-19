# ソースコード

COEIROINK Coreの公開ソースは[COEIROINK Core](https://github.com/cyrne1-7208/coeiroink_core)で参照できます。セットアップ手順は本体のREADMEを参照してください。

## 収録内容と依存の参照先

- **リリースZIPとwheel**：リリースZIPには本体ソース、`pyproject.toml`、`uv.lock`、ビルド用ファイルを含みます。CoreのwheelもPythonソースを含みます。
- **依存パッケージのソース**：使用バージョンと配布元は`uv.lock`に記録しています。Pythonパッケージのアーカイブは`sdist.url`、Git取得の依存は`source.git`を確認してください。
- **OpenCL拡張**：上流コードのコミット、適用パッチ、ビルド設定は`native/opencl`のREADMEと`CMakeLists.txt`にあります。

## 再配布とEngineのソース案内

- **EngineパッケージとDockerイメージ**：GPL、LGPL、MPL依存のソースアーカイブも収録しています。場所と再ビルド手順は[Engineのソース案内](https://github.com/cyrne1-7208/coeiroink_engine/blob/main/licenses/SOURCES.md)を参照してください。
- **再配布時の条件**：Core本体のLGPL-3.0-onlyは依存ライブラリのライセンスを変更しません。依存バイナリを加えて再配布する場合は、そのライセンス本文とGPL、LGPL、MPLなどで求められる対応ソースも提供してください。
