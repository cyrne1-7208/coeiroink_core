# ソースコード

CoreのリリースZIPは、本体ソース、`pyproject.toml`、`uv.lock`、ビルド用ファイルを含みます。CoreのwheelもPythonソースを含みます。公開ソースは [COEIROINK Core](https://github.com/cyrne1-7208/coeiroink_core) で参照できます。

セットアップ方法は本体のREADMEを参照してください。使用する依存のバージョンと配布元は `uv.lock` に記録されています。Pythonパッケージのソースアーカイブは `sdist.url`、Gitから取得するパッケージのソースは `source.git` に示されています。

EngineのリリースパッケージとDockerイメージには、GPL・LGPL・MPL依存のソースアーカイブも収録します。場所と再ビルド方法は [Engineのソース案内](https://github.com/cyrne1-7208/coeiroink_engine/blob/main/licenses/SOURCES.md) を参照してください。

OpenCL拡張を再ビルドする場合は `native/opencl` のREADMEとCMakeLists.txtを参照してください。上流コードのコミット、適用するパッチ、ビルド設定は同ディレクトリにあります。

Core本体のLGPL-3.0-onlyは、依存ライブラリのライセンスを変更するものではありません。Coreに依存のバイナリを加えて再配布するときは、その依存のライセンス本文と、GPL・LGPL・MPLなどで求められる対応ソースも提供してください。
