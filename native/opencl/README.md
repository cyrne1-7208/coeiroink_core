# COEIROINK OpenCLバックエンド

CUDAを利用できないLinux環境のGPUで、COEIROINK CoreのPyTorch推論を実行するためのバックエンドです。`pytorch_dlprim`と`dlprimitives`を固定したコミットからビルドし、PyTorchのPrivateUse1デバイスを`ocl`として登録します。

Coreが追加したVITS演算と、pytorch_dlprimが対応する演算はOpenCLで実行します。未登録の演算は警告を出し、CPUへフォールバックします。警告はそのまま表示されます。

VITSの行列積と畳み込みでは、タイルの構成を文章の長さに依存させません。畳み込みの時間長をOpenCLカーネルの実行時引数として渡すため、生成する音声の長さが変わってもコンパイル済みカーネルを再利用できます。

ビルドにはOpenCL C++ヘッダー、ICDローダー、SQLite 3の開発用ヘッダーが必要です。導入方法とuv extraについては、[ルートのREADME](../../README.md)を参照してください。GPUドライバはwheelに含まれないため、利用するGPUに対応したドライバを別途インストールしてください。

DLPrimitivesは、AMD・Intel GPU向けのコンパイル済みカーネルを`$HOME/.dlprimitives/cache.db`へ保存します。同じデバイスとドライバを使う次回の起動では、このキャッシュを再利用します。NVIDIAでは、ドライバ内蔵のキャッシュを利用します。保存先は`DLPRIM_CACHE_DIR`で変更でき、`DLPRIM_CACHE_DISABLE=1`で無効にできます。
