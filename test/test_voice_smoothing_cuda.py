"""GPUがないビルド環境でも、音声補正のCUDAカーネルを実際にコンパイルする。"""

import pytest

from coeirocore.voice_smoothing_gpu import _source


def test_cuda_kernels_compile_with_nvrtc():
    nvrtc = pytest.importorskip("cupy_backends.cuda.libs.nvrtc")
    program = nvrtc.createProgram(_source(), "voice_smoothing.cu", (), ())
    try:
        # 実デバイスを参照せず、対応するGPU世代を指定して全カーネルを検査する。
        nvrtc.compileProgram(program, ("--std=c++11", "--gpu-architecture=compute_75"))
        assert nvrtc.getPTX(program)
    except nvrtc.NVRTCError as error:
        raise RuntimeError(nvrtc.getProgramLog(program)) from error
    finally:
        nvrtc.destroyProgram(program)
