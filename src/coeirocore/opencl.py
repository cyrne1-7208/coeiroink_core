"""OpenCL実行時にVITSの1D畳み込みを2D演算へ写像する。"""

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_CONVOLUTION = torch.ops.aten.convolution.default
_RANDN_LIKE = torch.ops.aten.randn_like.default


def _expand_dimension(value: Any, trailing_value: int) -> Any:
    """1D畳み込みの引数へ幅方向の値を追加する。"""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 1:
            return [value[0], trailing_value]
        return value
    return [value, trailing_value]


class OpenCLVitsMode(TorchDispatchMode):
    """OpenCLの2D畳み込み実装でVITSの1D畳み込みを実行するモード。"""

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...] | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        call_kwargs = {} if kwargs is None else kwargs

        if func is _CONVOLUTION:
            # VITSが使用する1D畳み込みだけを2Dへ写像し、それ以外の畳み込みは元のディスパッチへ渡す。
            converted = self._convert_convolution(args)
            if converted is not None:
                result = func(*converted, **call_kwargs)
                return result.squeeze(-1)

        if func is _RANDN_LIKE and args:
            input_tensor = args[0]
            if (
                isinstance(input_tensor, torch.Tensor)
                and input_tensor.device.type == "ocl"
                and not input_tensor.is_contiguous()
            ):
                # OpenCLバックエンドのrandn_likeは非連続strideを扱えないため、同じ形状の連続テンソルへ正規化する。
                args = (input_tensor.contiguous(), *args[1:])

        return func(*args, **call_kwargs)

    @staticmethod
    def _convert_convolution(
        args: tuple[Any, ...],
    ) -> tuple[Any, ...] | None:
        """入力と重みへダミー幅を追加し、1D畳み込みの各引数を等価な2D形式へ変換する。"""

        if len(args) < 9:
            return None
        input_tensor, weight = args[0], args[1]
        if not isinstance(input_tensor, torch.Tensor) or not isinstance(
            weight, torch.Tensor
        ):
            return None
        if input_tensor.ndim != 3 or weight.ndim != 3:
            return None

        converted = list(args)
        converted[0] = input_tensor.unsqueeze(-1)
        converted[1] = weight.unsqueeze(-1)
        converted[3] = _expand_dimension(args[3], 1)
        converted[4] = _expand_dimension(args[4], 0)
        converted[5] = _expand_dimension(args[5], 1)
        converted[7] = _expand_dimension(args[7], 0)
        return tuple(converted)
