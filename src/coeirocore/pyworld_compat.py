"""非推奨のPyWorldパッケージ初期化を通さず、ネイティブ拡張を読み込む。"""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def _find_extension(package_path: Path) -> Path:
    spec = importlib.machinery.PathFinder.find_spec(
        "pyworld.pyworld", [str(package_path)]
    )
    if spec is None or spec.loader is None or spec.origin is None:
        raise ImportError(
            f"PyWorld extension was not found in package directory: {package_path}"
        )
    return Path(spec.origin).resolve()


def _load_extension(extension_path: Path) -> ModuleType:
    # 末尾をpyworldに保つことで、C拡張が公開するPyInit_pyworldをそのまま利用できる。
    spec = importlib.util.spec_from_file_location(
        "_coeirocore_pyworld.pyworld", extension_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Failed to create a PyWorld extension spec: {extension_path}"
        )

    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    return extension


@lru_cache(maxsize=1)
def load_pyworld() -> ModuleType:
    """非推奨のpkg_resourcesを使うラッパーを避けてPyWorldを返す。"""

    try:
        distribution = importlib.metadata.distribution("pyworld")
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError("PyWorld is not installed") from error

    package_path = Path(distribution.locate_file("pyworld")).resolve()
    if not package_path.is_dir():
        raise ImportError(f"PyWorld package directory was not found: {package_path}")

    extension = _load_extension(_find_extension(package_path))
    extension.__version__ = distribution.version
    return extension
