"""非推奨のパッケージラッパーを実行せずPyWorldを読み込みます。
PyWorld 0.3.5の初期化処理は``pkg_resources``でバージョンを取得しますが、コンパイル済み拡張はそのラッパーに依存しません。
拡張を直接読み込み通常の``pyworld``パッケージとして公開し、公開関数を変えずに標準ライブラリのメタデータAPIを使います。
"""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType


def _load_extension(package_path: Path) -> ModuleType:
    spec = importlib.machinery.PathFinder.find_spec(
        "pyworld.pyworld", [str(package_path)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"PyWorld extension was not found in package directory: {package_path}"
        )

    extension = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = extension
    try:
        spec.loader.exec_module(extension)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return extension


def load_pyworld() -> ModuleType:
    """非推奨の``pkg_resources``importを避けてPyWorldを返します。"""

    existing = sys.modules.get("pyworld")
    if existing is not None:
        return existing

    try:
        package_path = Path(
            importlib.metadata.distribution("pyworld").locate_file("pyworld")
        ).resolve()
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError("PyWorld is not installed") from error

    if not package_path.is_dir():
        raise ImportError(f"PyWorld package directory was not found: {package_path}")

    package = types.ModuleType("pyworld")
    package.__file__ = str(package_path / "__init__.py")
    package.__path__ = [str(package_path)]
    package.__package__ = "pyworld"
    package.__spec__ = importlib.machinery.ModuleSpec(
        "pyworld", loader=None, is_package=True
    )
    # 拡張がpyworld.pyworldを参照するため、先に仮のパッケージを登録します。
    sys.modules["pyworld"] = package

    try:
        extension = _load_extension(package_path)
        for name in dir(extension):
            if not name.startswith("__"):
                setattr(package, name, getattr(extension, name))
        package.pyworld = extension
        package.__version__ = importlib.metadata.version("pyworld")
    except Exception:
        # 拡張の読込に失敗した場合は、次の試行が壊れたモジュールを再利用しないよう両方を戻します。
        sys.modules.pop("pyworld", None)
        sys.modules.pop("pyworld.pyworld", None)
        raise

    return package
