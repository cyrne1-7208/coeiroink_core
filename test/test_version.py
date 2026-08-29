import tomllib
from pathlib import Path

import coeirocore

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _project_version(path: Path) -> str:
    with path.open("rb") as project_file:
        return tomllib.load(project_file)["project"]["version"]


def _load_toml(path: Path) -> dict:
    with path.open("rb") as toml_file:
        return tomllib.load(toml_file)


def _locked_versions(package_name: str) -> set[str]:
    lock = _load_toml(REPOSITORY_ROOT / "uv.lock")
    return {
        package["version"]
        for package in lock["package"]
        if package["name"] == package_name and "version" in package
    }


def test_core_and_native_package_versions_match() -> None:
    """Core本体と同時に配布するOpenCL拡張のバージョン同期漏れを検出する。"""

    project = _load_toml(REPOSITORY_ROOT / "pyproject.toml")
    core_version = project["project"]["version"]
    opencl_version = _project_version(
        REPOSITORY_ROOT / "native" / "opencl" / "pyproject.toml"
    )
    opencl_requirement = next(
        requirement
        for requirement in project["project"]["optional-dependencies"]["opencl"]
        if requirement.startswith("coeiroink-opencl==")
    )
    opencl_metadata = next(
        metadata
        for metadata in project["tool"]["uv"]["dependency-metadata"]
        if metadata["name"] == "coeiroink-opencl"
    )

    assert coeirocore.__version__ == core_version
    assert opencl_version == core_version
    assert opencl_requirement.split("==", 1)[1].split(";", 1)[0] == core_version
    assert opencl_metadata["version"] == core_version
    assert _locked_versions("coeirocore") == {core_version}
    assert _locked_versions("coeiroink-opencl") == {core_version}
