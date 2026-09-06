"""Local docker/ must not shadow PyPI docker. No GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_image_recipes_are_not_named_docker() -> None:
    """CWD ``docker/`` was a namespace package that blocked ``import docker.errors``."""
    assert not (ROOT / "docker").exists()
    assert (ROOT / "dockerfiles" / "rgbt-detection-v2-cuda" / "Dockerfile").is_file()


def test_pypi_docker_errors_importable_when_sdk_installed() -> None:
    docker = pytest.importorskip("docker")
    pytest.importorskip("docker.errors")
    from docker.errors import DockerException

    assert hasattr(docker, "from_env")
    assert DockerException is not None
    file_hint = str(getattr(docker, "__file__", "") or "").replace("\\", "/")
    path_hint = " ".join(str(p) for p in (getattr(docker, "__path__", None) or [])).replace("\\", "/")
    root = str(ROOT.resolve()).replace("\\", "/")
    combined = f"{file_hint} {path_hint}"
    assert f"{root}/dockerfiles" not in combined
    assert f"{root}/docker/" not in combined


def test_freeze_execute_deps_import_without_docker_errors_crash() -> None:
    """Manager execute path must be able to import live_runner deps (no GPU)."""
    from scientist_lab.adapters.dfine.cuda_runner import make_cuda_live_runner

    assert callable(make_cuda_live_runner)
    pytest.importorskip("docker.errors")
    from scientist_lab.services.experiment_service import ExperimentService

    assert ExperimentService is not None
