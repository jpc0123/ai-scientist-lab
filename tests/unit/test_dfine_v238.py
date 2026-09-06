"""v2.3.8 unit tests — package / API version freeze for seal prep."""

from __future__ import annotations

from pathlib import Path

from scientist_lab import API_VERSION, __version__


ROOT = Path(__file__).resolve().parents[2]


def test_package_version_is_2_3_0():
    assert __version__ == "2.3.0"
    assert API_VERSION == "v2.3.0"
    assert __version__ == API_VERSION.lstrip("v")


def test_pyproject_and_web_aligned():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.3.0"' in pyproject
    web = (ROOT / "web" / "package.json").read_text(encoding="utf-8")
    assert '"version": "2.3.0"' in web


def test_acceptance_manifest_present():
    manifest = ROOT / "docs" / "acceptance" / "v2.3" / "version_manifest.json"
    text = manifest.read_text(encoding="utf-8")
    assert '"tag_target": "v2.3.0"' in text
    assert "v2.3.8" in text
