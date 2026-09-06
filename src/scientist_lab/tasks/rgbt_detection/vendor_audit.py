"""Offline audit helpers for vendored D-FINE (v2.3.1; zero GPU / zero network)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


PINNED_COMMIT = "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
BASELINE_IMPLEMENTATION = "dfine_s_vendored_v0_8_9"
VENDOR_REL = Path("third_party") / "DFINE"
TRAIN_REL = VENDOR_REL / "train.py"


def vendor_md_path(project_root: Path | str) -> Path:
    return Path(project_root) / "third_party" / "VENDOR.md"


def resolve_dfine_vendor_root(project_root: Path | str) -> Path | None:
    """Return the vendored DFINE root if ``train.py`` is present."""
    root = Path(project_root).resolve()
    candidate = root / VENDOR_REL
    if (candidate / "train.py").is_file():
        return candidate
    return None


def audit_dfine_vendor_pin(project_root: Path | str) -> dict[str, Any]:
    """Check vendor presence + VENDOR.md pin (offline; no Docker/GPU)."""
    root = Path(project_root).resolve()
    vendor = resolve_dfine_vendor_root(root)
    vendor_md = vendor_md_path(root)
    md_text = vendor_md.read_text(encoding="utf-8") if vendor_md.is_file() else ""
    pin_ok = PINNED_COMMIT in md_text
    train_ok = vendor is not None
    return {
        "ok": bool(train_ok and pin_ok and vendor_md.is_file()),
        "vendor_root": str(vendor) if vendor else None,
        "train_py": str(vendor / "train.py") if vendor else None,
        "vendor_md": str(vendor_md) if vendor_md.is_file() else None,
        "pinned_commit": PINNED_COMMIT,
        "pin_documented": pin_ok,
        "gpu_used": False,
        "network_used": False,
    }


def stage_dfine_vendor(
    workspace_dir: Path | str,
    *,
    project_root: Path | str,
) -> dict[str, Any]:
    """Copy vendored DFINE into a workspace (same intent as Docker staging)."""
    root = Path(project_root).resolve()
    src = resolve_dfine_vendor_root(root)
    if src is None:
        return {
            "ok": False,
            "error": "vendored DFINE train.py missing",
            "dest": None,
        }
    dest = Path(workspace_dir).resolve() / "third_party" / "DFINE"
    if dest.exists():
        return {
            "ok": True,
            "copied": False,
            "src": str(src),
            "dest": str(dest),
            "already_present": True,
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return {
        "ok": (dest / "train.py").is_file(),
        "copied": True,
        "src": str(src),
        "dest": str(dest),
        "already_present": False,
    }


# Image recipes live here. Must not be named ``docker/`` — that shadows PyPI ``docker``.
DOCKERFILES_REL = Path("dockerfiles")


def cuda_dockerfile_present(project_root: Path | str) -> bool:
    path = (
        Path(project_root).resolve()
        / DOCKERFILES_REL
        / "rgbt-detection-v2-cuda"
        / "Dockerfile"
    )
    return path.is_file()
