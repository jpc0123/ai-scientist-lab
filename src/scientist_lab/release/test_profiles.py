"""Internal test profiles for controlled merge (v1.9.4).

Commands are registry-only — never accepted from frontend, LLM, or PatchProposal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


ProfileId = Literal["syntax", "unit", "smoke", "full_regression", "acceptance"]


class TestProfile(BaseModel):
    profile_id: ProfileId
    commands: list[list[str]]
    timeout_seconds: int = 600
    required: bool = True
    description: str = ""


def _python(project_root: Path) -> str:
    win = Path(project_root) / ".venv" / "Scripts" / "python.exe"
    if win.is_file():
        return str(win.resolve())
    nix = Path(project_root) / ".venv" / "bin" / "python"
    if nix.is_file():
        return str(nix.resolve())
    return "python"


def build_profile_registry(project_root: Path) -> dict[str, TestProfile]:
    py = _python(project_root)
    return {
        "syntax": TestProfile(
            profile_id="syntax",
            description="Compile release/merge modules",
            timeout_seconds=180,
            commands=[[py, "-m", "compileall", "-q", "src/scientist_lab/release"]],
        ),
        "unit": TestProfile(
            profile_id="unit",
            description="Unit tests for merge/release",
            timeout_seconds=600,
            commands=[
                [
                    py,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/unit/test_merge_v19.py",
                    "tests/unit/test_release_v18.py",
                ]
            ],
        ),
        "smoke": TestProfile(
            profile_id="smoke",
            description="Short merge smoke",
            timeout_seconds=300,
            commands=[
                [py, "-m", "pytest", "-q", "tests/unit/test_merge_v19.py"],
            ],
        ),
        "full_regression": TestProfile(
            profile_id="full_regression",
            description="Full pytest suite",
            timeout_seconds=3600,
            commands=[[py, "-m", "pytest", "-q"]],
        ),
        "acceptance": TestProfile(
            profile_id="acceptance",
            description="v1.8 acceptance script (offline)",
            timeout_seconds=600,
            commands=[[py, "scripts/accept_v18.py"]],
        ),
    }


def require_profile(project_root: Path, profile_id: str) -> TestProfile:
    registry = build_profile_registry(project_root)
    if profile_id not in registry:
        raise ValueError(
            f"unknown test profile {profile_id!r}; "
            f"allowed={sorted(registry)}"
        )
    return registry[profile_id]


def list_profile_ids(project_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "profile_id": p.profile_id,
            "description": p.description,
            "timeout_seconds": p.timeout_seconds,
            "required": p.required,
            "command_count": len(p.commands),
        }
        for p in build_profile_registry(project_root).values()
    ]
