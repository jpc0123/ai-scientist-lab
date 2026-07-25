"""Allow-listed sandbox test profile registry (v2.2.6).

Only profiles named here may run. No arbitrary shell / pytest discovery.
"""

from __future__ import annotations

from typing import Any


SANDBOX_TEST_REGISTRY: dict[str, dict[str, Any]] = {
    "smoke": {
        "id": "smoke",
        "description": "Sandbox layout + written-file path/UTF-8 checks",
        "runs_shell": False,
        "includes": ["layout", "written_files"],
    },
    "syntax": {
        "id": "syntax",
        "description": "Smoke + ast.parse for written .py files",
        "runs_shell": False,
        "includes": ["layout", "written_files", "python_syntax"],
    },
    "unit": {
        "id": "unit",
        "description": (
            "Syntax + registered in-process unit hooks "
            "(Digits entrypoint compile; forbidden-import scan)"
        ),
        "runs_shell": False,
        "includes": [
            "layout",
            "written_files",
            "python_syntax",
            "registered_unit_hooks",
        ],
    },
    "mock_experiment": {
        "id": "mock_experiment",
        "description": "Syntax + deterministic mock note / marker checks",
        "runs_shell": False,
        "includes": [
            "layout",
            "written_files",
            "python_syntax",
            "mock_experiment",
        ],
    },
}


def list_sandbox_test_profiles() -> list[dict[str, Any]]:
    return [dict(SANDBOX_TEST_REGISTRY[k]) for k in sorted(SANDBOX_TEST_REGISTRY)]


def require_sandbox_test_profile(profile: str) -> str:
    name = (profile or "").strip()
    if name not in SANDBOX_TEST_REGISTRY:
        allowed = ", ".join(sorted(SANDBOX_TEST_REGISTRY))
        raise ValueError(
            f"unknown sandbox test profile {profile!r}; "
            f"allowed=[{allowed}] (registry only; no arbitrary shell)"
        )
    return name
