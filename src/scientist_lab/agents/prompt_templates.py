"""Prompt template loading with version stamps."""

from __future__ import annotations

from pathlib import Path


def prompts_root() -> Path:
    # scientist-lab/prompts relative to package parents[2] == project root
    return Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> tuple[str, str]:
    """Return (prompt_text, prompt_version)."""
    path = prompts_root() / name
    if not path.is_file():
        return (f"# missing prompt: {name}\n", f"missing:{name}")
    text = path.read_text(encoding="utf-8")
    version = path.stem
    return text, version
