"""Path / file allow-deny policy for restricted patches (v1.6.3)."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import PurePosixPath


DEFAULT_ALLOWED_PREFIXES: tuple[str, ...] = (
    "experiment_apps/rgbt_detection_real/adapters/",
    "experiment_apps/rgbt_detection_real/configs/",
    "src/scientist_lab/tasks/rgbt_detection/",
)

DEFAULT_DENIED_PREFIXES: tuple[str, ...] = (
    ".venv/",
    ".git/",
    ".github/",
    "src/scientist_lab/llm/",
)

DEFAULT_DENIED_NAMES: tuple[str, ...] = (
    "Dockerfile",
    "Dockerfile.*",
    "docker-compose.yml",
    "docker-compose.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-*.txt",
    ".env",
    ".env.*",
)

DEFAULT_DENIED_SUFFIXES: tuple[str, ...] = (
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
)

DEFAULT_DENIED_SUBSTRINGS: tuple[str, ...] = (
    "credentials",
    "secret",
    "api_key",
)


@dataclass(frozen=True)
class PathPolicy:
    allowed_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_PREFIXES
    denied_prefixes: tuple[str, ...] = DEFAULT_DENIED_PREFIXES
    denied_names: tuple[str, ...] = DEFAULT_DENIED_NAMES
    denied_suffixes: tuple[str, ...] = DEFAULT_DENIED_SUFFIXES
    denied_substrings: tuple[str, ...] = DEFAULT_DENIED_SUBSTRINGS
    extra_denied_paths: tuple[str, ...] = (
        "src/scientist_lab/llm/config.py",
    )

    def normalize(self, path: str) -> str:
        text = (path or "").strip().replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        return text

    def is_absolute(self, path: str) -> bool:
        text = self.normalize(path)
        if not text:
            return False
        if text.startswith("/") or text.startswith("~"):
            return True
        # Windows drive
        if len(text) >= 2 and text[1] == ":":
            return True
        return os.path.isabs(path)

    def has_escape(self, path: str) -> bool:
        parts = PurePosixPath(self.normalize(path)).parts
        return any(part == ".." for part in parts)

    def is_denied_name(self, path: str) -> bool:
        name = PurePosixPath(self.normalize(path)).name
        for pattern in self.denied_names:
            if fnmatch.fnmatch(name, pattern):
                return True
        low = name.lower()
        for token in self.denied_substrings:
            if token in low:
                return True
        for suffix in self.denied_suffixes:
            if low.endswith(suffix):
                return True
        return False

    def is_allowed(self, path: str) -> tuple[bool, str]:
        """Return (ok, reason_if_not)."""
        norm = self.normalize(path)
        if not norm:
            return False, "empty path"
        if self.is_absolute(norm):
            return False, "absolute path forbidden"
        if self.has_escape(norm):
            return False, "path escape (..) forbidden"
        if norm in self.extra_denied_paths:
            return False, f"explicitly denied path: {norm}"
        for prefix in self.denied_prefixes:
            if norm.startswith(prefix) or norm == prefix.rstrip("/"):
                return False, f"denied prefix: {prefix}"
        if self.is_denied_name(norm):
            return False, f"denied filename/pattern: {norm}"
        for prefix in self.allowed_prefixes:
            if norm.startswith(prefix):
                return True, ""
        return False, "path not under allowed prefixes"
