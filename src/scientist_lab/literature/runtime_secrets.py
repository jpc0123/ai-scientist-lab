"""Local literature runtime secrets (never YAML / DB / outputs).

Stored under ``{runtime_dir}/literature_secrets.env`` (gitignored via ``runtime/*``).
Independent of ``LLM_API_KEY`` — Semantic Scholar uses ``SEMANTIC_SCHOLAR_API_KEY``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from scientist_lab.literature.factory import ENV_S2_KEY
from scientist_lab.literature.semantic_scholar import S2_DEFAULT_BASE

ENV_BASE_URL = "SEMANTIC_SCHOLAR_BASE_URL"

_MANAGED_KEYS = (ENV_S2_KEY, ENV_BASE_URL)
SECRETS_FILENAME = "literature_secrets.env"


def secrets_path(runtime_dir: Path) -> Path:
    return Path(runtime_dir) / SECRETS_FILENAME


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key in _MANAGED_KEYS:
            out[key] = value
    return out


def _format_env_file(values: dict[str, str]) -> str:
    lines = [
        "# Scientist Lab literature runtime secrets — DO NOT COMMIT",
        "# Written by Web/API literature config. Keys never returned by GET APIs.",
        "# This is NOT LLM_API_KEY. Live search needs SEMANTIC_SCHOLAR_API_KEY.",
        "",
    ]
    for key in _MANAGED_KEYS:
        if key in values and values[key] != "":
            val = str(values[key]).replace("\n", "").replace("\r", "")
            lines.append(f"{key}={val}")
    lines.append("")
    return "\n".join(lines)


def apply_runtime_literature_env(runtime_dir: Path) -> dict[str, str]:
    path = secrets_path(runtime_dir)
    parsed = _parse_env_file(path)
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        os.environ[key] = value
        applied[key] = "[set]" if key == ENV_S2_KEY else value
    return applied


def write_runtime_literature_env(
    runtime_dir: Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    clear_api_key: bool = False,
) -> Path:
    root = Path(runtime_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = secrets_path(root)
    current = _parse_env_file(path)

    if clear_api_key:
        current.pop(ENV_S2_KEY, None)
        os.environ.pop(ENV_S2_KEY, None)
    elif api_key is not None and str(api_key).strip():
        current[ENV_S2_KEY] = str(api_key).strip()
    if base_url is not None:
        text = str(base_url).strip()
        if text:
            current[ENV_BASE_URL] = text
        else:
            current.pop(ENV_BASE_URL, None)
            os.environ.pop(ENV_BASE_URL, None)

    path.write_text(_format_env_file(current), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    for key, value in current.items():
        os.environ[key] = value
    return path


def _fingerprint(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return digest[:12]


def literature_config_status(*, runtime_dir: Path) -> dict[str, Any]:
    apply_runtime_literature_env(runtime_dir)
    file_values = _parse_env_file(secrets_path(runtime_dir))
    key = str(os.environ.get(ENV_S2_KEY) or file_values.get(ENV_S2_KEY) or "").strip()
    base = str(
        os.environ.get(ENV_BASE_URL)
        or file_values.get(ENV_BASE_URL)
        or S2_DEFAULT_BASE
    ).strip()
    present = bool(key)
    return {
        "provider": "semantic_scholar" if present else "fake",
        "live_default": False,
        "api_key_present": present,
        "api_key_fingerprint": _fingerprint(key) if present else "",
        "base_url": base,
        "secrets_file_present": secrets_path(runtime_dir).is_file(),
        "secrets_file": SECRETS_FILENAME,
        "ready_for_live_search": present,
        "missing_for_live": [] if present else [ENV_S2_KEY],
        "notes": [
            "SEMANTIC_SCHOLAR_API_KEY is independent of LLM_API_KEY.",
            "API key is never returned by this endpoint.",
            "Without a key, LiteratureRetriever stays on the fake offline corpus.",
            "Live search still requires an explicit --live / live=True on the call.",
            "LiteratureEvidence is not ExperimentEvidence and cannot enter ClaimGate.",
        ],
        "signup_url": "https://www.semanticscholar.org/product/api",
    }
