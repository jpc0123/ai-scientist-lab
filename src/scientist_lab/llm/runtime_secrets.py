"""Local LLM runtime secrets (never YAML / DB / outputs).

Stored under ``{runtime_dir}/llm_secrets.env`` (gitignored via ``runtime/*``).
Applied into ``os.environ`` so ``load_llm_config()`` picks them up without restart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scientist_lab.llm.config import load_llm_config

ENV_PROVIDER = "LLM_PROVIDER"
ENV_BASE_URL = "LLM_BASE_URL"
ENV_API_KEY = "LLM_API_KEY"
ENV_MODEL = "LLM_MODEL"
ENV_TIMEOUT = "LLM_TIMEOUT_SECONDS"
ENV_ALLOW_NETWORK = "LLM_ALLOW_NETWORK"

_MANAGED_KEYS = (
    ENV_PROVIDER,
    ENV_BASE_URL,
    ENV_API_KEY,
    ENV_MODEL,
    ENV_TIMEOUT,
    ENV_ALLOW_NETWORK,
)

SECRETS_FILENAME = "llm_secrets.env"


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
        "# Scientist Lab LLM runtime secrets — DO NOT COMMIT",
        "# Written by Web/API model config. Keys never returned by GET APIs.",
        "",
    ]
    for key in _MANAGED_KEYS:
        if key in values and values[key] != "":
            # Keep value on one line; escape newlines.
            val = str(values[key]).replace("\n", "").replace("\r", "")
            lines.append(f"{key}={val}")
    lines.append("")
    return "\n".join(lines)


def apply_runtime_llm_env(runtime_dir: Path) -> dict[str, str]:
    """Load runtime secrets file into ``os.environ`` (overlay).

    Returns the keys that were applied (values never include raw api key in
    return — only key names).
    """
    path = secrets_path(runtime_dir)
    parsed = _parse_env_file(path)
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        os.environ[key] = value
        applied[key] = "[set]" if key == ENV_API_KEY else value
    return applied


def read_runtime_llm_values(runtime_dir: Path) -> dict[str, str]:
    """Read file contents (may include api key — caller must not log/return)."""
    return _parse_env_file(secrets_path(runtime_dir))


def write_runtime_llm_env(
    runtime_dir: Path,
    *,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    api_key: str | None = None,
    allow_network: bool | None = None,
    clear_api_key: bool = False,
) -> Path:
    """Merge updates into secrets file and apply to ``os.environ``.

    - ``api_key=None`` and not ``clear_api_key`` → keep existing key
    - ``clear_api_key=True`` → remove key from file and environ
    """
    root = Path(runtime_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = secrets_path(root)
    current = _parse_env_file(path)

    if provider is not None:
        current[ENV_PROVIDER] = str(provider).strip() or "mock"
    if base_url is not None:
        text = str(base_url).strip()
        if text:
            current[ENV_BASE_URL] = text
        else:
            current.pop(ENV_BASE_URL, None)
            os.environ.pop(ENV_BASE_URL, None)
    if model is not None:
        text = str(model).strip()
        if text:
            current[ENV_MODEL] = text
        else:
            current.pop(ENV_MODEL, None)
            os.environ.pop(ENV_MODEL, None)
    if timeout_seconds is not None:
        current[ENV_TIMEOUT] = str(float(timeout_seconds))
    if clear_api_key:
        current.pop(ENV_API_KEY, None)
        os.environ.pop(ENV_API_KEY, None)
    elif api_key is not None and str(api_key).strip():
        current[ENV_API_KEY] = str(api_key).strip()
    if allow_network is not None:
        current[ENV_ALLOW_NETWORK] = "true" if allow_network else "false"

    path.write_text(_format_env_file(current), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    for key, value in current.items():
        os.environ[key] = value
    return path


def llm_config_status(
    *,
    runtime_dir: Path,
    default_profile_id: str | None = None,
) -> dict[str, Any]:
    """Masked status for GET APIs (never includes raw api_key)."""
    apply_runtime_llm_env(runtime_dir)
    cfg = load_llm_config(require_real=False)
    allow = str(os.environ.get(ENV_ALLOW_NETWORK) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    file_exists = secrets_path(runtime_dir).is_file()
    payload = cfg.safe_dict()
    missing: list[str] = []
    if cfg.is_real:
        if not cfg.api_key_present:
            missing.append("LLM_API_KEY")
        if not cfg.base_url:
            missing.append("LLM_BASE_URL")
        if not cfg.model:
            missing.append("LLM_MODEL")
    payload.update(
        {
            "allow_network": allow,
            "secrets_file_present": file_exists,
            "secrets_file": SECRETS_FILENAME,
            "default_profile_id": default_profile_id,
            "ready_for_real_calls": bool(cfg.is_real and allow and not missing),
            "missing_for_real": missing,
            "notes": [
                "API key is never returned by this endpoint.",
                "Real calls still require explicit allow_network on the action "
                "plus LLM_ALLOW_NETWORK=true.",
                "Default provider remains mock until you save openai-compatible.",
            ],
        }
    )
    return payload
