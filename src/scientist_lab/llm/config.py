"""LLM provider configuration and API-key boundary (v1.4.1).

Keys are never written into logs, repr, audit payloads, or usage reports.
Real-provider settings are validated only when a real provider is selected.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


LLMProviderName = Literal["mock", "fake", "replay", "openai-compatible"]
REAL_PROVIDERS: frozenset[str] = frozenset({"openai-compatible"})

_ENV_PROVIDER = "LLM_PROVIDER"
_ENV_BASE_URL = "LLM_BASE_URL"
_ENV_API_KEY = "LLM_API_KEY"
_ENV_MODEL = "LLM_MODEL"
_ENV_TIMEOUT = "LLM_TIMEOUT_SECONDS"

# Patterns that may appear if a raw key leaks into text.
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*([^\s,;\"']+)"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"\bsk-[A-Za-z0-9\-_]{8,}\b"),
    re.compile(r'(?i)"(api[_-]?key|access_token|token)"\s*:\s*"[^"]+"'),
)


from scientist_lab.llm.errors import (
    InvalidLLMConfigError,
    MissingAPIKeyError,
)

# Re-export for v1.4.1 call sites.
__all_config_errors__ = ("MissingAPIKeyError", "InvalidLLMConfigError")


def redact_secrets(text: str, *, placeholder: str = "[REDACTED]") -> str:
    """Best-effort scrub of API keys / bearer tokens from free text."""
    if not text:
        return text
    out = text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(
            lambda m: (
                f"{m.group(1)}={placeholder}"
                if m.lastindex and m.lastindex >= 2
                else placeholder
            ),
            out,
        )
    return out


def mask_secret(value: str | None, *, visible: int = 0) -> str:
    """Never reveal key prefixes/suffixes (v1.4 security policy)."""
    if not value:
        return ""
    return "[REDACTED]"


class LLMConfig(BaseModel):
    """Runtime LLM settings. Default provider is always mock."""

    provider: LLMProviderName = "mock"
    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0.0)

    # Optional offline modes keep working without real config.
    # Real validation is opt-in via require_real_ready().

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any) -> str:
        text = str(value or "mock").strip().lower()
        aliases = {
            "openai": "openai-compatible",
            "openai_compatible": "openai-compatible",
            "openai-compat": "openai-compatible",
        }
        return aliases.get(text, text)

    @model_validator(mode="after")
    def _strip_empty(self) -> LLMConfig:
        if self.base_url is not None and not self.base_url.strip():
            self.base_url = None
        if self.model is not None and not self.model.strip():
            self.model = None
        return self

    @property
    def is_real(self) -> bool:
        return self.provider in REAL_PROVIDERS

    @property
    def api_key_present(self) -> bool:
        if self.api_key is None:
            return False
        return bool(self.api_key.get_secret_value())

    def api_key_value(self) -> str | None:
        if self.api_key is None:
            return None
        value = self.api_key.get_secret_value()
        return value or None

    def require_real_ready(self) -> None:
        """Validate real-provider config. No-op for mock/fake/replay."""
        if not self.is_real:
            return
        if not self.api_key_present:
            raise MissingAPIKeyError(
                "LLM_API_KEY is required when LLM_PROVIDER="
                f"{self.provider!r}. Set the environment variable or pass "
                "api_key explicitly. The key is never logged."
            )
        if not self.base_url:
            raise InvalidLLMConfigError(
                f"LLM_BASE_URL is required when LLM_PROVIDER={self.provider!r}"
            )
        if not self.model:
            raise InvalidLLMConfigError(
                f"LLM_MODEL is required when LLM_PROVIDER={self.provider!r}"
            )

    def safe_dict(self) -> dict[str, Any]:
        """Serializable view with the API key masked (never raw)."""
        raw = self.api_key_value()
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "api_key_present": self.api_key_present,
            "api_key_fingerprint": mask_secret(raw) if raw else "",
            "is_real": self.is_real,
        }

    def __repr__(self) -> str:
        return (
            "LLMConfig("
            f"provider={self.provider!r}, "
            f"base_url={self.base_url!r}, "
            f"model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"api_key={'[REDACTED]' if self.api_key_present else None}"
            ")"
        )

    __str__ = __repr__


def load_llm_config(
    *,
    environ: dict[str, str] | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    require_real: bool | None = None,
) -> LLMConfig:
    """Load config from env + optional explicit overrides.

    Default provider is mock. Real-provider readiness is checked only when
    ``require_real`` is True, or when the resolved provider is real and
    ``require_real`` is left as None (auto).
    """
    env = environ if environ is not None else dict(os.environ)
    resolved_provider = (
        provider
        if provider is not None
        else env.get(_ENV_PROVIDER, "mock")
    )
    resolved_base = (
        base_url if base_url is not None else env.get(_ENV_BASE_URL) or None
    )
    resolved_key = api_key if api_key is not None else env.get(_ENV_API_KEY) or None
    resolved_model = model if model is not None else env.get(_ENV_MODEL) or None
    if timeout_seconds is not None:
        resolved_timeout = float(timeout_seconds)
    else:
        raw_timeout = env.get(_ENV_TIMEOUT)
        resolved_timeout = float(raw_timeout) if raw_timeout else 60.0

    config = LLMConfig(
        provider=resolved_provider,  # type: ignore[arg-type]
        base_url=resolved_base,
        api_key=SecretStr(resolved_key) if resolved_key else None,
        model=resolved_model,
        timeout_seconds=resolved_timeout,
    )

    should_require = require_real
    if should_require is None:
        should_require = config.is_real
    if should_require:
        config.require_real_ready()
    return config
