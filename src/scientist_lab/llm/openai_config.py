"""OpenAI-compatible config helpers: URL normalize, header whitelist (v1.4.2)."""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from scientist_lab.llm.errors import (
    InvalidLLMConfigError,
    MissingAPIKeyError,
    MissingBaseURLError,
    MissingModelError,
)


ApiMode = Literal["chat_completions", "responses"]

_BLOCKED_EXTRA_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
    }
)
_CRLF = re.compile(r"[\r\n]")


def normalize_base_url(url: str, *, allow_http_localhost: bool = True) -> str:
    text = (url or "").strip()
    if not text:
        raise InvalidLLMConfigError("LLM_BASE_URL must not be empty")
    # Reject secrets embedded in URL query.
    lower = text.lower()
    if "api_key=" in lower or "access_token=" in lower or "token=" in lower:
        raise InvalidLLMConfigError("LLM_BASE_URL must not contain API key query params")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        raise InvalidLLMConfigError(
            f"LLM_BASE_URL scheme must be http or https, got {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise InvalidLLMConfigError("LLM_BASE_URL must be an absolute URL")
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http":
        if not allow_http_localhost or host not in {"localhost", "127.0.0.1", "::1"}:
            raise InvalidLLMConfigError(
                "LLM_BASE_URL must use https except for localhost/127.0.0.1"
            )
    # Strip trailing slashes from path (keep empty path as "").
    path = (parsed.path or "").rstrip("/")
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        # Drop query entirely for safety; keys in query already rejected above.
        raise InvalidLLMConfigError("LLM_BASE_URL must not include a query string")
    return normalized


def join_api_url(base_url: str, path: str) -> str:
    base = normalize_base_url(base_url)
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def build_auth_headers(
    *,
    api_key: str,
    organization: str | None = None,
    project: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    if not api_key:
        raise MissingAPIKeyError("API key required to build Authorization header")
    if _CRLF.search(api_key):
        raise InvalidLLMConfigError("API key must not contain newline characters")

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if organization:
        if _CRLF.search(organization):
            raise InvalidLLMConfigError("organization header value invalid")
        headers["OpenAI-Organization"] = organization
    if project:
        if _CRLF.search(project):
            raise InvalidLLMConfigError("project header value invalid")
        headers["OpenAI-Project"] = project

    for key, value in (extra_headers or {}).items():
        name = str(key)
        if _CRLF.search(name) or _CRLF.search(str(value)):
            raise InvalidLLMConfigError("extra_headers must not contain CR/LF")
        if name.lower() in _BLOCKED_EXTRA_HEADERS:
            raise InvalidLLMConfigError(
                f"extra_headers must not override protected header: {name}"
            )
        headers[name] = str(value)
    return headers


class OpenAICompatibleConfig(BaseModel):
    """Full real-provider settings (extends v1.4.1 LLMConfig fields)."""

    provider: Literal["openai-compatible"] = "openai-compatible"
    base_url: str
    api_key: SecretStr
    model: str
    api_mode: ApiMode = "chat_completions"
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    connect_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_retries: int = Field(default=1, ge=0, le=5)
    retry_base_seconds: float = Field(default=1.0, ge=0.1, le=30)
    retry_max_seconds: float = Field(default=20.0, ge=1, le=120)
    max_concurrency: int = Field(default=1, ge=1, le=16)
    default_max_output_tokens: int = Field(default=2048, ge=1, le=65536)
    allow_network: bool = False
    verify_tls: bool = True
    organization: str | None = None
    project: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def _normalize_url(cls, value: str) -> str:
        return normalize_base_url(value)

    @model_validator(mode="after")
    def _require_secrets(self) -> OpenAICompatibleConfig:
        if not self.api_key.get_secret_value():
            raise MissingAPIKeyError("api_key is required for openai-compatible")
        if not (self.model or "").strip():
            raise MissingModelError("model is required for openai-compatible")
        if not (self.base_url or "").strip():
            raise MissingBaseURLError("base_url is required for openai-compatible")
        return self

    def api_key_value(self) -> str:
        return self.api_key.get_secret_value()

    def request_headers(self) -> dict[str, str]:
        return build_auth_headers(
            api_key=self.api_key_value(),
            organization=self.organization,
            project=self.project,
            extra_headers=self.extra_headers,
        )

    def safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": "configured" if self.base_url else None,
            "api_key": "[REDACTED]",
            "model": self.model,
            "api_mode": self.api_mode,
            "timeout_seconds": self.timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "allow_network": self.allow_network,
            "max_retries": self.max_retries,
            "max_concurrency": self.max_concurrency,
        }

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleConfig("
            f"provider={self.provider!r}, "
            f"base_url={'configured' if self.base_url else None!r}, "
            f"model={self.model!r}, "
            f"api_mode={self.api_mode!r}, "
            f"allow_network={self.allow_network!r}, "
            "api_key='[REDACTED]'"
            ")"
        )

    __str__ = __repr__


def openai_config_from_llm_config(
    *,
    base_url: str,
    api_key: str,
    model: str,
    allow_network: bool = False,
    api_mode: ApiMode = "chat_completions",
    timeout_seconds: float = 60.0,
    connect_timeout_seconds: float = 10.0,
    **kwargs: Any,
) -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        base_url=base_url,
        api_key=SecretStr(api_key),
        model=model,
        allow_network=allow_network,
        api_mode=api_mode,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        **kwargs,
    )
