"""Literature retrieval errors. Never include API keys in messages."""

from __future__ import annotations

from scientist_lab.llm.errors import (
    LLMConfigurationError,
    MissingAPIKeyError,
    RealProviderNotEnabledError,
    UnsupportedProviderError,
)


class LiteratureError(Exception):
    """LiteratureRetriever refused a call."""


class LiteratureGateError(LiteratureError):
    """Paper failed Literature Evidence Gate (not ClaimGate)."""


class LiteratureProviderError(LiteratureError):
    """Provider HTTP / parse failure."""


__all__ = [
    "LLMConfigurationError",
    "LiteratureError",
    "LiteratureGateError",
    "LiteratureProviderError",
    "MissingAPIKeyError",
    "RealProviderNotEnabledError",
    "UnsupportedProviderError",
]
