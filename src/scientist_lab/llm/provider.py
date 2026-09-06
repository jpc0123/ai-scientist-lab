"""LLMProvider protocol and request fingerprinting."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from scientist_lab.llm.models import LLMRequest, LLMResponse


def request_fingerprint(request: LLMRequest | dict[str, Any]) -> str:
    """Stable hash used as the ReplayProvider lookup key."""
    if isinstance(request, LLMRequest):
        payload = {
            "purpose": request.purpose,
            "messages": request.messages,
            "response_schema": request.response_schema,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
    else:
        payload = {
            "purpose": request.get("purpose"),
            "messages": request.get("messages"),
            "response_schema": request.get("response_schema"),
            "temperature": request.get("temperature", 0.0),
            "max_tokens": request.get("max_tokens"),
        }
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@runtime_checkable
class LLMProvider(Protocol):
    """Unified provider surface. Implementations must not require network by default."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class BaseLLMProvider(ABC):
    """Optional ABC helper for concrete providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
