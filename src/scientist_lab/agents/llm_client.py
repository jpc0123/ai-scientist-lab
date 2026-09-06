"""Optional LLM client. Planner/Critic must work without a live model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LLMUnavailableError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str: ...


@dataclass
class NullLLMClient:
    """Default: LLM layer unavailable; rules/Mock continue to work."""

    reason: str = "LLM client not configured"

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        raise LLMUnavailableError(self.reason)


@dataclass
class ScriptedLLMClient:
    """Test helper: returns scripted responses (optionally failing once)."""

    responses: list[str]
    fail_first: bool = False
    _calls: int = 0

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        self._calls += 1
        if self.fail_first and self._calls == 1:
            raise LLMUnavailableError("forced first failure")
        if not self.responses:
            raise LLMUnavailableError("no scripted responses")
        return self.responses.pop(0)
