"""Retry policy for OpenAI-compatible LLM calls (v1.4.3)."""

from __future__ import annotations

import random
from typing import Callable

from pydantic import BaseModel, Field

from scientist_lab.llm.errors import LLMError


class RetryPolicy(BaseModel):
    max_retries: int = Field(default=1, ge=0, le=5)
    base_delay_seconds: float = Field(default=1.0, ge=0.0, le=30)
    max_delay_seconds: float = Field(default=20.0, ge=0.1, le=120)
    jitter: bool = True

    def delay_seconds(self, attempt: int) -> float:
        """attempt is 0-based index of the retry about to happen."""
        base = float(self.base_delay_seconds) * (2 ** max(0, int(attempt)))
        delay = min(float(self.max_delay_seconds), base)
        if self.jitter:
            delay += random.uniform(0.0, min(0.25, delay * 0.1 + 0.01))
        return delay

    def should_retry(self, exc: BaseException, *, attempt: int) -> bool:
        """attempt is the failed attempt index (0 = first try failed)."""
        if attempt >= int(self.max_retries):
            return False
        if isinstance(exc, LLMError):
            return bool(exc.retryable)
        return False


def run_with_retries(
    fn: Callable[[], object],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] | None = None,
    on_attempt: Callable[[int, BaseException | None], None] | None = None,
) -> object:
    """Execute ``fn`` with retry. ``on_attempt(attempt_index, error_or_None)``."""
    sleeper = sleep or (lambda _s: None)
    last_exc: BaseException | None = None
    for attempt in range(int(policy.max_retries) + 1):
        try:
            result = fn()
            if on_attempt is not None:
                on_attempt(attempt, None)
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if on_attempt is not None:
                on_attempt(attempt, exc)
            if not policy.should_retry(exc, attempt=attempt):
                raise
            sleeper(policy.delay_seconds(attempt))
    assert last_exc is not None
    raise last_exc
