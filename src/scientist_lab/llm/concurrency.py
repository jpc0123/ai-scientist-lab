"""In-process concurrency gate for real LLM calls (v1.4.3)."""

from __future__ import annotations

import threading
import time
from typing import Any


from scientist_lab.llm.errors import LLMConcurrencyLimitError


class ConcurrencyGate:
    """Process-local semaphore. Default max_concurrency=1."""

    def __init__(self, max_concurrency: int = 1, *, wait_timeout_seconds: float = 30.0) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = int(max_concurrency)
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self._sem = threading.Semaphore(self.max_concurrency)
        self._active = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        ok = self._sem.acquire(timeout=self.wait_timeout_seconds)
        if not ok:
            raise LLMConcurrencyLimitError(
                f"LLM concurrency limit reached (max={self.max_concurrency}); "
                f"waited {self.wait_timeout_seconds}s"
            )
        with self._lock:
            self._active += 1

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
        self._sem.release()

    def __enter__(self) -> ConcurrencyGate:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.release()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_concurrency": self.max_concurrency,
                "active": self._active,
                "wait_timeout_seconds": self.wait_timeout_seconds,
            }


def sleep_interruptible(seconds: float) -> None:
    """Simple sleep helper (tests may monkeypatch)."""
    if seconds > 0:
        time.sleep(seconds)
