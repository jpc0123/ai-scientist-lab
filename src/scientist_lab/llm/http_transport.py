"""HTTP transport abstraction for OpenAI-compatible providers (v1.4.2).

Default tests use MockTransport — never opens network sockets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from scientist_lab.llm.config import redact_secrets
from scientist_lab.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMPermissionError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)


@dataclass
class HttpResponse:
    status_code: int
    body: str
    headers: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@runtime_checkable
class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        timeout_seconds: float = 60.0,
        connect_timeout_seconds: float = 10.0,
    ) -> HttpResponse: ...


def map_http_error(status_code: int, body: str) -> Exception | None:
    """Map HTTP status to a typed LLMError, or None if 2xx."""
    if 200 <= status_code < 300:
        return None
    redacted = redact_secrets((body or "")[:500], placeholder="[REDACTED]")
    if status_code == 401:
        return LLMAuthenticationError(f"authentication failed (HTTP 401): {redacted}")
    if status_code == 403:
        return LLMPermissionError(f"permission denied (HTTP 403): {redacted}")
    if status_code == 404:
        return LLMInvalidRequestError(f"endpoint or model not found (HTTP 404): {redacted}")
    if status_code == 408:
        return LLMTimeoutError(f"request timeout (HTTP 408): {redacted}")
    if status_code == 429:
        return LLMRateLimitError(f"rate limited (HTTP 429): {redacted}")
    if status_code == 400:
        return LLMInvalidRequestError(f"invalid request (HTTP 400): {redacted}")
    if 500 <= status_code <= 599:
        return LLMServerError(f"provider server error (HTTP {status_code}): {redacted}")
    return LLMInvalidRequestError(f"unexpected HTTP {status_code}: {redacted}")


class MockTransport:
    """In-memory HTTP transport for offline unit tests."""

    def __init__(
        self,
        *,
        responses: list[HttpResponse] | None = None,
        default_response: HttpResponse | None = None,
    ) -> None:
        self._queue = list(responses or [])
        self.default_response = default_response
        self.calls: list[dict[str, Any]] = []

    def enqueue(self, response: HttpResponse) -> None:
        self._queue.append(response)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        timeout_seconds: float = 60.0,
        connect_timeout_seconds: float = 10.0,
    ) -> HttpResponse:
        # Never log Authorization values.
        safe_headers = {
            k: ("[REDACTED]" if k.lower() in {"authorization", "proxy-authorization", "x-api-key", "cookie"} else v)
            for k, v in (headers or {}).items()
        }
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": safe_headers,
                "json_body": json_body,
                "timeout_seconds": timeout_seconds,
                "connect_timeout_seconds": connect_timeout_seconds,
            }
        )
        if self._queue:
            return self._queue.pop(0)
        if self.default_response is not None:
            return self.default_response
        raise LLMConnectionError("MockTransport has no queued responses")


class HttpxTransport:
    """Real httpx transport. Only construct when allow_network is True."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        timeout_seconds: float = 60.0,
        connect_timeout_seconds: float = 10.0,
    ) -> HttpResponse:
        import time

        import httpx

        started = time.perf_counter()
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=connect_timeout_seconds,
        )
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                resp = client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"HTTP timeout: {type(exc).__name__}") from None
        except httpx.TransportError as exc:
            raise LLMConnectionError(
                f"HTTP connection error: {type(exc).__name__}"
            ) from None

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        header_map = {str(k): str(v) for k, v in resp.headers.items()}
        return HttpResponse(
            status_code=int(resp.status_code),
            body=resp.text or "",
            headers=header_map,
            elapsed_ms=elapsed_ms,
        )
