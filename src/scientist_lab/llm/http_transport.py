"""HTTP transport abstraction for OpenAI-compatible providers (v1.4.2).

Default tests use MockTransport — never opens network sockets.
TLS stays verified by default. Corporate/self-signed CAs must be trusted
via SSL_CERT_FILE / REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE / LLM_CA_BUNDLE
or the Windows ROOT store — never by disabling verification.
"""

from __future__ import annotations

import os
import ssl
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

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

# Prefer explicit CA bundles, then OS trust, then httpx/certifi.
_CA_BUNDLE_ENVS = (
    "LLM_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)
_SERVER_AUTH_OID = "1.3.6.1.5.5.7.3.1"


def _ca_bundle_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    env = environ if environ is not None else os.environ
    for key in _CA_BUNDLE_ENVS:
        raw = str(env.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if path.is_file():
            return str(path), key
    return None


def _windows_root_ssl_context() -> ssl.SSLContext | None:
    """Trust the Windows ROOT store (corporate MITM CAs live here).

    Does not disable hostname checks. Returns None when the store is unused.
    """
    if sys.platform != "win32" or not hasattr(ssl, "enum_certificates"):
        return None
    try:
        entries = ssl.enum_certificates("ROOT")
    except OSError:
        return None
    pem_parts: list[str] = []
    for der, encoding, trust in entries:
        if encoding != "x509_asn":
            continue
        if trust is not True:
            if not (isinstance(trust, (set, frozenset)) and _SERVER_AUTH_OID in trust):
                continue
        try:
            pem_parts.append(ssl.DER_cert_to_PEM_cert(der))
        except (ValueError, TypeError, ssl.SSLError):
            continue
    if not pem_parts:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(cadata="".join(pem_parts))
    return ctx


def resolve_tls_verify(
    *,
    verify_tls: bool = True,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool | str | ssl.SSLContext, str]:
    """Return httpx ``verify=`` value and a non-secret source label.

    Default remains verified. ``verify_tls=False`` is never implied by missing
    env; callers must pass it explicitly.

    Trust order:
    1. ``LLM_CA_BUNDLE`` (explicit project CA)
    2. Windows ROOT store (corporate MITM CAs live here; conda ``SSL_CERT_FILE``
       must not hide them)
    3. ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` / ``CURL_CA_BUNDLE``
    4. httpx/certifi default
    """
    if not verify_tls:
        return False, "verify_tls_disabled"
    env = environ if environ is not None else os.environ
    llm_raw = str(env.get("LLM_CA_BUNDLE") or "").strip()
    if llm_raw:
        path = Path(llm_raw)
        if path.is_file():
            return str(path), "LLM_CA_BUNDLE"
    windows_ctx = _windows_root_ssl_context()
    if windows_ctx is not None:
        return windows_ctx, "windows_root_store"
    bundle = _ca_bundle_from_env(environ)
    if bundle is not None:
        return bundle[0], bundle[1]
    return True, "default"


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

    def __init__(self, *, verify: bool | str | ssl.SSLContext | None = None) -> None:
        if verify is None:
            verify, source = resolve_tls_verify()
        else:
            source = "explicit"
        self.verify = verify
        self.verify_source = source

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
            with httpx.Client(
                timeout=timeout,
                follow_redirects=False,
                verify=self.verify,
            ) as client:
                resp = client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"HTTP timeout: {type(exc).__name__}") from None
        except httpx.TransportError as exc:
            detail = redact_secrets(str(exc)[:400])
            raise LLMConnectionError(
                f"HTTP connection error: {type(exc).__name__}: {detail} "
                f"(tls_verify_source={self.verify_source})"
            ) from None

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        header_map = {str(k): str(v) for k, v in resp.headers.items()}
        return HttpResponse(
            status_code=int(resp.status_code),
            body=resp.text or "",
            headers=header_map,
            elapsed_ms=elapsed_ms,
        )
