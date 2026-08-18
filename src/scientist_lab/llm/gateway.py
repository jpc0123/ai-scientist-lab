"""LLM Gateway facade over the existing provider layer.

Not a fifth Agent. Planner/Reviewer call this as a cognitive backend.
Reuses FakeProvider / OpenAICompatibleProvider / ReplayProvider via factory.
Does not write Memory, KEEP/DISCARD, or ClaimGate verdicts.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

from scientist_lab.domain.models import new_id
from scientist_lab.llm.audit import build_usage_from_messages
from scientist_lab.llm.config import load_llm_config, redact_secrets
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.factory import create_llm_provider
from scientist_lab.llm.models import LLMRequest, LLMResponse
from scientist_lab.llm.openai_config import openai_config_from_llm_config
from scientist_lab.llm.provider import BaseLLMProvider, LLMProvider, request_fingerprint
from scientist_lab.llm.schema_parser import parse_and_validate


class GatewayError(ValueError):
    """Gateway refused to complete a call (config / live gate / empty)."""


class ScriptedProvider(BaseLLMProvider):
    """Deterministic scripted completions for fail-closed tests. No network."""

    def __init__(
        self,
        content: str | Callable[[LLMRequest], str],
        *,
        model: str = "scripted-llm-v1",
    ) -> None:
        self._content = content
        self._model = model

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        raw = self._content(request) if callable(self._content) else str(self._content)
        parsed: dict[str, Any] | None = None
        errors: list[str] = []
        try:
            parsed, errors = parse_and_validate(raw, request.response_schema)
        except (ValueError, json.JSONDecodeError) as exc:
            errors = [str(exc)]
            parsed = None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LLMResponse(
            request_id=new_id("llmreq"),
            content=raw,
            parsed_json=parsed,
            usage=build_usage_from_messages(request.messages, raw),
            latency_ms=elapsed_ms,
            provider=self.name,
            model=self.model,
            request_fingerprint=request_fingerprint(request),
            schema_valid=bool(parsed) and not errors,
            schema_errors=errors,
            created_at=datetime.now(timezone.utc).replace(microsecond=0),
        )


def resolve_gateway_provider(
    *,
    live: bool = False,
    provider: str | None = None,
    allow_network: bool = False,
    environ: dict[str, str] | None = None,
    openai_config: Any = None,
    transport: Any = None,
) -> LLMProvider:
    """Resolve the existing provider stack. live=True requires env API key."""
    if live:
        cfg = load_llm_config(
            environ=environ,
            provider=provider or "openai-compatible",
            require_real=True,
        )
        if not cfg.api_key_present:
            raise MissingAPIKeyError(
                "LLM_API_KEY is required for live Gateway calls; key stays in env"
            )
        oai = openai_config or openai_config_from_llm_config(
            base_url=str(cfg.base_url or ""),
            api_key=str(cfg.api_key_value() or ""),
            model=str(cfg.model or ""),
            allow_network=True,
            timeout_seconds=float(cfg.timeout_seconds),
        )
        return create_llm_provider(
            "openai-compatible",
            allow_network=True,
            environ=environ,
            openai_config=oai,
            transport=transport,
        )

    name = (provider or "mock").strip().lower()
    if name in {"real", "openai", "openai-compatible", "openai_compatible"}:
        raise RealProviderNotEnabledError(
            "real Gateway provider requires live=True (and LLM_API_KEY); "
            "default mock path stays offline"
        )
    return create_llm_provider(
        name,
        allow_network=allow_network,
        environ=environ,
        openai_config=openai_config,
        transport=transport,
    )


def complete_chat(
    request: LLMRequest,
    *,
    provider: LLMProvider | None = None,
    live: bool = False,
    provider_name: str | None = None,
    environ: dict[str, str] | None = None,
    transport: Any = None,
) -> LLMResponse:
    """One chat completion through the existing provider layer. No GPU."""
    impl = provider or resolve_gateway_provider(
        live=live,
        provider=provider_name,
        environ=environ,
        transport=transport,
    )
    response = impl.complete(request)
    if response.content:
        meta = dict(response.metadata or {})
        meta["raw_redacted"] = redact_secrets(response.content)
        return response.model_copy(update={"metadata": meta})
    return response
