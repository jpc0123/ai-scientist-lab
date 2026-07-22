"""OpenAICompatibleProvider — chat_completions / responses via HttpTransport (v1.4.2).

Never constructs a real network client unless allow_network=True and an
HttpTransport is supplied (or HttpxTransport when explicitly enabled).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from scientist_lab.domain.models import new_id
from scientist_lab.llm.errors import (
    LLMEmptyResponseError,
    RealProviderNotEnabledError,
    StructuredOutputValidationError,
    UnsupportedAPIModeError,
)
from scientist_lab.llm.http_transport import (
    HttpTransport,
    HttpxTransport,
    map_http_error,
)
from scientist_lab.llm.models import LLMRequest, LLMResponse, TokenUsage
from scientist_lab.llm.openai_config import OpenAICompatibleConfig, join_api_url
from scientist_lab.llm.provider import BaseLLMProvider, request_fingerprint
from scientist_lab.llm.schema_parser import parse_and_validate


def _message_dicts(request: LLMRequest) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in request.messages:
        role = str(item.get("role") or "user")
        content = item.get("content")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        out.append({"role": role, "content": str(content or "")})
    return out


def build_chat_completions_body(
    request: LLMRequest,
    config: OpenAICompatibleConfig,
) -> dict[str, Any]:
    max_tokens = request.max_tokens or config.default_max_output_tokens
    body: dict[str, Any] = {
        "model": config.model,
        "messages": _message_dicts(request),
        "temperature": float(request.temperature),
        "max_tokens": int(max_tokens),
    }
    # Prefer JSON object when schema present (broad compatibility).
    if request.response_schema is not None:
        body["response_format"] = {"type": "json_object"}
    return body


def build_responses_body(
    request: LLMRequest,
    config: OpenAICompatibleConfig,
) -> dict[str, Any]:
    """Minimal Responses API body (explicit api_mode only)."""
    max_tokens = request.max_tokens or config.default_max_output_tokens
    # Flatten messages into input text for maximum compatibility.
    parts = []
    for msg in _message_dicts(request):
        parts.append(f"{msg['role']}: {msg['content']}")
    body: dict[str, Any] = {
        "model": config.model,
        "input": "\n".join(parts),
        "temperature": float(request.temperature),
        "max_output_tokens": int(max_tokens),
    }
    return body


def extract_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMEmptyResponseError("chat completions response has no choices")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if content is None:
        raise LLMEmptyResponseError("chat completions message content is empty")
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and "text" in block:
                texts.append(str(block.get("text") or ""))
        content = "".join(texts)
    text = str(content).strip()
    if not text:
        raise LLMEmptyResponseError("chat completions content is blank")
    return text


def extract_responses_content(payload: dict[str, Any]) -> str:
    # Common shapes: output_text, or output[].content[].text
    if payload.get("output_text"):
        text = str(payload["output_text"]).strip()
        if text:
            return text
    outputs = payload.get("output") or payload.get("outputs") or []
    texts: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        for block in item.get("content") or []:
            if isinstance(block, dict) and block.get("text"):
                texts.append(str(block["text"]))
            elif isinstance(block, str):
                texts.append(block)
    text = "".join(texts).strip()
    if not text:
        # Fallback: some gateways nest under response
        nested = payload.get("response")
        if isinstance(nested, dict):
            return extract_responses_content(nested)
        raise LLMEmptyResponseError("responses API returned no text content")
    return text


def parse_usage(payload: dict[str, Any]) -> tuple[TokenUsage, bool]:
    """Return (usage, usage_unknown). Never invent precise counts."""
    usage = payload.get("usage")
    if not isinstance(usage, dict) or not usage:
        return TokenUsage(), True
    prompt = usage.get("prompt_tokens")
    if prompt is None:
        prompt = usage.get("input_tokens")
    completion = usage.get("completion_tokens")
    if completion is None:
        completion = usage.get("output_tokens")
    total = usage.get("total_tokens")
    unknown = prompt is None and completion is None and total is None
    if unknown:
        return TokenUsage(), True
    p = int(prompt or 0)
    c = int(completion or 0)
    t = int(total) if total is not None else (p + c)
    return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=t), False


def provider_request_id_from(
    headers: dict[str, str],
    payload: dict[str, Any],
) -> str | None:
    for key, value in headers.items():
        if key.lower() in {"x-request-id", "x-openai-request-id", "request-id"}:
            return str(value)
    rid = payload.get("id")
    return str(rid) if rid else None


class OpenAICompatibleProvider(BaseLLMProvider):
    """Real OpenAI-compatible HTTP provider behind an HttpTransport."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        transport: HttpTransport | None = None,
    ) -> None:
        self.config = config
        if transport is not None:
            self._transport = transport
        elif config.allow_network:
            self._transport = HttpxTransport()
        else:
            raise RealProviderNotEnabledError(
                "OpenAICompatibleProvider requires allow_network=True "
                "or an explicit HttpTransport (e.g. MockTransport for tests)"
            )

    @property
    def name(self) -> str:
        return "openai-compatible"

    @property
    def model(self) -> str:
        return self.config.model

    def _endpoint_and_body(
        self, request: LLMRequest
    ) -> tuple[str, dict[str, Any]]:
        mode = self.config.api_mode
        if mode == "chat_completions":
            url = join_api_url(self.config.base_url, "/chat/completions")
            return url, build_chat_completions_body(request, self.config)
        if mode == "responses":
            url = join_api_url(self.config.base_url, "/responses")
            return url, build_responses_body(request, self.config)
        raise UnsupportedAPIModeError(f"unsupported api_mode: {mode!r}")

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        url, body = self._endpoint_and_body(request)
        headers = self.config.request_headers()
        http_resp = self._transport.request(
            "POST",
            url,
            headers=headers,
            json_body=body,
            timeout_seconds=self.config.timeout_seconds,
            connect_timeout_seconds=self.config.connect_timeout_seconds,
        )
        mapped = map_http_error(http_resp.status_code, http_resp.body)
        if mapped is not None:
            raise mapped

        try:
            payload = json.loads(http_resp.body or "{}")
        except json.JSONDecodeError as exc:
            raise StructuredOutputValidationError(
                "provider returned non-JSON body"
            ) from exc
        if not isinstance(payload, dict):
            raise StructuredOutputValidationError("provider JSON root must be an object")

        if self.config.api_mode == "chat_completions":
            content = extract_chat_content(payload)
        else:
            content = extract_responses_content(payload)

        usage, usage_unknown = parse_usage(payload)
        parsed, schema_errors = parse_and_validate(content, request.response_schema)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if http_resp.elapsed_ms:
            elapsed_ms = float(http_resp.elapsed_ms)

        req_id = provider_request_id_from(http_resp.headers, payload)
        return LLMResponse(
            request_id=new_id("llmreq"),
            content=content,
            parsed_json=parsed,
            usage=usage,
            latency_ms=elapsed_ms,
            provider=self.name,
            model=self.model,
            request_fingerprint=request_fingerprint(request),
            schema_valid=not schema_errors,
            schema_errors=schema_errors,
            created_at=datetime.now(timezone.utc).replace(microsecond=0),
            provider_request_id=req_id,
            usage_unknown=usage_unknown,
            api_mode=self.config.api_mode,
            metadata={
                "usage_unknown": usage_unknown,
                "api_mode": self.config.api_mode,
            },
        )
