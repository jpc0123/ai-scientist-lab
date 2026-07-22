"""OpenAICompatibleProvider — chat_completions / responses via HttpTransport.

v1.4.2: MockTransport / mapping / schema parse
v1.4.3: retry, budget gate, concurrency, one-shot schema repair
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

from scientist_lab.domain.models import new_id
from scientist_lab.llm.budget import LLMBudget
from scientist_lab.llm.concurrency import ConcurrencyGate
from scientist_lab.llm.errors import (
    LLMEmptyResponseError,
    LLMError,
    RealProviderNotEnabledError,
    StructuredOutputValidationError,
    UnsupportedAPIModeError,
)
from scientist_lab.llm.http_transport import (
    HttpResponse,
    HttpTransport,
    HttpxTransport,
    map_http_error,
)
from scientist_lab.llm.models import LLMRequest, LLMResponse, TokenUsage
from scientist_lab.llm.openai_config import OpenAICompatibleConfig, join_api_url
from scientist_lab.llm.provider import BaseLLMProvider, request_fingerprint
from scientist_lab.llm.retry_policy import RetryPolicy, run_with_retries
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
    if request.response_schema is not None:
        body["response_format"] = {"type": "json_object"}
    return body


def build_responses_body(
    request: LLMRequest,
    config: OpenAICompatibleConfig,
) -> dict[str, Any]:
    max_tokens = request.max_tokens or config.default_max_output_tokens
    parts = [f"{m['role']}: {m['content']}" for m in _message_dicts(request)]
    return {
        "model": config.model,
        "input": "\n".join(parts),
        "temperature": float(request.temperature),
        "max_output_tokens": int(max_tokens),
    }


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
        nested = payload.get("response")
        if isinstance(nested, dict):
            return extract_responses_content(nested)
        raise LLMEmptyResponseError("responses API returned no text content")
    return text


def parse_usage(payload: dict[str, Any]) -> tuple[TokenUsage, bool]:
    usage = payload.get("usage")
    if not isinstance(usage, dict) or not usage:
        return TokenUsage(), True
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
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


def build_repair_request(
    original: LLMRequest,
    *,
    bad_content: str,
    schema_errors: list[str],
) -> LLMRequest:
    """One-shot structured repair request (same provider/model)."""
    errors = "; ".join((schema_errors or [])[:8])
    snippet = (bad_content or "")[:1500]
    messages = [
        {
            "role": "system",
            "content": (
                "Repair the JSON so it matches the schema. "
                "Return ONLY a single JSON object. No markdown."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "schema_errors": errors,
                    "invalid_output": snippet,
                    "instruction": "Fix validation issues only; do not invent science.",
                },
                ensure_ascii=False,
            ),
        },
    ]
    return LLMRequest(
        purpose=original.purpose,
        messages=messages,
        response_schema=original.response_schema,
        temperature=0.0,
        max_tokens=original.max_tokens,
        metadata={
            **dict(original.metadata or {}),
            "schema_repair": True,
        },
    )


class OpenAICompatibleProvider(BaseLLMProvider):
    """Real OpenAI-compatible HTTP provider behind an HttpTransport."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        transport: HttpTransport | None = None,
        budget: LLMBudget | None = None,
        concurrency: ConcurrencyGate | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        enable_schema_repair: bool = True,
    ) -> None:
        self.config = config
        self.budget = budget
        self.concurrency = concurrency or ConcurrencyGate(
            max_concurrency=int(config.max_concurrency or 1)
        )
        self.retry_policy = retry_policy or RetryPolicy(
            max_retries=int(config.max_retries),
            base_delay_seconds=float(config.retry_base_seconds),
            max_delay_seconds=float(config.retry_max_seconds),
            jitter=True,
        )
        self._sleep = sleep
        self.enable_schema_repair = enable_schema_repair
        self.attempt_log: list[dict[str, Any]] = []

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
            return (
                join_api_url(self.config.base_url, "/chat/completions"),
                build_chat_completions_body(request, self.config),
            )
        if mode == "responses":
            return (
                join_api_url(self.config.base_url, "/responses"),
                build_responses_body(request, self.config),
            )
        raise UnsupportedAPIModeError(f"unsupported api_mode: {mode!r}")

    def _http_once(self, request: LLMRequest) -> tuple[HttpResponse, dict[str, Any], str]:
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
        return http_resp, payload, content

    def _to_response(
        self,
        request: LLMRequest,
        *,
        content: str,
        payload: dict[str, Any],
        http_resp: HttpResponse,
        started: float,
        attempt_count: int,
        repaired: bool,
        prior_errors: list[str] | None = None,
    ) -> LLMResponse:
        usage, usage_unknown = parse_usage(payload)
        parsed, schema_errors = parse_and_validate(content, request.response_schema)
        elapsed_ms = float(http_resp.elapsed_ms or ((time.perf_counter() - started) * 1000.0))
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
                "attempt_count": attempt_count,
                "schema_repaired": repaired,
                "prior_schema_errors": list(prior_errors or [])[:8],
                "attempt_log": list(self.attempt_log),
            },
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        self.attempt_log = []

        if self.budget is not None:
            projected = request.max_tokens or self.config.default_max_output_tokens
            self.budget.check_can_call(projected_output_tokens=int(projected))

        def _do_http() -> tuple[HttpResponse, dict[str, Any], str]:
            return self._http_once(request)

        with self.concurrency:
            http_resp, payload, content = run_with_retries(  # type: ignore[misc]
                _do_http,
                self.retry_policy,
                sleep=self._sleep,
                on_attempt=lambda idx, err: self.attempt_log.append(
                    {
                        "attempt": idx,
                        "ok": err is None,
                        "error_type": type(err).__name__ if err else None,
                        "retryable": bool(getattr(err, "retryable", False))
                        if isinstance(err, LLMError)
                        else False,
                    }
                ),
            )

        response = self._to_response(
            request,
            content=content,
            payload=payload,
            http_resp=http_resp,
            started=started,
            attempt_count=len(self.attempt_log) or 1,
            repaired=False,
        )

        # One-shot schema repair (counts as extra budgeted call).
        if (
            self.enable_schema_repair
            and request.response_schema is not None
            and response.schema_errors
            and not (request.metadata or {}).get("schema_repair")
        ):
            prior = list(response.schema_errors)
            repair_req = build_repair_request(
                request, bad_content=response.content, schema_errors=prior
            )
            if self.budget is not None:
                self.budget.check_can_call(
                    projected_output_tokens=int(
                        repair_req.max_tokens or self.config.default_max_output_tokens
                    )
                )
            with self.concurrency:
                repair_http, repair_payload, repair_content = self._http_once(repair_req)
            repaired = self._to_response(
                request,
                content=repair_content,
                payload=repair_payload,
                http_resp=repair_http,
                started=started,
                attempt_count=(len(self.attempt_log) or 1) + 1,
                repaired=True,
                prior_errors=prior,
            )
            # Record original failed attempt usage then repaired usage.
            if self.budget is not None:
                self.budget.record(
                    response.usage,
                    usage_unknown=response.usage_unknown,
                )
                budget_meta = self.budget.record(
                    repaired.usage,
                    usage_unknown=repaired.usage_unknown,
                )
                repaired.metadata["budget"] = budget_meta
            if repaired.schema_errors:
                raise StructuredOutputValidationError(
                    "structured output invalid after one repair attempt",
                    issues=list(repaired.schema_errors)[:8],
                )
            return repaired

        if self.budget is not None:
            budget_meta = self.budget.record(
                response.usage,
                usage_unknown=response.usage_unknown,
            )
            response.metadata["budget"] = budget_meta
        return response
