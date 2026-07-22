"""Auditing wrapper around any LLMProvider (v1.3.4)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scientist_lab.domain.models import new_id
from scientist_lab.llm.models import LLMCallRecord, LLMRequest, LLMResponse, TokenUsage
from scientist_lab.llm.provider import LLMProvider, request_fingerprint
from scientist_lab.llm.repository import LLMCallRepository
from scientist_lab.llm.schema_parser import parse_and_validate


class AuditingProvider:
    """Delegate to an inner provider and persist request/response audit records."""

    def __init__(
        self,
        inner: LLMProvider,
        repository: LLMCallRepository,
        *,
        project_id: str | None = None,
        validate_schema: bool = True,
    ) -> None:
        self._inner = inner
        self._repo = repository
        self.project_id = project_id
        self.validate_schema = validate_schema

    @property
    def name(self) -> str:
        return f"audit:{self._inner.name}"

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def inner(self) -> LLMProvider:
        return self._inner

    @property
    def repository(self) -> LLMCallRepository:
        return self._repo

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._inner.complete(request)
        schema_valid = True
        schema_errors: list[str] = []
        parsed = response.parsed_json
        if self.validate_schema and request.response_schema is not None:
            try:
                if parsed is None:
                    parsed, schema_errors = parse_and_validate(
                        response.content, request.response_schema
                    )
                else:
                    from scientist_lab.llm.schema_parser import validate_against_schema

                    schema_errors = validate_against_schema(
                        parsed, request.response_schema
                    )
            except Exception as exc:  # noqa: BLE001
                schema_valid = False
                schema_errors = [str(exc)]
            schema_valid = schema_valid and not schema_errors

        fp = response.request_fingerprint or request_fingerprint(request)
        request_id = response.request_id or new_id("llmreq")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        enriched = response.model_copy(
            update={
                "request_id": request_id,
                "request_fingerprint": fp,
                "parsed_json": parsed if parsed is not None else response.parsed_json,
                "schema_valid": schema_valid,
                "schema_errors": schema_errors,
                "created_at": response.created_at or now,
            }
        )
        record = LLMCallRecord(
            request_id=enriched.request_id,
            request_fingerprint=enriched.request_fingerprint,
            purpose=request.purpose,
            provider=enriched.provider,
            model=enriched.model,
            request=request.model_dump(mode="json"),
            response=enriched.model_dump(mode="json"),
            usage=enriched.usage
            if isinstance(enriched.usage, TokenUsage)
            else TokenUsage.model_validate(enriched.usage),
            latency_ms=enriched.latency_ms,
            schema_valid=enriched.schema_valid,
            schema_errors=list(enriched.schema_errors),
            created_at=enriched.created_at,
            project_id=self.project_id
            or str((request.metadata or {}).get("project_id") or "")
            or None,
        )
        self._repo.save(record)
        return enriched


def estimate_tokens(text: str) -> int:
    """Rough offline token estimate (chars/4). No network."""
    return max(1, (len(text or "") + 3) // 4)


def build_usage_from_messages(
    messages: list[dict[str, Any]], content: str
) -> TokenUsage:
    prompt = "\n".join(str(m.get("content") or "") for m in messages)
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(content)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
