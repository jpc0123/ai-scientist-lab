"""ReplayProvider — replay audited responses by request fingerprint (offline)."""

from __future__ import annotations

from datetime import datetime, timezone

from scientist_lab.llm.models import LLMRequest, LLMResponse
from scientist_lab.llm.provider import BaseLLMProvider, request_fingerprint
from scientist_lab.llm.repository import LLMCallRepository
from scientist_lab.llm.schema_parser import parse_and_validate


class ReplayMissError(KeyError):
    """Raised when no audited call exists for the request fingerprint."""


class ReplayProvider(BaseLLMProvider):
    """Never calls the network; only returns previously audited responses."""

    def __init__(
        self,
        repository: LLMCallRepository,
        *,
        model: str = "replay-v1",
        validate_schema: bool = True,
    ) -> None:
        self._repo = repository
        self._model = model
        self.validate_schema = validate_schema

    @property
    def name(self) -> str:
        return "replay"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        fp = request_fingerprint(request)
        try:
            record = self._repo.require_by_fingerprint(fp)
        except KeyError as exc:
            raise ReplayMissError(str(exc)) from exc

        response_blob = dict(record.response or {})
        content = str(response_blob.get("content") or "")
        parsed = response_blob.get("parsed_json")
        schema_errors: list[str] = []
        schema_valid = bool(response_blob.get("schema_valid", True))
        if self.validate_schema and request.response_schema is not None:
            try:
                parsed, schema_errors = parse_and_validate(
                    content, request.response_schema
                )
                schema_valid = not schema_errors
            except Exception as exc:  # noqa: BLE001
                schema_valid = False
                schema_errors = [str(exc)]

        return LLMResponse(
            request_id=str(response_blob.get("request_id") or record.request_id),
            content=content,
            parsed_json=parsed if isinstance(parsed, dict) else None,
            usage=record.usage,
            latency_ms=float(response_blob.get("latency_ms") or record.latency_ms or 0.0),
            provider=self.name,
            model=str(response_blob.get("model") or record.model or self.model),
            request_fingerprint=fp,
            schema_valid=schema_valid,
            schema_errors=schema_errors
            or list(response_blob.get("schema_errors") or []),
            created_at=datetime.now(timezone.utc).replace(microsecond=0),
        )
