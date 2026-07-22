"""v1.4.3: retry, budget, concurrency, schema repair (MockTransport, zero network)."""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from scientist_lab.llm.budget import LLMBudget, ModelPricing
from scientist_lab.llm.concurrency import ConcurrencyGate
from scientist_lab.llm.errors import (
    LLMBudgetExceededError,
    LLMConcurrencyLimitError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    StructuredOutputValidationError,
)
from scientist_lab.llm.http_transport import HttpResponse, MockTransport
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.openai_compatible_provider import OpenAICompatibleProvider
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.llm.retry_policy import RetryPolicy
from scientist_lab.llm.schema_parser import PLANNER_OUTPUT_SCHEMA


def _cfg(**kwargs) -> OpenAICompatibleConfig:
    base = dict(
        base_url="https://api.example.com/v1",
        api_key=SecretStr("sk-test-secret-key-value"),
        model="gpt-test",
        allow_network=False,
        max_retries=1,
        max_concurrency=1,
    )
    base.update(kwargs)
    return OpenAICompatibleConfig(**base)


def _valid_planner_json() -> str:
    return json.dumps(
        {
            "project_id": "project_rgbt_003",
            "reasoning_summary": "ok",
            "candidates": [],
            "stop_recommended": True,
            "stop_reason": "fixture",
        }
    )


def _chat_body(content: str, *, usage: dict | None = None) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "choices": [{"message": {"content": content}}],
            "usage": usage
            if usage is not None
            else {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
    )


def _req() -> LLMRequest:
    return LLMRequest(
        purpose="planner",
        messages=[{"role": "user", "content": "plan"}],
        response_schema=PLANNER_OUTPUT_SCHEMA,
        temperature=0.0,
        max_tokens=128,
    )


def test_429_retries_then_succeeds():
    sleeps: list[float] = []
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=429, body='{"error":"rate"}'),
            HttpResponse(status_code=200, body=_chat_body(_valid_planner_json())),
        ]
    )
    provider = OpenAICompatibleProvider(
        _cfg(),
        transport=transport,
        retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.01, jitter=False),
        sleep=sleeps.append,
        enable_schema_repair=False,
    )
    response = provider.complete(_req())
    assert response.schema_valid is True
    assert len(transport.calls) == 2
    assert sleeps  # one backoff
    assert any(a.get("error_type") == "LLMRateLimitError" for a in provider.attempt_log)


def test_400_does_not_retry():
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=400, body='{"error":"bad"}'),
            HttpResponse(status_code=200, body=_chat_body(_valid_planner_json())),
        ]
    )
    provider = OpenAICompatibleProvider(
        _cfg(),
        transport=transport,
        retry_policy=RetryPolicy(max_retries=2, jitter=False),
        sleep=lambda _s: None,
        enable_schema_repair=False,
    )
    with pytest.raises(LLMInvalidRequestError):
        provider.complete(_req())
    assert len(transport.calls) == 1


def test_budget_blocks_before_network():
    transport = MockTransport(
        default_response=HttpResponse(
            status_code=200, body=_chat_body(_valid_planner_json())
        )
    )
    budget = LLMBudget(project_id="p", max_requests=0)
    provider = OpenAICompatibleProvider(
        _cfg(), transport=transport, budget=budget, enable_schema_repair=False
    )
    with pytest.raises(LLMBudgetExceededError):
        provider.complete(_req())
    assert transport.calls == []


def test_budget_records_usage():
    transport = MockTransport(
        default_response=HttpResponse(
            status_code=200, body=_chat_body(_valid_planner_json())
        )
    )
    budget = LLMBudget(
        project_id="p",
        max_requests=5,
        pricing=ModelPricing(
            input_cost_per_million_tokens=1.0,
            output_cost_per_million_tokens=2.0,
        ),
    )
    provider = OpenAICompatibleProvider(
        _cfg(), transport=transport, budget=budget, enable_schema_repair=False
    )
    response = provider.complete(_req())
    assert budget.used_requests == 1
    assert budget.used_total_tokens == 12
    assert response.metadata["budget"]["cost_status"] == "estimated"


def test_usage_unknown_when_missing():
    transport = MockTransport(
        default_response=HttpResponse(
            status_code=200, body=_chat_body(_valid_planner_json(), usage={})
        )
    )
    budget = LLMBudget(project_id="p", max_requests=5)
    provider = OpenAICompatibleProvider(
        _cfg(), transport=transport, budget=budget, enable_schema_repair=False
    )
    response = provider.complete(_req())
    assert response.usage_unknown is True
    assert budget.used_requests == 1
    assert budget.used_total_tokens == 0
    assert response.metadata["budget"]["cost_status"] == "unknown"


def test_schema_repair_once_then_ok():
    bad = '{"project_id":"project_rgbt_003"}'  # missing required fields
    good = _valid_planner_json()
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=200, body=_chat_body(bad)),
            HttpResponse(status_code=200, body=_chat_body(good)),
        ]
    )
    budget = LLMBudget(project_id="p", max_requests=5)
    provider = OpenAICompatibleProvider(
        _cfg(),
        transport=transport,
        budget=budget,
        enable_schema_repair=True,
        sleep=lambda _s: None,
    )
    response = provider.complete(_req())
    assert response.schema_valid is True
    assert response.metadata.get("schema_repaired") is True
    assert len(transport.calls) == 2
    assert budget.used_requests == 2  # original + repair


def test_schema_repair_second_failure_raises():
    bad = '{"project_id":"project_rgbt_003"}'
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=200, body=_chat_body(bad)),
            HttpResponse(status_code=200, body=_chat_body(bad)),
        ]
    )
    provider = OpenAICompatibleProvider(
        _cfg(), transport=transport, enable_schema_repair=True, sleep=lambda _s: None
    )
    with pytest.raises(StructuredOutputValidationError):
        provider.complete(_req())


def test_concurrency_limit():
    gate = ConcurrencyGate(max_concurrency=1, wait_timeout_seconds=0.05)
    gate.acquire()
    with pytest.raises(LLMConcurrencyLimitError):
        gate.acquire()
    gate.release()
