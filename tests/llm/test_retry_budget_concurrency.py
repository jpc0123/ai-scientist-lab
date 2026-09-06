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


def _reviewer_schema() -> dict:
    from scientist_lab.llm.reviewer_contract import REVIEWER_CONTRACT_SCHEMA

    return REVIEWER_CONTRACT_SCHEMA


def _valid_reviewer_json() -> str:
    return json.dumps(
        {
            "observation": "VALID evidence judged KEEP by DecisionRubric.",
            "hypothesis_status": "not_a_claim",
            "interpretation": "KEEP is a next-action, not ClaimGate SUPPORTED.",
            "alternative_explanations": ["Single-seed slice noise."],
            "next_research_priority": "Replicate on a second seed before any G2 claim.",
            "evidence_refs": [{"run_id": "run_x", "metric": "APS_lowlight"}],
            "created_from": ["run_x"],
            "confidence": "medium",
        }
    )


def _reviewer_req() -> LLMRequest:
    return LLMRequest(
        purpose="reviewer",
        messages=[{"role": "user", "content": "review"}],
        response_schema=_reviewer_schema(),
        temperature=0.0,
        max_tokens=256,
        metadata={"reviewer_contract": "semantic"},
    )


def test_repair_request_reviewer_does_not_ask_for_selected():
    from scientist_lab.llm.openai_compatible_provider import build_repair_request

    req = build_repair_request(
        _reviewer_req(),
        bad_content='{"observation":"x"}',
        schema_errors=["$.hypothesis_status: missing required property"],
    )
    blob = json.loads(req.messages[1]["content"])
    assert "selected" not in blob.get("required_root_keys")
    assert "selected_required" not in blob
    assert "hypothesis_status" in blob["required_root_keys"]
    assert "Do NOT add selected" in blob["instruction"]
    assert "not_a_claim" in blob["instruction"]
    assert "Root JSON must include selected" not in blob["instruction"]
    assert blob["response_schema"]["required"][0] == "observation"


def test_repair_request_planner_still_mentions_selected():
    from scientist_lab.llm.openai_compatible_provider import build_repair_request
    from scientist_lab.llm.planner_contract import PLANNER_CONTRACT_SCHEMA

    original = LLMRequest(
        purpose="planner",
        messages=[{"role": "user", "content": "plan"}],
        response_schema=PLANNER_CONTRACT_SCHEMA,
        temperature=0.0,
        metadata={"planner_contract": "experiment_plan"},
    )
    req = build_repair_request(
        original,
        bad_content="{}",
        schema_errors=["$.selected: missing required property"],
    )
    blob = json.loads(req.messages[1]["content"])
    assert blob["required_root_keys"] == ["selected"]
    assert "requested_module" in blob["selected_required"]
    assert "selected" in blob["instruction"]


def test_schema_repair_reviewer_once_then_ok():
    missing = '{"observation":"VALID evidence judged KEEP.","hypothesis_status":"not_a_claim"}'
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=200, body=_chat_body(missing)),
            HttpResponse(status_code=200, body=_chat_body(_valid_reviewer_json())),
        ]
    )
    provider = OpenAICompatibleProvider(
        _cfg(),
        transport=transport,
        enable_schema_repair=True,
        sleep=lambda _s: None,
    )
    response = provider.complete(_reviewer_req())
    assert response.schema_valid is True
    assert response.metadata.get("schema_repaired") is True
    repair_body = json.loads(transport.calls[1]["json_body"]["messages"][1]["content"])
    assert "Root JSON must include selected" not in repair_body.get("instruction", "")
    assert "Do NOT add selected" in repair_body["instruction"]
    assert "hypothesis_status" in repair_body["required_root_keys"]


def test_schema_repair_reviewer_planner_shaped_second_still_fail_closed():
    missing = '{"observation":"VALID evidence judged KEEP."}'
    planner_shaped = '{"selected":{"requested_module":"fusion","hypothesis":"x","proposed_changes":[{"target":"fusion","summary":"F3"}]}}'
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=200, body=_chat_body(missing)),
            HttpResponse(status_code=200, body=_chat_body(planner_shaped)),
        ]
    )
    provider = OpenAICompatibleProvider(
        _cfg(), transport=transport, enable_schema_repair=True, sleep=lambda _s: None
    )
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        provider.complete(_reviewer_req())
    err = excinfo.value
    assert err.issues
    assert any("observation" in item or "hypothesis_status" in item for item in err.issues)
    assert err.content
    assert err.prior_issues


def test_valid_reviewer_json_without_selected_does_not_repair():
    transport = MockTransport(
        responses=[HttpResponse(status_code=200, body=_chat_body(_valid_reviewer_json()))]
    )
    provider = OpenAICompatibleProvider(
        _cfg(), transport=transport, enable_schema_repair=True, sleep=lambda _s: None
    )
    response = provider.complete(_reviewer_req())
    assert response.schema_valid is True
    assert "selected" not in (response.parsed_json or {})
    assert response.metadata.get("schema_repaired") is False
    assert len(transport.calls) == 1


def test_reviewer_purpose_does_not_validate_planner_selected_schema():
    from scientist_lab.llm.planner_contract import PLANNER_CONTRACT_SCHEMA

    transport = MockTransport(
        responses=[HttpResponse(status_code=200, body=_chat_body(_valid_reviewer_json()))]
    )
    provider = OpenAICompatibleProvider(
        _cfg(), transport=transport, enable_schema_repair=True, sleep=lambda _s: None
    )
    wrong = LLMRequest(
        purpose="reviewer",
        messages=[{"role": "user", "content": "review"}],
        response_schema=PLANNER_CONTRACT_SCHEMA,
        temperature=0.0,
        max_tokens=256,
        metadata={"reviewer_contract": "semantic"},
    )
    response = provider.complete(wrong)
    assert response.schema_valid is True
    assert "selected" not in (response.parsed_json or {})
    assert len(transport.calls) == 1


def test_reviewer_missing_fields_fail_closed_without_selected():
    incomplete = '{"observation":"VALID evidence judged KEEP."}'
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=200, body=_chat_body(incomplete)),
            HttpResponse(status_code=200, body=_chat_body(incomplete)),
        ]
    )
    provider = OpenAICompatibleProvider(
        _cfg(), transport=transport, enable_schema_repair=True, sleep=lambda _s: None
    )
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        provider.complete(_reviewer_req())
    err = excinfo.value
    assert any("hypothesis_status" in item or "interpretation" in item for item in err.issues)
    assert not any("selected" in item for item in err.issues)
    assert not any("selected" in item for item in (err.prior_issues or []))


def test_concurrency_limit():
    gate = ConcurrencyGate(max_concurrency=1, wait_timeout_seconds=0.05)
    gate.acquire()
    with pytest.raises(LLMConcurrencyLimitError):
        gate.acquire()
    gate.release()
