"""v1.4.2 OpenAI-compatible provider tests — MockTransport only, zero network."""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from scientist_lab.llm.errors import (
    LLMAuthenticationError,
    RealProviderNotEnabledError,
    UnsupportedProviderError,
)
from scientist_lab.llm.factory import create_llm_provider
from scientist_lab.llm.http_transport import HttpResponse, MockTransport
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.openai_compatible_provider import (
    OpenAICompatibleProvider,
    build_chat_completions_body,
    build_responses_body,
    extract_chat_content,
)
from scientist_lab.llm.openai_config import (
    OpenAICompatibleConfig,
    build_auth_headers,
    join_api_url,
    normalize_base_url,
)
from scientist_lab.llm.schema_parser import PLANNER_OUTPUT_SCHEMA


def _cfg(**kwargs) -> OpenAICompatibleConfig:
    base = dict(
        base_url="https://api.example.com/v1",
        api_key=SecretStr("sk-test-secret-key-value"),
        model="gpt-test",
        allow_network=False,
        api_mode="chat_completions",
    )
    base.update(kwargs)
    return OpenAICompatibleConfig(**base)


def _planner_request() -> LLMRequest:
    return LLMRequest(
        purpose="planner",
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": '{"project_id":"project_rgbt_003"}'},
        ],
        response_schema=PLANNER_OUTPUT_SCHEMA,
        temperature=0.0,
        max_tokens=512,
        metadata={"project_id": "project_rgbt_003"},
    )


def _chat_ok_body() -> str:
    payload = {
        "id": "chatcmpl-test-001",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "project_id": "project_rgbt_003",
                            "reasoning_summary": "mock transport planner",
                            "candidates": [],
                            "stop_recommended": True,
                            "stop_reason": "no candidates in fixture",
                        },
                        ensure_ascii=False,
                    ),
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }
    return json.dumps(payload)


def test_factory_default_is_fake_for_mock():
    provider = create_llm_provider("mock")
    assert provider.name == "fake"


def test_factory_blocks_real_without_network_gate():
    with pytest.raises(RealProviderNotEnabledError):
        create_llm_provider(
            "openai-compatible",
            allow_network=False,
            openai_config=_cfg(allow_network=False),
        )


def test_factory_mock_transport_offline():
    transport = MockTransport(
        default_response=HttpResponse(
            status_code=200,
            body=_chat_ok_body(),
            headers={"x-request-id": "req-abc"},
            elapsed_ms=12.5,
        )
    )
    provider = create_llm_provider(
        "openai-compatible",
        allow_network=False,
        openai_config=_cfg(),
        transport=transport,
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    response = provider.complete(_planner_request())
    assert response.provider == "openai-compatible"
    assert response.model == "gpt-test"
    assert response.provider_request_id in {"req-abc", "chatcmpl-test-001"}
    assert response.usage.total_tokens == 30
    assert response.usage_unknown is False
    assert response.parsed_json is not None
    assert transport.calls
    # Authorization must exist on the wire call but never in logged headers.
    assert "Authorization" not in transport.calls[0]["headers"] or transport.calls[0][
        "headers"
    ].get("Authorization") == "[REDACTED]"
    assert "sk-test-secret-key-value" not in json.dumps(transport.calls)


def test_chat_completions_url_and_body():
    cfg = _cfg()
    req = _planner_request()
    url = join_api_url(cfg.base_url, "/chat/completions")
    assert url == "https://api.example.com/v1/chat/completions"
    body = build_chat_completions_body(req, cfg)
    assert body["model"] == "gpt-test"
    assert body["messages"][0]["role"] == "system"
    assert body["response_format"]["type"] == "json_object"


def test_responses_mode_mapping():
    cfg = _cfg(api_mode="responses")
    body = build_responses_body(_planner_request(), cfg)
    assert "input" in body
    assert body["max_output_tokens"] == 512
    transport = MockTransport(
        default_response=HttpResponse(
            status_code=200,
            body=json.dumps(
                {
                    "id": "resp_1",
                    "output_text": '{"project_id":"project_rgbt_003","reasoning_summary":"r","candidates":[],"stop_recommended":true}',
                }
            ),
            headers={},
        )
    )
    provider = OpenAICompatibleProvider(cfg, transport=transport)
    response = provider.complete(_planner_request())
    assert response.api_mode == "responses"
    assert response.parsed_json is not None


def test_markdown_json_content_parses():
    fenced = "```json\n{\"project_id\":\"project_rgbt_003\",\"reasoning_summary\":\"x\",\"candidates\":[],\"stop_recommended\":true}\n```"
    payload = {
        "choices": [{"message": {"content": fenced}}],
        "usage": {},
    }
    text = extract_chat_content(payload)
    transport = MockTransport(
        default_response=HttpResponse(status_code=200, body=json.dumps(payload))
    )
    provider = OpenAICompatibleProvider(_cfg(), transport=transport)
    response = provider.complete(_planner_request())
    assert response.usage_unknown is True
    assert response.parsed_json is not None
    assert text.startswith("```") or "project_id" in text


def test_normalize_base_url_rules():
    assert normalize_base_url("https://api.example.com/v1/") == "https://api.example.com/v1"
    assert normalize_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    with pytest.raises(Exception):
        normalize_base_url("http://evil.example.com/v1")
    with pytest.raises(Exception):
        normalize_base_url("https://api.example.com/v1?api_key=secret")


def test_extra_headers_cannot_override_authorization():
    with pytest.raises(Exception):
        build_auth_headers(
            api_key="sk-test",
            extra_headers={"Authorization": "Bearer hijack"},
        )


def test_auth_header_present_but_not_in_repr():
    cfg = _cfg()
    headers = cfg.request_headers()
    assert headers["Authorization"].startswith("Bearer sk-")
    assert "sk-test-secret-key-value" not in repr(cfg)
    assert "[REDACTED]" in repr(cfg)


def test_http_401_maps_to_auth_error():
    transport = MockTransport(
        default_response=HttpResponse(
            status_code=401,
            body='{"error":"invalid_api_key","api_key":"sk-should-redact"}',
        )
    )
    provider = OpenAICompatibleProvider(_cfg(), transport=transport)
    with pytest.raises(LLMAuthenticationError) as exc:
        provider.complete(_planner_request())
    assert "sk-should-redact" not in str(exc.value)


def test_unsupported_provider():
    with pytest.raises(UnsupportedProviderError):
        create_llm_provider("claude-direct")
