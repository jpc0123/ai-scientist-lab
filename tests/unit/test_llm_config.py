"""v1.4.1: LLM config and API-key boundary tests (offline, no network)."""

from __future__ import annotations

import pytest

from scientist_lab.llm.config import (
    InvalidLLMConfigError,
    LLMConfig,
    MissingAPIKeyError,
    load_llm_config,
    mask_secret,
    redact_secrets,
)


def test_default_config_is_mock_without_env():
    cfg = load_llm_config(environ={})
    assert cfg.provider == "mock"
    assert cfg.is_real is False
    assert cfg.api_key_present is False
    # No exception: mock does not require a key.


def test_mock_ignores_missing_real_fields():
    cfg = load_llm_config(
        environ={
            "LLM_PROVIDER": "mock",
            # Intentionally no LLM_API_KEY / BASE_URL / MODEL
        }
    )
    cfg.require_real_ready()  # no-op


def test_real_provider_requires_key_base_url_model():
    with pytest.raises(MissingAPIKeyError):
        load_llm_config(
            environ={
                "LLM_PROVIDER": "openai-compatible",
                "LLM_BASE_URL": "https://example.com/v1",
                "LLM_MODEL": "gpt-test",
            },
            require_real=True,
        )

    with pytest.raises(InvalidLLMConfigError, match="LLM_BASE_URL"):
        load_llm_config(
            environ={
                "LLM_PROVIDER": "openai-compatible",
                "LLM_API_KEY": "sk-test-secret-key-value",
                "LLM_MODEL": "gpt-test",
            },
            require_real=True,
        )

    with pytest.raises(InvalidLLMConfigError, match="LLM_MODEL"):
        load_llm_config(
            environ={
                "LLM_PROVIDER": "openai-compatible",
                "LLM_API_KEY": "sk-test-secret-key-value",
                "LLM_BASE_URL": "https://example.com/v1",
            },
            require_real=True,
        )


def test_real_provider_ok_with_full_config():
    cfg = load_llm_config(
        environ={
            "LLM_PROVIDER": "openai-compatible",
            "LLM_API_KEY": "sk-test-secret-key-value",
            "LLM_BASE_URL": "https://example.com/v1",
            "LLM_MODEL": "gpt-test",
            "LLM_TIMEOUT_SECONDS": "30",
        }
    )
    assert cfg.is_real is True
    assert cfg.timeout_seconds == 30.0
    assert cfg.api_key_value() == "sk-test-secret-key-value"


def test_repr_and_safe_dict_never_leak_key():
    secret = "sk-super-secret-should-not-leak"
    cfg = LLMConfig(
        provider="openai-compatible",
        base_url="https://example.com/v1",
        api_key=secret,  # type: ignore[arg-type]
        model="gpt-test",
    )
    text = repr(cfg)
    assert secret not in text
    assert "[REDACTED]" in text
    dumped = cfg.safe_dict()
    assert secret not in str(dumped)
    assert dumped["api_key_present"] is True
    assert dumped["api_key_fingerprint"] == "[REDACTED]"


def test_redact_secrets_scrubs_bearer_and_sk():
    raw = (
        "Authorization: Bearer sk-abcdef1234567890 "
        "api_key=sk-zzzzzzzzzzzzzzzz error"
    )
    cleaned = redact_secrets(raw)
    assert "sk-abcdef1234567890" not in cleaned
    assert "sk-zzzzzzzzzzzzzzzz" not in cleaned
    assert "REDACTED" in cleaned


def test_mask_secret_keeps_tail_only():
    assert mask_secret("sk-abcdefghijklmnop") == "[REDACTED]"
    assert "mnop" not in mask_secret("sk-abcdefghijklmnop")


def test_openai_alias_maps_to_compatible():
    cfg = load_llm_config(
        environ={"LLM_PROVIDER": "openai"},
        require_real=False,
    )
    assert cfg.provider == "openai-compatible"


def test_explicit_override_beats_env():
    cfg = load_llm_config(
        environ={"LLM_PROVIDER": "openai-compatible", "LLM_API_KEY": "env-key"},
        provider="mock",
        require_real=False,
    )
    assert cfg.provider == "mock"
