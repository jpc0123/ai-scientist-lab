"""Offline LLM Provider layer + config boundary (v1.3 / v1.4.1)."""

from scientist_lab.llm.audit import AuditingProvider, build_usage_from_messages
from scientist_lab.llm.config import (
    InvalidLLMConfigError,
    LLMConfig,
    MissingAPIKeyError,
    load_llm_config,
    mask_secret,
    redact_secrets,
)
from scientist_lab.llm.context_codec import planning_context_to_planner_request
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.limits import (
    LimitingProvider,
    ProviderLimitExceeded,
    ProviderLimits,
    UsageLedger,
    estimate_cost_usd,
)
from scientist_lab.llm.models import LLMCallRecord, LLMRequest, LLMResponse, TokenUsage
from scientist_lab.llm.provider import BaseLLMProvider, LLMProvider, request_fingerprint
from scientist_lab.llm.replay_provider import ReplayMissError, ReplayProvider
from scientist_lab.llm.repository import LLMCallRepository
from scientist_lab.llm.schema_parser import (
    CRITIC_REVIEW_SCHEMA,
    PLANNER_OUTPUT_SCHEMA,
    SchemaValidationError,
    extract_json_object,
    parse_and_validate,
    validate_against_schema,
)

__all__ = [
    "AuditingProvider",
    "BaseLLMProvider",
    "CRITIC_REVIEW_SCHEMA",
    "FakeProvider",
    "InvalidLLMConfigError",
    "LLMCallRecord",
    "LLMCallRepository",
    "LLMConfig",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LimitingProvider",
    "MissingAPIKeyError",
    "PLANNER_OUTPUT_SCHEMA",
    "ProviderLimitExceeded",
    "ProviderLimits",
    "ReplayMissError",
    "ReplayProvider",
    "SchemaValidationError",
    "TokenUsage",
    "UsageLedger",
    "build_usage_from_messages",
    "estimate_cost_usd",
    "extract_json_object",
    "load_llm_config",
    "mask_secret",
    "parse_and_validate",
    "planning_context_to_planner_request",
    "redact_secrets",
    "request_fingerprint",
    "validate_against_schema",
]

API_VERSION = "v1.4.1"
