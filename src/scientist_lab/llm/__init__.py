"""Offline LLM Provider layer (v1.3: Fake / Replay / Audit / Schema / Limits)."""

from scientist_lab.llm.audit import AuditingProvider, build_usage_from_messages
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
    "LLMCallRecord",
    "LLMCallRepository",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LimitingProvider",
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
    "parse_and_validate",
    "planning_context_to_planner_request",
    "request_fingerprint",
    "validate_against_schema",
]

API_VERSION = "v1.3.8"
