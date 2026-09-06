"""LLM Provider layer (v1.3–v1.4.3)."""

from scientist_lab.llm.audit import AuditingProvider, build_usage_from_messages
from scientist_lab.llm.budget import LLMBudget, ModelPricing
from scientist_lab.llm.concurrency import ConcurrencyGate
from scientist_lab.llm.config import (
    InvalidLLMConfigError,
    LLMConfig,
    MissingAPIKeyError,
    load_llm_config,
    mask_secret,
    redact_secrets,
)
from scientist_lab.llm.context_codec import planning_context_to_planner_request
from scientist_lab.llm.eval_suite import (
    SuiteEvalReport,
    real_eval_gates,
    run_llm_eval_suite,
    write_suite_report,
)
from scientist_lab.llm.errors import (
    LLMAuthenticationError,
    LLMBudgetExceededError,
    LLMError,
    LLMRateLimitError,
    RealProviderNotEnabledError,
    StructuredOutputValidationError,
    UnsupportedAPIModeError,
    UnsupportedProviderError,
)
from scientist_lab.llm.factory import create_llm_provider
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.gateway import (
    GatewayError,
    ScriptedProvider,
    complete_chat,
    resolve_gateway_provider,
)
from scientist_lab.llm.http_transport import HttpResponse, HttpTransport, MockTransport
from scientist_lab.llm.limits import (
    LimitingProvider,
    ProviderLimitExceeded,
    ProviderLimits,
    UsageLedger,
    estimate_cost_usd,
)
from scientist_lab.llm.models import LLMCallRecord, LLMRequest, LLMResponse, TokenUsage
from scientist_lab.llm.openai_compatible_provider import OpenAICompatibleProvider
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.llm.plan_replay import load_replay_bundle, run_llm_plan_replay
from scientist_lab.llm.planner_contract import (
    PLANNER_CONTRACT_SCHEMA,
    PLANNER_SYSTEM_PROMPT,
    PlannerContractError,
    PlannerContractInput,
    build_contract_input,
    build_planner_request,
    infer_discarded_modules,
    parse_planner_completion,
    prompt_hash,
)
from scientist_lab.llm.provider import BaseLLMProvider, LLMProvider, request_fingerprint
from scientist_lab.llm.replay_provider import ReplayMissError, ReplayProvider
from scientist_lab.llm.repository import LLMCallRepository
from scientist_lab.llm.retry_policy import RetryPolicy
from scientist_lab.llm.review_replay import run_llm_review_replay
from scientist_lab.llm.reviewer_contract import (
    REVIEWER_CONTRACT_SCHEMA,
    REVIEWER_SYSTEM_PROMPT,
    ReviewerContractError,
    ReviewerContractInput,
    build_reviewer_request,
    parse_reviewer_completion,
    proposal_to_research_lesson,
)
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
    "ConcurrencyGate",
    "FakeProvider",
    "GatewayError",
    "HttpResponse",
    "HttpTransport",
    "InvalidLLMConfigError",
    "LLMAuthenticationError",
    "LLMBudget",
    "LLMBudgetExceededError",
    "LLMCallRecord",
    "LLMCallRepository",
    "LLMConfig",
    "LLMError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LimitingProvider",
    "MissingAPIKeyError",
    "MockTransport",
    "ModelPricing",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProvider",
    "PLANNER_CONTRACT_SCHEMA",
    "PLANNER_OUTPUT_SCHEMA",
    "PLANNER_SYSTEM_PROMPT",
    "PlannerContractError",
    "PlannerContractInput",
    "REVIEWER_CONTRACT_SCHEMA",
    "REVIEWER_SYSTEM_PROMPT",
    "ReviewerContractError",
    "ReviewerContractInput",
    "ProviderLimitExceeded",
    "ProviderLimits",
    "RealProviderNotEnabledError",
    "ReplayMissError",
    "ReplayProvider",
    "RetryPolicy",
    "ScriptedProvider",
    "SuiteEvalReport",
    "SchemaValidationError",
    "StructuredOutputValidationError",
    "TokenUsage",
    "UnsupportedAPIModeError",
    "UnsupportedProviderError",
    "UsageLedger",
    "build_contract_input",
    "build_planner_request",
    "build_reviewer_request",
    "build_usage_from_messages",
    "complete_chat",
    "create_llm_provider",
    "estimate_cost_usd",
    "extract_json_object",
    "infer_discarded_modules",
    "load_llm_config",
    "load_replay_bundle",
    "mask_secret",
    "parse_and_validate",
    "parse_planner_completion",
    "parse_reviewer_completion",
    "planning_context_to_planner_request",
    "prompt_hash",
    "proposal_to_research_lesson",
    "real_eval_gates",
    "redact_secrets",
    "request_fingerprint",
    "resolve_gateway_provider",
    "run_llm_eval_suite",
    "run_llm_plan_replay",
    "run_llm_review_replay",
    "validate_against_schema",
    "write_suite_report",
]

API_VERSION = "v1.4.4"
