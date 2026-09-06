"""Planner/Critic adapters over LLMProvider (v1.3–v1.4.5).

Default production path remains MockPlanner / MockCritic.
Fake / Replay stay offline. Real requires explicit allow_network (+ env gates)
and never silently falls back to mock.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr

from scientist_lab.agents.critic import Critic, CriticReview, MockCritic
from scientist_lab.agents.models import CandidateExperiment, PlanningContext, PlannerOutput
from scientist_lab.agents.legacy_planner import MockPlanner, Planner
from scientist_lab.llm.audit import AuditingProvider
from scientist_lab.llm.context_codec import planning_context_to_planner_request
from scientist_lab.llm.context_sanitizer import sanitize_planning_context
from scientist_lab.llm.errors import (
    LLMError,
    MissingAPIKeyError,
    RealProviderNotEnabledError,
)
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.http_transport import HttpTransport
from scientist_lab.llm.limits import LimitingProvider, ProviderLimits
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.openai_compatible_provider import OpenAICompatibleProvider
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.llm.provider import LLMProvider
from scientist_lab.llm.replay_provider import ReplayProvider
from scientist_lab.llm.repository import LLMCallRepository
from scientist_lab.llm.schema_parser import CRITIC_REVIEW_SCHEMA


ProviderMode = Literal["mock", "fake", "replay", "real", "openai-compatible"]


def normalize_provider_mode(mode: str | None) -> str:
    text = (mode or "mock").strip().lower()
    aliases = {
        "openai": "openai-compatible",
        "openai_compatible": "openai-compatible",
        "real": "real",
    }
    return aliases.get(text, text)


def _env_truthy(name: str, environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else dict(os.environ)
    return str(env.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def load_openai_config_for_runtime(
    *,
    allow_network: bool,
    environ: dict[str, str] | None = None,
    openai_config: OpenAICompatibleConfig | None = None,
) -> OpenAICompatibleConfig:
    if openai_config is not None:
        return openai_config.model_copy(update={"allow_network": allow_network})
    env = environ if environ is not None else dict(os.environ)
    key = (env.get("LLM_API_KEY") or "").strip()
    base = (env.get("LLM_BASE_URL") or "").strip()
    model = (env.get("LLM_MODEL") or "").strip()
    if not key:
        raise MissingAPIKeyError(
            "LLM_API_KEY is required for --provider real. The key is never logged."
        )
    if not base:
        raise RealProviderNotEnabledError("LLM_BASE_URL is required for --provider real")
    if not model:
        raise RealProviderNotEnabledError("LLM_MODEL is required for --provider real")
    timeout = float(env.get("LLM_TIMEOUT_SECONDS") or 60.0)
    return OpenAICompatibleConfig(
        base_url=base,
        api_key=SecretStr(key),
        model=model,
        allow_network=allow_network,
        timeout_seconds=timeout,
    )


def candidate_to_critic_request(
    candidate: CandidateExperiment,
    context: PlanningContext,
    *,
    system_prompt: str = (
        "You are a constrained experiment critic. "
        "Return JSON matching the provided schema only."
    ),
) -> LLMRequest:
    payload = {
        "project_id": context.project_id,
        "research_goal": context.research_goal,
        "protocol_id": context.protocol_id,
        "current_best_node_id": context.current_best_node_id,
        "candidate": candidate.model_dump(mode="json"),
        "allowed_parameter_changes": list(context.allowed_parameter_changes or []),
        "tested_parameter_fingerprints": list(
            context.tested_parameter_fingerprints or []
        ),
    }
    return LLMRequest(
        purpose="critic",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ],
        response_schema=CRITIC_REVIEW_SCHEMA,
        temperature=0.0,
        metadata={
            "project_id": context.project_id,
            "candidate_id": candidate.candidate_id,
            "current_best_node_id": context.current_best_node_id,
        },
    )


class ProviderPlanner(Planner):
    """Planner backed by an LLMProvider (Fake / Replay / Real)."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        sanitize: bool = False,
        requested_provider: str | None = None,
    ) -> None:
        self.provider = provider
        self.sanitize = sanitize
        self.requested_provider = requested_provider or str(
            getattr(provider, "name", "provider")
        )

    @property
    def model_provider(self) -> str:
        return str(getattr(self.provider, "name", "provider"))

    @property
    def model_name(self) -> str:
        return str(getattr(self.provider, "model", "unknown"))

    def plan(self, context: PlanningContext) -> PlannerOutput:
        ctx = sanitize_planning_context(context) if self.sanitize else context
        request = planning_context_to_planner_request(ctx)
        try:
            response = self.provider.complete(request)
        except LLMError:
            raise
        if response.parsed_json is None:
            raise ValueError(
                f"planner provider returned no JSON "
                f"(provider={response.provider}, errors={response.schema_errors})"
            )
        if not response.schema_valid:
            raise ValueError(
                "planner provider JSON failed schema validation: "
                + "; ".join(response.schema_errors or ["unknown"])
            )
        return PlannerOutput.model_validate(response.parsed_json)


class ProviderCritic(Critic):
    """Critic backed by an LLMProvider (Fake / Replay / Real)."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        sanitize: bool = False,
        requested_provider: str | None = None,
    ) -> None:
        self.provider = provider
        self.sanitize = sanitize
        self.requested_provider = requested_provider or str(
            getattr(provider, "name", "provider")
        )

    @property
    def model_provider(self) -> str:
        return str(getattr(self.provider, "name", "provider"))

    @property
    def model_name(self) -> str:
        return str(getattr(self.provider, "model", "unknown"))

    def review(
        self,
        candidate: CandidateExperiment,
        context: PlanningContext,
    ) -> CriticReview:
        ctx = sanitize_planning_context(context) if self.sanitize else context
        request = candidate_to_critic_request(candidate, ctx)
        response = self.provider.complete(request)
        if response.parsed_json is None:
            raise ValueError(
                f"critic provider returned no JSON "
                f"(provider={response.provider}, errors={response.schema_errors})"
            )
        if not response.schema_valid:
            raise ValueError(
                "critic provider JSON failed schema validation: "
                + "; ".join(response.schema_errors or ["unknown"])
            )
        data = dict(response.parsed_json)
        data["candidate_id"] = candidate.candidate_id
        return CriticReview.model_validate(data)


def build_llm_provider(
    mode: str,
    *,
    audit_root: Path | str,
    project_id: str | None = None,
    limits: ProviderLimits | None = None,
    allow_network: bool = False,
    transport: HttpTransport | None = None,
    openai_config: OpenAICompatibleConfig | None = None,
    environ: dict[str, str] | None = None,
) -> LLMProvider | None:
    """Build provider stack. Returns None for mock mode."""
    resolved = normalize_provider_mode(mode)
    if resolved == "mock":
        return None
    repo = LLMCallRepository(Path(audit_root))
    if resolved == "fake":
        inner: LLMProvider = AuditingProvider(
            FakeProvider(),
            repo,
            project_id=project_id,
            validate_schema=True,
        )
    elif resolved == "replay":
        inner = ReplayProvider(repo, validate_schema=True)
    elif resolved in {"real", "openai-compatible"}:
        # Triple gate: CLI allow_network OR explicit offline MockTransport.
        env_ok = _env_truthy("LLM_ALLOW_NETWORK", environ)
        if transport is None:
            if not allow_network:
                raise RealProviderNotEnabledError(
                    "real provider blocked: pass --allow-network "
                    "(and set LLM_ALLOW_NETWORK=true for live HTTP)"
                )
            if not env_ok:
                raise RealProviderNotEnabledError(
                    "real provider blocked: set LLM_ALLOW_NETWORK=true"
                )
        cfg = load_openai_config_for_runtime(
            allow_network=bool(allow_network or transport is not None),
            environ=environ,
            openai_config=openai_config,
        )
        if transport is None:
            cfg = cfg.model_copy(update={"allow_network": True})
        real = OpenAICompatibleProvider(cfg, transport=transport)
        inner = AuditingProvider(
            real,
            repo,
            project_id=project_id,
            validate_schema=True,
        )
    else:
        raise ValueError(f"unsupported provider mode: {mode}")
    if limits is not None:
        return LimitingProvider(inner, limits)
    return inner


def build_planner_critic(
    mode: ProviderMode | str = "mock",
    *,
    audit_root: Path | str | None = None,
    project_id: str | None = None,
    limits: ProviderLimits | None = None,
    allow_network: bool = False,
    transport: HttpTransport | None = None,
    openai_config: OpenAICompatibleConfig | None = None,
    environ: dict[str, str] | None = None,
) -> tuple[Planner, Critic]:
    """Factory used by AgentPlanningService / ExperimentService."""
    resolved = normalize_provider_mode(mode)
    if resolved == "mock":
        return MockPlanner(), MockCritic()
    if audit_root is None:
        raise ValueError("audit_root is required for non-mock provider modes")
    provider = build_llm_provider(
        resolved,
        audit_root=audit_root,
        project_id=project_id,
        limits=limits,
        allow_network=allow_network,
        transport=transport,
        openai_config=openai_config,
        environ=environ,
    )
    assert provider is not None
    sanitize = resolved in {"real", "openai-compatible"}
    return (
        ProviderPlanner(
            provider, sanitize=sanitize, requested_provider=resolved
        ),
        ProviderCritic(
            provider, sanitize=sanitize, requested_provider=resolved
        ),
    )
