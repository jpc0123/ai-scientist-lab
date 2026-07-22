"""Planner/Critic adapters over LLMProvider (v1.3.6 / v1.3.7).

Default production path remains MockPlanner / MockCritic.
Fake / Replay modes go through AuditingProvider and never open network sockets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from scientist_lab.agents.critic import Critic, CriticReview, MockCritic
from scientist_lab.agents.models import CandidateExperiment, PlanningContext, PlannerOutput
from scientist_lab.agents.planner import MockPlanner, Planner
from scientist_lab.llm.audit import AuditingProvider
from scientist_lab.llm.context_codec import planning_context_to_planner_request
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.limits import LimitingProvider, ProviderLimits
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.provider import LLMProvider
from scientist_lab.llm.replay_provider import ReplayProvider
from scientist_lab.llm.repository import LLMCallRepository
from scientist_lab.llm.schema_parser import CRITIC_REVIEW_SCHEMA


ProviderMode = Literal["mock", "fake", "replay"]


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
    """Planner backed by an LLMProvider (Fake / Replay / future Real)."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    @property
    def model_provider(self) -> str:
        return str(getattr(self.provider, "name", "provider"))

    @property
    def model_name(self) -> str:
        return str(getattr(self.provider, "model", "unknown"))

    def plan(self, context: PlanningContext) -> PlannerOutput:
        request = planning_context_to_planner_request(context)
        response = self.provider.complete(request)
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
    """Critic backed by an LLMProvider (Fake / Replay / future Real)."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

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
        request = candidate_to_critic_request(candidate, context)
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
        # Ensure candidate_id alignment even if provider echoes a fixed id.
        data["candidate_id"] = candidate.candidate_id
        return CriticReview.model_validate(data)


def build_llm_provider(
    mode: ProviderMode,
    *,
    audit_root: Path | str,
    project_id: str | None = None,
    limits: ProviderLimits | None = None,
) -> LLMProvider | None:
    """Build Fake/Replay provider stack. Returns None for mock mode.

    Stack (fake): LimitingProvider → AuditingProvider → FakeProvider
    Stack (replay): LimitingProvider → ReplayProvider
    """
    if mode == "mock":
        return None
    repo = LLMCallRepository(Path(audit_root))
    if mode == "fake":
        inner: LLMProvider = AuditingProvider(
            FakeProvider(),
            repo,
            project_id=project_id,
            validate_schema=True,
        )
    elif mode == "replay":
        inner = ReplayProvider(repo, validate_schema=True)
    else:
        raise ValueError(f"unsupported provider mode: {mode}")
    if limits is not None:
        return LimitingProvider(inner, limits)
    return inner


def build_planner_critic(
    mode: ProviderMode = "mock",
    *,
    audit_root: Path | str | None = None,
    project_id: str | None = None,
    limits: ProviderLimits | None = None,
) -> tuple[Planner, Critic]:
    """Factory used by AgentPlanningService / ExperimentService."""
    if mode == "mock":
        return MockPlanner(), MockCritic()
    if audit_root is None:
        raise ValueError("audit_root is required for fake/replay provider modes")
    provider = build_llm_provider(
        mode,
        audit_root=audit_root,
        project_id=project_id,
        limits=limits,
    )
    assert provider is not None
    return ProviderPlanner(provider), ProviderCritic(provider)
