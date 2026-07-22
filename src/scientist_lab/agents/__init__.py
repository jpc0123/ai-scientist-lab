from __future__ import annotations

from typing import Any

from scientist_lab.agents.critic import CriticReview, MockCritic
from scientist_lab.agents.models import (
    CandidateExperiment,
    CandidateVerification,
    PlanningContext,
    PlannerOutput,
)
from scientist_lab.agents.planner import MockPlanner, Planner
from scientist_lab.agents.service import AgentPlanningService

__all__ = [
    "AgentPlanningService",
    "CandidateExperiment",
    "CandidateVerification",
    "CriticReview",
    "MockCritic",
    "MockPlanner",
    "Planner",
    "PlanningContext",
    "PlannerOutput",
    "ProviderCritic",
    "ProviderPlanner",
    "build_planner_critic",
]


def __getattr__(name: str) -> Any:
    # Lazy export to avoid llm ↔ agents circular imports at package import time.
    if name in {"ProviderCritic", "ProviderPlanner", "build_planner_critic"}:
        from scientist_lab.agents import provider_bridge

        return getattr(provider_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
