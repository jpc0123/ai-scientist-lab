"""Domain models for real multi-round research loops (v2.1.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from scientist_lab.domain.models import new_id, utc_now_iso


RealLoopStatus = Literal[
    "created",
    "baseline_ready",
    "round_1_planning",
    "round_1_reviewing",
    "round_1_waiting_approval",
    "round_1_executing",
    "round_1_feedback_ready",
    "round_2_planning",
    "round_2_reviewing",
    "completed",
    "provider_failed",
    "planner_failed",
    "critic_failed",
    "candidate_rejected",
    "execution_failed",
    "feedback_incomplete",
    "quality_gate_blocked",
    "cancelled",
]

REAL_LOOP_STATUSES: tuple[str, ...] = (
    "created",
    "baseline_ready",
    "round_1_planning",
    "round_1_reviewing",
    "round_1_waiting_approval",
    "round_1_executing",
    "round_1_feedback_ready",
    "round_2_planning",
    "round_2_reviewing",
    "completed",
    "provider_failed",
    "planner_failed",
    "critic_failed",
    "candidate_rejected",
    "execution_failed",
    "feedback_incomplete",
    "quality_gate_blocked",
    "cancelled",
)

FAILURE_STATUSES: frozenset[str] = frozenset(
    {
        "provider_failed",
        "planner_failed",
        "critic_failed",
        "candidate_rejected",
        "execution_failed",
        "feedback_incomplete",
        "quality_gate_blocked",
        "cancelled",
    }
)

RoundStatus = Literal[
    "pending",
    "planning",
    "reviewing",
    "waiting_approval",
    "executing",
    "feedback_ready",
    "completed",
    "failed",
    "cancelled",
]


class ResearchLoopRound(BaseModel):
    round_id: str
    session_id: str
    round_number: int

    planning_context_id: str = ""
    planner_call_id: str | None = None
    critic_call_ids: list[str] = Field(default_factory=list)

    candidate_ids: list[str] = Field(default_factory=list)
    approved_candidate_id: str | None = None

    contract_id: str | None = None
    iteration_id: str | None = None
    execution_node_id: str | None = None

    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)

    # v2.1.2: persisted feedback used to build the next PlanningContext
    feedback_summary_json: dict[str, Any] = Field(default_factory=dict)
    planning_context_json: dict[str, Any] = Field(default_factory=dict)
    # v2.1.3: provider audit for plan/review calls
    provider_audit_json: dict[str, Any] = Field(default_factory=dict)
    plan_id: str | None = None

    status: RoundStatus = "pending"
    created_at: str = ""
    completed_at: str | None = None

    @classmethod
    def create(cls, *, session_id: str, round_number: int) -> ResearchLoopRound:
        now = utc_now_iso()
        return cls(
            round_id=new_id("rlround"),
            session_id=session_id,
            round_number=round_number,
            created_at=now,
        )


class RealResearchLoopSession(BaseModel):
    session_id: str
    project_id: str
    tree_id: str | None = None

    provider_profile_id: str
    protocol_id: str

    status: RealLoopStatus = "created"
    current_round: int = 0
    required_rounds: int = 2

    baseline_node_ids: list[str] = Field(default_factory=list)
    generated_candidate_ids: list[str] = Field(default_factory=list)
    execution_node_ids: list[str] = Field(default_factory=list)

    fallback_allowed: bool = False
    fallback_used: bool = False
    real_only: bool = True

    error_type: str | None = None
    error_message: str | None = None

    created_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    @classmethod
    def create(
        cls,
        *,
        project_id: str,
        provider_profile_id: str,
        protocol_id: str,
        required_rounds: int = 2,
        baseline_node_ids: list[str] | None = None,
        tree_id: str | None = None,
        fallback_allowed: bool = False,
    ) -> RealResearchLoopSession:
        now = utc_now_iso()
        return cls(
            session_id=new_id("realloop"),
            project_id=project_id,
            tree_id=tree_id,
            provider_profile_id=provider_profile_id,
            protocol_id=protocol_id,
            status="created",
            current_round=0,
            required_rounds=max(2, int(required_rounds)),
            baseline_node_ids=list(baseline_node_ids or []),
            fallback_allowed=bool(fallback_allowed),
            fallback_used=False,
            real_only=not bool(fallback_allowed),
            created_at=now,
            updated_at=now,
        )

    def to_view(self) -> dict:
        return self.model_dump(mode="json")


class FeedbackUseVerification(BaseModel):
    """Six deterministic dims proving Round N used Round N-1 feedback (v2.1.6)."""

    context_contains_round_1_metrics: bool = False
    context_contains_round_1_evidence: bool = False
    context_contains_round_1_decision: bool = False
    output_references_new_evidence: bool = False
    output_changes_experiment_plan: bool = False
    output_avoids_duplicate_candidate: bool = False
    pass_status: bool = False
    issues: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class FeedbackUsageRecord(BaseModel):
    feedback_usage_id: str
    session_id: str
    source_round_id: str
    target_round_id: str
    source_round_number: int
    target_round_number: int
    context_sha256: str = ""
    plan_id: str | None = None
    verified: bool = False
    verification: FeedbackUseVerification = Field(default_factory=FeedbackUseVerification)
    created_at: str = ""

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        source_round_id: str,
        target_round_id: str,
        source_round_number: int,
        target_round_number: int,
        context_sha256: str = "",
        plan_id: str | None = None,
        verification: FeedbackUseVerification | None = None,
    ) -> FeedbackUsageRecord:
        ver = verification or FeedbackUseVerification()
        return cls(
            feedback_usage_id=new_id("fbuse"),
            session_id=session_id,
            source_round_id=source_round_id,
            target_round_id=target_round_id,
            source_round_number=int(source_round_number),
            target_round_number=int(target_round_number),
            context_sha256=context_sha256,
            plan_id=plan_id,
            verified=bool(ver.pass_status),
            verification=ver,
            created_at=utc_now_iso(),
        )
