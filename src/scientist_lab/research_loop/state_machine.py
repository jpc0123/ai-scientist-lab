"""State machine for RealResearchLoopSession (v2.1.1)."""

from __future__ import annotations

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.research_loop.errors import InvalidRealLoopTransition
from scientist_lab.research_loop.models import FAILURE_STATUSES, RealResearchLoopSession


# Happy-path transitions; failure/cancel can be entered from most active states.
_ACTIVE = {
    "created",
    "baseline_ready",
    "round_1_planning",
    "round_1_reviewing",
    "round_1_waiting_approval",
    "round_1_executing",
    "round_1_feedback_ready",
    "round_2_planning",
    "round_2_reviewing",
}

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "created": {"baseline_ready", "cancelled", "quality_gate_blocked"},
    "baseline_ready": {
        "round_1_planning",
        "cancelled",
        "quality_gate_blocked",
        "provider_failed",
    },
    "round_1_planning": {
        "round_1_reviewing",
        "planner_failed",
        "provider_failed",
        "cancelled",
        "quality_gate_blocked",
    },
    "round_1_reviewing": {
        "round_1_waiting_approval",
        "critic_failed",
        "candidate_rejected",
        "provider_failed",
        "cancelled",
        "quality_gate_blocked",
    },
    "round_1_waiting_approval": {
        "round_1_executing",
        "candidate_rejected",
        "cancelled",
    },
    "round_1_executing": {
        "round_1_feedback_ready",
        "execution_failed",
        "cancelled",
    },
    "round_1_feedback_ready": {
        "round_2_planning",
        "feedback_incomplete",
        "cancelled",
        "completed",  # allow early stop if required_rounds == 1 (future)
    },
    "round_2_planning": {
        "round_2_reviewing",
        "planner_failed",
        "provider_failed",
        "cancelled",
        "quality_gate_blocked",
    },
    "round_2_reviewing": {
        "completed",
        "critic_failed",
        "candidate_rejected",
        "provider_failed",
        "cancelled",
        "quality_gate_blocked",
    },
    "completed": set(),
}

for _status in _ACTIVE:
    ALLOWED_TRANSITIONS.setdefault(_status, set())
    ALLOWED_TRANSITIONS[_status] |= set(FAILURE_STATUSES)

# Terminal failure states have no outgoing transitions.
for _fail in FAILURE_STATUSES:
    ALLOWED_TRANSITIONS.setdefault(_fail, set())

NEXT_ACTIONS: dict[str, str] = {
    "created": "Attach/confirm baseline nodes, then mark baseline_ready.",
    "baseline_ready": "Run real-loop-plan (network gated) for round 1.",
    "round_1_planning": "Waiting for Planner output / review step.",
    "round_1_reviewing": "Waiting for Critic + rule verifiers.",
    "round_1_waiting_approval": "Human: real-loop-approve --candidate <id>, then real-loop-execute.",
    "round_1_executing": "Digits done; next: real-loop-record-execution-feedback.",
    "round_1_feedback_ready": "Evidence ready; next: real-loop-next-round (Round 2 context).",
    "round_2_planning": "Run real-loop-plan for round 2, then real-loop-verify-feedback.",
    "round_2_reviewing": "Run real-loop-verify-feedback, then critic/approval.",
    "completed": "Export Replay Bundle with real-loop-export; freeze or archive.",
    "provider_failed": "Inspect provider audit; retry explicitly (no silent fallback).",
    "planner_failed": "Inspect planner audit; retry planning explicitly.",
    "critic_failed": "Inspect critic audit; retry review explicitly.",
    "candidate_rejected": "Start a new plan step or cancel the session.",
    "execution_failed": "Inspect execution; do not auto-rerun expensive jobs.",
    "feedback_incomplete": "Repair Evidence/Claim feedback before round 2.",
    "quality_gate_blocked": "Select a qualified Model Profile.",
    "cancelled": "Session cancelled.",
}


def transition(
    session: RealResearchLoopSession, target_status: str
) -> RealResearchLoopSession:
    allowed = ALLOWED_TRANSITIONS.get(session.status, set())
    if target_status not in allowed:
        raise InvalidRealLoopTransition(
            f"Cannot transition real-loop from {session.status} to {target_status}"
        )
    session.status = target_status  # type: ignore[assignment]
    session.touch()
    if target_status == "completed" or target_status in FAILURE_STATUSES:
        session.completed_at = utc_now_iso()
    return session


def require_status(session: RealResearchLoopSession, expected: str) -> None:
    if session.status != expected:
        raise InvalidRealLoopTransition(
            f"Real-loop {session.session_id} must be in '{expected}', "
            f"but is '{session.status}'"
        )


def can_enter_round_2(session: RealResearchLoopSession) -> bool:
    return session.status == "round_1_feedback_ready" and session.required_rounds >= 2
