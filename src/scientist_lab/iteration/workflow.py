from __future__ import annotations

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.iteration.models import IterationSession


class InvalidIterationTransition(ValueError):
    """Raised when an IterationSession status transition is not allowed."""


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "created": {
        "feedback_ready",
        "failed",
    },
    "feedback_ready": {
        "proposal_ready",
        "stopped_no_recommendation",
        "failed",
    },
    "proposal_ready": {
        "waiting_approval",
        "failed",
    },
    "waiting_approval": {
        "approved",
        "rejected",
        "cancelled",
    },
    "approved": {
        "running",
        "failed",
    },
    "running": {
        "comparing",
        "failed",
        "cancelled",
    },
    "comparing": {
        "waiting_decision",
        "failed",
    },
    "waiting_decision": {
        "completed",
        "cancelled",
    },
}


def transition(session: IterationSession, target_status: str) -> IterationSession:
    allowed = ALLOWED_TRANSITIONS.get(session.status, set())
    if target_status not in allowed:
        raise InvalidIterationTransition(
            f"Cannot transition from {session.status} to {target_status}"
        )
    session.status = target_status  # type: ignore[assignment]
    session.updated_at = utc_now_iso()
    return session


def require_status(session: IterationSession, expected: str) -> None:
    if session.status != expected:
        raise InvalidIterationTransition(
            f"Iteration {session.iteration_id} must be in '{expected}', "
            f"but is '{session.status}'"
        )
