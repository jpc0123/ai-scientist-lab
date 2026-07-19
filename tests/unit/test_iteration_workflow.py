from __future__ import annotations

import pytest

from scientist_lab.iteration.models import IterationSession
from scientist_lab.iteration.workflow import (
    ALLOWED_TRANSITIONS,
    InvalidIterationTransition,
    require_status,
    transition,
)


def _session(status: str = "created") -> IterationSession:
    return IterationSession(
        iteration_id="iter_test",
        project_id="project_001",
        source_baseline_node_id="node_003",
        source_candidate_node_id="node_004",
        status=status,  # type: ignore[arg-type]
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def test_happy_path_transitions():
    session = _session("created")
    transition(session, "feedback_ready")
    transition(session, "proposal_ready")
    transition(session, "waiting_approval")
    transition(session, "approved")
    transition(session, "running")
    transition(session, "comparing")
    transition(session, "waiting_decision")
    transition(session, "completed")
    assert session.status == "completed"


def test_reject_path():
    session = _session("waiting_approval")
    transition(session, "rejected")
    assert session.status == "rejected"
    with pytest.raises(InvalidIterationTransition):
        transition(session, "approved")


def test_stopped_no_recommendation_is_terminal_from_feedback():
    session = _session("feedback_ready")
    transition(session, "stopped_no_recommendation")
    assert "stopped_no_recommendation" not in ALLOWED_TRANSITIONS
    with pytest.raises(InvalidIterationTransition):
        transition(session, "waiting_approval")


def test_require_status():
    session = _session("waiting_approval")
    require_status(session, "waiting_approval")
    with pytest.raises(InvalidIterationTransition):
        require_status(session, "waiting_decision")
