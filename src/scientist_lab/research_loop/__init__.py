"""v2.1 real research loop — session models, state machine, and offline service."""

from scientist_lab.research_loop.errors import (
    InvalidRealLoopTransition,
    RealLoopError,
    RealLoopNotFoundError,
    RealLoopValidationError,
)
from scientist_lab.research_loop.feedback import (
    build_round_feedback_summary,
    require_round_feedback_for_planning,
)
from scientist_lab.research_loop.feedback_use_verifier import verify_feedback_use
from scientist_lab.research_loop.models import (
    FAILURE_STATUSES,
    REAL_LOOP_STATUSES,
    FeedbackUsageRecord,
    FeedbackUseVerification,
    RealResearchLoopSession,
    ResearchLoopRound,
    RealLoopStatus,
)
from scientist_lab.research_loop.replay_bundle import (
    assert_bundle_redacted,
    build_replay_bundle,
    load_replay_bundle,
)
from scientist_lab.research_loop.service import RealResearchLoopService

__all__ = [
    "FAILURE_STATUSES",
    "REAL_LOOP_STATUSES",
    "FeedbackUsageRecord",
    "FeedbackUseVerification",
    "InvalidRealLoopTransition",
    "RealLoopError",
    "RealLoopNotFoundError",
    "RealLoopStatus",
    "RealLoopValidationError",
    "RealResearchLoopService",
    "RealResearchLoopSession",
    "ResearchLoopRound",
    "assert_bundle_redacted",
    "build_replay_bundle",
    "build_round_feedback_summary",
    "load_replay_bundle",
    "require_round_feedback_for_planning",
    "verify_feedback_use",
]
