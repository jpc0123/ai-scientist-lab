from scientist_lab.iteration.models import ApprovalRecord, IterationSession
from scientist_lab.iteration.workflow import (
    ALLOWED_TRANSITIONS,
    InvalidIterationTransition,
    transition,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ApprovalRecord",
    "InvalidIterationTransition",
    "IterationService",
    "IterationSession",
    "transition",
]


def __getattr__(name: str):
    if name == "IterationService":
        from scientist_lab.iteration.service import IterationService

        return IterationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
