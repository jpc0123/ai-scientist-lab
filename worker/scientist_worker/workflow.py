from __future__ import annotations

from scientist_worker.models import WORKER_TRANSITIONS


class InvalidWorkerTransition(ValueError):
    pass


def transition(current: str, target: str) -> str:
    allowed = WORKER_TRANSITIONS.get(current, set())
    if target not in allowed and target != current:
        # terminal states stay put
        if current in {
            "completed",
            "failed",
            "cancelled",
            "timed_out",
            "validation_failed",
            "prepare_failed",
            "artifact_failed",
        }:
            raise InvalidWorkerTransition(
                f"job already terminal status={current}; cannot move to {target}"
            )
        raise InvalidWorkerTransition(
            f"invalid worker transition {current} -> {target}; "
            f"allowed={sorted(allowed)}"
        )
    return target
