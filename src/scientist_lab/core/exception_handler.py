"""Map execution failures to RETRY_EXECUTION vs HUMAN_REQUIRED.

Does not KEEP/DISCARD. Does not start GPU.
"""

from __future__ import annotations

from typing import Any, Mapping

from scientist_lab.core.state_machine import OrchestrationAction

_RETRYABLE = {"OOM", "CUDA_ERROR", "IO_ERROR"}


def next_exception_action(
    result: Mapping[str, Any] | None,
    *,
    evidence_status: str | None = None,
) -> str | None:
    if evidence_status and evidence_status != "NOT_APPLICABLE":
        return None
    execution = dict((result or {}).get("execution") or {})
    status = str(execution.get("status") or "")
    if status not in {"failed", "timeout"}:
        return None
    error_type = str(execution.get("error_type") or "")
    if status == "timeout" or error_type in _RETRYABLE:
        return OrchestrationAction.RETRY_EXECUTION.value
    return OrchestrationAction.NEED_HUMAN.value
