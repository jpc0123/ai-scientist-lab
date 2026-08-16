"""Read-only scientific_outcome projection. Not a second reviewer."""

from __future__ import annotations

from scientist_lab.core.state_machine import EvidenceStatus, RunState


def project_scientific_outcome(
    *,
    run_state: RunState | str,
    evidence_status: EvidenceStatus | str,
    primary_delta: float | None,
    constraints_ok: bool = True,
    positive_threshold: float = 0.0,
    negative_threshold: float = 0.0,
) -> str:
    """Map existing facts to POSITIVE/NEGATIVE/INCONCLUSIVE/NOT_EVALUATED.

    Does not read review_decision. KEEP may still apply to INCONCLUSIVE, etc.
    """
    rs = run_state.value if isinstance(run_state, RunState) else run_state
    ev = evidence_status.value if isinstance(evidence_status, EvidenceStatus) else evidence_status

    if rs in {"FAILED", "BLOCKED"}:
        return "NOT_EVALUATED"
    if ev in {"INVALID", "INCOMPLETE", "NOT_APPLICABLE", "PENDING"}:
        return "NOT_EVALUATED"
    if rs != "COMPLETED" and rs != "MEMORY_WRITTEN":
        return "NOT_EVALUATED"
    if not constraints_ok:
        return "INCONCLUSIVE"
    if primary_delta is None:
        return "INCONCLUSIVE"
    if primary_delta > positive_threshold:
        return "POSITIVE"
    if primary_delta < negative_threshold:
        return "NEGATIVE"
    return "INCONCLUSIVE"
