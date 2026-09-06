"""Three-field experiment run state machine (Architecture Freeze).

run_state / evidence_status / review_decision are separate facts.
Manager reads these facts; it does not invent them via prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RunState(str, Enum):
    CREATED = "CREATED"
    PLANNED = "PLANNED"
    MATERIALIZED = "MATERIALIZED"  # candidate contract exists
    GATED = "GATED"
    CONTRACTED = "CONTRACTED"  # frozen contract ready (alias path after approve)
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    MEMORY_WRITTEN = "MEMORY_WRITTEN"
    STOPPED = "STOPPED"


class EvidenceStatus(str, Enum):
    PENDING = "PENDING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    VALID = "VALID"
    INVALID = "INVALID"
    INCOMPLETE = "INCOMPLETE"


class ReviewDecisionValue(str, Enum):
    PENDING = "PENDING"
    KEEP = "KEEP"
    DISCARD = "DISCARD"
    REPLICATE = "REPLICATE"
    VALIDATE = "VALIDATE"
    ESCALATE = "ESCALATE"
    STOP = "STOP"


class OrchestrationAction(str, Enum):
    NEED_PLAN = "NEED_PLAN"
    NEED_MATERIALIZE = "NEED_MATERIALIZE"
    NEED_GATE = "NEED_GATE"
    NEED_HUMAN = "NEED_HUMAN"
    NEED_EXECUTION = "NEED_EXECUTION"
    NEED_PARSE = "NEED_PARSE"
    NEED_EVIDENCE_CHECK = "NEED_EVIDENCE_CHECK"
    NEED_REVIEW = "NEED_REVIEW"
    NEED_MEMORY = "NEED_MEMORY"
    RETRY_EXECUTION = "RETRY_EXECUTION"
    NEXT_ROUND = "NEXT_ROUND"
    STOP = "STOP"
    PROTOCOL_AMENDMENT_REQUIRED = "PROTOCOL_AMENDMENT_REQUIRED"
    NOVELTY_EXHAUSTED = "NOVELTY_EXHAUSTED"
    IDLE = "IDLE"


class InvalidTransition(ValueError):
    pass


# Legal run_state transitions (orchestration fact only).
RUN_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.PLANNED, RunState.STOPPED},
    RunState.PLANNED: {RunState.MATERIALIZED, RunState.BLOCKED, RunState.STOPPED},
    # Auto risk: MATERIALIZED -> APPROVED; approval_required: MATERIALIZED -> GATED (wait human)
    RunState.MATERIALIZED: {
        RunState.GATED,
        RunState.APPROVED,
        RunState.BLOCKED,
        RunState.STOPPED,
    },
    RunState.GATED: {
        RunState.APPROVED,
        RunState.BLOCKED,
        RunState.STOPPED,
    },
    RunState.APPROVED: {RunState.CONTRACTED, RunState.RUNNING, RunState.BLOCKED},
    RunState.CONTRACTED: {RunState.RUNNING, RunState.BLOCKED},
    RunState.RUNNING: {RunState.COMPLETED, RunState.FAILED, RunState.BLOCKED},
    RunState.COMPLETED: {RunState.MEMORY_WRITTEN, RunState.STOPPED},
    RunState.FAILED: {
        RunState.APPROVED,  # RETRY_EXECUTION re-enter after fix
        RunState.RUNNING,
        RunState.STOPPED,
        RunState.MEMORY_WRITTEN,  # record failure then stop/next
    },
    RunState.BLOCKED: {RunState.STOPPED, RunState.PLANNED},  # replan after block
    RunState.MEMORY_WRITTEN: {RunState.STOPPED},
    RunState.STOPPED: set(),
}


@dataclass(frozen=True)
class ExperimentRunState:
    run_state: RunState
    evidence_status: EvidenceStatus = EvidenceStatus.PENDING
    review_decision: ReviewDecisionValue = ReviewDecisionValue.PENDING

    @classmethod
    def initial(cls) -> ExperimentRunState:
        return cls(
            run_state=RunState.CREATED,
            evidence_status=EvidenceStatus.PENDING,
            review_decision=ReviewDecisionValue.PENDING,
        )


def transition(
    state: ExperimentRunState,
    *,
    run_state: RunState | None = None,
    evidence_status: EvidenceStatus | None = None,
    review_decision: ReviewDecisionValue | None = None,
) -> ExperimentRunState:
    """Apply a partial update with legality checks."""
    new_run = run_state if run_state is not None else state.run_state
    new_ev = evidence_status if evidence_status is not None else state.evidence_status
    new_rev = review_decision if review_decision is not None else state.review_decision

    if run_state is not None and run_state != state.run_state:
        allowed = RUN_TRANSITIONS.get(state.run_state, set())
        if run_state not in allowed:
            raise InvalidTransition(
                f"Illegal run_state transition: {state.run_state.value} -> {run_state.value}"
            )

    new_state = ExperimentRunState(
        run_state=new_run,
        evidence_status=new_ev,
        review_decision=new_rev,
    )
    _assert_field_consistency(new_state)
    return new_state


def _assert_field_consistency(state: ExperimentRunState) -> None:
    """Cross-field invariants from Architecture Freeze."""
    rs, ev, rd = state.run_state, state.evidence_status, state.review_decision

    # Scientific decision only after VALID evidence (or still PENDING).
    if rd not in (ReviewDecisionValue.PENDING,) and ev not in (
        EvidenceStatus.VALID,
    ):
        # ESCALATE/STOP may happen on INVALID path only if we explicitly allow STOP;
        # architecture: invalid evidence must NOT enter scientific KEEP/DISCARD/etc.
        if rd in {
            ReviewDecisionValue.KEEP,
            ReviewDecisionValue.DISCARD,
            ReviewDecisionValue.REPLICATE,
            ReviewDecisionValue.VALIDATE,
        }:
            raise InvalidTransition(
                f"review_decision={rd.value} requires evidence_status=VALID, got {ev.value}"
            )

    if rs == RunState.FAILED and ev == EvidenceStatus.VALID:
        raise InvalidTransition("FAILED run cannot have VALID evidence")

    if rs in {RunState.CREATED, RunState.PLANNED, RunState.MATERIALIZED, RunState.GATED}:
        if ev not in {EvidenceStatus.PENDING, EvidenceStatus.NOT_APPLICABLE}:
            raise InvalidTransition(
                f"{rs.value} cannot have evidence_status={ev.value}"
            )
        if rd != ReviewDecisionValue.PENDING:
            raise InvalidTransition(f"{rs.value} requires review_decision=PENDING")

    if rs == RunState.RUNNING and rd != ReviewDecisionValue.PENDING:
        raise InvalidTransition("RUNNING requires review_decision=PENDING")

    if rs == RunState.BLOCKED and rd not in {
        ReviewDecisionValue.PENDING,
        ReviewDecisionValue.ESCALATE,
        ReviewDecisionValue.STOP,
    }:
        raise InvalidTransition("BLOCKED cannot carry KEEP/DISCARD/REPLICATE/VALIDATE")


def next_orchestration_action(state: ExperimentRunState) -> OrchestrationAction:
    """Deterministic Manager hint from facts (not LLM)."""
    rs, ev, rd = state.run_state, state.evidence_status, state.review_decision

    if rs == RunState.CREATED:
        return OrchestrationAction.NEED_PLAN
    if rs == RunState.PLANNED:
        return OrchestrationAction.NEED_MATERIALIZE
    if rs == RunState.MATERIALIZED:
        return OrchestrationAction.NEED_GATE
    if rs == RunState.GATED:
        return OrchestrationAction.NEED_HUMAN  # waiting approval path; auto-approve skips here
    if rs in {RunState.APPROVED, RunState.CONTRACTED}:
        return OrchestrationAction.NEED_EXECUTION
    if rs == RunState.RUNNING:
        return OrchestrationAction.NEED_PARSE
    if rs == RunState.COMPLETED:
        if ev in {EvidenceStatus.PENDING}:
            return OrchestrationAction.NEED_EVIDENCE_CHECK
        if ev in {
            EvidenceStatus.INVALID,
            EvidenceStatus.INCOMPLETE,
            EvidenceStatus.NOT_APPLICABLE,
        }:
            return OrchestrationAction.NEED_MEMORY  # record; no scientific review
        if ev == EvidenceStatus.VALID and rd == ReviewDecisionValue.PENDING:
            return OrchestrationAction.NEED_REVIEW
        if rd != ReviewDecisionValue.PENDING:
            return OrchestrationAction.NEED_MEMORY
        return OrchestrationAction.NEED_EVIDENCE_CHECK
    if rs == RunState.FAILED:
        return OrchestrationAction.RETRY_EXECUTION
    if rs == RunState.BLOCKED:
        return OrchestrationAction.STOP
    if rs == RunState.MEMORY_WRITTEN:
        if rd == ReviewDecisionValue.STOP:
            return OrchestrationAction.STOP
        if rd == ReviewDecisionValue.ESCALATE:
            return OrchestrationAction.NEED_HUMAN
        return OrchestrationAction.NEXT_ROUND
    if rs == RunState.STOPPED:
        return OrchestrationAction.STOP
    return OrchestrationAction.IDLE


def apply_happy_path_negative_result(state: ExperimentRunState) -> ExperimentRunState:
    """Helper scenario: COMPLETED + VALID + DISCARD (scientific negative)."""
    s = state
    for step in (
        (RunState.PLANNED, None, None),
        (RunState.MATERIALIZED, None, None),
        (RunState.GATED, None, None),
        (RunState.APPROVED, None, None),
        (RunState.RUNNING, None, None),
        (RunState.COMPLETED, EvidenceStatus.PENDING, None),
        (None, EvidenceStatus.VALID, None),
        (None, None, ReviewDecisionValue.DISCARD),
        (RunState.MEMORY_WRITTEN, None, None),
    ):
        s = transition(
            s,
            run_state=step[0],
            evidence_status=step[1],
            review_decision=step[2],
        )
    return s
