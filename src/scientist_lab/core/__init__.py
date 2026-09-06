"""Autonomous-loop core: schemas, state machine (Architecture Freeze / M1)."""

from scientist_lab.core.claim_gate import ClaimGate, ClaimVerdict, evaluate_claim
from scientist_lab.core.decision_rubric import RubricResult, evaluate_rubric
from scientist_lab.core.evidence_validator import EvidenceValidator, EvidenceVerdict
from scientist_lab.core.exception_handler import next_exception_action
from scientist_lab.core.gate_engine import GateEngine, GateStatus, GateVerdict
from scientist_lab.core.git_manager import GitManager, GitManagerError, GitPointers, is_git_repo
from scientist_lab.core.invariants import (
    InvariantError,
    assert_discard_is_not_module_ineffective,
    assert_keep_is_not_claim,
    assert_lesson_has_evidence,
    assert_memory_refs_resolvable,
    assert_plan_memory_policy,
    evaluate_stop_rules,
)
from scientist_lab.core.manager import Manager, ManagerStep
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.next_plan import (
    build_candidate_next_plan,
    gate_candidate_next_plan,
    memory_refs_from_run,
)
from scientist_lab.core.planner import (
    PlanPacket,
    Planner,
    PlanRefused,
    propose_and_gate_next,
)
from scientist_lab.core.result_parser import ResultParser
from scientist_lab.core.reviewer import ReviewPacket, ReviewRefused, Reviewer
from scientist_lab.core.schema_registry import (
    SCHEMA_DIR,
    SchemaValidationError,
    list_schema_names,
    load_json,
    validate_document,
    validate_named,
)
from scientist_lab.core.scientific_outcome import project_scientific_outcome
from scientist_lab.core.state_machine import (
    EvidenceStatus,
    ExperimentRunState,
    InvalidTransition,
    OrchestrationAction,
    ReviewDecisionValue,
    RunState,
    next_orchestration_action,
    transition,
)

__all__ = [
    "SCHEMA_DIR",
    "SchemaValidationError",
    "list_schema_names",
    "load_json",
    "validate_document",
    "validate_named",
    "EvidenceStatus",
    "ExperimentRunState",
    "InvalidTransition",
    "OrchestrationAction",
    "ReviewDecisionValue",
    "RunState",
    "next_orchestration_action",
    "transition",
    "RubricResult",
    "evaluate_rubric",
    "InvariantError",
    "assert_lesson_has_evidence",
    "assert_memory_refs_resolvable",
    "assert_plan_memory_policy",
    "assert_keep_is_not_claim",
    "assert_discard_is_not_module_ineffective",
    "evaluate_stop_rules",
    "ClaimGate",
    "ClaimVerdict",
    "evaluate_claim",
    "project_scientific_outcome",
    "GateEngine",
    "GateStatus",
    "GateVerdict",
    "EvidenceValidator",
    "EvidenceVerdict",
    "ResultParser",
    "next_exception_action",
    "GitManager",
    "GitManagerError",
    "GitPointers",
    "is_git_repo",
    "MemoryWriter",
    "build_candidate_next_plan",
    "gate_candidate_next_plan",
    "memory_refs_from_run",
    "Planner",
    "PlanPacket",
    "PlanRefused",
    "propose_and_gate_next",
    "Manager",
    "ManagerStep",
    "Reviewer",
    "ReviewPacket",
    "ReviewRefused",
]
