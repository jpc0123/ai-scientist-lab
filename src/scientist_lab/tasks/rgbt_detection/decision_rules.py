"""Decision-type restrictions for RGB-T debug / smoke / Fast Eval iterations."""

from __future__ import annotations

from typing import Any


ALLOWED_SMOKE_DECISION_TYPES = frozenset(
    {
        "pipeline_validated",
        "ready_for_fast_eval",
        "debug_failure_fixed",
        "pipeline_validation",
    }
)

ALLOWED_EXPLORATORY_DECISION_TYPES = frozenset(
    {
        "ready_for_full_evaluation",
        "exploratory_improvement",
        "exploratory_tradeoff",
        "pipeline_validated",
        "ready_for_fast_eval",
    }
)

FORBIDDEN_SMOKE_DECISION_TYPES = frozenset(
    {
        "performance_winner",
        "best_model",
        "superior_method",
        "state_of_the_art",
        "sota",
        "statistically_significant",
    }
)


def is_smoke_detection_context(
    *,
    task_type: str | None,
    execution_mode: str | None,
    claim_level: str | None = None,
) -> bool:
    if (task_type or "").strip() != "rgbt_detection":
        return False
    claim = (claim_level or "").strip()
    if claim == "exploratory_comparison":
        return True
    mode = (execution_mode or "").strip()
    if mode in {"smoke_train", "validate_data", "smoke_test", "fast_eval"}:
        return True
    return claim == "pipeline_validation_only"


def is_exploratory_fast_eval_context(
    *,
    task_type: str | None,
    execution_mode: str | None,
    claim_level: str | None = None,
    evaluation_scope: str | None = None,
) -> bool:
    if (task_type or "").strip() != "rgbt_detection":
        return False
    if (claim_level or "").strip() == "exploratory_comparison":
        return True
    if (evaluation_scope or "").strip() == "fast_eval_subset":
        return True
    return False


def validate_smoke_decision(
    decision_type: str,
    *,
    evidence_strength: str = "weak",
    claim_level: str | None = None,
) -> dict[str, Any]:
    dtype = (decision_type or "").strip()
    strength = (evidence_strength or "").strip() or "weak"
    claim = (claim_level or "").strip() or "pipeline_validation_only"

    if dtype in FORBIDDEN_SMOKE_DECISION_TYPES:
        raise ValueError(
            f"detection claim gate forbids decision_type={dtype!r}; "
            f"use exploratory or pipeline decision types instead"
        )

    if claim == "exploratory_comparison":
        allowed = ALLOWED_EXPLORATORY_DECISION_TYPES
        if dtype not in allowed:
            raise ValueError(
                f"exploratory Fast Eval only allows decision_type in "
                f"{sorted(allowed)}; got {dtype!r}"
            )
        if strength in {"strong", "moderate"}:
            strength = "weak"
        return {
            "decision_type": dtype,
            "evidence_strength": strength,
            "claim_level": "exploratory_comparison",
        }

    if dtype not in ALLOWED_SMOKE_DECISION_TYPES:
        raise ValueError(
            f"smoke detection only allows decision_type in "
            f"{sorted(ALLOWED_SMOKE_DECISION_TYPES)}; got {dtype!r}"
        )
    if strength in {"strong", "moderate"}:
        strength = "weak"
    return {
        "decision_type": dtype,
        "evidence_strength": strength,
        "claim_level": "pipeline_validation_only",
    }


def node_is_smoke_detection(node_contract: dict[str, Any] | None) -> bool:
    contract = node_contract or {}
    task_config = contract.get("task_config") or {}
    return is_smoke_detection_context(
        task_type=contract.get("task_type"),
        execution_mode=contract.get("execution_mode"),
        claim_level=task_config.get("claim_level"),
    )


def node_is_exploratory_fast_eval(node_contract: dict[str, Any] | None) -> bool:
    contract = node_contract or {}
    task_config = contract.get("task_config") or {}
    return is_exploratory_fast_eval_context(
        task_type=contract.get("task_type"),
        execution_mode=contract.get("execution_mode"),
        claim_level=task_config.get("claim_level"),
        evaluation_scope=task_config.get("evaluation_scope"),
    )


def node_is_fast_eval(node_contract: dict[str, Any] | None) -> bool:
    contract = node_contract or {}
    if (contract.get("task_type") or "").strip() != "rgbt_detection":
        return False
    return (contract.get("execution_mode") or "").strip() == "fast_eval"


def node_is_smoke_train_only(node_contract: dict[str, Any] | None) -> bool:
    """Smoke/validate pipeline context excluding fast_eval multi-seed runs."""
    if not node_is_smoke_detection(node_contract):
        return False
    if node_is_exploratory_fast_eval(node_contract):
        return False
    return not node_is_fast_eval(node_contract)
