"""Decision-type restrictions for RGB-T debug / smoke iterations."""

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
    mode = (execution_mode or "").strip()
    if mode in {"smoke_train", "validate_data", "smoke_test"}:
        return True
    return (claim_level or "").strip() == "pipeline_validation_only"


def validate_smoke_decision(
    decision_type: str,
    *,
    evidence_strength: str = "weak",
) -> dict[str, Any]:
    dtype = (decision_type or "").strip()
    strength = (evidence_strength or "").strip() or "weak"

    if dtype in FORBIDDEN_SMOKE_DECISION_TYPES:
        raise ValueError(
            f"smoke detection forbids decision_type={dtype!r}; "
            f"allowed={sorted(ALLOWED_SMOKE_DECISION_TYPES)}"
        )
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
