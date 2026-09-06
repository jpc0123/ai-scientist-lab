"""Claim gates: keep smoke / short-budget runs from asserting method superiority."""

from __future__ import annotations

from typing import Any


SMOKE_LIKE_PROTOCOLS = frozenset(
    {
        "smoke",
        "fast_eval",
        "debug",
        "execution_validation",
        "multi_seed_execution_validation",
        "multi_seed_exploratory",
        "metric_validity",
        "budget_scale_exploratory",
        "formal_candidate",
        "mechanism_diagnosis",
        "diagnostic_only",
        "baseline_resolution_probe",
        "gate_e_debug",
        "debug_only",
    }
)

FORMAL_PROTOCOLS = frozenset({"formal", "method_comparison"})


class ClaimRejected(ValueError):
    """Raised when a run's protocol forbids the requested claim class."""


def resolve_protocol(
    *,
    contract: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    execution_mode: str | None = None,
) -> str:
    params = dict(parameters or {})
    ctr = dict(contract or {})
    explicit = (
        params.get("protocol")
        or ctr.get("protocol")
        or (ctr.get("task_config") or {}).get("protocol")
    )
    if explicit:
        return str(explicit).strip().lower()
    mode = str(
        execution_mode
        or ctr.get("execution_mode")
        or params.get("execution_mode")
        or "smoke"
    ).strip().lower()
    if mode in FORMAL_PROTOCOLS:
        return "formal"
    if mode in {
        "fast_eval",
        "smoke",
        "smoke_train",
        "debug",
        "metric_validity",
        "multi_seed_exploratory",
        "budget_scale_exploratory",
        "formal_candidate",
        "mechanism_diagnosis",
        "diagnostic_only",
        "baseline_resolution_probe",
        "gate_e_debug",
        "debug_only",
    }:
        if mode in {
            "metric_validity",
            "multi_seed_exploratory",
            "budget_scale_exploratory",
            "formal_candidate",
            "mechanism_diagnosis",
            "diagnostic_only",
            "baseline_resolution_probe",
            "gate_e_debug",
            "debug_only",
        }:
            return mode
        return "smoke"
    return mode or "smoke"


def allow_scientific_claims(protocol: str) -> bool:
    return str(protocol).strip().lower() in FORMAL_PROTOCOLS


def assert_claim_allowed(
    claim_type: str,
    *,
    protocol: str,
    reason: str | None = None,
) -> None:
    """Reject performance-superiority claims from smoke / execution-only runs."""
    claim = str(claim_type).strip().lower()
    proto = str(protocol).strip().lower()
    if claim in {
        "performance_superiority",
        "method_superiority",
        "beats_baseline",
        "scientific_comparison",
    } and not allow_scientific_claims(proto):
        raise ClaimRejected(
            reason
            or (
                f"Claim {claim!r} rejected: protocol={proto!r} is not formal. "
                "Smoke / short-budget runs are execution tests, not method comparisons."
            )
        )


def claim_gate_metadata(protocol: str) -> dict[str, Any]:
    proto = str(protocol).strip().lower()
    if proto == "metric_validity":
        purpose = "metric_validity"
    elif proto == "multi_seed_exploratory":
        purpose = "multi_seed_exploratory"
    elif proto == "budget_scale_exploratory":
        purpose = "budget_scale_exploratory"
    elif proto == "formal_candidate":
        purpose = "formal_candidate"
    elif proto in {"mechanism_diagnosis", "diagnostic_only"}:
        purpose = "mechanism_diagnosis"
    elif proto == "baseline_resolution_probe":
        purpose = "baseline_resolution_probe"
    elif proto in {"gate_e_debug", "debug_only"}:
        purpose = "gate_e_debug"
    elif allow_scientific_claims(proto):
        purpose = "method_comparison"
    else:
        purpose = "execution_validation"
    return {
        "protocol": proto,
        "allow_scientific_claims": allow_scientific_claims(proto),
        "purpose": purpose,
        "formal_performance_claims_allowed": False
        if purpose in {
            "execution_validation",
            "metric_validity",
            "multi_seed_exploratory",
            "budget_scale_exploratory",
            "formal_candidate",
            "mechanism_diagnosis",
            "baseline_resolution_probe",
            "gate_e_debug",
        }
        else allow_scientific_claims(proto),
    }
