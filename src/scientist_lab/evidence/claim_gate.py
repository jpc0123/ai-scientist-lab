"""Rule-based evidence strength and claim restriction gate (no LLM)."""

from __future__ import annotations

from typing import Any, Literal

EvidenceStrength = Literal["weak", "moderate", "strong"]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def assess_evidence_strength(
    *,
    seed_count: int,
    execution_mode: str | None = None,
    evaluation_scope: str | None = None,
    implementation: str | None = None,
    claim_level: str | None = None,
    has_protocol: bool = False,
    has_ablation: bool = False,
    formal_implementation: bool = False,
    stable_direction: bool = False,
    requested_type: str = "paired_comparison",
) -> dict[str, Any]:
    """Return strength tags + limitations. Never invent strong scientific claims."""
    limitations: list[str] = []
    mode = (execution_mode or "").strip()
    scope = (evaluation_scope or "").strip()
    impl = (implementation or "").strip().lower().replace("-", "_")
    claim = (claim_level or "").strip()

    is_fast_eval = mode == "fast_eval" or "fast_eval" in scope
    is_stand_in = impl in {"", "stand_in", "standin", "torch_mini_standin", "numpy"}
    is_formal = formal_implementation and not is_stand_in

    if seed_count < 3:
        limitations.append(f"Fewer than 3 matched seeds ({seed_count}).")
    if is_fast_eval:
        limitations.append("Fast Eval subset / budget only.")
    if is_stand_in:
        limitations.append("Stand-in implementation was used.")
        limitations.append(
            "Stand-in evidence cannot support formal DFINE performance claims."
        )
    if not has_ablation:
        limitations.append("No ablation evidence attached.")
    if not has_protocol:
        limitations.append("No ExperimentProtocol linkage.")
    if claim == "exploratory_comparison":
        limitations.append("Claim level is exploratory_comparison.")

    # Type gating
    effective_type = requested_type
    if requested_type == "repeated_experiment" and seed_count < 2:
        effective_type = "single_execution"
        limitations.append(
            "Single seed cannot support repeated_experiment evidence; "
            "downgraded to single_execution."
        )
    elif requested_type == "paired_comparison" and seed_count < 2:
        limitations.append(
            "Paired comparison with fewer than 2 shared seeds is weak and limited."
        )

    # Scientific strength caps
    scientific: EvidenceStrength = "weak"
    engineering: str = "weak"

    if (
        seed_count >= 3
        and has_protocol
        and is_formal
        and not is_fast_eval
        and stable_direction
        and has_ablation
    ):
        scientific = "moderate"
        engineering = "moderate"
    elif seed_count >= 3 and has_protocol and stable_direction and is_formal:
        scientific = "weak"
        engineering = "moderate_engineering_evidence"
        limitations.append(
            "Engineering signal may be moderate, but scientific evidence remains weak "
            "without full benchmark / ablation / formal DFINE acceptance."
        )
    elif seed_count >= 3 and (is_stand_in or is_fast_eval):
        scientific = "weak"
        engineering = "moderate_engineering_evidence" if stable_direction else "weak"
        limitations.append("weak_scientific_evidence under Fast Eval / stand-in caps.")

    # Hard caps from plan §6.4
    if is_fast_eval or is_stand_in:
        if scientific == "strong":
            scientific = "weak"
        if scientific == "moderate":
            scientific = "weak"
            limitations.append(
                "Fast Eval / stand-in evidence cannot be labeled moderate/strong "
                "scientific strength."
            )

    # Strong is intentionally unreachable without full formal conditions.
    if scientific == "strong" and (is_fast_eval or is_stand_in or seed_count < 5):
        scientific = "weak"
        limitations.append("Strong evidence requirements are not met.")

    overall: EvidenceStrength = scientific
    if engineering.startswith("moderate") and overall == "weak":
        # Keep overall scientific-facing strength weak; expose engineering separately.
        overall = "weak"

    blocked_claims = [
        "state-of-the-art / SOTA performance",
        "full RGBT-Tiny benchmark improvement",
        "formal DFINE superiority" if is_stand_in else None,
        "statistically significant improvement",
    ]
    blocked_claims = [item for item in blocked_claims if item]

    return {
        "evidence_strength": overall,
        "scientific_evidence_level": (
            "weak_scientific_evidence" if overall == "weak" else overall
        ),
        "engineering_evidence_level": engineering,
        "limitations": limitations,
        "effective_evidence_type": effective_type,
        "blocked_claim_templates": blocked_claims,
        "allows_strong": False if (is_fast_eval or is_stand_in) else overall == "strong",
    }


def assert_not_strong_for_fast_eval(strength: str, *, is_fast_eval: bool) -> str:
    if is_fast_eval and strength == "strong":
        return "weak"
    return strength
