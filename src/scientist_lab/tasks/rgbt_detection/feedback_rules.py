from __future__ import annotations

from typing import Any


SMOKE_OR_VALIDATE = {"smoke_train", "validate_data", "smoke_test"}


def apply_detection_claim_gate(
    *,
    execution_mode: str,
    evaluation_scope: str = "debug_subset",
) -> dict[str, Any]:
    """Restrict claims for debug / smoke detection runs."""
    mode = (execution_mode or "").strip()
    if mode not in SMOKE_OR_VALIDATE and evaluation_scope != "debug_subset":
        return {
            "claim_level": "unrestricted",
            "allowed_claims": [],
            "blocked_claims": [],
        }

    return {
        "claim_level": "pipeline_validation_only",
        "allowed_claims": [
            "The RGB-T pipeline completed successfully on the debug subset.",
            "RGB and thermal data can be read and paired.",
            "Detection metric files were generated.",
        ],
        "blocked_claims": [
            {
                "claim": "Fusion improves detection performance.",
                "reason": "Smoke training is insufficient for performance conclusions.",
            },
            {
                "claim": "The method achieves state-of-the-art performance.",
                "reason": "No full-dataset matched baseline comparison exists.",
            },
            {
                "claim": "The method has statistically significant gains.",
                "reason": "Debug subset and smoke_train do not support significance claims.",
            },
        ],
        "max_evidence_strength": "weak",
        "forbid_performance_support": True,
    }


def annotate_feedback_for_detection(
    feedback: dict[str, Any],
    *,
    execution_mode: str,
) -> dict[str, Any]:
    gate = apply_detection_claim_gate(execution_mode=execution_mode)
    payload = dict(feedback)
    payload["claim_gate"] = gate
    if gate.get("forbid_performance_support"):
        payload["evidence_strength"] = "weak"
        if payload.get("hypothesis_status") == "supported_with_repeated_evidence":
            payload["hypothesis_status"] = "inconclusive"
            payload.setdefault("uncertainties", []).append(
                "Smoke/debug detection runs cannot support performance hypotheses."
            )
        payload.setdefault("uncertainties", []).append(
            "Performance conclusions are blocked by claim gate "
            f"(execution_mode={execution_mode})."
        )
        # Downgrade parameter-change recommendations that imply capacity/SOTA chasing.
        recs = []
        for item in payload.get("recommendations") or []:
            rec = dict(item)
            if rec.get("recommendation_type") in {"expand_range", "intermediate_value"}:
                rec["status"] = "deferred"
                rec["rationale"] = (
                    "Deferred: smoke/debug detection mode cannot justify "
                    "performance-oriented parameter search. "
                    f"Original: {rec.get('rationale', '')}"
                )
            recs.append(rec)

        # Keep a pipeline-only follow-up so iterate-start can still propose one node.
        has_active_param = any(
            rec.get("status", "active") == "active" and rec.get("parameter_changes")
            for rec in recs
        )
        if not has_active_param:
            recs.append(
                {
                    "recommendation_type": "ablation",
                    "priority": 0.72,
                    "rationale": (
                        "Pipeline follow-up under claim gate: run a shorter smoke "
                        "(epochs=1) to confirm the detection loop remains stable. "
                        "This is not a performance claim."
                    ),
                    "parameter_changes": {"epochs": 1},
                    "status": "active",
                }
            )
        payload["recommendations"] = recs
        payload["recommended_action"] = "verify"
    return payload
