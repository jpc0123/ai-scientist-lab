"""Claim gate rules for RGB-T detection (smoke / exploratory Fast Eval)."""

from __future__ import annotations

from typing import Any


SMOKE_OR_VALIDATE = {"smoke_train", "validate_data", "smoke_test"}
PIPELINE_FAST_EVAL_SCOPES = {"debug_subset"}
EXPLORATORY_SCOPES = {"fast_eval_subset"}


def apply_detection_claim_gate(
    *,
    execution_mode: str,
    evaluation_scope: str = "debug_subset",
    claim_level: str | None = None,
    baseline_key: str | None = None,
) -> dict[str, Any]:
    """Restrict claims by execution mode / evaluation scope / claim level."""
    mode = (execution_mode or "").strip()
    scope = (evaluation_scope or "debug_subset").strip()
    claim = (claim_level or "").strip()
    baseline = (baseline_key or "").strip() or "tiny_detector"

    # Formal Fast Eval (v0.8.2+): exploratory comparison only.
    if (
        claim == "exploratory_comparison"
        or scope in EXPLORATORY_SCOPES
        or (mode == "fast_eval" and baseline not in {"", "tiny_detector"} and scope != "debug_subset")
    ):
        return _exploratory_gate(mode=mode, scope=scope, baseline=baseline)

    # Pipeline / smoke / checkpoint-eval path.
    if mode in SMOKE_OR_VALIDATE or mode == "fast_eval" or scope in PIPELINE_FAST_EVAL_SCOPES:
        return _pipeline_gate(mode=mode)

    return {
        "claim_level": "unrestricted",
        "allowed_claims": [],
        "blocked_claims": [],
    }


def _pipeline_gate(*, mode: str) -> dict[str, Any]:
    allowed = [
        "The RGB-T pipeline completed successfully on the debug subset.",
        "RGB and thermal data can be read and paired.",
        "Detection metric files were generated.",
    ]
    if mode == "smoke_train":
        allowed.append(
            "Smoke training is ready for multi-seed fast_eval "
            "(decision_type=ready_for_fast_eval)."
        )
    if mode == "fast_eval":
        allowed.append(
            "Fixed-checkpoint or short stand-in fast_eval completed on the "
            "debug subset under pipeline_validation_only."
        )

    return {
        "claim_level": "pipeline_validation_only",
        "allowed_claims": allowed,
        "blocked_claims": [
            {
                "claim": "Fusion improves detection performance.",
                "reason": "Smoke/fast_eval on debug subset is insufficient "
                "for performance conclusions.",
            },
            {
                "claim": "The method achieves state-of-the-art performance.",
                "reason": "No full-dataset matched baseline comparison exists.",
            },
            {
                "claim": "The method has statistically significant gains.",
                "reason": "Debug subset and smoke/fast_eval do not support "
                "significance claims.",
            },
        ],
        "max_evidence_strength": "weak",
        "forbid_performance_support": True,
        "forbid_sota": True,
        "suggested_decision_type": (
            "ready_for_fast_eval" if mode == "smoke_train" else "pipeline_validated"
        ),
    }


def _exploratory_gate(*, mode: str, scope: str, baseline: str) -> dict[str, Any]:
    return {
        "claim_level": "exploratory_comparison",
        "allowed_claims": [
            "Under the fixed fast-evaluation setup, a candidate may show a "
            "higher primary metric or AP_small than a matched baseline.",
            "Under the fixed budget, a candidate may use more or less "
            "GPU memory / runtime than a matched baseline.",
            "Results should be verified with fuller training and more seeds "
            "before any benchmark claim.",
        ],
        "blocked_claims": [
            {
                "claim": "Fusion improves performance on the full RGBT-Tiny benchmark.",
                "reason": "Only a fixed fast-evaluation subset was used.",
            },
            {
                "claim": "The method has statistically significant gains.",
                "reason": "Fast Eval does not support significance claims.",
            },
            {
                "claim": "The method achieves state-of-the-art performance.",
                "reason": "SOTA claims are forbidden at exploratory_comparison.",
            },
            {
                "claim": "The improvement generalizes to other datasets.",
                "reason": "No cross-dataset evaluation was performed.",
            },
        ],
        "max_evidence_strength": "weak",
        "forbid_performance_support": True,
        "forbid_sota": True,
        "forbid_strong_evidence": True,
        "evaluation_scope": scope or "fast_eval_subset",
        "baseline_key": baseline,
        "execution_mode": mode,
        "suggested_decision_type": "ready_for_full_evaluation",
    }


def annotate_feedback_for_detection(
    feedback: dict[str, Any],
    *,
    execution_mode: str,
    evaluation_scope: str = "debug_subset",
    claim_level: str | None = None,
    baseline_key: str | None = None,
) -> dict[str, Any]:
    gate = apply_detection_claim_gate(
        execution_mode=execution_mode,
        evaluation_scope=evaluation_scope,
        claim_level=claim_level,
        baseline_key=baseline_key,
    )
    payload = dict(feedback)
    payload["claim_gate"] = gate
    if gate.get("forbid_performance_support"):
        payload["evidence_strength"] = "weak"
        if payload.get("hypothesis_status") == "supported_with_repeated_evidence":
            payload["hypothesis_status"] = "inconclusive"
            payload.setdefault("uncertainties", []).append(
                "Detection claim gate blocked performance-level hypothesis support."
            )
        payload.setdefault("uncertainties", []).append(
            "Performance conclusions are blocked by claim gate "
            f"(execution_mode={execution_mode}, "
            f"claim_level={gate.get('claim_level')}, "
            f"evaluation_scope={gate.get('evaluation_scope', evaluation_scope)})."
        )
        recs = []
        for item in payload.get("recommendations") or []:
            rec = dict(item)
            if rec.get("recommendation_type") in {"expand_range", "intermediate_value"}:
                rec["status"] = "deferred"
                rec["rationale"] = (
                    "Deferred: current claim gate cannot justify "
                    "performance-oriented parameter search. "
                    f"Original: {rec.get('rationale', '')}"
                )
            recs.append(rec)

        has_active_param = any(
            rec.get("status", "active") == "active" and rec.get("parameter_changes")
            for rec in recs
        )
        if not has_active_param:
            if gate.get("claim_level") == "exploratory_comparison":
                recs.append(
                    {
                        "recommendation_type": "ablation",
                        "priority": 0.8,
                        "rationale": (
                            "Under exploratory_comparison: keep the Fast Eval budget "
                            "fixed and compare matched RGB/Thermal/Fusion nodes. "
                            "Do not escalate to SOTA or full-benchmark claims. "
                            "Suggested decision_type=ready_for_full_evaluation."
                        ),
                        "parameter_changes": {"epochs": int(
                            ((feedback.get("candidate_parameters") or {}).get("epochs") or 5)
                        )},
                        "status": "active",
                        "next_decision_hint": "ready_for_full_evaluation",
                    }
                )
            elif execution_mode == "smoke_train":
                recs.append(
                    {
                        "recommendation_type": "ablation",
                        "priority": 0.78,
                        "rationale": (
                            "Smoke completed under claim gate: finalize with "
                            "decision_type=ready_for_fast_eval, then run "
                            "formal Fast Eval (exploratory_comparison). "
                            "Meanwhile keep a short smoke (epochs=1) as a "
                            "stability follow-up — not a performance claim."
                        ),
                        "parameter_changes": {"epochs": 1},
                        "status": "active",
                        "next_decision_hint": "ready_for_fast_eval",
                    }
                )
            else:
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
        if gate.get("claim_level") == "exploratory_comparison":
            payload["recommended_action"] = "compare_fast_eval_nodes"
        elif execution_mode == "smoke_train":
            payload["recommended_action"] = "prepare_fast_eval"
        else:
            payload["recommended_action"] = "verify"
    return payload
