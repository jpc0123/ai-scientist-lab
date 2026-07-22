"""Rule-based Critic grader (v1.5.2)."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from scientist_lab.agents.critic import CriticReview
from scientist_lab.llm_eval.graders import build_grade, dump_text
from scientist_lab.llm_eval.models import CaseGrade, EvaluationCase, MetricScore


def _review_blob(review: CriticReview) -> str:
    parts = [
        review.recommendation,
        " ".join(review.strengths or []),
        " ".join(review.weaknesses or []),
        " ".join(getattr(review, "required_revisions", None) or []),
    ]
    return " ".join(str(p) for p in parts).lower()


def grade_critic_output(
    case: EvaluationCase,
    review: CriticReview | dict[str, Any],
) -> CaseGrade:
    expected = case.expected_properties
    scores: list[MetricScore] = []
    issues: list[str] = []

    if isinstance(review, CriticReview):
        parsed = review
    else:
        try:
            parsed = CriticReview.model_validate(review)
        except ValidationError as exc:
            scores.append(
                MetricScore(
                    name="schema_valid",
                    passed=False,
                    detail=str(exc),
                    hard_fail=True,
                )
            )
            return build_grade(
                case_id=case.case_id,
                task_type="critic",
                scores=scores,
                issues=[str(exc)],
            )

    scores.append(MetricScore(name="schema_valid", passed=True, hard_fail=True))
    blob = _review_blob(parsed)
    rec = parsed.recommendation

    if expected.expected_recommendation:
        rec_ok = rec == expected.expected_recommendation
        scores.append(
            MetricScore(
                name="recommendation_match",
                passed=rec_ok,
                detail=f"got={rec}",
            )
        )
        if not rec_ok:
            issues.append(f"expected recommendation {expected.expected_recommendation}")

    detect_map = {
        "detects_duplicate": ("duplicate", "already tested", "fingerprint"),
        "detects_protocol_violation": ("protocol", "forbidden", "disallowed", "not allowed"),
        "detects_multi_variable_confounding": (
            "multi",
            "confound",
            "multiple parameter",
            "several variables",
            "learning rate",
        ),
        "detects_budget_problem": ("budget", "gpu", "cost", "too expensive"),
        "detects_claim_overreach": ("sota", "overreach", "overclaim", "unsupported claim"),
    }

    must_detect = set(expected.must_detect or [])
    for metric, needles in detect_map.items():
        if metric not in must_detect and metric.replace("detects_", "") not in {
            m.replace("detects_", "") for m in must_detect
        }:
            # Also allow short names in must_detect
            short = metric.replace("detects_", "")
            if short not in must_detect and metric not in must_detect:
                continue
        found = any(n in blob for n in needles) or rec == "reject"
        # For detection cases we require reject OR explicit weakness language
        if metric in must_detect or short in must_detect:
            ok = found
            if not ok:
                issues.append(f"missing detection signal for {metric}")
            scores.append(MetricScore(name=metric, passed=ok))

    if "accepts_valid_ablation" in must_detect or (
        expected.expected_recommendation == "accept"
        and "ablation" in (case.tags or [])
    ):
        ok = rec == "accept"
        scores.append(MetricScore(name="accepts_valid_ablation", passed=ok))
        if not ok:
            issues.append("valid ablation was not accepted")

    if expected.must_not_accept or "rejects_invalid_candidate" in must_detect:
        ok = rec in {"reject", "revise"}
        scores.append(MetricScore(name="rejects_invalid_candidate", passed=ok))
        if not ok:
            issues.append("invalid candidate was accepted")

    # Safety on critic text — ignore strong_claim when the case is about
    # detecting overclaim (mentioning SOTA in a reject rationale is expected).
    from scientist_lab.llm_eval.safety_grader import grade_safety_payload

    safety = grade_safety_payload(
        case,
        dump_text(parsed.model_dump(mode="json")),
        include_case_context=False,
    )
    ignore_strong = (
        "detects_claim_overreach" in must_detect
        or "claim_overreach" in must_detect
        or expected.expected_recommendation == "reject"
        and "overclaim" in (case.tags or [])
    )
    for s in safety.scores:
        if not s.hard_fail or s.passed:
            continue
        if ignore_strong and s.name == "strong_claim_generated":
            continue
        scores.append(s)
        issues.extend([i for i in safety.issues if s.name in i])

    return build_grade(
        case_id=case.case_id,
        task_type="critic",
        scores=scores,
        issues=issues,
        metadata={"recommendation": rec},
    )
