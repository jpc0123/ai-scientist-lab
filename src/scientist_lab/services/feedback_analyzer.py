from __future__ import annotations

from typing import Any

from scientist_lab.domain.feedback import (
    ExperimentRecommendation,
    FeedbackInput,
    FeedbackReport,
)

DURATION_INCREASE_THRESHOLD = 0.5


def _status(comparison: dict[str, Any]) -> str:
    return str(comparison.get("hypothesis_status") or "inconclusive")


def _param_changes(comparison: dict[str, Any], candidate_contract: dict) -> dict[str, Any]:
    changes = comparison.get("parameter_changes") or {}
    if changes:
        return changes
    # Fallback: empty
    _ = candidate_contract
    return {}


def _evidence_strength(comparison: dict[str, Any]) -> str:
    status = _status(comparison)
    if status == "invalid_comparison":
        return "weak"
    shared = comparison.get("shared_seeds") or []
    n = len(shared)
    # Never claim "strong" with a small seed set.
    if n < 10:
        if status in {
            "supported_with_repeated_evidence",
            "rejected_with_repeated_evidence",
        }:
            return "moderate" if n >= 3 else "weak"
        if comparison.get("practically_equivalent"):
            return "moderate" if n >= 5 else "weak"
        return "weak"

    wins = int(comparison.get("candidate_win_count") or 0)
    mean_delta = comparison.get("mean_delta")
    if status == "supported_with_repeated_evidence":
        if (
            wins >= max(4, int(0.8 * n))
            and isinstance(mean_delta, (int, float))
            and abs(mean_delta) >= 0.005
        ):
            return "strong"
        return "moderate"
    if status == "rejected_with_repeated_evidence":
        return "moderate"
    return "weak"


def _duration_increase_percent(comparison: dict[str, Any]) -> float | None:
    base = comparison.get("baseline_duration_mean")
    cand = comparison.get("candidate_duration_mean")
    if not isinstance(base, (int, float)) or not isinstance(cand, (int, float)):
        return None
    if float(base) <= 0:
        return None
    return (float(cand) - float(base)) / float(base)


def _intermediate_for_numeric(from_v: Any, to_v: Any) -> Any | None:
    if isinstance(from_v, bool) or isinstance(to_v, bool):
        return None
    if isinstance(from_v, int) and isinstance(to_v, int):
        mid = int(round((from_v + to_v) / 2))
        if mid != from_v and mid != to_v:
            return mid
        return None
    if isinstance(from_v, float) and isinstance(to_v, float):
        mid = (from_v + to_v) / 2.0
        if mid != from_v and mid != to_v:
            return mid
    return None


def _expand_for_numeric(from_v: Any, to_v: Any) -> Any | None:
    if isinstance(from_v, bool) or isinstance(to_v, bool):
        return None
    if isinstance(from_v, (int, float)) and isinstance(to_v, (int, float)):
        direction = 1 if to_v >= from_v else -1
        span = abs(float(to_v) - float(from_v))
        if span == 0:
            return None
        expanded = float(to_v) + direction * span
        if isinstance(from_v, int) and isinstance(to_v, int):
            return int(round(expanded))
        return expanded
    return None


def build_invalid_comparison_feedback(data: FeedbackInput) -> FeedbackReport:
    comparison = data.comparison
    issues = (comparison.get("verification") or {}).get("blocking_issues") or []
    return FeedbackReport(
        baseline_node_id=data.baseline_node_id,
        candidate_node_id=data.candidate_node_id,
        hypothesis_status="invalid_comparison",
        evidence_strength="weak",
        positive_findings=[],
        negative_findings=[
            "Comparison is not valid for performance conclusions.",
            *[str(item) for item in issues],
        ],
        tradeoffs=[],
        uncertainties=["Fix comparability issues before claiming gains or losses."],
        recommendations=[
            ExperimentRecommendation(
                recommendation_type="stop",
                priority=0.95,
                rationale="Do not generate performance claims from an invalid comparison.",
                parameter_changes={},
            ),
            ExperimentRecommendation(
                recommendation_type="repeat_seeds",
                priority=0.7,
                rationale="Re-run both nodes with matched seeds and a single changed parameter.",
                parameter_changes={},
            ),
        ],
        recommended_action="revise",
        comparison_summary={
            "mean_delta": comparison.get("mean_delta"),
            "shared_seeds": comparison.get("shared_seeds"),
        },
    )


def build_supported_feedback(data: FeedbackInput) -> FeedbackReport:
    comparison = data.comparison
    primary = comparison.get("primary_metric") or "accuracy"
    mean_delta = comparison.get("mean_delta")
    wins = comparison.get("candidate_win_count")
    shared = comparison.get("shared_seeds") or []
    changes = _param_changes(comparison, data.candidate_contract)

    positives = [
        f"Candidate shows a repeated-evidence improvement on mean {primary}"
        + (f" (delta={mean_delta:+.6f})." if isinstance(mean_delta, (int, float)) else "."),
        f"Candidate won {wins} of {len(shared)} paired seeds.",
    ]
    negatives: list[str] = []
    tradeoffs: list[str] = []
    uncertainties = [
        "Only one dataset was evaluated.",
        "Evidence is based on a limited set of random seeds, not a formal significance test.",
    ]

    secondary = (comparison.get("candidate_aggregate") or {}).get(
        "aggregate_metrics"
    ) or {}
    if "log_loss" in secondary and "log_loss" in (
        (comparison.get("baseline_aggregate") or {}).get("aggregate_metrics") or {}
    ):
        b_ll = secondary.get("log_loss", {}).get("mean")
        # use baseline from comparison aggregates
        base_ll = (
            (comparison.get("baseline_aggregate") or {})
            .get("aggregate_metrics", {})
            .get("log_loss", {})
            .get("mean")
        )
        if isinstance(b_ll, (int, float)) and isinstance(base_ll, (int, float)) and b_ll < base_ll:
            positives.append("Mean log loss decreased for the candidate.")

    duration_pct = _duration_increase_percent(comparison)
    recommendations: list[ExperimentRecommendation] = []

    for key, change in changes.items():
        mid = _intermediate_for_numeric(change.get("from"), change.get("to"))
        expanded = _expand_for_numeric(change.get("from"), change.get("to"))
        if mid is not None:
            priority = 0.92
            rationale = (
                "Test whether most of the gain can be retained at an intermediate value."
            )
            rec_type = "intermediate_value"
            if duration_pct is not None and duration_pct > DURATION_INCREASE_THRESHOLD:
                tradeoffs.append(
                    "Performance improved, but mean execution time increased substantially."
                )
                rationale = "Search for a better performance-cost balance via an intermediate value."
                rec_type = "efficiency_tradeoff"
                priority = 0.94
            recommendations.append(
                ExperimentRecommendation(
                    recommendation_type=rec_type,  # type: ignore[arg-type]
                    priority=priority,
                    rationale=rationale,
                    parameter_changes={key: mid},
                )
            )
        if expanded is not None:
            recommendations.append(
                ExperimentRecommendation(
                    recommendation_type="expand_range",
                    priority=0.65,
                    rationale="Determine whether performance has saturated beyond the current candidate.",
                    parameter_changes={key: expanded},
                )
            )

    if not recommendations:
        recommendations.append(
            ExperimentRecommendation(
                recommendation_type="repeat_seeds",
                priority=0.6,
                rationale="Improvement looks supportive; gather more seeds before expanding scope.",
                parameter_changes={},
            )
        )

    if not changes:
        uncertainties.append("No parameter changes were recorded between nodes.")

    for key in changes:
        uncertainties.append(
            f"Intermediate values of '{key}' between the compared settings were not tested."
        )

    strength = _evidence_strength(comparison)
    action = "continue" if recommendations else "verify"
    return FeedbackReport(
        baseline_node_id=data.baseline_node_id,
        candidate_node_id=data.candidate_node_id,
        hypothesis_status="supported_with_repeated_evidence",
        evidence_strength=strength,  # type: ignore[arg-type]
        positive_findings=positives,
        negative_findings=negatives,
        tradeoffs=tradeoffs,
        uncertainties=uncertainties,
        recommendations=sorted(recommendations, key=lambda r: r.priority, reverse=True),
        recommended_action=action,  # type: ignore[arg-type]
        comparison_summary={
            "primary_metric": primary,
            "mean_delta": mean_delta,
            "candidate_win_count": wins,
            "baseline_win_count": comparison.get("baseline_win_count"),
            "parameter_changes": changes,
            "stable_improvement": comparison.get("stable_improvement"),
        },
    )


def build_rejected_feedback(data: FeedbackInput) -> FeedbackReport:
    comparison = data.comparison
    primary = comparison.get("primary_metric") or "accuracy"
    mean_delta = comparison.get("mean_delta")
    changes = _param_changes(comparison, data.candidate_contract)
    recommendations: list[ExperimentRecommendation] = []
    for key, change in changes.items():
        recommendations.append(
            ExperimentRecommendation(
                recommendation_type="ablation",
                priority=0.85,
                rationale=(
                    f"Candidate worsened mean {primary}; consider reverting '{key}' "
                    "or trying the opposite direction with a smaller step."
                ),
                parameter_changes={key: change.get("from")},
            )
        )
    if not recommendations:
        recommendations.append(
            ExperimentRecommendation(
                recommendation_type="stop",
                priority=0.8,
                rationale="Repeated evidence rejects the candidate change; pause expansion.",
                parameter_changes={},
            )
        )
    return FeedbackReport(
        baseline_node_id=data.baseline_node_id,
        candidate_node_id=data.candidate_node_id,
        hypothesis_status="rejected_with_repeated_evidence",
        evidence_strength=_evidence_strength(comparison),  # type: ignore[arg-type]
        positive_findings=[],
        negative_findings=[
            f"Candidate failed to improve mean {primary}"
            + (f" (delta={mean_delta:+.6f})." if isinstance(mean_delta, (int, float)) else "."),
            f"Baseline won {comparison.get('baseline_win_count')} paired seeds.",
        ],
        tradeoffs=[],
        uncertainties=["Rejection is based on the current seed set and single dataset."],
        recommendations=recommendations,
        recommended_action="revise",
        comparison_summary={
            "mean_delta": mean_delta,
            "parameter_changes": changes,
        },
    )


def build_efficiency_tradeoff_feedback(data: FeedbackInput) -> FeedbackReport:
    comparison = data.comparison
    primary = comparison.get("primary_metric") or "accuracy"
    mean_delta = comparison.get("mean_delta")
    wins = comparison.get("candidate_win_count")
    base_wins = comparison.get("baseline_win_count")
    shared = comparison.get("shared_seeds") or []
    runtime_change = comparison.get("runtime_change_ratio")
    std_ratio = comparison.get("std_ratio")
    tradeoff_status = comparison.get("tradeoff_status")

    positives: list[str] = [
        f"Mean {primary} is practically equivalent between the two nodes"
        + (f" (delta={mean_delta:+.6f})." if isinstance(mean_delta, (int, float)) else "."),
    ]
    if isinstance(runtime_change, (int, float)) and runtime_change < 0:
        positives.append(
            f"Candidate reduced mean runtime by approximately {abs(runtime_change) * 100:.1f}%."
        )

    tradeoffs: list[str] = []
    if comparison.get("stability_relation") == "baseline_better":
        tradeoffs.append("Candidate accuracy variance was higher than baseline.")
        if isinstance(std_ratio, (int, float)):
            tradeoffs.append(
                f"Candidate primary-metric std is about {std_ratio:.2f}x baseline std."
            )
    if wins is not None and base_wins is not None:
        tradeoffs.append(
            f"Candidate won only {wins} of {len(shared)} paired comparisons "
            f"(baseline won {base_wins})."
        )

    return FeedbackReport(
        baseline_node_id=data.baseline_node_id,
        candidate_node_id=data.candidate_node_id,
        hypothesis_status="inconclusive",
        evidence_strength=_evidence_strength(comparison),  # type: ignore[arg-type]
        positive_findings=positives,
        negative_findings=[
            "Candidate is not shown to be overall superior on primary accuracy.",
        ],
        tradeoffs=tradeoffs,
        uncertainties=[
            f"Only {len(shared)} shared seeds were evaluated.",
            "Do not treat practical equivalence as statistical significance.",
            "Prefer recording an efficiency-tradeoff decision over inventing a new width.",
        ],
        recommendations=[
            ExperimentRecommendation(
                recommendation_type="repeat_seeds",
                priority=0.92,
                rationale=(
                    "Add more matched seeds before choosing between stability-first "
                    "and efficiency-first nodes."
                ),
                parameter_changes={},
            ),
            ExperimentRecommendation(
                recommendation_type="efficiency_tradeoff",
                priority=0.85,
                rationale=(
                    "Treat the candidate as a potential efficiency tradeoff: comparable "
                    f"mean {primary}, lower runtime, weaker stability. Avoid expanding "
                    "capacity further until this tradeoff is clarified."
                ),
                parameter_changes={},
            ),
            ExperimentRecommendation(
                recommendation_type="stop",
                priority=0.7,
                rationale=(
                    "Do not auto-generate a farther intermediate width (e.g. 112) yet; "
                    "first decide whether efficiency or stability is the selection goal."
                ),
                parameter_changes={},
            ),
        ],
        recommended_action="verify",
        comparison_summary={
            "mean_delta": mean_delta,
            "practically_equivalent": True,
            "runtime_change_ratio": runtime_change,
            "std_ratio": std_ratio,
            "tradeoff_status": tradeoff_status,
            "performance_relation": comparison.get("performance_relation"),
            "efficiency_relation": comparison.get("efficiency_relation"),
            "stability_relation": comparison.get("stability_relation"),
            "overall_decision": comparison.get("overall_decision"),
            "candidate_win_count": wins,
            "baseline_win_count": base_wins,
            "shared_seeds": shared,
        },
    )


def build_inconclusive_feedback(data: FeedbackInput) -> FeedbackReport:
    comparison = data.comparison
    if comparison.get("practically_equivalent"):
        return build_efficiency_tradeoff_feedback(data)

    shared = comparison.get("shared_seeds") or []
    changes = _param_changes(comparison, data.candidate_contract)
    primary = comparison.get("primary_metric") or "accuracy"
    mean_delta = comparison.get("mean_delta")
    wins = comparison.get("candidate_win_count")
    base_wins = comparison.get("baseline_win_count")

    positives: list[str] = []
    if isinstance(mean_delta, (int, float)) and abs(mean_delta) > 0.001 and mean_delta > 0:
        positives.append(
            f"Mean {primary} rose (delta={mean_delta:+.6f}), "
            "but paired seed wins are not stable enough for a support claim."
        )
    elif isinstance(mean_delta, (int, float)) and abs(mean_delta) <= 0.001:
        positives.append(
            f"Mean {primary} is practically equivalent "
            f"(delta={mean_delta:+.6f})."
        )

    tradeoffs: list[str] = []
    duration_pct = _duration_increase_percent(comparison)
    if duration_pct is not None and duration_pct > DURATION_INCREASE_THRESHOLD:
        tradeoffs.append(
            "Candidate mean runtime increased substantially relative to baseline."
        )
    runtime_change = comparison.get("runtime_change_ratio")
    if isinstance(runtime_change, (int, float)) and runtime_change < -0.05:
        positives.append(
            f"Candidate reduced mean runtime by approximately {abs(runtime_change) * 100:.1f}%."
        )

    recommendations: list[ExperimentRecommendation] = [
        ExperimentRecommendation(
            recommendation_type="repeat_seeds",
            priority=0.9,
            rationale="Increase the number of matched random seeds before claiming a stable gain.",
            parameter_changes={},
        )
    ]

    # Intermediate only when not practically equivalent — explore smaller step.
    for key, change in changes.items():
        mid = _intermediate_for_numeric(change.get("from"), change.get("to"))
        if mid is not None:
            rec_type = (
                "efficiency_tradeoff"
                if duration_pct is not None and duration_pct > DURATION_INCREASE_THRESHOLD
                else "intermediate_value"
            )
            recommendations.append(
                ExperimentRecommendation(
                    recommendation_type=rec_type,  # type: ignore[arg-type]
                    priority=0.88,
                    rationale=(
                        "Results are inconclusive: try a smaller intermediate step "
                        f"for '{key}' to separate signal from seed noise / cost."
                    ),
                    parameter_changes={key: mid},
                )
            )

    if not any(item.parameter_changes for item in recommendations):
        recommendations.append(
            ExperimentRecommendation(
                recommendation_type="ablation",
                priority=0.55,
                rationale="If the change looks noisy, try a smaller parameter step.",
                parameter_changes={},
            )
        )

    return FeedbackReport(
        baseline_node_id=data.baseline_node_id,
        candidate_node_id=data.candidate_node_id,
        hypothesis_status="inconclusive",
        evidence_strength=_evidence_strength(comparison),  # type: ignore[arg-type]
        positive_findings=positives,
        negative_findings=[
            "Repeated evidence is inconclusive for a stable improvement claim.",
            f"Candidate won {wins} paired seeds; baseline won {base_wins} "
            f"(need majority of {len(shared)}).",
        ],
        tradeoffs=tradeoffs,
        uncertainties=[
            f"{len(shared)} shared seeds were evaluated.",
            "Mean delta alone is not enough when paired wins are unstable.",
        ],
        recommendations=sorted(recommendations, key=lambda r: r.priority, reverse=True),
        recommended_action="verify",
        comparison_summary={
            "mean_delta": mean_delta,
            "candidate_win_count": wins,
            "baseline_win_count": base_wins,
            "shared_seeds": shared,
            "parameter_changes": changes,
            "practically_equivalent": comparison.get("practically_equivalent"),
            "tradeoff_status": comparison.get("tradeoff_status"),
        },
    )


def enrich_v09_feedback_fields(
    report: FeedbackReport, data: FeedbackInput
) -> FeedbackReport:
    """Fill structured v0.9 feedback fields; never invent strong scientific claims."""
    comparison = data.comparison or {}
    candidate = data.candidate_contract or {}
    task_config = dict(candidate.get("task_config") or {})
    claim_level = str(
        comparison.get("claim_level")
        or task_config.get("claim_level")
        or ""
    )
    execution_mode = str(candidate.get("execution_mode") or "")
    evaluation_scope = str(task_config.get("evaluation_scope") or "")
    implementation = str(
        task_config.get("implementation")
        or comparison.get("implementation")
        or ""
    ).lower()
    shared = list(comparison.get("shared_seeds") or [])
    primary = str(comparison.get("primary_metric") or "primary_metric")
    mean_delta = comparison.get("mean_delta")

    is_fast_eval = execution_mode == "fast_eval" or "fast_eval" in evaluation_scope
    is_stand_in = implementation in {
        "",
        "stand_in",
        "standin",
        "torch_mini_standin",
        "numpy",
    }
    weak = report.evidence_strength == "weak" or is_fast_eval or is_stand_in

    scientific: list[str] = []
    if report.hypothesis_status == "invalid_comparison":
        scientific.append(
            "Comparison is invalid; no scientific performance interpretation is allowed."
        )
    elif weak:
        scientific.append(
            "Under weak / exploratory evidence, interpret metric deltas as "
            "protocol-local signals only — not full-benchmark conclusions."
        )
        if isinstance(mean_delta, (int, float)):
            scientific.append(
                f"Exploratory signal: mean {primary} delta={mean_delta:+.6f} "
                f"across {len(shared)} matched seeds."
            )
    elif report.hypothesis_status == "supported_with_repeated_evidence":
        scientific.append(
            f"Repeated matched-seed evidence supports a candidate gain on {primary}."
        )
    elif report.hypothesis_status == "rejected_with_repeated_evidence":
        scientific.append(
            f"Repeated matched-seed evidence rejects a candidate gain on {primary}."
        )
    else:
        scientific.append(
            "Evidence is inconclusive for a stable scientific performance claim."
        )

    engineering = [
        "Compared nodes completed under the recorded contracts.",
    ]
    changes = comparison.get("parameter_changes") or {}
    if changes:
        engineering.append(
            "Observed controlled parameter changes: "
            + ", ".join(sorted(str(k) for k in changes.keys()))
            + "."
        )
    if is_stand_in:
        engineering.append(
            "Stand-in implementation was used; treat results as engineering "
            "pipeline evidence, not formal DFINE acceptance."
        )
    if is_fast_eval:
        engineering.append("Fast Eval budget / subset was used.")

    performance = list(report.positive_findings) + list(report.negative_findings)
    relations = comparison.get("metric_relations") or {}
    for metric, relation in relations.items():
        if relation in {"candidate_better", "baseline_better", "tie"}:
            performance.append(f"Metric relation {metric}={relation}.")

    resource: list[str] = list(report.tradeoffs)
    resource_relations = comparison.get("resource_relations") or {}
    for metric, relation in resource_relations.items():
        if relation == "missing":
            resource.append(f"Resource metric '{metric}' is missing.")
        elif relation:
            resource.append(f"Resource relation {metric}={relation}.")

    gaps = list(report.uncertainties)
    if len(shared) < 3:
        gaps.append(f"Fewer than 3 matched seeds ({len(shared)}).")
    if is_fast_eval:
        gaps.append(
            "Fast Eval subset only; no full-training / full RGBT-Tiny evaluation yet."
        )
    if is_stand_in:
        gaps.append("Formal DFINE / vendor acceptance evidence is missing.")
    if claim_level in {"", "exploratory_comparison", "pipeline_validation_only"}:
        gaps.append(f"Claim level remains {claim_level or 'unspecified'}.")

    restrictions = [
        "Do not claim state-of-the-art / SOTA performance from this comparison.",
        "Do not claim statistically significant improvement without a formal test.",
    ]
    if weak or is_fast_eval or is_stand_in:
        restrictions.append(
            "Do not claim full RGBT-Tiny benchmark improvement from Fast Eval "
            "or stand-in evidence."
        )
        restrictions.append(
            "Do not claim formal DFINE superiority from stand-in results."
        )

    next_experiments = [
        item.rationale
        for item in report.recommendations
        if getattr(item, "status", "active") == "active"
    ]
    if weak and is_fast_eval:
        next_experiments.append(
            "Keep matched seeds and protocol fixed; escalate only after formal "
            "implementation / fuller budget acceptance."
        )

    return report.model_copy(
        update={
            "scientific_interpretation": _unique(scientific),
            "engineering_findings": _unique(engineering),
            "performance_findings": _unique(performance),
            "resource_tradeoffs": _unique(resource),
            "evidence_gaps": _unique(gaps),
            "claim_restrictions": _unique(restrictions),
            "recommended_next_experiments": _unique(next_experiments),
        }
    )


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def analyze_feedback(data: FeedbackInput) -> FeedbackReport:
    from scientist_lab.services.recommendation_filter import (
        annotate_recommendations_against_tested,
        collect_tested_parameter_index,
    )

    comparison = data.comparison
    verification = comparison.get("verification") or {}
    if verification.get("valid") is False or _status(comparison) == "invalid_comparison":
        report = build_invalid_comparison_feedback(data)
    elif comparison.get("practically_equivalent") or comparison.get("tradeoff_status") in {
        "candidate_is_efficiency_tradeoff",
        "candidate_pareto_preferred",
        "practically_equivalent_no_clear_efficiency_win",
        "baseline_more_efficient_at_parity",
    }:
        report = build_efficiency_tradeoff_feedback(data)
    else:
        status = _status(comparison)
        if status == "supported_with_repeated_evidence":
            report = build_supported_feedback(data)
        elif status == "rejected_with_repeated_evidence":
            report = build_rejected_feedback(data)
        else:
            report = build_inconclusive_feedback(data)

    source_contract = data.candidate_contract or data.baseline_contract or {}
    source_parameters = dict(source_contract.get("parameters") or {})
    tested_index = collect_tested_parameter_index(data.existing_node_parameters)
    report.recommendations = annotate_recommendations_against_tested(
        report.recommendations,
        source_parameters=source_parameters,
        tested_index=tested_index,
        tested_nodes=data.existing_node_parameters,
        defer_intermediate=_status(comparison) == "supported_with_repeated_evidence",
        source_environment_key=str(source_contract.get("environment_key") or ""),
        source_dataset_reference=str(source_contract.get("dataset_reference") or ""),
        source_code_reference=str(source_contract.get("code_reference") or ""),
    )
    return enrich_v09_feedback_fields(report, data)
