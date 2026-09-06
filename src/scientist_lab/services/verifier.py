from __future__ import annotations

from typing import Any

from scientist_lab.domain import JobStatus
from scientist_lab.domain.comparison import (
    ComparisonResult,
    HypothesisStatus,
    VerificationReport,
)
from scientist_lab.domain.models import ExecutionAttempt

# Training params that should stay fixed under single-variable comparison
CRITICAL_TRAINING_PARAMS = (
    "learning_rate",
    "epochs",
    "hidden_units",
    "batch_size",
    "test_size",
)

LOWER_IS_BETTER = frozenset({"log_loss", "loss"})

DEFAULT_MINIMUM_IMPROVEMENT = 0.002


def _metrics_blob(attempt: ExecutionAttempt) -> dict[str, Any]:
    return (attempt.result_json or {}).get("metrics") or {}


def _metric_values(attempt: ExecutionAttempt) -> dict[str, Any]:
    blob = _metrics_blob(attempt)
    values = dict(blob.get("metrics") or {})
    training = blob.get("training") or {}
    if "duration_seconds" in training and isinstance(
        training["duration_seconds"], (int, float)
    ):
        values.setdefault("duration_seconds", training["duration_seconds"])
    return values


def _primary_metric(attempt: ExecutionAttempt) -> str | None:
    return _metrics_blob(attempt).get("primary_metric")


def _contract(attempt: ExecutionAttempt) -> dict[str, Any]:
    return (attempt.result_json or {}).get("contract") or {}


def _parameters(attempt: ExecutionAttempt) -> dict[str, Any]:
    return dict(_contract(attempt).get("parameters") or {})


def verify_fair_comparison(
    baseline: ExecutionAttempt,
    candidate: ExecutionAttempt,
) -> VerificationReport:
    warnings: list[str] = []
    blocking: list[str] = []

    if baseline.status != JobStatus.COMPLETED:
        blocking.append(f"Baseline status is {baseline.status}, expected completed")
    if candidate.status != JobStatus.COMPLETED:
        blocking.append(f"Candidate status is {candidate.status}, expected completed")

    metrics_a = _metric_values(baseline)
    metrics_b = _metric_values(candidate)
    if not metrics_a:
        blocking.append("Baseline metrics.json is missing or empty")
    if not metrics_b:
        blocking.append("Candidate metrics.json is missing or empty")

    primary_a = _primary_metric(baseline)
    primary_b = _primary_metric(candidate)
    if not primary_a or not primary_b:
        blocking.append("primary_metric missing on one or both executions")
    elif primary_a != primary_b:
        blocking.append(
            f"primary_metric mismatch: baseline={primary_a}, candidate={primary_b}"
        )
    elif primary_a not in metrics_a or primary_b not in metrics_b:
        blocking.append(f"primary_metric '{primary_a}' not present in metrics values")

    contract_a = _contract(baseline)
    contract_b = _contract(candidate)
    if not contract_a or not contract_b:
        blocking.append("Contract snapshot missing on one or both executions")
    else:
        if contract_a.get("dataset_reference") != contract_b.get("dataset_reference"):
            blocking.append("dataset_reference differs")
        if contract_a.get("code_reference") != contract_b.get("code_reference"):
            blocking.append("code_reference differs")
        if contract_a.get("environment_key") != contract_b.get("environment_key"):
            blocking.append("environment_key differs")
        if contract_a.get("entrypoint") != contract_b.get("entrypoint"):
            blocking.append("entrypoint differs")
        if contract_a.get("seed") != contract_b.get("seed"):
            blocking.append(
                f"seed differs: baseline={contract_a.get('seed')}, "
                f"candidate={contract_b.get('seed')}"
            )

        params_a = _parameters(baseline)
        params_b = _parameters(candidate)
        test_a = params_a.get("test_size")
        test_b = params_b.get("test_size")
        if test_a != test_b:
            blocking.append(f"test_size differs: baseline={test_a}, candidate={test_b}")

        changed_critical = [
            key
            for key in CRITICAL_TRAINING_PARAMS
            if params_a.get(key) != params_b.get(key)
        ]
        if len(changed_critical) > 1:
            blocking.append(
                "Multiple critical training parameters changed "
                f"({', '.join(changed_critical)}); single-variable principle violated"
            )
        elif len(changed_critical) == 0 and baseline.status == JobStatus.COMPLETED:
            warnings.append(
                "No critical training parameter differences detected between runs"
            )

    warnings.append("Only one random seed was evaluated.")

    return VerificationReport(
        valid=len(blocking) == 0,
        warnings=warnings,
        blocking_issues=blocking,
    )


def judge_hypothesis(
    *,
    experiment_valid: bool,
    primary_metric: str | None,
    primary_metric_delta: float | None,
    minimum_improvement: float = DEFAULT_MINIMUM_IMPROVEMENT,
) -> HypothesisStatus:
    if not experiment_valid or primary_metric is None or primary_metric_delta is None:
        return HypothesisStatus.INCONCLUSIVE

    delta = primary_metric_delta
    if primary_metric in LOWER_IS_BETTER:
        # Improvement means delta negative (candidate lower)
        if delta < -minimum_improvement:
            return HypothesisStatus.SUPPORTED
        if delta > minimum_improvement:
            return HypothesisStatus.REJECTED
        return HypothesisStatus.INCONCLUSIVE

    if delta > minimum_improvement:
        return HypothesisStatus.SUPPORTED
    if delta < -minimum_improvement:
        return HypothesisStatus.REJECTED
    return HypothesisStatus.INCONCLUSIVE


def build_parameter_changes(
    baseline: ExecutionAttempt, candidate: ExecutionAttempt
) -> dict[str, Any]:
    params_a = _parameters(baseline)
    params_b = _parameters(candidate)
    keys = sorted(set(params_a) | set(params_b))
    changes: dict[str, Any] = {}
    for key in keys:
        if params_a.get(key) != params_b.get(key):
            changes[key] = {"from": params_a.get(key), "to": params_b.get(key)}
    return changes


def build_metric_changes(
    baseline: ExecutionAttempt, candidate: ExecutionAttempt
) -> dict[str, Any]:
    metrics_a = _metric_values(baseline)
    metrics_b = _metric_values(candidate)
    keys = sorted(set(metrics_a) | set(metrics_b))
    changes: dict[str, Any] = {}
    for key in keys:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        entry: dict[str, Any] = {"baseline": va, "candidate": vb, "delta": None}
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            entry["delta"] = float(vb) - float(va)
        changes[key] = entry
    return changes


def build_conclusion(
    *,
    hypothesis_status: HypothesisStatus,
    experiment_valid: bool,
    parameter_changes: dict[str, Any],
    primary_metric: str | None,
    primary_metric_delta: float | None,
    verification: VerificationReport,
) -> str:
    if not experiment_valid:
        issues = "; ".join(verification.blocking_issues) or "comparison invalid"
        return (
            "Comparison is not valid for causal claims about the parameter change. "
            f"Blocking issues: {issues}. Hypothesis status is inconclusive."
        )

    changed = ", ".join(parameter_changes.keys()) or "no critical parameters"
    if primary_metric is None or primary_metric_delta is None:
        return (
            f"Fair comparison succeeded (changed: {changed}), but primary metric "
            "delta is unavailable. Hypothesis status is inconclusive."
        )

    points = abs(primary_metric_delta) * 100
    direction = "improved" if primary_metric_delta > 0 else "worsened"
    if primary_metric in LOWER_IS_BETTER:
        direction = "improved" if primary_metric_delta < 0 else "worsened"

    if hypothesis_status == HypothesisStatus.SUPPORTED:
        return (
            f"In this single-seed comparison, changing {changed} {direction} "
            f"{primary_metric} by about {points:.2f} percentage points "
            f"(delta={primary_metric_delta:+.6f}). Current evidence supports the hypothesis."
        )
    if hypothesis_status == HypothesisStatus.REJECTED:
        return (
            f"In this single-seed comparison, changing {changed} {direction} "
            f"{primary_metric} by about {points:.2f} percentage points "
            f"(delta={primary_metric_delta:+.6f}). Current evidence goes against the hypothesis."
        )
    return (
        f"In this single-seed comparison, changing {changed} produced a "
        f"{primary_metric} delta of {primary_metric_delta:+.6f}, which is within "
        "the minimum-improvement band. Hypothesis status is inconclusive."
    )


def compare_attempts(
    baseline: ExecutionAttempt,
    candidate: ExecutionAttempt,
    *,
    minimum_improvement: float = DEFAULT_MINIMUM_IMPROVEMENT,
) -> ComparisonResult:
    verification = verify_fair_comparison(baseline, candidate)
    parameter_changes = build_parameter_changes(baseline, candidate)
    metric_changes = build_metric_changes(baseline, candidate)

    primary = _primary_metric(baseline) or _primary_metric(candidate)
    primary_delta: float | None = None
    relative: float | None = None
    if primary and primary in metric_changes:
        raw_delta = metric_changes[primary].get("delta")
        if isinstance(raw_delta, (int, float)):
            primary_delta = float(raw_delta)
            baseline_val = metric_changes[primary].get("baseline")
            if (
                isinstance(baseline_val, (int, float))
                and float(baseline_val) != 0
                and primary_delta is not None
            ):
                relative = (primary_delta / float(baseline_val)) * 100.0

    experiment_valid = verification.valid
    hypothesis_status = judge_hypothesis(
        experiment_valid=experiment_valid,
        primary_metric=primary,
        primary_metric_delta=primary_delta,
        minimum_improvement=minimum_improvement,
    )
    conclusion = build_conclusion(
        hypothesis_status=hypothesis_status,
        experiment_valid=experiment_valid,
        parameter_changes=parameter_changes,
        primary_metric=primary,
        primary_metric_delta=primary_delta,
        verification=verification,
    )

    return ComparisonResult(
        baseline_node_id=baseline.node_id,
        candidate_node_id=candidate.node_id,
        baseline_execution_id=baseline.execution_id,
        candidate_execution_id=candidate.execution_id,
        parameter_changes=parameter_changes,
        metric_changes=metric_changes,
        primary_metric=primary,
        primary_metric_delta=primary_delta,
        relative_improvement_percent=relative,
        verification=verification,
        experiment_valid=experiment_valid,
        hypothesis_status=hypothesis_status,
        conclusion=conclusion,
        execution_a=baseline.model_dump(),
        execution_b=candidate.model_dump(),
    )
