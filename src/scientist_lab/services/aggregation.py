from __future__ import annotations

from math import ceil
from statistics import mean, pstdev, stdev
from typing import Any

from scientist_lab.domain import JobStatus
from scientist_lab.domain.comparison import HypothesisStatus, VerificationReport
from scientist_lab.domain.models import (
    ExecutionAttempt,
    ExperimentArtifact,
    ExperimentNode,
    new_id,
    utc_now_iso,
)
from scientist_lab.storage.artifact_store import sha256_file, write_json
from scientist_lab.storage.repositories import Repository

DEFAULT_MEAN_DELTA = 0.002
DEFAULT_STD_RATIO = 1.5
DEFAULT_WIN_RATIO = 0.6
AGGREGATION_POLICY = "latest_success_per_seed"
PRACTICAL_EQUIVALENCE_THRESHOLD = 0.001
RUNTIME_MEANINGFUL_RATIO = 0.05
STABILITY_MEANINGFUL_RATIO = 1.25


def _metric_values(attempt: ExecutionAttempt) -> dict[str, Any]:
    blob = (attempt.result_json or {}).get("metrics") or {}
    values = dict(blob.get("metrics") or {})
    training = blob.get("training") or {}
    if "duration_seconds" in training and isinstance(
        training["duration_seconds"], (int, float)
    ):
        values.setdefault("duration_seconds", float(training["duration_seconds"]))

    # Optional resource / summary blobs stored on the execution result.
    for key in ("peak_gpu_memory_mb", "parameter_count", "duration_seconds"):
        if key in values and isinstance(values[key], (int, float)):
            continue
        for container_name in ("resource_usage", "model_summary", "resources"):
            container = (attempt.result_json or {}).get(container_name) or {}
            if isinstance(container.get(key), (int, float)):
                values.setdefault(key, float(container[key]))
                break
        # Also accept nested under metrics.json-style top-level fields.
        if isinstance(blob.get(key), (int, float)):
            values.setdefault(key, float(blob[key]))
    return values


def _primary_metric(attempt: ExecutionAttempt) -> str | None:
    blob = (attempt.result_json or {}).get("metrics") or {}
    return blob.get("primary_metric")


def _contract(attempt: ExecutionAttempt) -> dict[str, Any]:
    return (attempt.result_json or {}).get("contract") or {}


def _seed_of(attempt: ExecutionAttempt) -> int | None:
    seed = _contract(attempt).get("seed")
    return int(seed) if isinstance(seed, int) else None


def _summary(values: list[float]) -> dict[str, float | str | int]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std_type": "sample",
            "ddof": 1,
        }
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "std_type": "sample",
        "ddof": 1,
    }


class NodeAggregationService:
    def __init__(self, repo: Repository, outputs_root) -> None:
        self.repo = repo
        self.outputs_root = outputs_root

    def list_completed_with_metrics(
        self, node_id: str, limit: int = 200
    ) -> list[ExecutionAttempt]:
        attempts = self.repo.list_attempts(limit=limit, node_id=node_id)
        return [
            item
            for item in attempts
            if item.status == JobStatus.COMPLETED
            and (item.result_json or {}).get("metrics")
        ]

    def best_attempt_per_seed(
        self, node_id: str
    ) -> dict[int, ExecutionAttempt]:
        """Keep the newest completed attempt for each seed (latest_success_per_seed)."""
        by_seed: dict[int, ExecutionAttempt] = {}
        for attempt in self.list_completed_with_metrics(node_id):
            seed = _seed_of(attempt)
            if seed is None:
                continue
            prev = by_seed.get(seed)
            if prev is None:
                by_seed[seed] = attempt
                continue
            prev_key = prev.completed_at or prev.created_at
            cur_key = attempt.completed_at or attempt.created_at
            if cur_key >= prev_key:
                by_seed[seed] = attempt
        return by_seed

    def aggregate(self, node_id: str) -> dict[str, Any]:
        node = self.repo.get_node(node_id)
        if node is None:
            raise KeyError(f"未找到 node: {node_id}")

        by_seed = self.best_attempt_per_seed(node_id)
        if not by_seed:
            raise KeyError(f"节点 {node_id} 没有带 metrics 的 completed 执行")

        seeds = sorted(by_seed)
        metric_buckets: dict[str, list[float]] = {}
        primary_counter: dict[str, int] = {}
        executions: list[dict[str, Any]] = []

        for seed in seeds:
            attempt = by_seed[seed]
            values = _metric_values(attempt)
            primary = _primary_metric(attempt)
            if isinstance(primary, str):
                primary_counter[primary] = primary_counter.get(primary, 0) + 1
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    metric_buckets.setdefault(key, []).append(float(value))
            executions.append(
                {
                    "seed": seed,
                    "execution_id": attempt.execution_id,
                    "metrics": values,
                    "primary_metric": primary,
                }
            )

        aggregate_metrics = {
            key: _summary(vals) for key, vals in sorted(metric_buckets.items())
        }
        primary_metric = None
        if primary_counter:
            primary_metric = max(primary_counter.items(), key=lambda item: item[1])[0]

        payload = {
            "node_id": node_id,
            "project_id": node.project_id,
            "aggregation_policy": AGGREGATION_POLICY,
            "seed_count": len(seeds),
            "seeds": seeds,
            "primary_metric": primary_metric,
            "aggregate_metrics": aggregate_metrics,
            "executions": executions,
            "updated_at": utc_now_iso(),
        }

        aggregate_path = self._write_aggregate_file(node, payload)
        payload["aggregate_path"] = str(aggregate_path)
        self._persist_on_node(node, payload)
        self._register_artifact(node, by_seed[seeds[-1]], aggregate_path, payload)
        return payload

    def _write_aggregate_file(
        self, node: ExperimentNode, payload: dict[str, Any]
    ):
        path = self.outputs_root / node.project_id / node.node_id / "aggregate_metrics.json"
        write_json(path, payload)
        return path

    def _persist_on_node(self, node: ExperimentNode, payload: dict[str, Any]) -> None:
        feedback = dict(node.feedback_json or {})
        feedback["aggregate_metrics"] = payload
        node.feedback_json = feedback
        node.updated_at = utc_now_iso()
        self.repo.upsert_node(node)

    def _register_artifact(
        self,
        node: ExperimentNode,
        attempt: ExecutionAttempt,
        aggregate_path,
        payload: dict[str, Any],
    ) -> None:
        existing = self.repo.list_artifacts(attempt.execution_id)
        # Drop previous node_aggregate markers on this attempt to avoid duplicates.
        # Repository has no delete; we only add if path marker not present.
        marker = f"node_aggregate:{node.node_id}"
        if any(
            (a.metadata_json or {}).get("marker") == marker for a in existing
        ):
            return
        artifact = ExperimentArtifact(
            artifact_id=new_id("art"),
            execution_id=attempt.execution_id,
            artifact_type="node_aggregate",
            relative_path=str(aggregate_path),
            size_bytes=aggregate_path.stat().st_size,
            sha256=sha256_file(aggregate_path),
            metadata_json={
                "marker": marker,
                "node_id": node.node_id,
                "seed_count": payload.get("seed_count"),
                "aggregate_path": str(aggregate_path),
            },
            created_at=utc_now_iso(),
        )
        self.repo.add_artifact(artifact)


def build_tradeoff_assessment(
    *,
    mean_delta: float | None,
    baseline_duration: float | None,
    candidate_duration: float | None,
    baseline_std: float | None,
    candidate_std: float | None,
    mean_delta_threshold: float = DEFAULT_MEAN_DELTA,
    equivalence_threshold: float = PRACTICAL_EQUIVALENCE_THRESHOLD,
) -> dict[str, Any]:
    """Assess performance / efficiency / stability relations beyond accuracy wins."""
    practically_equivalent = (
        mean_delta is not None and abs(float(mean_delta)) <= equivalence_threshold
    )

    if practically_equivalent:
        performance_relation = "practically_equivalent"
    elif mean_delta is not None and float(mean_delta) >= mean_delta_threshold:
        performance_relation = "candidate_better"
    elif mean_delta is not None and float(mean_delta) <= -mean_delta_threshold:
        performance_relation = "baseline_better"
    else:
        performance_relation = "unclear"

    runtime_change_ratio: float | None = None
    if (
        isinstance(baseline_duration, (int, float))
        and isinstance(candidate_duration, (int, float))
        and float(baseline_duration) > 0
    ):
        runtime_change_ratio = (
            float(candidate_duration) - float(baseline_duration)
        ) / float(baseline_duration)

    if runtime_change_ratio is None:
        efficiency_relation = "unknown"
    elif runtime_change_ratio <= -RUNTIME_MEANINGFUL_RATIO:
        efficiency_relation = "candidate_better"
    elif runtime_change_ratio >= RUNTIME_MEANINGFUL_RATIO:
        efficiency_relation = "baseline_better"
    else:
        efficiency_relation = "similar"

    std_ratio: float | None = None
    if (
        isinstance(baseline_std, (int, float))
        and isinstance(candidate_std, (int, float))
        and float(baseline_std) > 1e-12
    ):
        std_ratio = float(candidate_std) / float(baseline_std)

    if std_ratio is None:
        stability_relation = "unknown"
    elif std_ratio >= STABILITY_MEANINGFUL_RATIO:
        stability_relation = "baseline_better"
    elif std_ratio <= 1.0 / STABILITY_MEANINGFUL_RATIO:
        stability_relation = "candidate_better"
    else:
        stability_relation = "similar"

    runtime_reduction_ratio = (
        -runtime_change_ratio if runtime_change_ratio is not None else None
    )
    pareto_preferred = (
        practically_equivalent
        and runtime_change_ratio is not None
        and runtime_change_ratio <= -RUNTIME_MEANINGFUL_RATIO
        and stability_relation != "baseline_better"
    )
    efficiency_tradeoff = (
        practically_equivalent
        and runtime_change_ratio is not None
        and runtime_change_ratio <= -RUNTIME_MEANINGFUL_RATIO
        and stability_relation == "baseline_better"
    )

    if efficiency_tradeoff:
        tradeoff_status = "candidate_is_efficiency_tradeoff"
        overall_decision = "candidate_is_efficiency_tradeoff"
    elif pareto_preferred:
        tradeoff_status = "candidate_pareto_preferred"
        overall_decision = "candidate_preferred_on_efficiency"
    elif practically_equivalent and efficiency_relation == "baseline_better":
        tradeoff_status = "baseline_more_efficient_at_parity"
        overall_decision = "keep_baseline_unless_other_goals"
    elif practically_equivalent:
        tradeoff_status = "practically_equivalent_no_clear_efficiency_win"
        overall_decision = "inconclusive_tradeoff"
    else:
        tradeoff_status = "not_applicable"
        overall_decision = "use_primary_metric_rules"

    return {
        "practically_equivalent": practically_equivalent,
        "practical_equivalence_threshold": equivalence_threshold,
        "performance_relation": performance_relation,
        "runtime_change_ratio": runtime_change_ratio,
        "runtime_reduction_ratio": runtime_reduction_ratio,
        "efficiency_relation": efficiency_relation,
        "std_ratio": std_ratio,
        "stability_relation": stability_relation,
        "pareto_preferred": pareto_preferred,
        "tradeoff_status": tradeoff_status,
        "overall_decision": overall_decision,
    }


def compare_node_groups(
    aggregation: NodeAggregationService,
    baseline_node_id: str,
    candidate_node_id: str,
    *,
    mean_delta_threshold: float = DEFAULT_MEAN_DELTA,
    std_ratio_limit: float = DEFAULT_STD_RATIO,
    win_ratio: float = DEFAULT_WIN_RATIO,
) -> dict[str, Any]:
    baseline = aggregation.aggregate(baseline_node_id)
    candidate = aggregation.aggregate(candidate_node_id)

    baseline_by_seed = {
        item["seed"]: item for item in baseline["executions"]
    }
    candidate_by_seed = {
        item["seed"]: item for item in candidate["executions"]
    }
    shared_seeds = sorted(set(baseline_by_seed) & set(candidate_by_seed))

    verification = VerificationReport(valid=True, warnings=[], blocking_issues=[])
    if len(shared_seeds) < 2:
        verification.valid = False
        verification.blocking_issues.append(
            f"Need at least 2 shared seeds for repeated comparison; got {len(shared_seeds)}"
        )

    primary = baseline.get("primary_metric") or candidate.get("primary_metric") or "accuracy"
    if baseline.get("primary_metric") and candidate.get("primary_metric"):
        if baseline["primary_metric"] != candidate["primary_metric"]:
            verification.valid = False
            verification.blocking_issues.append("primary_metric differs between nodes")

    # Fairness: non-seed contract fields should match on paired runs
    sample_b = aggregation.best_attempt_per_seed(baseline_node_id).get(
        shared_seeds[0]
    ) if shared_seeds else None
    sample_c = aggregation.best_attempt_per_seed(candidate_node_id).get(
        shared_seeds[0]
    ) if shared_seeds else None
    if sample_b and sample_c:
        cb = _contract(sample_b)
        cc = _contract(sample_c)
        for field in ("dataset_reference", "code_reference", "environment_key", "entrypoint"):
            if cb.get(field) != cc.get(field):
                verification.valid = False
                verification.blocking_issues.append(f"{field} differs between nodes")
        pb = dict(cb.get("parameters") or {})
        pc = dict(cc.get("parameters") or {})
        from scientist_lab.services.verifier import CRITICAL_TRAINING_PARAMS

        changed = [
            key
            for key in CRITICAL_TRAINING_PARAMS
            if pb.get(key) != pc.get(key)
        ]
        if len(changed) > 1:
            verification.valid = False
            verification.blocking_issues.append(
                "Multiple critical training parameters changed: " + ", ".join(changed)
            )

    paired: list[dict[str, Any]] = []
    deltas: list[float] = []
    candidate_wins = 0
    baseline_wins = 0
    ties = 0

    for seed in shared_seeds:
        b_metrics = baseline_by_seed[seed]["metrics"]
        c_metrics = candidate_by_seed[seed]["metrics"]
        b_val = b_metrics.get(primary)
        c_val = c_metrics.get(primary)
        delta = None
        if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
            delta = float(c_val) - float(b_val)
            deltas.append(delta)
            if delta > 0:
                candidate_wins += 1
            elif delta < 0:
                baseline_wins += 1
            else:
                ties += 1
        paired.append(
            {
                "seed": seed,
                "baseline_execution_id": baseline_by_seed[seed]["execution_id"],
                "candidate_execution_id": candidate_by_seed[seed]["execution_id"],
                "baseline_value": b_val,
                "candidate_value": c_val,
                "delta": delta,
            }
        )

    baseline_mean = (baseline.get("aggregate_metrics") or {}).get(primary, {}).get("mean")
    candidate_mean = (candidate.get("aggregate_metrics") or {}).get(primary, {}).get("mean")
    baseline_std = (baseline.get("aggregate_metrics") or {}).get(primary, {}).get("std", 0.0)
    candidate_std = (candidate.get("aggregate_metrics") or {}).get(primary, {}).get("std", 0.0)

    mean_delta = None
    if isinstance(baseline_mean, (int, float)) and isinstance(candidate_mean, (int, float)):
        mean_delta = float(candidate_mean) - float(baseline_mean)
    elif deltas:
        mean_delta = float(mean(deltas))

    required_wins = max(1, ceil(len(shared_seeds) * win_ratio)) if shared_seeds else 1
    stable_improvement = False
    stable_degradation = False
    if (
        verification.valid
        and mean_delta is not None
        and isinstance(baseline_std, (int, float))
        and isinstance(candidate_std, (int, float))
    ):
        std_ok = candidate_std <= max(baseline_std * std_ratio_limit, 1e-12)
        stable_improvement = (
            verification.valid
            and mean_delta >= mean_delta_threshold
            and candidate_wins >= required_wins
            and std_ok
        )
        stable_degradation = (
            verification.valid
            and mean_delta <= -mean_delta_threshold
            and baseline_wins >= required_wins
            and baseline_std <= max(candidate_std * std_ratio_limit, 1e-12)
        )

    if not verification.valid:
        hypothesis_status = HypothesisStatus.INVALID_COMPARISON
        conclusion = (
            "Node-group comparison is invalid. "
            + "; ".join(verification.blocking_issues)
        )
    elif stable_improvement:
        hypothesis_status = HypothesisStatus.SUPPORTED_WITH_REPEATED_EVIDENCE
        conclusion = (
            f"Across {len(shared_seeds)} shared seeds, candidate improved mean "
            f"{primary} by {mean_delta:+.6f} and won {candidate_wins}/{len(shared_seeds)} "
            "paired comparisons. Current repeated evidence supports the hypothesis."
        )
    elif stable_degradation:
        hypothesis_status = HypothesisStatus.REJECTED_WITH_REPEATED_EVIDENCE
        conclusion = (
            f"Across {len(shared_seeds)} shared seeds, candidate worsened mean "
            f"{primary} by {mean_delta:+.6f} and lost {baseline_wins}/{len(shared_seeds)} "
            "paired comparisons. Current repeated evidence goes against the hypothesis."
        )
    else:
        hypothesis_status = HypothesisStatus.INCONCLUSIVE
        conclusion = (
            f"Across {len(shared_seeds)} shared seeds, mean {primary} delta="
            f"{mean_delta if mean_delta is not None else 'n/a'}, "
            f"candidate wins={candidate_wins}, baseline wins={baseline_wins}. "
            "Evidence is insufficient for a stable improvement claim."
        )

    if verification.valid and len(shared_seeds) < 5:
        verification.warnings.append(
            f"Only {len(shared_seeds)} shared seeds were evaluated."
        )
    if verification.valid and len(shared_seeds) <= 5:
        verification.warnings.append(
            "Evidence strength should not be overstated with a small seed set."
        )

    parameter_changes: dict[str, Any] = {}
    if sample_b and sample_c:
        pb = dict(_contract(sample_b).get("parameters") or {})
        pc = dict(_contract(sample_c).get("parameters") or {})
        for key in sorted(set(pb) | set(pc)):
            if pb.get(key) != pc.get(key):
                parameter_changes[key] = {"from": pb.get(key), "to": pc.get(key)}

    baseline_duration = (
        (baseline.get("aggregate_metrics") or {}).get("duration_seconds") or {}
    ).get("mean")
    candidate_duration = (
        (candidate.get("aggregate_metrics") or {}).get("duration_seconds") or {}
    ).get("mean")

    tradeoff = build_tradeoff_assessment(
        mean_delta=mean_delta,
        baseline_duration=baseline_duration
        if isinstance(baseline_duration, (int, float))
        else None,
        candidate_duration=candidate_duration
        if isinstance(candidate_duration, (int, float))
        else None,
        baseline_std=baseline_std if isinstance(baseline_std, (int, float)) else None,
        candidate_std=candidate_std if isinstance(candidate_std, (int, float)) else None,
        mean_delta_threshold=mean_delta_threshold,
    )

    # Refine conclusion language when metrics are practically equivalent.
    if verification.valid and tradeoff["practically_equivalent"]:
        runtime_txt = ""
        ratio = tradeoff.get("runtime_change_ratio")
        if isinstance(ratio, (int, float)):
            pct = abs(float(ratio)) * 100
            if float(ratio) < 0:
                runtime_txt = f" Candidate mean runtime is about {pct:.1f}% lower."
            elif float(ratio) > 0:
                runtime_txt = f" Candidate mean runtime is about {pct:.1f}% higher."
        stability_txt = ""
        if tradeoff["stability_relation"] == "baseline_better":
            stability_txt = " Baseline remains more stable (lower primary-metric std)."
        elif tradeoff["stability_relation"] == "candidate_better":
            stability_txt = " Candidate is more stable (lower primary-metric std)."
        conclusion = (
            f"Across {len(shared_seeds)} shared seeds, mean {primary} is practically "
            f"equivalent (delta={mean_delta:+.6f})."
            f"{runtime_txt}{stability_txt}"
        )

    # Highlight single-seed dominated deltas.
    if deltas:
        max_abs = max(abs(value) for value in deltas)
        if max_abs >= 0.02 and abs(float(mean_delta or 0.0)) > 0:
            for item in paired:
                delta = item.get("delta")
                if isinstance(delta, (int, float)) and abs(delta) == max_abs:
                    verification.warnings.append(
                        f"Mean delta is partly driven by seed {item['seed']} "
                        f"(paired delta={delta:+.6f})."
                    )
                    break

    sample_contract = _contract(sample_c) if sample_c else (
        _contract(sample_b) if sample_b else None
    )

    payload = {
        "baseline_node_id": baseline_node_id,
        "candidate_node_id": candidate_node_id,
        "shared_seeds": shared_seeds,
        "primary_metric": primary,
        "parameter_changes": parameter_changes,
        "baseline_accuracy_mean": baseline_mean if primary == "accuracy" else None,
        "candidate_accuracy_mean": candidate_mean if primary == "accuracy" else None,
        "baseline_primary_mean": baseline_mean,
        "candidate_primary_mean": candidate_mean,
        "baseline_primary_std": baseline_std,
        "candidate_primary_std": candidate_std,
        "baseline_duration_mean": baseline_duration,
        "candidate_duration_mean": candidate_duration,
        "mean_delta": mean_delta,
        "candidate_win_count": candidate_wins,
        "baseline_win_count": baseline_wins,
        "tie_count": ties,
        "required_win_count": required_wins,
        "stable_improvement": stable_improvement,
        "paired_deltas": paired,
        "baseline_aggregate": baseline,
        "candidate_aggregate": candidate,
        "verification": verification.model_dump(),
        "hypothesis_status": str(hypothesis_status),
        "conclusion": conclusion,
        "population_std_note": None if len(deltas) < 2 else float(pstdev(deltas)),
        **tradeoff,
    }

    from scientist_lab.services.detection_comparison import enrich_group_comparison

    return enrich_group_comparison(payload, sample_contract=sample_contract)
