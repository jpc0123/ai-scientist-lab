"""Build EvidenceRecord objects from comparison / execution payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scientist_lab.domain.models import new_id
from scientist_lab.evidence.claim_gate import assess_evidence_strength
from scientist_lab.evidence.models import EvidenceRecord


def _contract_from_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    for key in ("candidate_aggregate", "baseline_aggregate"):
        agg = comparison.get(key) or {}
        for item in agg.get("executions") or []:
            # executions in aggregate may not embed full contract; handled by caller
            _ = item
    return {}


def collect_execution_ids(comparison: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in comparison.get("paired_deltas") or []:
        for key in ("baseline_execution_id", "candidate_execution_id"):
            value = item.get(key)
            if value and value not in ids:
                ids.append(str(value))
    if not ids:
        for key in ("baseline_aggregate", "candidate_aggregate"):
            for item in (comparison.get(key) or {}).get("executions") or []:
                value = item.get("execution_id")
                if value and value not in ids:
                    ids.append(str(value))
    return ids


def build_paired_comparison_evidence(
    comparison: dict[str, Any],
    *,
    project_id: str,
    source_artifact_ids: list[str] | None = None,
    protocol_id: str | None = None,
    sample_contract: dict[str, Any] | None = None,
    has_ablation: bool = False,
    formal_implementation: bool = False,
    evidence_id: str | None = None,
    comparison_path: str | None = None,
) -> EvidenceRecord:
    shared_seeds = list(comparison.get("shared_seeds") or [])
    seed_count = len(shared_seeds)
    contract = dict(sample_contract or _contract_from_comparison(comparison))
    task_config = dict(contract.get("task_config") or {})

    implementation = str(
        task_config.get("implementation")
        or comparison.get("implementation")
        or ""
    )
    execution_mode = str(contract.get("execution_mode") or "")
    evaluation_scope = str(task_config.get("evaluation_scope") or "")
    claim_level = str(
        comparison.get("claim_level")
        or task_config.get("claim_level")
        or ""
    )
    resolved_protocol = protocol_id or contract.get("protocol_id")

    assessment = assess_evidence_strength(
        seed_count=seed_count,
        execution_mode=execution_mode,
        evaluation_scope=evaluation_scope,
        implementation=implementation,
        claim_level=claim_level,
        has_protocol=bool(resolved_protocol),
        has_ablation=has_ablation,
        formal_implementation=formal_implementation,
        stable_direction=bool(comparison.get("stable_improvement")),
        requested_type="paired_comparison"
        if seed_count >= 2
        else "single_execution",
    )

    metric_summary: dict[str, Any] = {
        "primary_metric": comparison.get("primary_metric"),
        "mean_delta": comparison.get("mean_delta"),
        "paired_win_count": comparison.get("paired_win_count")
        or comparison.get("candidate_win_count"),
        "paired_loss_count": comparison.get("paired_loss_count")
        or comparison.get("baseline_win_count"),
        "seed_count": seed_count,
        "shared_seeds": shared_seeds,
        "metric_relations": comparison.get("metric_relations") or {},
        "resource_relations": comparison.get("resource_relations") or {},
    }
    if isinstance(comparison.get("mean_delta"), (int, float)):
        primary = comparison.get("primary_metric") or "primary"
        metric_summary[f"{primary}_mean_delta"] = comparison.get("mean_delta")
    for key, value in (comparison.get("metric_mean_deltas") or {}).items():
        if value is not None:
            metric_summary[f"{key}_mean_delta"] = value

    verification = comparison.get("verification") or {}
    valid = bool(verification.get("valid", True)) and bool(
        assessment.get("effective_evidence_type")
    )

    node_ids = [
        nid
        for nid in (
            comparison.get("baseline_node_id"),
            comparison.get("candidate_node_id"),
        )
        if nid
    ]

    return EvidenceRecord(
        evidence_id=evidence_id or new_id("evidence"),
        project_id=project_id,
        evidence_type=assessment["effective_evidence_type"],  # type: ignore[arg-type]
        source_node_ids=node_ids,
        source_execution_ids=collect_execution_ids(comparison),
        source_artifact_ids=list(source_artifact_ids or []),
        protocol_id=str(resolved_protocol) if resolved_protocol else None,
        metric_summary=metric_summary,
        evidence_strength=assessment["evidence_strength"],  # type: ignore[arg-type]
        engineering_evidence_level=assessment.get("engineering_evidence_level"),
        scientific_evidence_level=assessment.get("scientific_evidence_level"),
        limitations=list(assessment.get("limitations") or []),
        valid=valid,
        claim_level=claim_level or None,
        comparison_path=comparison_path,
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
    )


def build_resource_comparison_evidence(
    comparison: dict[str, Any],
    *,
    project_id: str,
    paired_evidence_id: str | None = None,
    source_artifact_ids: list[str] | None = None,
    protocol_id: str | None = None,
    sample_contract: dict[str, Any] | None = None,
) -> EvidenceRecord:
    """Secondary evidence focused on resource tradeoffs."""
    base = build_paired_comparison_evidence(
        comparison,
        project_id=project_id,
        source_artifact_ids=source_artifact_ids,
        protocol_id=protocol_id,
        sample_contract=sample_contract,
    )
    summary = dict(base.metric_summary)
    summary["resource_relations"] = comparison.get("resource_relations") or {}
    summary["resource_mean_deltas"] = comparison.get("resource_mean_deltas") or {}
    if paired_evidence_id:
        summary["paired_evidence_id"] = paired_evidence_id
    limitations = list(base.limitations)
    relations = comparison.get("resource_relations") or {}
    if any(value == "missing" for value in relations.values()):
        limitations.append("One or more resource metrics are missing.")
    return base.model_copy(
        update={
            "evidence_id": new_id("evidence"),
            "evidence_type": "resource_comparison",
            "metric_summary": summary,
            "limitations": limitations,
        }
    )
