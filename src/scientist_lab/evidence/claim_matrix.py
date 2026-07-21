"""Build Claim Support Matrix from EvidenceRecords (rule-based, no LLM)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scientist_lab.evidence.models import (
    ClaimSupportMatrix,
    EvidenceRecord,
    ScientificClaim,
)


def _limitations_blob(records: list[EvidenceRecord]) -> str:
    parts: list[str] = []
    for record in records:
        parts.extend(record.limitations)
        if record.scientific_evidence_level:
            parts.append(str(record.scientific_evidence_level))
        if record.engineering_evidence_level:
            parts.append(str(record.engineering_evidence_level))
        if record.claim_level:
            parts.append(str(record.claim_level))
        parts.append(str(record.evidence_strength))
    return " ".join(parts).lower()


def _is_fast_eval_or_standin(records: list[EvidenceRecord]) -> bool:
    blob = _limitations_blob(records)
    if "fast eval" in blob or "stand-in" in blob or "stand_in" in blob:
        return True
    for record in records:
        if record.claim_level == "exploratory_comparison":
            return True
        summary = record.metric_summary or {}
        if str(summary.get("evaluation_scope") or "").startswith("fast_eval"):
            return True
    return False


def _paired_records(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    return [
        item
        for item in records
        if item.evidence_type == "paired_comparison" and item.valid
    ]


def _metric_relation(
    records: list[EvidenceRecord], metric: str
) -> str | None:
    for record in _paired_records(records):
        relations = (record.metric_summary or {}).get("metric_relations") or {}
        if metric in relations:
            return str(relations[metric])
        delta_key = f"{metric}_mean_delta"
        if delta_key in (record.metric_summary or {}):
            delta = record.metric_summary.get(delta_key)
            if isinstance(delta, (int, float)):
                if delta > 0:
                    return "candidate_better"
                if delta < 0:
                    return "baseline_better"
                return "tie"
    return None


def _supporting_ids(
    records: list[EvidenceRecord], *, evidence_types: list[str] | None = None
) -> list[str]:
    ids: list[str] = []
    allowed = set(evidence_types or [])
    for record in records:
        if allowed and record.evidence_type not in allowed:
            continue
        if record.evidence_id not in ids:
            ids.append(record.evidence_id)
    return ids


def evaluate_exploratory_metric_claim(
    *,
    project_id: str,
    claim_id: str,
    claim_text: str,
    metric: str,
    records: list[EvidenceRecord],
) -> ScientificClaim:
    required = ["paired_comparison"]
    paired = _paired_records(records)
    if not paired:
        return ScientificClaim(
            claim_id=claim_id,
            project_id=project_id,
            claim_text=claim_text,
            claim_type="exploratory_metric_comparison",
            required_evidence_types=required,
            support_status="unsupported",
            supporting_evidence_ids=[],
            limitations=["No paired_comparison evidence available."],
            reason="Missing required paired comparison evidence.",
        )

    relation = _metric_relation(records, metric)
    ids = _supporting_ids(paired, evidence_types=required)
    limitations: list[str] = []
    for record in paired:
        limitations.extend(record.limitations)

    if relation == "candidate_better":
        # Exploratory / Fast Eval protocol claims may be "supported" on direction,
        # even when scientific evidence_strength remains weak.
        return ScientificClaim(
            claim_id=claim_id,
            project_id=project_id,
            claim_text=claim_text,
            claim_type="exploratory_metric_comparison",
            required_evidence_types=required,
            support_status="supported",
            supporting_evidence_ids=ids,
            limitations=list(dict.fromkeys(limitations)),
            reason=None,
        )

    if relation is None:
        return ScientificClaim(
            claim_id=claim_id,
            project_id=project_id,
            claim_text=claim_text,
            claim_type="exploratory_metric_comparison",
            required_evidence_types=required,
            support_status="unsupported",
            supporting_evidence_ids=ids,
            limitations=list(dict.fromkeys(limitations + [f"Metric {metric} not present in evidence."])),
            reason=f"Evidence does not include relation for {metric}.",
        )

    return ScientificClaim(
        claim_id=claim_id,
        project_id=project_id,
        claim_text=claim_text,
        claim_type="exploratory_metric_comparison",
        required_evidence_types=required,
        support_status="unsupported",
        supporting_evidence_ids=ids,
        limitations=list(dict.fromkeys(limitations)),
        reason=f"Metric relation for {metric} is {relation}, not candidate_better.",
    )


def evaluate_full_benchmark_claim(
    *,
    project_id: str,
    records: list[EvidenceRecord],
) -> ScientificClaim:
    claim_text = "Fusion improves performance on the full RGBT-Tiny benchmark."
    required = ["paired_comparison", "repeated_experiment"]
    ids = _supporting_ids(records)
    if _is_fast_eval_or_standin(records) or not records:
        return ScientificClaim(
            claim_id="claim_full_rgbt_tiny",
            project_id=project_id,
            claim_text=claim_text,
            claim_type="full_benchmark",
            required_evidence_types=required,
            support_status="blocked",
            supporting_evidence_ids=ids,
            limitations=[
                "Full RGBT-Tiny benchmark claim requires full_train budget "
                "and formal (non-stand-in) implementation."
            ],
            reason="Only Fast Eval and stand-in evidence are available."
            if records
            else "No evidence available for full benchmark claim.",
        )
    # Without explicit full-train formal evidence, remain blocked.
    return ScientificClaim(
        claim_id="claim_full_rgbt_tiny",
        project_id=project_id,
        claim_text=claim_text,
        claim_type="full_benchmark",
        required_evidence_types=required,
        support_status="blocked",
        supporting_evidence_ids=ids,
        limitations=["Full benchmark acceptance criteria are not met."],
        reason="Full training / formal DFINE acceptance is incomplete.",
    )


def evaluate_sota_claim(*, project_id: str, records: list[EvidenceRecord]) -> ScientificClaim:
    return ScientificClaim(
        claim_id="claim_sota",
        project_id=project_id,
        claim_text="The method achieves state-of-the-art performance.",
        claim_type="sota",
        required_evidence_types=["paired_comparison", "repeated_experiment", "ablation"],
        support_status="blocked",
        supporting_evidence_ids=_supporting_ids(records),
        limitations=["SOTA claims require matched published benchmark comparisons."],
        reason="No matched published benchmark comparison exists.",
    )


def evaluate_formal_dfine_claim(
    *, project_id: str, records: list[EvidenceRecord]
) -> ScientificClaim:
    ids = _supporting_ids(records)
    blob = _limitations_blob(records)
    if "stand-in" in blob or "stand_in" in blob or not records:
        return ScientificClaim(
            claim_id="claim_formal_dfine",
            project_id=project_id,
            claim_text="The method demonstrates formal DFINE superiority.",
            claim_type="formal_implementation",
            required_evidence_types=["paired_comparison"],
            support_status="blocked",
            supporting_evidence_ids=ids,
            limitations=["Stand-in evidence cannot support formal DFINE claims."],
            reason="Stand-in / non-vendor DFINE evidence cannot support this claim.",
        )
    return ScientificClaim(
        claim_id="claim_formal_dfine",
        project_id=project_id,
        claim_text="The method demonstrates formal DFINE superiority.",
        claim_type="formal_implementation",
        required_evidence_types=["paired_comparison"],
        support_status="unsupported",
        supporting_evidence_ids=ids,
        limitations=["Formal DFINE claim still requires stronger acceptance evidence."],
        reason="Formal DFINE evidence is incomplete.",
    )


def build_claim_support_matrix(
    *,
    project_id: str,
    records: list[EvidenceRecord],
    protocol_id: str | None = None,
) -> ClaimSupportMatrix:
    resolved_protocol = protocol_id
    if not resolved_protocol:
        for record in records:
            if record.protocol_id:
                resolved_protocol = record.protocol_id
                break

    claims = [
        evaluate_exploratory_metric_claim(
            project_id=project_id,
            claim_id="claim_fast_eval_ap_small",
            claim_text=(
                "Fusion produced higher AP_small under the fixed Fast Eval protocol."
            ),
            metric="AP_small",
            records=records,
        ),
        evaluate_exploratory_metric_claim(
            project_id=project_id,
            claim_id="claim_fast_eval_map",
            claim_text=(
                "Fusion produced higher mAP50_95 under the fixed Fast Eval protocol."
            ),
            metric="mAP50_95",
            records=records,
        ),
        evaluate_full_benchmark_claim(project_id=project_id, records=records),
        evaluate_sota_claim(project_id=project_id, records=records),
        evaluate_formal_dfine_claim(project_id=project_id, records=records),
    ]

    return ClaimSupportMatrix(
        project_id=project_id,
        protocol_id=resolved_protocol,
        claims=claims,
        evidence_ids=[item.evidence_id for item in records],
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
    )


def matrix_export_dict(matrix: ClaimSupportMatrix) -> dict[str, Any]:
    """CLI / JSON shape aligned with the plan example."""
    claims_out: list[dict[str, Any]] = []
    for claim in matrix.claims:
        item: dict[str, Any] = {
            "claim_id": claim.claim_id,
            "claim": claim.claim_text,
            "claim_type": claim.claim_type,
            "support_status": claim.support_status,
            "evidence": claim.supporting_evidence_ids,
            "required_evidence_types": claim.required_evidence_types,
            "limitations": claim.limitations,
        }
        if claim.reason:
            item["reason"] = claim.reason
        claims_out.append(item)
    return {
        "project_id": matrix.project_id,
        "protocol_id": matrix.protocol_id,
        "evidence_ids": matrix.evidence_ids,
        "created_at": matrix.created_at.isoformat() if matrix.created_at else None,
        "matrix_path": matrix.matrix_path,
        "claims": claims_out,
    }
