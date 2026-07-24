"""Structured Evidence / Claim / Report views for v2.0.5."""

from __future__ import annotations

from typing import Any


EVIDENCE_TYPE_LABELS: dict[str, str] = {
    "single_execution": "单次执行",
    "repeated_experiment": "重复实验",
    "paired_comparison": "配对比较",
    "ablation": "消融",
    "resource_comparison": "资源比较",
    "failure_analysis": "失败分析",
    "patch_evidence": "补丁证据",
}

CLAIM_STATUS_LABELS: dict[str, str] = {
    "supported": "已支持",
    "partially_supported": "部分支持",
    "unsupported": "未支持",
    "blocked": "已阻断",
}


def _as_record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def enrich_evidence(record: dict[str, Any]) -> dict[str, Any]:
    evidence_type = str(record.get("evidence_type") or "")
    payload = dict(record)
    payload["evidence_type_label"] = EVIDENCE_TYPE_LABELS.get(
        evidence_type, evidence_type or "未知"
    )
    payload["summary"] = {
        "evidence_id": record.get("evidence_id"),
        "project_id": record.get("project_id"),
        "evidence_type": evidence_type,
        "evidence_type_label": payload["evidence_type_label"],
        "evidence_strength": record.get("evidence_strength"),
        "valid": bool(record.get("valid", True)),
        "protocol_id": record.get("protocol_id"),
        "source_node_ids": list(record.get("source_node_ids") or []),
        "source_execution_ids": list(record.get("source_execution_ids") or []),
        "source_artifact_ids": list(record.get("source_artifact_ids") or []),
        "metric_summary": _as_record(record.get("metric_summary")),
        "limitations": list(record.get("limitations") or []),
        "claim_level": record.get("claim_level"),
        "created_at": record.get("created_at"),
    }
    return payload


def enrich_claim(claim: dict[str, Any]) -> dict[str, Any]:
    status = str(claim.get("support_status") or "unsupported")
    evidence_ids = list(
        claim.get("supporting_evidence_ids")
        or claim.get("evidence")
        or []
    )
    payload = dict(claim)
    payload["claim_text"] = claim.get("claim_text") or claim.get("claim") or ""
    payload["supporting_evidence_ids"] = evidence_ids
    payload["support_status_label"] = CLAIM_STATUS_LABELS.get(status, status)
    payload["next_evidence_needed"] = [
        item
        for item in (claim.get("required_evidence_types") or [])
        if item not in {
            # Keep required types visible; UI can still show gaps.
        }
    ]
    # Prefer explicit next-step gaps when present.
    if claim.get("next_evidence_needed"):
        payload["next_evidence_needed"] = list(claim["next_evidence_needed"])
    elif status in {"unsupported", "partially_supported", "blocked"}:
        payload["next_evidence_needed"] = list(
            claim.get("required_evidence_types") or []
        )
    else:
        payload["next_evidence_needed"] = []
    payload["block_reason"] = claim.get("reason") or (
        "blocked by claim gate" if status == "blocked" else None
    )
    return payload


def enrich_claim_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    claims = [enrich_claim(_as_record(item)) for item in (matrix.get("claims") or [])]
    by_status: dict[str, int] = {
        "supported": 0,
        "partially_supported": 0,
        "unsupported": 0,
        "blocked": 0,
    }
    for claim in claims:
        status = str(claim.get("support_status") or "unsupported")
        if status in by_status:
            by_status[status] += 1
        else:
            by_status[status] = by_status.get(status, 0) + 1
    payload = dict(matrix)
    payload["claims"] = claims
    payload["status_counts"] = by_status
    payload["claim_count"] = len(claims)
    payload["sections"] = {
        "overview": {
            "project_id": matrix.get("project_id"),
            "protocol_id": matrix.get("protocol_id"),
            "claim_count": len(claims),
            "evidence_count": len(matrix.get("evidence_ids") or []),
            "status_counts": by_status,
            "matrix_path": matrix.get("matrix_path"),
        },
        "claims": claims,
    }
    return payload


def build_report_sections(report: dict[str, Any]) -> dict[str, Any]:
    verification = _as_record(report.get("verification"))
    return {
        "overview": {
            "report_id": report.get("report_id"),
            "project_id": report.get("project_id"),
            "status": report.get("status"),
            "research_goal": report.get("research_goal"),
            "protocol_id": report.get("protocol_id"),
            "tree_id": report.get("tree_id"),
            "verification_valid": verification.get("valid"),
            "created_at": report.get("created_at"),
        },
        "key_results": {
            "metrics": list(report.get("metrics_table") or []),
            "stability": list(report.get("stability_table") or []),
            "resources": list(report.get("resource_table") or []),
            "ablations": list(report.get("ablation_table") or []),
            "failures": list(report.get("failure_table") or []),
        },
        "key_path": list(report.get("key_path") or []),
        "claims": {
            "supported": list(report.get("supported_claims") or []),
            "blocked": list(report.get("blocked_claims") or []),
            "conclusions": list(report.get("conclusions") or []),
        },
        "limitations": list(report.get("limitations") or []),
        "open_evidence_gaps": list(report.get("open_evidence_gaps") or []),
        "recommended_next_experiments": list(
            report.get("recommended_next_experiments") or []
        ),
        "reproducibility": {
            "dataset_references": list(report.get("dataset_references") or []),
            "environment_keys": list(report.get("environment_keys") or []),
            "image_references": list(report.get("image_references") or []),
            "context_sha256": report.get("context_sha256"),
            "generator_version": report.get("generator_version"),
            "markdown_path": report.get("markdown_path"),
            "json_path": report.get("json_path"),
            "summary_path": report.get("summary_path"),
        },
        "audit": {
            "status": "linked" if report.get("json_path") else "not_built",
            "note": "在审计中心构建 Audit Bundle 以完成封存",
        },
    }


def enrich_report(report: dict[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    payload["sections"] = build_report_sections(report)
    payload["export_options"] = ["markdown", "json", "audit_bundle"]
    return payload
