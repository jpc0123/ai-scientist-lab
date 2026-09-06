"""Claim-aware report verifier (v1.2.6). Never allow weak evidence → strong claims."""

from __future__ import annotations

from typing import Any

from scientist_lab.reporting.report_generator import FORBIDDEN_PHRASES, _is_forbidden
from scientist_lab.reporting.models import ResearchReport


def verify_research_report(report: ResearchReport) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []

    if not report.conclusions:
        blocking.append("Report has no conclusions.")

    for item in report.conclusions:
        if not item.text or not str(item.text).strip():
            blocking.append(f"{item.conclusion_id}: empty conclusion text")
            continue
        if _is_forbidden(item.text):
            blocking.append(
                f"{item.conclusion_id}: forbidden strong-claim phrasing detected"
            )
        status = str(item.support_status or "")
        if status in {"blocked", "unsupported"} and item.strength in {
            "strong",
            "moderate",
        }:
            blocking.append(
                f"{item.conclusion_id}: blocked/unsupported claim cannot have "
                f"strength={item.strength}"
            )
        if status == "supported" and not item.evidence_ids:
            warnings.append(
                f"{item.conclusion_id}: supported conclusion lacks evidence_ids"
            )
        if not item.limitations:
            warnings.append(f"{item.conclusion_id}: missing limitations")
        # Require claim linkage except for scoped metric observations.
        if item.claim_id is None and "关键路径末端" not in item.text and "尚无足够" not in item.text:
            warnings.append(
                f"{item.conclusion_id}: conclusion has no claim_id linkage"
            )

    # Blocked claims must not appear as supported conclusions with strong wording.
    blocked_ids = {
        str(c.get("claim_id"))
        for c in report.blocked_claims
        if str(c.get("support_status")) == "blocked"
    }
    for item in report.conclusions:
        if item.claim_id and item.claim_id in blocked_ids:
            if item.strength in {"strong", "moderate"} or "优于" in item.text:
                blocking.append(
                    f"{item.conclusion_id}: blocked claim {item.claim_id} overstated"
                )

    # Evidence strength sanity: weak evidence should not back "strong" conclusions.
    weak_ids = {
        str(e.get("evidence_id"))
        for e in report.evidence_records
        if str(e.get("evidence_strength") or "") == "weak"
    }
    for item in report.conclusions:
        if item.strength == "strong" and any(
            eid in weak_ids for eid in item.evidence_ids
        ):
            blocking.append(
                f"{item.conclusion_id}: strong conclusion backed only by weak evidence"
            )

    valid = not blocking
    return {
        "valid": valid,
        "blocking_issues": blocking,
        "warnings": warnings,
        "forbidden_phrase_list": list(FORBIDDEN_PHRASES),
        "conclusion_count": len(report.conclusions),
    }
