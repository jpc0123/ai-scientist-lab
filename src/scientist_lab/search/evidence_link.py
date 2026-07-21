"""Link evaluated TreeNodes to EvidenceRecords / Claim Matrix (v1.1.8)."""

from __future__ import annotations

from typing import Any


def extract_open_gaps(
    claim_matrix: dict[str, Any] | None,
    evidence_records: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Collect open evidence / claim gaps (stable, de-duplicated order)."""
    gaps: list[str] = []
    matrix = dict(claim_matrix or {})
    for claim in matrix.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        status = str(claim.get("support_status") or "")
        if status not in {"partially_supported", "unsupported", "blocked"}:
            continue
        text = str(
            claim.get("claim_text")
            or claim.get("claim")
            or claim.get("reason")
            or status
        ).strip()
        if text:
            gaps.append(f"{status}: {text}")

    for record in evidence_records or []:
        if not isinstance(record, dict):
            continue
        for limitation in record.get("limitations") or []:
            text = str(limitation).strip()
            if text:
                gaps.append(text)

    seen: set[str] = set()
    unique: list[str] = []
    for item in gaps:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def diff_gaps(
    before: list[str] | None,
    after: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (resolved_gaps, new_gaps) comparing before → after snapshots."""
    before_set = set(before or [])
    after_set = set(after or [])
    resolved = [g for g in (before or []) if g not in after_set]
    new = [g for g in (after or []) if g not in before_set]
    return resolved, new


def collect_related_evidence_ids(
    evidence_records: list[dict[str, Any]] | None,
    *,
    experiment_node_ids: list[str],
) -> list[str]:
    """Pick evidence whose source_node_ids intersect the given experiment nodes."""
    wanted = {str(x) for x in experiment_node_ids if x}
    if not wanted:
        return []
    ids: list[str] = []
    for record in evidence_records or []:
        if not isinstance(record, dict):
            continue
        sources = {
            str(x) for x in (record.get("source_node_ids") or []) if x
        }
        if not sources.intersection(wanted):
            continue
        eid = str(record.get("evidence_id") or "").strip()
        if eid and eid not in ids:
            ids.append(eid)
    return ids


def evidence_link_payload(
    *,
    tree_node_id: str,
    experiment_node_id: str,
    evidence_ids: list[str],
    resolved_evidence_gaps: list[str],
    new_evidence_gaps: list[str],
    claim_matrix_path: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tree_node_id": tree_node_id,
        "experiment_node_id": experiment_node_id,
        "evidence_ids": list(evidence_ids),
        "resolved_evidence_gaps": list(resolved_evidence_gaps),
        "new_evidence_gaps": list(new_evidence_gaps),
        "claim_matrix_path": claim_matrix_path,
        "notes": list(notes or []),
    }
