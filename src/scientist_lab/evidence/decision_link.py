"""Link ExperimentDecision records to Evidence / Claim Support Matrix."""

from __future__ import annotations

from typing import Any, Literal

from scientist_lab.evidence.models import EvidenceRecord

EvidenceStrengthName = Literal["weak", "moderate", "strong"]

_STRENGTH_RANK: dict[str, int] = {"weak": 0, "moderate": 1, "strong": 2}
_RANK_STRENGTH: dict[int, EvidenceStrengthName] = {
    0: "weak",
    1: "moderate",
    2: "strong",
}


def cap_evidence_strength(
    requested: str,
    records: list[EvidenceRecord],
) -> EvidenceStrengthName:
    """Decision strength cannot exceed the weakest supporting evidence."""
    req = (requested or "weak").strip().lower()
    req_rank = _STRENGTH_RANK.get(req, 0)
    if not records:
        return _RANK_STRENGTH[req_rank]
    min_rank = min(
        _STRENGTH_RANK.get(item.evidence_strength, 0) for item in records
    )
    return _RANK_STRENGTH[min(req_rank, min_rank)]


def select_supporting_evidence(
    records: list[EvidenceRecord],
    *,
    node_ids: set[str],
    explicit_ids: list[str] | None = None,
) -> list[EvidenceRecord]:
    """Prefer explicitly listed IDs; else evidence touching decision nodes."""
    by_id = {item.evidence_id: item for item in records}
    if explicit_ids:
        missing = [eid for eid in explicit_ids if eid not in by_id]
        if missing:
            raise KeyError(f"evidence not found: {', '.join(missing)}")
        return [by_id[eid] for eid in explicit_ids]

    if not node_ids:
        return list(records)

    matched = [
        item
        for item in records
        if node_ids.intersection(item.source_node_ids or [])
    ]
    return matched if matched else list(records)


def resolve_protocol_id(
    *,
    explicit: str | None,
    selected_contract: dict[str, Any] | None,
    evidence_records: list[EvidenceRecord],
) -> str | None:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    contract = selected_contract or {}
    value = contract.get("protocol_id")
    if value:
        return str(value)
    for record in evidence_records:
        if record.protocol_id:
            return record.protocol_id
    return None
