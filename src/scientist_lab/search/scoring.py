"""Node score and expansion priority for finite experiment trees (v1.1.2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


STRENGTH_MAP: dict[str, float] = {
    "weak": 0.30,
    "moderate": 0.65,
    "strong": 0.90,
}

NODE_SCORE_WEIGHTS = {
    "performance": 0.40,
    "stability": 0.20,
    "efficiency": 0.15,
    "evidence": 0.15,
    "protocol": 0.10,
}

EXPANSION_WEIGHTS = {
    "node_score": 0.30,
    "information_gap": 0.30,
    "expected_gain": 0.20,
    "novelty": 0.10,
    "cost_effectiveness": 0.10,
}


class ScoreBreakdown(BaseModel):
    experiment_node_id: str
    tree_node_id: str | None = None
    node_type: str = "root"

    performance_score: float
    stability_score: float
    efficiency_score: float
    evidence_strength_score: float
    protocol_compliance_score: float
    node_score: float

    information_gap_score: float
    expected_information_gain: float
    novelty_score: float
    cost_effectiveness: float
    expansion_priority: float

    penalties: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    primary_metric: str | None = None
    seed_count: int = 0
    evidence_gap_count: int = 0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _aggregate_blob(aggregate: dict[str, Any] | None) -> dict[str, Any]:
    if not aggregate:
        return {}
    # Accept either aggregate_node payload or nested feedback["aggregate_metrics"].
    if "aggregate_metrics" in aggregate and isinstance(
        aggregate.get("aggregate_metrics"), dict
    ):
        inner = aggregate["aggregate_metrics"]
        # feedback stores full payload under aggregate_metrics key
        if "aggregate_metrics" in inner and isinstance(
            inner.get("aggregate_metrics"), dict
        ):
            return inner
        if any(
            isinstance(v, dict) and "mean" in v for v in inner.values()
        ):
            return {
                "primary_metric": aggregate.get("primary_metric")
                or inner.get("primary_metric"),
                "seed_count": aggregate.get("seed_count") or inner.get("seed_count") or 0,
                "aggregate_metrics": inner,
            }
        return aggregate
    return aggregate


def performance_score(
    aggregate: dict[str, Any] | None,
    *,
    reference_mean: float | None = None,
) -> tuple[float, str | None, list[str]]:
    notes: list[str] = []
    blob = _aggregate_blob(aggregate)
    primary = blob.get("primary_metric")
    metrics = blob.get("aggregate_metrics") or {}
    if not primary or primary not in metrics:
        # Prefer common detection / accuracy keys.
        for key in ("mAP50_95", "AP_small", "mAP50", "accuracy", "val_accuracy"):
            if key in metrics:
                primary = key
                break
    if not primary or primary not in metrics:
        notes.append("No aggregate primary metric; using neutral performance.")
        return 0.35, None, notes

    mean = _as_float((metrics.get(primary) or {}).get("mean"))
    if mean is None:
        notes.append(f"Primary metric {primary} has no mean; neutral performance.")
        return 0.35, str(primary), notes

    # Absolute score for metrics already in [0, 1].
    if 0.0 <= mean <= 1.0:
        score = mean
    else:
        # Soft saturation for unbounded metrics.
        score = mean / (abs(mean) + 1.0)

    if reference_mean is not None and abs(reference_mean) > 1e-12:
        rel = (mean - reference_mean) / abs(reference_mean)
        score = _clamp(0.5 + 0.5 * _clamp(rel, -1.0, 1.0))
        notes.append(
            f"Performance relative to reference mean={reference_mean:.4f}."
        )
    return _clamp(score), str(primary), notes


def stability_score(aggregate: dict[str, Any] | None) -> tuple[float, int, list[str]]:
    notes: list[str] = []
    blob = _aggregate_blob(aggregate)
    seed_count = int(blob.get("seed_count") or 0)
    primary = blob.get("primary_metric")
    metrics = blob.get("aggregate_metrics") or {}
    if not primary or primary not in metrics:
        for key in ("mAP50_95", "AP_small", "mAP50", "accuracy"):
            if key in metrics:
                primary = key
                break

    if seed_count <= 0:
        notes.append("No completed seeds; low stability.")
        return 0.25, 0, notes
    if seed_count == 1:
        notes.append("Single seed; stability capped.")
        return 0.40, 1, notes
    if seed_count == 2:
        base = 0.55
    else:
        base = 0.70

    std = _as_float((metrics.get(primary) or {}).get("std")) if primary else None
    mean = _as_float((metrics.get(primary) or {}).get("mean")) if primary else None
    if std is None or mean is None:
        notes.append("Missing std/mean; seed-count based stability only.")
        return _clamp(base), seed_count, notes

    # Coefficient of variation style penalty.
    denom = max(abs(mean), 1e-6)
    cv = abs(std) / denom
    # cv=0 → +0.25, cv>=0.5 → -0.35
    adj = 0.25 - min(0.60, cv) * 1.2
    score = _clamp(base + adj)
    if cv > 0.25:
        notes.append(f"High variance (cv={cv:.3f}).")
    return score, seed_count, notes


def efficiency_score(aggregate: dict[str, Any] | None) -> tuple[float, list[str]]:
    notes: list[str] = []
    blob = _aggregate_blob(aggregate)
    metrics = blob.get("aggregate_metrics") or {}
    duration = _as_float((metrics.get("duration_seconds") or {}).get("mean"))
    gpu_mem = _as_float((metrics.get("peak_gpu_memory_mb") or {}).get("mean"))

    if duration is None and gpu_mem is None:
        notes.append("No resource metrics; neutral efficiency.")
        return 0.50, notes

    parts: list[float] = []
    if duration is not None:
        # 60s → ~0.83, 300s → 0.5, 900s → ~0.25
        parts.append(1.0 / (1.0 + duration / 300.0))
    if gpu_mem is not None:
        parts.append(1.0 / (1.0 + gpu_mem / 4096.0))
    return _clamp(sum(parts) / len(parts)), notes


def evidence_strength_score(
    evidence_records: list[dict[str, Any]] | None,
    *,
    experiment_node_id: str | None = None,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    records = list(evidence_records or [])
    if experiment_node_id:
        filtered = [
            r
            for r in records
            if experiment_node_id
            in {
                r.get("source_node_id"),
                r.get("source_baseline_node_id"),
                r.get("source_candidate_node_id"),
                r.get("node_id_a"),
                r.get("node_id_b"),
            }
            or experiment_node_id in (r.get("related_node_ids") or [])
        ]
        if filtered:
            records = filtered

    if not records:
        notes.append("No evidence records; weak evidence strength.")
        return 0.25, notes

    values: list[float] = []
    for record in records:
        strength = str(
            record.get("evidence_strength")
            or record.get("scientific_evidence_level")
            or "weak"
        ).lower()
        values.append(STRENGTH_MAP.get(strength, 0.30))
    score = sum(values) / len(values)
    notes.append(f"Evidence strength from {len(values)} record(s).")
    return _clamp(score), notes


def protocol_compliance_score(
    *,
    has_protocol: bool,
    protocol_valid: bool = True,
    contract_protocol_match: bool = True,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    if not has_protocol:
        notes.append("No protocol linked.")
        return 0.20, notes
    if not protocol_valid:
        notes.append("Protocol verification failed.")
        return 0.25, notes
    if not contract_protocol_match:
        notes.append("Contract protocol_id mismatch.")
        return 0.40, notes
    return 1.0, notes


def information_gap_score(
    claim_matrix: dict[str, Any] | None,
    evidence_records: list[dict[str, Any]] | None = None,
) -> tuple[float, int, list[str]]:
    notes: list[str] = []
    gaps: list[str] = []
    matrix = dict(claim_matrix or {})
    for claim in matrix.get("claims") or []:
        status = str(claim.get("support_status") or "")
        if status in {"partially_supported", "unsupported", "blocked"}:
            text = str(
                claim.get("claim")
                or claim.get("claim_text")
                or claim.get("reason")
                or status
            )
            gaps.append(f"{status}: {text}")
    for record in evidence_records or []:
        for limitation in record.get("limitations") or []:
            text = str(limitation)
            if text:
                gaps.append(text)

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for item in gaps:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)

    count = len(unique)
    # More open gaps → higher value of exploring further.
    score = _clamp(count / 5.0)
    if count == 0:
        notes.append("No open evidence gaps.")
        score = 0.15
    else:
        notes.append(f"{count} evidence gap(s) open.")
    return score, count, notes


def expected_information_gain(
    *,
    node_type: str,
    information_gap: float,
    stability: float,
    status: str,
) -> float:
    if status in {"pruned", "failed", "stopped"}:
        return 0.05
    type_bonus = {
        "root": 0.55,
        "improve": 0.60,
        "ablation": 0.85,
        "replication": 0.45,
        "debug": 0.35,
        "efficiency": 0.55,
    }.get(node_type, 0.50)
    # Unstable nodes gain from replication-oriented expansion.
    instability_boost = (1.0 - stability) * 0.15
    return _clamp(0.45 * type_bonus + 0.45 * information_gap + instability_boost)


def novelty_score(
    *,
    duplicate_fingerprint: bool = False,
    similar_node_count: int = 0,
) -> float:
    if duplicate_fingerprint:
        return 0.10
    if similar_node_count <= 0:
        return 0.80
    if similar_node_count == 1:
        return 0.55
    return _clamp(0.80 - 0.15 * similar_node_count)


def cost_effectiveness(
    remaining_budget: dict[str, Any] | None,
    *,
    estimated_gpu_hours: float = 1.0,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    rem = dict(remaining_budget or {})
    nodes_left = rem.get("max_new_nodes")
    gpu_left = rem.get("max_total_gpu_hours")
    if nodes_left is None and gpu_left is None:
        notes.append("No budget remaining info; neutral cost effectiveness.")
        return 0.55, notes

    node_score = 1.0
    if isinstance(nodes_left, (int, float)):
        if nodes_left <= 0:
            notes.append("No remaining node budget.")
            return 0.05, notes
        node_score = _clamp(float(nodes_left) / 3.0)

    gpu_score = 1.0
    if isinstance(gpu_left, (int, float)):
        if float(gpu_left) < float(estimated_gpu_hours):
            notes.append("Insufficient remaining GPU hours.")
            return 0.10, notes
        gpu_score = _clamp(float(gpu_left) / max(float(estimated_gpu_hours) * 3.0, 1e-6))

    return _clamp(0.5 * node_score + 0.5 * gpu_score), notes


def compute_scores(
    *,
    experiment_node_id: str,
    tree_node_id: str | None = None,
    node_type: str = "root",
    status: str = "created",
    aggregate: dict[str, Any] | None = None,
    evidence_records: list[dict[str, Any]] | None = None,
    claim_matrix: dict[str, Any] | None = None,
    remaining_budget: dict[str, Any] | None = None,
    has_protocol: bool = True,
    protocol_valid: bool = True,
    contract_protocol_match: bool = True,
    reference_mean: float | None = None,
    duplicate_fingerprint: bool = False,
    similar_node_count: int = 0,
    parent_failed_streak: int = 0,
) -> ScoreBreakdown:
    notes: list[str] = []
    penalties: list[str] = []

    perf, primary, n1 = performance_score(aggregate, reference_mean=reference_mean)
    notes.extend(n1)
    stab, seed_count, n2 = stability_score(aggregate)
    notes.extend(n2)
    eff, n3 = efficiency_score(aggregate)
    notes.extend(n3)
    evid, n4 = evidence_strength_score(
        evidence_records, experiment_node_id=experiment_node_id
    )
    notes.extend(n4)
    proto, n5 = protocol_compliance_score(
        has_protocol=has_protocol,
        protocol_valid=protocol_valid,
        contract_protocol_match=contract_protocol_match,
    )
    notes.extend(n5)

    node_score = _clamp(
        NODE_SCORE_WEIGHTS["performance"] * perf
        + NODE_SCORE_WEIGHTS["stability"] * stab
        + NODE_SCORE_WEIGHTS["efficiency"] * eff
        + NODE_SCORE_WEIGHTS["evidence"] * evid
        + NODE_SCORE_WEIGHTS["protocol"] * proto
    )

    info_gap, gap_count, n6 = information_gap_score(claim_matrix, evidence_records)
    notes.extend(n6)
    gain = expected_information_gain(
        node_type=node_type,
        information_gap=info_gap,
        stability=stab,
        status=status,
    )
    novelty = novelty_score(
        duplicate_fingerprint=duplicate_fingerprint,
        similar_node_count=similar_node_count,
    )
    cost, n7 = cost_effectiveness(remaining_budget)
    notes.extend(n7)

    expansion = (
        EXPANSION_WEIGHTS["node_score"] * node_score
        + EXPANSION_WEIGHTS["information_gap"] * info_gap
        + EXPANSION_WEIGHTS["expected_gain"] * gain
        + EXPANSION_WEIGHTS["novelty"] * novelty
        + EXPANSION_WEIGHTS["cost_effectiveness"] * cost
    )

    # Explicit penalties (v1.1.2).
    rem = dict(remaining_budget or {})
    nodes_left = rem.get("max_new_nodes")
    if isinstance(nodes_left, (int, float)) and float(nodes_left) <= 1:
        expansion -= 0.15
        penalties.append("Near node budget limit (-0.15).")
    if stab < 0.40:
        expansion -= 0.10
        penalties.append("Highly unstable results (-0.10).")
    if duplicate_fingerprint:
        expansion -= 0.20
        penalties.append("Duplicate configuration (-0.20).")
    if parent_failed_streak >= 2:
        expansion -= 0.10
        penalties.append("Parent failed streak >= 2 (-0.10).")
    if gap_count == 0 and status in {"evaluated", "selected"}:
        expansion -= 0.05
        penalties.append("No clear evidence goal left (-0.05).")

    expansion = _clamp(expansion)

    return ScoreBreakdown(
        experiment_node_id=experiment_node_id,
        tree_node_id=tree_node_id,
        node_type=node_type,
        performance_score=round(perf, 4),
        stability_score=round(stab, 4),
        efficiency_score=round(eff, 4),
        evidence_strength_score=round(evid, 4),
        protocol_compliance_score=round(proto, 4),
        node_score=round(node_score, 4),
        information_gap_score=round(info_gap, 4),
        expected_information_gain=round(gain, 4),
        novelty_score=round(novelty, 4),
        cost_effectiveness=round(cost, 4),
        expansion_priority=round(expansion, 4),
        penalties=penalties,
        notes=notes,
        primary_metric=primary,
        seed_count=seed_count,
        evidence_gap_count=gap_count,
    )
