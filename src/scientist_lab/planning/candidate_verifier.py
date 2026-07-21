"""Fingerprint helpers and CandidateVerifier (rule layer, no LLM)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from scientist_lab.agents.models import (
    BLOCKED_PARAMETER_KEYS,
    CandidateExperiment,
    CandidateVerification,
    PlanningContext,
)


def parameter_fingerprint(changes: dict[str, Any]) -> str:
    payload = json.dumps(changes or {}, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def context_sha256(context: PlanningContext | dict[str, Any]) -> str:
    if isinstance(context, PlanningContext):
        data = context.model_dump(mode="json")
    else:
        data = dict(context)
    data.pop("context_sha256", None)
    payload = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def output_sha256(output: dict[str, Any]) -> str:
    payload = json.dumps(output or {}, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CandidateVerifier:
    """Reject illegal / unsafe / duplicate / over-budget candidates."""

    def verify(
        self,
        candidate: CandidateExperiment,
        context: PlanningContext,
        *,
        known_node_ids: set[str] | None = None,
        known_evidence_ids: set[str] | None = None,
    ) -> CandidateVerification:
        blocking: list[str] = []
        warnings: list[str] = []

        node_ids = known_node_ids or {
            str(item.get("node_id"))
            for item in context.nodes
            if item.get("node_id")
        }
        evidence_ids = known_evidence_ids or {
            str(item.get("evidence_id"))
            for item in context.evidence_records
            if item.get("evidence_id")
        }

        if candidate.parent_node_id not in node_ids:
            blocking.append(f"parent_node_id not found: {candidate.parent_node_id}")

        changes = dict(candidate.parameter_changes or {})
        if not changes:
            blocking.append("parameter_changes must be non-empty")

        allowed = set(context.allowed_parameter_changes or [])
        protocol = dict(context.protocol or {})
        if not allowed:
            allowed = set(protocol.get("allowed_variables") or [])

        for key, value in changes.items():
            if key in BLOCKED_PARAMETER_KEYS:
                blocking.append(f"blocked safety field in parameter_changes: {key}")
            elif key not in allowed:
                blocking.append(
                    f"parameter {key} is not in allowed_parameter_changes"
                )

        from scientist_lab.agents.tunable import (
            tunables_for_protocol,
            validate_tunable_value,
        )

        tunables = tunables_for_protocol(protocol)
        for key, value in changes.items():
            issue = validate_tunable_value(key, value, tunables=tunables)
            if issue:
                blocking.append(issue)

        fixed = dict(protocol.get("fixed_parameters") or {})
        for key, value in changes.items():
            if key in fixed and fixed[key] != value and key not in allowed:
                blocking.append(
                    f"attempts to override fixed parameter {key}={fixed[key]!r}"
                )

        # Protocol claim exaggeration heuristics
        text_blob = " ".join(
            [
                candidate.hypothesis,
                candidate.rationale,
                " ".join(candidate.claim_limitations or []),
            ]
        ).lower()
        for phrase in (
            "state-of-the-art",
            "sota",
            "proven",
            "verified scientific",
            "full rgbt-tiny",
        ):
            if phrase in text_blob and "claim_limitations" not in phrase:
                if phrase in (candidate.hypothesis + candidate.rationale).lower():
                    warnings.append(
                        f"possible claim exaggeration language: {phrase!r}"
                    )

        for gap in candidate.evidence_gap_addressed or []:
            # Gaps are free text; if they look like evidence ids, check existence.
            if gap.startswith("evidence_") and gap not in evidence_ids:
                blocking.append(f"evidence gap references missing evidence_id: {gap}")

        fingerprint = parameter_fingerprint(changes) if changes else None
        if fingerprint and fingerprint in set(context.tested_parameter_fingerprints or []):
            blocking.append(
                f"duplicate parameter fingerprint already tested: {fingerprint}"
            )

        budget = dict(context.remaining_budget or {})
        max_nodes = budget.get("max_new_nodes")
        if isinstance(max_nodes, int) and max_nodes <= 0:
            blocking.append("remaining budget max_new_nodes is 0")

        estimated = dict(candidate.estimated_cost or {})
        gpu_hours = estimated.get("gpu_hours")
        remaining_gpu = budget.get("max_total_gpu_hours")
        if (
            isinstance(gpu_hours, (int, float))
            and isinstance(remaining_gpu, (int, float))
            and float(gpu_hours) > float(remaining_gpu)
        ):
            blocking.append(
                f"estimated gpu_hours {gpu_hours} exceeds remaining {remaining_gpu}"
            )

        if not candidate.evidence_gap_addressed:
            warnings.append("candidate does not address an explicit evidence gap")

        if len(changes) > 2:
            warnings.append(
                "multiple parameters changed; risk of confounding factors"
            )

        return CandidateVerification(
            candidate_id=candidate.candidate_id,
            valid=len(blocking) == 0,
            blocking_issues=blocking,
            warnings=warnings,
            parameter_fingerprint=fingerprint,
            estimated_cost=estimated,
        )

    def verify_all(
        self,
        candidates: list[CandidateExperiment],
        context: PlanningContext,
    ) -> list[CandidateVerification]:
        return [self.verify(item, context) for item in candidates]
