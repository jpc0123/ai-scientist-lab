"""Build filtered PlanningContext from project state (no host secrets)."""

from __future__ import annotations

import json
from typing import Any, Callable

from scientist_lab.agents.models import (
    DEFAULT_BLOCKED_ACTIONS,
    PlanningContext,
    RoundFeedbackSummary,
)
from scientist_lab.planning.candidate_verifier import context_sha256, parameter_fingerprint


SENSITIVE_KEYS = frozenset(
    {
        "host_path",
        "absolute_path",
        "local_path",
        "auth_token",
        "password",
        "secret",
        "api_key",
        "endpoint_token",
    }
)


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in SENSITIVE_KEYS:
                continue
            if key_l.endswith("_path") and key_l not in {
                "matrix_path",
                "comparison_path",
                "evidence_path",
                "aggregate_path",
            }:
                # Drop host filesystem paths from planning context.
                if isinstance(item, str) and _looks_absolute(item):
                    continue
            cleaned[key] = _scrub(item)
        return cleaned
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, str) and _looks_absolute(value):
        return "<redacted_path>"
    return value


def _looks_absolute(value: str) -> bool:
    text = value or ""
    if len(text) > 2 and text[1:3] == ":\\":
        return True
    if text.startswith("/home/") or text.startswith("/Users/") or text.startswith("/var/"):
        return True
    return False


def _node_summary(node: Any) -> dict[str, Any]:
    contract = dict(getattr(node, "contract_json", None) or {})
    params = dict(contract.get("parameters") or {})
    return {
        "node_id": getattr(node, "node_id", None),
        "project_id": getattr(node, "project_id", None),
        "status": str(getattr(node, "status", "")),
        "hypothesis": getattr(node, "hypothesis", None),
        "parameters": params,
        "protocol_id": contract.get("protocol_id"),
        "execution_mode": contract.get("execution_mode"),
        "task_config": {
            key: value
            for key, value in dict(contract.get("task_config") or {}).items()
            if key
            in {
                "claim_level",
                "evaluation_scope",
                "implementation",
                "primary_metric",
            }
        },
        "parameter_fingerprint": parameter_fingerprint(params),
    }


def build_planning_context(
    *,
    project_id: str,
    research_goal: str,
    protocol: dict[str, Any] | None,
    nodes: list[Any],
    evidence_records: list[dict[str, Any]] | None = None,
    claim_support_matrix: dict[str, Any] | None = None,
    comparisons: list[dict[str, Any]] | None = None,
    current_best_node_id: str | None = None,
    remaining_budget: dict[str, Any] | None = None,
    extra_blocked_actions: list[str] | None = None,
    loop_session_id: str | None = None,
    loop_round_number: int | None = None,
    round_feedback_summary: RoundFeedbackSummary | dict[str, Any] | None = None,
    recent_execution_summary: dict[str, Any] | None = None,
    previous_planner_hypothesis: str | None = None,
    previous_parameter_changes: dict[str, Any] | None = None,
    patch_feedback_records: list[dict[str, Any]] | None = None,
) -> PlanningContext:
    protocol_payload = _scrub(dict(protocol or {}))
    node_payloads = [_scrub(_node_summary(node)) for node in nodes]
    fingerprints = sorted(
        {
            str(item.get("parameter_fingerprint"))
            for item in node_payloads
            if item.get("parameter_fingerprint")
        }
    )

    allowed = list(protocol_payload.get("allowed_variables") or [])
    blocked = list(DEFAULT_BLOCKED_ACTIONS)
    if extra_blocked_actions:
        blocked.extend(extra_blocked_actions)

    budget = dict(remaining_budget or {})
    budget.setdefault("max_new_nodes", 3)
    budget.setdefault("max_total_gpu_hours", 12)
    budget.setdefault("max_seeds_per_node", 3)

    # Prefer fusion formal node when present and no explicit best.
    best = current_best_node_id
    if best is None:
        for preferred in (
            "rgbt_formal_node_003",
            "rgbt_fast_node_003",
        ):
            if any(item.get("node_id") == preferred for item in node_payloads):
                best = preferred
                break
        if best is None and node_payloads:
            best = str(node_payloads[0].get("node_id"))

    feedback: RoundFeedbackSummary | None = None
    if round_feedback_summary is not None:
        if isinstance(round_feedback_summary, RoundFeedbackSummary):
            feedback = round_feedback_summary
        else:
            feedback = RoundFeedbackSummary.model_validate(
                _scrub(dict(round_feedback_summary))
            )

    context = PlanningContext(
        project_id=project_id,
        research_goal=research_goal or "Improve RGB-T detection under a fixed protocol.",
        protocol=protocol_payload,
        current_best_node_id=best,
        nodes=node_payloads,
        comparisons=_scrub(list(comparisons or [])),
        evidence_records=_scrub(list(evidence_records or [])),
        claim_support_matrix=_scrub(dict(claim_support_matrix or {})),
        tested_parameter_fingerprints=fingerprints,
        remaining_budget=budget,
        allowed_parameter_changes=allowed,
        blocked_actions=blocked,
        protocol_id=protocol_payload.get("protocol_id"),
        loop_session_id=loop_session_id,
        loop_round_number=loop_round_number,
        round_feedback_summary=feedback,
        recent_execution_summary=_scrub(dict(recent_execution_summary or {})),
        previous_planner_hypothesis=previous_planner_hypothesis,
        previous_parameter_changes=_scrub(dict(previous_parameter_changes or {})),
        patch_feedback_records=_scrub(list(patch_feedback_records or [])),
    )
    digest = context_sha256(context)
    return context.model_copy(update={"context_sha256": digest})


def load_claim_matrix_safe(
    loader: Callable[[str], dict[str, Any]],
    project_id: str,
) -> dict[str, Any]:
    try:
        return dict(loader(project_id) or {})
    except Exception:  # noqa: BLE001
        return {}


def dumps_context(context: PlanningContext) -> str:
    return json.dumps(context.model_dump(mode="json"), ensure_ascii=False, indent=2)
