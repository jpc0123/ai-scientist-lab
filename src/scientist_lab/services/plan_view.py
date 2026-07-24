"""Structured plan views for v2.0.4 Planning Center."""

from __future__ import annotations

from typing import Any


def _as_record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def candidate_detail(candidate: dict[str, Any]) -> dict[str, Any]:
    verification = _as_record(candidate.get("verification"))
    critic = _as_record(candidate.get("critic_review"))
    return {
        "candidate_id": candidate.get("candidate_id"),
        "title": candidate.get("title"),
        "hypothesis": candidate.get("hypothesis") or candidate.get("rationale"),
        "experiment_type": candidate.get("experiment_type"),
        "parent_node_id": candidate.get("parent_node_id"),
        "parameter_changes": candidate.get("parameter_changes")
        or candidate.get("parameters")
        or candidate.get("proposed_parameters"),
        "success_criteria": candidate.get("success_criteria"),
        "failure_criteria": candidate.get("failure_criteria"),
        "estimated_cost": candidate.get("estimated_cost"),
        "evidence_gaps_addressed": candidate.get("evidence_gaps_addressed")
        or candidate.get("resolved_evidence_gaps")
        or candidate.get("evidence_gaps"),
        "claim_restrictions": candidate.get("claim_restrictions"),
        "status": candidate.get("status"),
        "verification_valid": verification.get("valid"),
        "verification_issues": verification.get("issues") or [],
        "critic_recommendation": critic.get("recommendation"),
        "critic_summary": critic.get("summary") or critic.get("rationale"),
        "rank": candidate.get("rank"),
        "final_score": candidate.get("final_score"),
    }


def build_plan_sections(
    plan: dict[str, Any],
    *,
    budget_remaining: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = _as_record(plan.get("context_json"))
    planner_output = _as_record(plan.get("planner_output_json"))
    candidates = [
        candidate_detail(_as_record(item))
        for item in (plan.get("candidates") or [])
    ]
    ranked = sorted(
        [item for item in candidates if item.get("rank") is not None],
        key=lambda item: int(item.get("rank") or 999),
    )
    evidence_records = context.get("evidence_records") or []
    claim_matrix = _as_record(context.get("claim_support_matrix"))
    open_gaps = claim_matrix.get("open_gaps") or context.get("open_evidence_gaps") or []

    return {
        "overview": {
            "plan_id": plan.get("plan_id"),
            "project_id": plan.get("project_id"),
            "status": plan.get("status"),
            "candidate_count": len(candidates),
            "valid_candidate_count": sum(
                1 for item in candidates if item.get("verification_valid")
            ),
            "created_at": plan.get("created_at"),
            "updated_at": plan.get("updated_at"),
        },
        "context": {
            "research_goal": context.get("research_goal"),
            "current_best_node_id": context.get("current_best_node_id"),
            "node_count": len(context.get("nodes") or []),
            "evidence_count": len(evidence_records),
            "open_evidence_gaps": open_gaps,
            "remaining_budget": context.get("remaining_budget") or budget_remaining or {},
        },
        "planner": {
            "model_provider": plan.get("model_provider"),
            "model_name": plan.get("model_name"),
            "prompt_version": plan.get("prompt_version"),
            "context_sha256": plan.get("context_sha256"),
            "output_sha256": plan.get("output_sha256"),
            "reasoning_summary": planner_output.get("reasoning_summary"),
            "stop_recommended": planner_output.get("stop_recommended"),
            "stop_reason": planner_output.get("stop_reason"),
        },
        "candidates": candidates,
        "ranking": ranked,
        "budget": budget_remaining or {},
    }


def enrich_plan(plan: dict[str, Any], *, budget_remaining: dict[str, Any] | None = None) -> dict[str, Any]:
    sections = build_plan_sections(plan, budget_remaining=budget_remaining)
    ranked = sections["ranking"]
    payload = dict(plan)
    payload["sections"] = sections
    payload["ranking"] = [
        {
            "candidate_id": item.get("candidate_id"),
            "rank": item.get("rank"),
            "final_score": item.get("final_score"),
            "title": item.get("title"),
            "status": item.get("status"),
        }
        for item in ranked
    ]
    payload["budget_remaining"] = budget_remaining or sections["budget"]
    payload["approval_preview"] = {
        "candidate_count": len(sections["candidates"]),
        "ranked_count": len(ranked),
        "top_candidate_id": ranked[0]["candidate_id"] if ranked else None,
        "estimated_gpu_hours": sum(
            float(_as_record(item.get("estimated_cost")).get("gpu_hours") or 0)
            for item in sections["candidates"]
            if item.get("status") not in {"rejected"}
        ),
        "uses_real_llm": str(plan.get("model_provider") or "mock") not in {
            "mock",
            "mock-planner",
            "fake",
            "replay",
        },
        "quality_gate_passed": all(
            item.get("verification_valid") for item in sections["candidates"]
            if item.get("status") not in {"rejected"}
        ),
    }
    return payload


def enrich_tree(tree: dict[str, Any], *, budget_remaining: dict[str, Any] | None = None) -> dict[str, Any]:
    nodes = [ _as_record(item) for item in (tree.get("nodes") or []) ]
    depths = [int(item.get("depth") or 0) for item in nodes]
    payload = dict(tree)
    payload["limits"] = {
        "max_depth": tree.get("max_depth"),
        "max_nodes": tree.get("max_nodes"),
        "max_children_per_node": tree.get("max_children_per_node"),
        "current_depth": max(depths) if depths else 0,
        "current_node_count": tree.get("node_count") or len(nodes),
        "no_improvement_rounds": tree.get("no_improvement_rounds"),
        "remaining_budget": budget_remaining or {},
    }
    payload["node_summaries"] = [
        {
            "tree_node_id": item.get("tree_node_id"),
            "experiment_node_id": item.get("experiment_node_id"),
            "node_type": item.get("node_type"),
            "status": item.get("status"),
            "depth": item.get("depth"),
            "score": item.get("score"),
            "expansion_priority": item.get("expansion_priority"),
            "plan_id": item.get("plan_id"),
            "iteration_id": item.get("iteration_id"),
            "candidate_id": item.get("candidate_id"),
            "evidence_ids": item.get("evidence_ids") or [],
        }
        for item in nodes
    ]
    return payload
