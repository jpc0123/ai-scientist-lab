"""Helpers to build LLMRequest from PlanningContext without changing Planner."""

from __future__ import annotations

import json
from typing import Any

from scientist_lab.agents.models import PlanningContext
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.schema_parser import PLANNER_OUTPUT_SCHEMA


def planning_context_to_planner_request(
    context: PlanningContext,
    *,
    system_prompt: str = (
        "You are a constrained experiment planner. "
        "Return JSON matching the provided schema only."
    ),
) -> LLMRequest:
    """Serialize a stable PlanningContext subset into an LLMRequest.

    Uses a fingerprint-friendly payload (no volatile timestamps) so Fake→Replay
    round-trips remain stable across service rebuilds.
    """
    budget = dict(context.remaining_budget or {})
    payload = {
        "project_id": context.project_id,
        "research_goal": context.research_goal,
        "protocol_id": context.protocol_id
        or (context.protocol or {}).get("protocol_id"),
        "current_best_node_id": context.current_best_node_id,
        "allowed_parameter_changes": sorted(
            str(x) for x in (context.allowed_parameter_changes or [])
        ),
        "tested_parameter_fingerprints": sorted(
            str(x) for x in (context.tested_parameter_fingerprints or [])
        ),
        "remaining_budget": {
            "max_new_nodes": budget.get("max_new_nodes"),
            "max_gpu_hours": budget.get("max_gpu_hours")
            or budget.get("max_total_gpu_hours"),
        },
        "node_ids": [
            str(item.get("node_id"))
            for item in (context.nodes or [])
            if item.get("node_id")
        ],
        "nodes": [
            {
                "node_id": item.get("node_id"),
                "status": item.get("status"),
                "parameters": item.get("parameters") or {},
            }
            for item in (context.nodes or [])
            if item.get("node_id")
        ],
    }
    return LLMRequest(
        purpose="planner",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ],
        response_schema=PLANNER_OUTPUT_SCHEMA,
        temperature=0.0,
        metadata={
            "project_id": context.project_id,
            "current_best_node_id": context.current_best_node_id,
            "context_sha256": context.context_sha256,
        },
    )


def assert_no_network_imports() -> None:
    """Soft guard used in tests: provider modules must not import httpx/openai."""
    import scientist_lab.llm.fake_provider as fake
    import scientist_lab.llm.replay_provider as replay
    import scientist_lab.llm.audit as audit

    for module in (fake, replay, audit):
        text = open(module.__file__, encoding="utf-8").read().lower()
        for banned in ("httpx", "openai", "anthropic", "requests.get", "urllib.request"):
            if banned in text and "forbidden" not in text:
                # Allow mentioning banned names only in comments about avoidance.
                if f"no {banned}" in text or "never" in text:
                    continue
                if banned in ("httpx", "openai", "anthropic") and (
                    f"import {banned}" in text or f"from {banned}" in text
                ):
                    raise AssertionError(
                        f"{module.__file__} must not import network client: {banned}"
                    )
