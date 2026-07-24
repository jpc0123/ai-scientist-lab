"""Sanitize PlanningContext before sending to a real LLM (v1.4.5)."""

from __future__ import annotations

import re
from typing import Any

from scientist_lab.agents.models import PlanningContext


_ABS_PATH = re.compile(
    r"(?i)([a-z]:\\|\\\\|/Users/|/home/|/var/|/tmp/|D:\\\\|C:\\\\)[^\s\"']+"
)
_SECRETISH = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|worker[_-]?token|password)\s*[:=]\s*\S+"
)


def _scrub_text(value: str) -> str:
    text = _ABS_PATH.sub("[REDACTED_PATH]", value)
    text = _SECRETISH.sub(r"\1=[REDACTED]", text)
    return text


def _scrub_obj(value: Any) -> Any:
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, list):
        return [_scrub_obj(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            low = str(key).lower()
            if low in {
                "api_key",
                "authorization",
                "token",
                "password",
                "db_path",
                "database_path",
                "worker_token",
                "combined_log",
                "host_path",
            }:
                out[key] = "[REDACTED]"
            else:
                out[key] = _scrub_obj(item)
        return out
    return value


def sanitize_planning_context(context: PlanningContext) -> PlanningContext:
    """Return a copy safe for cloud prompts (no host paths / secrets)."""
    data = context.model_dump(mode="json")
    # Drop bulky / sensitive blobs that Planner does not need.
    data["nodes"] = [
        {
            "node_id": item.get("node_id"),
            "status": item.get("status"),
            "parameters": item.get("parameters") or {},
        }
        for item in (data.get("nodes") or [])
        if item.get("node_id")
    ]
    data["evidence_records"] = [
        {
            "evidence_id": item.get("evidence_id"),
            "claim_id": item.get("claim_id"),
            "summary": item.get("summary") or item.get("title"),
        }
        for item in (data.get("evidence_records") or [])[:20]
    ]
    # Keep round feedback (v2.1.2) but scrub nested strings/paths.
    if data.get("round_feedback_summary"):
        data["round_feedback_summary"] = _scrub_obj(data["round_feedback_summary"])
    if data.get("recent_execution_summary"):
        data["recent_execution_summary"] = _scrub_obj(data["recent_execution_summary"])
    if data.get("previous_parameter_changes"):
        data["previous_parameter_changes"] = _scrub_obj(
            data["previous_parameter_changes"]
        )
    cleaned = _scrub_obj(data)
    return PlanningContext.model_validate(cleaned)
