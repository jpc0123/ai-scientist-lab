"""Deterministic FakeProvider — no network, fixed structured outputs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from scientist_lab.domain.models import new_id
from scientist_lab.llm.audit import build_usage_from_messages
from scientist_lab.llm.models import LLMRequest, LLMResponse
from scientist_lab.llm.provider import BaseLLMProvider, request_fingerprint
from scientist_lab.llm.schema_parser import (
    CRITIC_REVIEW_SCHEMA,
    PLANNER_OUTPUT_SCHEMA,
    parse_and_validate,
)


def _project_id_from_request(request: LLMRequest) -> str:
    meta = dict(request.metadata or {})
    if meta.get("project_id"):
        return str(meta["project_id"])
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, dict) and content.get("project_id"):
            return str(content["project_id"])
        if isinstance(content, str) and "project_id" in content:
            try:
                blob = json.loads(content)
                if isinstance(blob, dict) and blob.get("project_id"):
                    return str(blob["project_id"])
            except Exception:  # noqa: BLE001
                pass
    return "project_unknown"


def _parent_from_request(request: LLMRequest) -> str:
    meta = dict(request.metadata or {})
    if meta.get("current_best_node_id"):
        return str(meta["current_best_node_id"])
    for message in request.messages:
        content = message.get("content")
        blob: dict[str, Any] | None = None
        if isinstance(content, dict):
            blob = content
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    blob = parsed
            except Exception:  # noqa: BLE001
                blob = None
        if not blob:
            continue
        if blob.get("current_best_node_id"):
            return str(blob["current_best_node_id"])
        nodes = blob.get("nodes") or []
        if nodes and isinstance(nodes[0], dict) and nodes[0].get("node_id"):
            return str(nodes[0]["node_id"])
    return "node_unknown"


def build_fake_planner_payload(request: LLMRequest) -> dict[str, Any]:
    project_id = _project_id_from_request(request)
    parent = _parent_from_request(request)
    return {
        "project_id": project_id,
        "reasoning_summary": (
            "FakeProvider fixed planner output for offline replay testing."
        ),
        "candidates": [
            {
                "candidate_id": "candidate_fake_ablation_001",
                "parent_node_id": parent,
                "title": "Fake ablation: RGB-only control",
                "hypothesis": (
                    "Removing fusion under a matched protocol should reduce AP_small "
                    "if fusion contributes."
                ),
                "experiment_type": "ablation",
                "parameter_changes": {
                    "input_mode": "rgb",
                    "fusion_method": "none",
                },
                "expected_outcomes": [
                    {
                        "metric": "AP_small",
                        "direction": "decrease",
                        "rationale": "Ablating fusion should hurt small-object AP.",
                    }
                ],
                "evidence_gap_addressed": [
                    "Missing controlled ablation for fusion contribution."
                ],
                "priority": 0.84,
                "rationale": "Deterministic fake candidate for provider-layer tests.",
                "claim_limitations": [
                    "FakeProvider output is not scientific evidence."
                ],
            }
        ],
        "stop_recommended": False,
        "stop_reason": None,
    }


def build_fake_critic_payload(request: LLMRequest) -> dict[str, Any]:
    candidate_id = str(
        (request.metadata or {}).get("candidate_id") or "candidate_fake_ablation_001"
    )
    return {
        "candidate_id": candidate_id,
        "scientific_validity": "valid",
        "novelty_status": "new",
        "expected_information_gain": 0.7,
        "cost_effectiveness": 0.65,
        "risk_level": "low",
        "strengths": ["Single-variable ablation under protocol constraints."],
        "weaknesses": ["FakeProvider review only; not a real scientific critique."],
        "required_revisions": [],
        "recommendation": "accept",
    }


class FakeProvider(BaseLLMProvider):
    """Returns deterministic structured JSON. Never opens network sockets."""

    def __init__(self, *, model: str = "fake-llm-v1") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        if request.purpose == "planner":
            payload = build_fake_planner_payload(request)
            schema = request.response_schema or PLANNER_OUTPUT_SCHEMA
        elif request.purpose == "critic":
            payload = build_fake_critic_payload(request)
            schema = request.response_schema or CRITIC_REVIEW_SCHEMA
        else:
            payload = {
                "ok": True,
                "echo_purpose": request.purpose,
                "message": "FakeProvider generic response.",
            }
            schema = request.response_schema

        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        parsed, errors = parse_and_validate(content, schema)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LLMResponse(
            request_id=new_id("llmreq"),
            content=content,
            parsed_json=parsed,
            usage=build_usage_from_messages(request.messages, content),
            latency_ms=elapsed_ms,
            provider=self.name,
            model=self.model,
            request_fingerprint=request_fingerprint(request),
            schema_valid=not errors,
            schema_errors=errors,
            created_at=datetime.now(timezone.utc).replace(microsecond=0),
        )
