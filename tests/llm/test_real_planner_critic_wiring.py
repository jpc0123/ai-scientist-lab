"""v1.4.5: real Planner/Critic wiring via MockTransport (zero network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from scientist_lab.agents.provider_bridge import build_planner_critic
from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.llm.errors import RealProviderNotEnabledError
from scientist_lab.llm.http_transport import HttpResponse, MockTransport
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    settings = Settings(
        project_root=root,
        db_path=tmp_path / "test.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=root / "experiment_app",
    ).resolve()
    return ExperimentService(settings=settings)


def _bootstrap(service: ExperimentService) -> None:
    now = "2026-01-01T00:00:00+00:00"
    service.protocols.create_from_path(EXAMPLES / "rgbt_protocol.json")
    service.set_budget("project_rgbt_003", max_new_nodes=3, max_gpu_hours=10)
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="Real provider wiring",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    contract = json.loads(
        (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
    )
    service.repo.upsert_node(
        ExperimentNode(
            node_id="rgbt_formal_node_003",
            project_id="project_rgbt_003",
            node_type=NodeType.BASELINE,
            stage=NodeStage.DONE,
            status=NodeStatus.SUCCEEDED,
            depth=0,
            contract_json=contract,
            created_at=now,
            updated_at=now,
        )
    )


def _cfg() -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        base_url="https://api.example.com/v1",
        api_key=SecretStr("sk-test-secret-key-value"),
        model="gpt-test",
        allow_network=False,
        api_mode="chat_completions",
    )


def _chat_body(content: dict) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-real-wiring",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content, ensure_ascii=False),
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 22,
                "total_tokens": 33,
            },
        }
    )


def _planner_payload() -> dict:
    return {
        "project_id": "project_rgbt_003",
        "reasoning_summary": "MockTransport real planner output",
        "candidates": [
            {
                "candidate_id": "candidate_real_ablation_001",
                "parent_node_id": "rgbt_formal_node_003",
                "title": "Real-path ablation: RGB-only control",
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
                "priority": 0.8,
                "rationale": "Deterministic MockTransport candidate.",
                "claim_limitations": ["Offline fixture only."],
            }
        ],
        "stop_recommended": False,
        "stop_reason": None,
    }


def _critic_payload() -> dict:
    return {
        "candidate_id": "candidate_real_ablation_001",
        "scientific_validity": "valid",
        "novelty_status": "new",
        "expected_information_gain": 0.7,
        "cost_effectiveness": 0.65,
        "risk_level": "low",
        "strengths": ["Single-variable ablation."],
        "weaknesses": ["Fixture review."],
        "required_revisions": [],
        "recommendation": "accept",
    }


def test_real_provider_blocked_without_allow_network(tmp_path: Path):
    with pytest.raises(RealProviderNotEnabledError):
        build_planner_critic(
            "real",
            audit_root=tmp_path / "llm",
            allow_network=False,
            openai_config=_cfg(),
            environ={"LLM_ALLOW_NETWORK": "true"},
        )


def test_plan_next_real_fails_closed_no_silent_mock(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    planned = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        provider="real",
        allow_network=False,
        openai_config=_cfg(),
    )
    assert planned["status"] == "real_provider_failed"
    assert planned["requested_provider"] == "real"
    assert planned["actual_provider"] is None
    assert planned["fallback_used"] is False
    assert "error_type" in planned
    assert "model_provider" not in planned or planned.get("model_provider") != "mock"


def test_plan_next_and_review_via_mock_transport(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=200, body=_chat_body(_planner_payload())),
            HttpResponse(status_code=200, body=_chat_body(_critic_payload())),
        ]
    )
    planned = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        provider="real",
        allow_network=False,
        transport=transport,
        openai_config=_cfg(),
    )
    assert planned["status"] != "real_provider_failed"
    assert planned["requested_provider"] == "real"
    assert planned["fallback_used"] is False
    assert "openai-compatible" in str(planned["actual_provider"])
    assert planned["candidates"]
    assert len(transport.calls) == 1

    reviewed = service.review_plan(
        planned["plan_id"],
        provider="real",
        allow_network=False,
        transport=transport,
        openai_config=_cfg(),
    )
    assert reviewed["status"] != "real_provider_failed"
    assert reviewed["requested_provider"] == "real"
    assert reviewed["fallback_used"] is False
    assert reviewed["reviews"]
    assert len(transport.calls) == 2


def test_default_plan_next_still_mock(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    planned = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
    )
    assert planned["model_provider"] == "mock"
    assert planned["requested_provider"] in {None, "mock"} or planned.get(
        "fallback_used"
    ) is False
