"""v2.1.3 real-only Planner/Critic gates (MockTransport; zero live network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from scientist_lab.domain import NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExperimentNode, new_id, utc_now_iso
from scientist_lab.llm.http_transport import HttpResponse, MockTransport
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.research_loop.errors import RealLoopProviderMismatchError
from scientist_lab.research_loop.provider_gate import (
    assert_no_fallback,
    assert_real_only_provider,
    check_profile_for_real_loop,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _service(tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "lab.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


@pytest.fixture()
def service(tmp_path: Path) -> ExperimentService:
    return _service(tmp_path)


def _cfg() -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        base_url="https://api.example.com/v1",
        api_key=SecretStr("sk-test-secret-key-value"),
        model="gpt-test",
        allow_network=False,
        api_mode="chat_completions",
    )


def _passing_scorecard(*, planner_prompt: str, critic_prompt: str) -> dict:
    return {
        "status": "completed",
        "suite_version": "eval_suite_v1",
        "planner": {
            "schema_valid_rate": 1.0,
            "protocol_compliance_rate": 1.0,
            "duplicate_candidate_rate": 0.0,
            "evidence_gap_relevance_rate": 0.95,
            "case_count": 10,
        },
        "critic": {"case_count": 5},
        "safety": {
            "pass": True,
            "pass_rate": 1.0,
            "violation_count": 0,
            "case_count": 5,
        },
        "operations": {"repair_rate": 0.0, "average_latency_ms": 50},
        "metadata": {
            "planner_prompt_version": planner_prompt,
            "critic_prompt_version": critic_prompt,
        },
    }


def _chat_body(content: dict) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-real-loop",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content, ensure_ascii=False),
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
    )


def _planner_payload(project_id: str, parent_id: str) -> dict:
    return {
        "project_id": project_id,
        "reasoning_summary": "Increase capacity slightly under Digits protocol.",
        "candidates": [
            {
                "candidate_id": "candidate_digits_hu112",
                "parent_node_id": parent_id,
                "title": "hidden_units=112",
                "hypothesis": "112 units may improve accuracy vs 64.",
                "experiment_type": "improve",
                "parameter_changes": {"hidden_units": 112},
                "expected_outcomes": [
                    {
                        "metric": "accuracy",
                        "direction": "increase",
                        "rationale": "Slightly larger MLP capacity.",
                    }
                ],
                "evidence_gap_addressed": ["Need mid-range hidden_units point."],
                "priority": 0.7,
                "rationale": "Interpolate between 64 and 128.",
                "claim_limitations": ["Exploratory Digits only."],
            }
        ],
        "stop_recommended": False,
        "stop_reason": None,
    }


def _critic_payload() -> dict:
    return {
        "candidate_id": "candidate_digits_hu112",
        "scientific_validity": "valid",
        "novelty_status": "new",
        "expected_information_gain": 0.6,
        "cost_effectiveness": 0.7,
        "risk_level": "low",
        "strengths": ["Single allowed variable."],
        "weaknesses": ["Fixture critic."],
        "required_revisions": [],
        "recommendation": "accept",
    }


def _seed_session(service: ExperimentService) -> dict[str, str]:
    project = service.create_project(
        title="Digits Real Only",
        research_question="real_only planning",
        research_goal="v2.1.3",
        task_type="general_ml",
        project_id="project_digits_real_only",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_openai_real_only",
        provider="openai-compatible",
        model="gpt-test",
        planner_prompt_version="planner_v1",
        critic_prompt_version="critic_v1",
        enabled=True,
    )
    service.llm_evals.upsert_profile(profile)
    service.llm_evals.save_evaluation(
        evaluation_id="eval_real_only_001",
        profile_id=profile.profile_id,
        suite_version="eval_suite_v1",
        status="completed",
        result=_passing_scorecard(
            planner_prompt="planner_v1", critic_prompt="critic_v1"
        ),
        report_path=None,
        case_rows=[],
    )
    service.set_budget(project["project_id"], max_new_nodes=5, max_gpu_hours=2.0)

    now = utc_now_iso()
    parent = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        node_type=NodeType.BASELINE,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=64",
        contract_json={
            "parameters": {"hidden_units": 64},
            "protocol_id": protocol.protocol_id,
        },
        created_at=now,
        updated_at=now,
    )
    service.repo.upsert_node(parent)

    created = service.real_loop_create(
        project["project_id"],
        profile_id=profile.profile_id,
        protocol_id=protocol.protocol_id,
        baseline_node_ids=[parent.node_id],
    )
    service._real_loop_service().mark_baseline_ready(created["session_id"])
    return {
        "session_id": created["session_id"],
        "project_id": project["project_id"],
        "profile_id": profile.profile_id,
        "parent_id": parent.node_id,
        "protocol_id": protocol.protocol_id,
    }


def test_assert_real_only_rejects_mock_and_replay():
    with pytest.raises(RealLoopProviderMismatchError):
        assert_real_only_provider("mock", real_only=True)
    with pytest.raises(RealLoopProviderMismatchError):
        assert_real_only_provider("replay", real_only=True)
    assert assert_real_only_provider("openai-compatible") == "openai-compatible"


def test_assert_no_fallback_detects_silent_mock():
    with pytest.raises(Exception):
        assert_no_fallback(
            requested_provider="openai-compatible",
            actual_provider="mock",
            fallback_allowed=False,
            fallback_used=False,
        )


def test_check_profile_offline_requires_quality_gate(service: ExperimentService):
    ids = _seed_session(service)
    report = service.real_loop_check_profile(ids["profile_id"])
    assert report["overall"] == "ok"
    assert report["network_used"] is False
    assert any(c["name"] == "quality_gate" and c["ok"] for c in report["checks"])

    # Mock profile must fail provider_type.
    service.llm_evals.upsert_profile(
        LLMModelProfile(
            profile_id="profile_mock_bad",
            provider="mock",
            model="mock",
            enabled=True,
        )
    )
    bad = check_profile_for_real_loop(
        profile=service.llm_evals.get_profile("profile_mock_bad"),
        evaluation=None,
        require_quality_gate=False,
    )
    assert bad["overall"] == "failed"


def test_plan_round_rejects_mock_without_candidates(service: ExperimentService):
    ids = _seed_session(service)
    result = service.real_loop_plan(
        ids["session_id"],
        round_number=1,
        provider="mock",
        allow_network=False,
    )
    assert result["ok"] is False
    assert result["candidates"] == []
    assert result["fallback_used"] is False
    assert result["provider_audit"]["fallback_allowed"] is False
    shown = service.real_loop_show(ids["session_id"])
    assert shown["status"] == "provider_failed"


def test_plan_and_review_via_mock_transport_real_only(service: ExperimentService):
    ids = _seed_session(service)
    transport = MockTransport(
        responses=[
            HttpResponse(
                status_code=200,
                body=_chat_body(
                    _planner_payload(ids["project_id"], ids["parent_id"])
                ),
            ),
            HttpResponse(status_code=200, body=_chat_body(_critic_payload())),
        ]
    )
    planned = service.real_loop_plan(
        ids["session_id"],
        round_number=1,
        provider="openai-compatible",
        allow_network=False,
        transport=transport,
        openai_config=_cfg(),
    )
    assert planned.get("ok", True) is not False
    assert planned["fallback_used"] is False
    assert planned["provider_audit"]["fallback_used"] is False
    assert planned["provider_audit"]["requested_provider"] in {
        "openai-compatible",
        "real",
    }
    assert planned["candidates"]
    assert planned["plan_id"]
    shown = service.real_loop_show(ids["session_id"])
    assert shown["status"] == "round_1_reviewing"

    reviewed = service.real_loop_review(
        ids["session_id"],
        round_number=1,
        provider="openai-compatible",
        allow_network=False,
        transport=transport,
        openai_config=_cfg(),
    )
    assert reviewed.get("ok", True) is not False
    assert reviewed["fallback_used"] is False
    assert reviewed["provider_audit"]["fallback_used"] is False
    shown2 = service.real_loop_show(ids["session_id"])
    assert shown2["status"] == "round_1_waiting_approval"
