"""v2.1.4 approve → Digits execute (mock entrypoint banned; run_seeds patched)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, new_id, utc_now_iso
from scientist_lab.llm.http_transport import HttpResponse, MockTransport
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.research_loop.execution_gate import assert_real_digits_contract
from scientist_lab.research_loop.errors import RealLoopValidationError
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


def _seed_through_review(service: ExperimentService) -> dict[str, str]:
    project = service.create_project(
        title="Digits Real Execute",
        research_question="real digits round 1",
        research_goal="v2.1.4",
        task_type="general_ml",
        project_id="project_digits_v214",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_openai_v214",
        provider="openai-compatible",
        model="gpt-test",
        planner_prompt_version="planner_v1",
        critic_prompt_version="critic_v1",
        enabled=True,
    )
    service.llm_evals.upsert_profile(profile)
    service.llm_evals.save_evaluation(
        evaluation_id="eval_v214_001",
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

    transport = MockTransport(
        responses=[
            HttpResponse(
                status_code=200,
                body=_chat_body(
                    _planner_payload(project["project_id"], parent.node_id)
                ),
            ),
            HttpResponse(status_code=200, body=_chat_body(_critic_payload())),
        ]
    )
    planned = service.real_loop_plan(
        created["session_id"],
        round_number=1,
        provider="openai-compatible",
        allow_network=False,
        transport=transport,
        openai_config=_cfg(),
    )
    assert planned.get("ok", True) is not False
    reviewed = service.real_loop_review(
        created["session_id"],
        round_number=1,
        provider="openai-compatible",
        allow_network=False,
        transport=transport,
        openai_config=_cfg(),
    )
    assert reviewed.get("ok", True) is not False
    shown = service.real_loop_show(created["session_id"])
    assert shown["status"] == "round_1_waiting_approval"
    return {
        "session_id": created["session_id"],
        "project_id": project["project_id"],
        "parent_id": parent.node_id,
        "candidate_id": "candidate_digits_hu112",
        "plan_id": planned["plan_id"],
    }


def _patch_run_seeds(monkeypatch: pytest.MonkeyPatch, service: ExperimentService):
    def _run(contract, seed_list, *, auto_aggregate=True, wait=True):
        assert str(contract.entrypoint) == "run_experiment.py"
        assert str(contract.environment_key) == "digits-mlp-v1"
        assert "mock" not in str(contract.entrypoint).lower()
        now = utc_now_iso()
        node = service.repo.get_node(contract.node_id)
        if node is None:
            service.repo.upsert_node(
                ExperimentNode(
                    node_id=contract.node_id,
                    project_id=contract.project_id,
                    parent_node_id=contract.parent_node_id,
                    node_type=NodeType.IMPROVEMENT,
                    stage=NodeStage.DONE if wait else NodeStage.EXECUTING,
                    status=NodeStatus.SUCCEEDED if wait else NodeStatus.RUNNING,
                    depth=1,
                    contract_json=contract.model_dump(mode="json"),
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            node.stage = NodeStage.DONE if wait else NodeStage.EXECUTING
            node.status = NodeStatus.SUCCEEDED if wait else NodeStatus.RUNNING
            node.updated_at = now
            service.repo.upsert_node(node)

        results = []
        for idx, seed in enumerate(seed_list, start=1):
            exec_id = f"exec_{contract.node_id}_{seed}"
            if wait:
                service.repo.upsert_attempt(
                    ExecutionAttempt(
                        execution_id=exec_id,
                        node_id=contract.node_id,
                        attempt_index=idx,
                        runner_profile="local",
                        status=JobStatus.COMPLETED,
                        image_reference="scientist-experiment:v2",
                        code_version="local:experiment_app",
                        dataset_version="sklearn:digits",
                        result_json={
                            "metrics": {
                                "primary_metric": "accuracy",
                                "metrics": {"accuracy": 0.95 + idx * 0.001},
                            },
                            "contract": {
                                **contract.model_dump(mode="json"),
                                "seed": int(seed),
                            },
                        },
                        created_at=now,
                        completed_at=now,
                    )
                )
            results.append(
                {
                    "seed": int(seed),
                    "execution_id": exec_id,
                    "status": "completed" if wait else "queued",
                    "metrics": {"primary_metric": "accuracy"} if wait else None,
                }
            )
        return {
            "node_id": contract.node_id,
            "results": results,
            "aggregate": {"ok": True} if (wait and auto_aggregate) else None,
        }

    monkeypatch.setattr(service, "run_seeds", _run)


def test_assert_real_digits_rejects_mock_entrypoint():
    with pytest.raises(RealLoopValidationError):
        assert_real_digits_contract(
            {
                "entrypoint": "run_mock_experiment.py",
                "environment_key": "digits-mlp-v1",
                "dataset_reference": "sklearn:digits",
            }
        )
    with pytest.raises(RealLoopValidationError):
        assert_real_digits_contract(
            {
                "entrypoint": "run_experiment.py",
                "environment_key": "scientist-experiment-v1",
                "dataset_reference": "sklearn:digits",
            }
        )


def test_approve_and_execute_round1(service: ExperimentService, monkeypatch: pytest.MonkeyPatch):
    ids = _seed_through_review(service)
    _patch_run_seeds(monkeypatch, service)

    approved = service.real_loop_approve(
        ids["session_id"],
        candidate_id=ids["candidate_id"],
        round_number=1,
        seeds=[42, 43],
    )
    assert approved["ok"] is True
    assert approved["mock_execution"] is False
    assert approved["iteration_id"]
    assert approved["execution_gate"]["environment_key"] == "digits-mlp-v1"
    assert approved["execution_gate"]["entrypoint"] == "run_experiment.py"
    shown = service.real_loop_show(ids["session_id"])
    assert shown["status"] == "round_1_waiting_approval"
    rounds = shown["rounds"]
    r1 = next(r for r in rounds if r["round_number"] == 1)
    assert r1["approved_candidate_id"] == ids["candidate_id"]
    assert r1["iteration_id"] == approved["iteration_id"]

    executed = service.real_loop_execute(
        ids["session_id"],
        round_number=1,
        seeds=[42, 43],
        wait=True,
    )
    assert executed["ok"] is True
    assert executed["mock_execution"] is False
    assert executed["status"] == "round_1_executing"
    assert executed["execution_node_id"]
    shown2 = service.real_loop_show(ids["session_id"])
    assert shown2["status"] == "round_1_executing"
    assert executed["execution_node_id"] in shown2["execution_node_ids"]


def test_execute_without_approve_fails(service: ExperimentService):
    ids = _seed_through_review(service)
    with pytest.raises(RealLoopValidationError, match="approve"):
        service.real_loop_execute(ids["session_id"], round_number=1)


def test_reject_round(service: ExperimentService):
    ids = _seed_through_review(service)
    rejected = service.real_loop_reject(
        ids["session_id"],
        candidate_id=ids["candidate_id"],
        reason="not worth the seed budget",
    )
    assert rejected["ok"] is True
    assert rejected["status"] == "candidate_rejected"
    shown = service.real_loop_show(ids["session_id"])
    assert shown["status"] == "candidate_rejected"
