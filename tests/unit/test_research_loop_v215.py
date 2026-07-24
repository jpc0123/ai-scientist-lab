"""v2.1.5 Evidence/Claim backfill → Round-2 PlanningContext."""

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
from scientist_lab.research_loop.errors import RealLoopValidationError
from scientist_lab.research_loop.feedback import normalize_comparison_for_feedback
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


@pytest.fixture()
def service(tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "lab.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def _cfg() -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        base_url="https://api.example.com/v1",
        api_key=SecretStr("sk-test-secret-key-value"),
        model="gpt-test",
        allow_network=False,
        api_mode="chat_completions",
    )


def _passing_scorecard() -> dict:
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
            "planner_prompt_version": "planner_v1",
            "critic_prompt_version": "critic_v1",
        },
    }


def _chat_body(content: dict) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-v215",
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
        "reasoning_summary": "Try mid-range capacity.",
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
                        "rationale": "More capacity.",
                    }
                ],
                "evidence_gap_addressed": ["Need mid-range hidden_units."],
                "priority": 0.7,
                "rationale": "Interpolate 64→128.",
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
        "strengths": ["Single variable."],
        "weaknesses": [],
        "required_revisions": [],
        "recommendation": "accept",
    }


def _seed_attempt(
    *,
    execution_id: str,
    node_id: str,
    project_id: str,
    seed: int,
    accuracy: float,
    hidden_units: int,
    attempt_index: int,
) -> ExecutionAttempt:
    now = utc_now_iso()
    return ExecutionAttempt(
        execution_id=execution_id,
        node_id=node_id,
        attempt_index=attempt_index,
        runner_profile="local",
        status=JobStatus.COMPLETED,
        image_reference="scientist-experiment:v2",
        code_version="local:experiment_app",
        dataset_version="sklearn:digits",
        result_json={
            "metrics": {
                "primary_metric": "accuracy",
                "metrics": {
                    "accuracy": accuracy,
                    "f1_macro": accuracy - 0.01,
                    "duration_seconds": 0.2,
                },
            },
            "contract": {
                "schema_version": "1.0",
                "project_id": project_id,
                "node_id": node_id,
                "seed": seed,
                "dataset_reference": "sklearn:digits",
                "code_reference": "local:experiment_app",
                "environment_key": "digits-mlp-v1",
                "entrypoint": "run_experiment.py",
                "execution_mode": "fast_eval",
                "parameters": {
                    "learning_rate": 0.001,
                    "epochs": 30,
                    "hidden_units": hidden_units,
                    "batch_size": 64,
                    "test_size": 0.2,
                },
                "task_config": {
                    "claim_level": "exploratory_comparison",
                    "primary_metric": "accuracy",
                },
            },
        },
        created_at=now,
        completed_at=now,
    )


def _seed_through_execute(
    service: ExperimentService, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    project = service.create_project(
        title="Digits Evidence Loop",
        research_question="evidence backfill",
        research_goal="v2.1.5",
        task_type="general_ml",
        project_id="project_digits_v215",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_openai_v215",
        provider="openai-compatible",
        model="gpt-test",
        planner_prompt_version="planner_v1",
        critic_prompt_version="critic_v1",
        enabled=True,
    )
    service.llm_evals.upsert_profile(profile)
    service.llm_evals.save_evaluation(
        evaluation_id="eval_v215_001",
        profile_id=profile.profile_id,
        suite_version="eval_suite_v1",
        status="completed",
        result=_passing_scorecard(),
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
    for idx, seed in enumerate([42, 43], start=1):
        service.repo.upsert_attempt(
            _seed_attempt(
                execution_id=f"exec_base_{seed}",
                node_id=parent.node_id,
                project_id=project["project_id"],
                seed=seed,
                accuracy=0.940 + idx * 0.001,
                hidden_units=64,
                attempt_index=idx,
            )
        )

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

    def _run(contract, seed_list, *, auto_aggregate=True, wait=True):
        assert str(contract.entrypoint) == "run_experiment.py"
        now_local = utc_now_iso()
        node = service.repo.get_node(contract.node_id)
        if node is None:
            service.repo.upsert_node(
                ExperimentNode(
                    node_id=contract.node_id,
                    project_id=contract.project_id,
                    parent_node_id=contract.parent_node_id,
                    node_type=NodeType.IMPROVEMENT,
                    stage=NodeStage.DONE,
                    status=NodeStatus.SUCCEEDED,
                    depth=1,
                    hypothesis=contract.hypothesis,
                    contract_json=contract.model_dump(mode="json"),
                    created_at=now_local,
                    updated_at=now_local,
                )
            )
        else:
            node.stage = NodeStage.DONE
            node.status = NodeStatus.SUCCEEDED
            node.updated_at = now_local
            service.repo.upsert_node(node)
        results = []
        for idx, seed in enumerate(seed_list, start=1):
            exec_id = f"exec_cand_{contract.node_id}_{seed}"
            service.repo.upsert_attempt(
                _seed_attempt(
                    execution_id=exec_id,
                    node_id=contract.node_id,
                    project_id=contract.project_id,
                    seed=int(seed),
                    accuracy=0.955 + idx * 0.001,
                    hidden_units=int((contract.parameters or {}).get("hidden_units") or 112),
                    attempt_index=idx,
                )
            )
            results.append(
                {
                    "seed": int(seed),
                    "execution_id": exec_id,
                    "status": "completed",
                    "metrics": {"primary_metric": "accuracy"},
                }
            )
        return {
            "node_id": contract.node_id,
            "results": results,
            "aggregate": {"ok": True} if auto_aggregate else None,
        }

    monkeypatch.setattr(service, "run_seeds", _run)

    approved = service.real_loop_approve(
        created["session_id"],
        candidate_id="candidate_digits_hu112",
        round_number=1,
        seeds=[42, 43],
    )
    executed = service.real_loop_execute(
        created["session_id"],
        round_number=1,
        seeds=[42, 43],
        wait=True,
    )
    assert executed["ok"] is True
    assert executed["status"] == "round_1_executing"
    return {
        "session_id": created["session_id"],
        "project_id": project["project_id"],
        "parent_id": parent.node_id,
        "execution_node_id": str(executed["execution_node_id"]),
        "iteration_id": approved["iteration_id"],
    }


def test_normalize_comparison_maps_mean_delta():
    normalized = normalize_comparison_for_feedback(
        {"primary_metric": "accuracy", "mean_delta": 0.012, "decision": "candidate_better"}
    )
    assert normalized["primary_metric_delta"] == 0.012
    assert normalized["outcome_label"] == "improved"


def test_record_execution_feedback_builds_evidence_and_round2_context(
    service: ExperimentService, monkeypatch: pytest.MonkeyPatch
):
    ids = _seed_through_execute(service, monkeypatch)
    shown = service.real_loop_show(ids["session_id"])
    assert shown["status"] == "round_1_executing"
    r1 = next(r for r in shown["rounds"] if r["round_number"] == 1)
    assert not r1.get("feedback_summary_json")

    fb = service.real_loop_record_execution_feedback(
        ids["session_id"],
        include_round2_context=True,
    )
    assert fb["ok"] is True
    assert fb["status"] == "round_1_feedback_ready"
    assert fb["evidence_ids"]
    assert fb["feedback_summary"]["evidence_added"]
    assert fb["feedback_summary"]["metric_deltas"]
    assert fb["round2_context"] is not None
    assert fb["round2_context"]["has_round_feedback"] is True
    assert fb["round2_context"]["evidence_count"] >= 1
    ctx = fb["round2_context"]["context"]
    assert ctx["round_feedback_summary"]["executed_node_id"] == ids["execution_node_id"]
    assert ctx.get("evidence_records") or fb["round2_context"]["evidence_count"] >= 1

    shown2 = service.real_loop_show(ids["session_id"])
    assert shown2["status"] == "round_1_feedback_ready"
    r1b = next(r for r in shown2["rounds"] if r["round_number"] == 1)
    assert r1b["evidence_ids"]
    assert r1b["feedback_summary_json"]

    nxt = service.real_loop_next_round(ids["session_id"])
    assert nxt["ok"] is True
    assert nxt["status"] == "round_2_planning"
    assert nxt["context"]["has_round_feedback"] is True
    assert nxt["context"]["claim_count"] >= 0


def test_record_execution_feedback_requires_executing(service: ExperimentService):
    project = service.create_project(
        title="Digits Gate",
        research_question="gate",
        research_goal="v2.1.5",
        task_type="general_ml",
        project_id="project_digits_v215_gate",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_gate_v215",
        provider="openai-compatible",
        model="gpt-test",
        enabled=True,
    )
    service.llm_evals.upsert_profile(profile)
    now = utc_now_iso()
    parent = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        node_type=NodeType.BASELINE,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=64",
        contract_json={"parameters": {"hidden_units": 64}},
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
    with pytest.raises(Exception):
        service.real_loop_record_execution_feedback(created["session_id"])
