from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.agents.models import CandidateExperiment, PlanningContext
from scientist_lab.agents.provider_bridge import (
    ProviderCritic,
    ProviderPlanner,
    build_planner_critic,
)
from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.llm import LLMCallRepository, ReplayProvider, request_fingerprint
from scientist_lab.llm.context_codec import planning_context_to_planner_request
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
            research_goal="Provider bridge",
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


def test_provider_planner_and_critic_fake_then_replay(tmp_path: Path):
    audit_root = tmp_path / "llm"
    planner, critic = build_planner_critic(
        "fake", audit_root=audit_root, project_id="project_rgbt_003"
    )
    assert isinstance(planner, ProviderPlanner)
    assert isinstance(critic, ProviderCritic)

    context = PlanningContext(
        project_id="project_rgbt_003",
        research_goal="g",
        protocol={"protocol_id": "protocol_rgbt_001", "allowed_variables": ["input_mode"]},
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        nodes=[{"node_id": "rgbt_formal_node_003", "status": "succeeded"}],
        remaining_budget={"max_new_nodes": 3, "max_gpu_hours": 10},
        allowed_parameter_changes=["input_mode", "fusion_method"],
    )
    output = planner.plan(context)
    assert output.project_id == "project_rgbt_003"
    assert output.candidates

    candidate = output.candidates[0]
    review = critic.review(candidate, context)
    assert review.candidate_id == candidate.candidate_id
    assert review.recommendation in {"accept", "revise", "reject"}

    # Replay stack reproduces planner JSON content.
    replay_planner, replay_critic = build_planner_critic(
        "replay", audit_root=audit_root, project_id="project_rgbt_003"
    )
    replayed = replay_planner.plan(context)
    assert replayed.model_dump(mode="json") == output.model_dump(mode="json")

    replayed_review = replay_critic.review(candidate, context)
    assert replayed_review.candidate_id == candidate.candidate_id
    assert replayed_review.recommendation == review.recommendation

    # Fingerprint index exists on disk.
    repo = LLMCallRepository(audit_root)
    fp = request_fingerprint(planning_context_to_planner_request(context))
    assert repo.get_by_fingerprint(fp) is not None


def test_experiment_service_plan_next_fake_provider(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    planned = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        provider="fake",
    )
    assert planned["model_provider"] in {"fake", "audit:fake"}
    assert planned["candidates"]
    assert planned["status"] in {"verified", "planner_failed", "generated"}

    reviewed = service.review_plan(planned["plan_id"], provider="fake")
    assert reviewed["status"] == "reviewed"
    assert reviewed["reviews"]

    # Replay should work against the same audit directory.
    replayed = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        provider="replay",
    )
    assert replayed["model_provider"] == "replay"
    assert replayed["reasoning_summary"] == planned["reasoning_summary"]


def test_default_remains_mock(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    planned = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        provider="mock",
    )
    assert planned["model_provider"] == "mock"
    assert planned["model_name"] == "mock-planner-v1"
