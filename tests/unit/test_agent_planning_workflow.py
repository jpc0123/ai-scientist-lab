from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.agents.critic import MockCritic
from scientist_lab.agents.models import CandidateExperiment
from scientist_lab.agents.ranker import rank_candidates
from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.planning.contract_generator import generate_contract_from_candidate
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
    service.set_budget(
        "project_rgbt_003",
        max_new_nodes=3,
        max_executions=15,
        max_gpu_hours=10,
    )
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="Improve small-object RGB-T detection under protocol.",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    service.repo.upsert_node(
        ExperimentNode(
            node_id="rgbt_formal_node_003",
            project_id="project_rgbt_003",
            node_type=NodeType.BASELINE,
            stage=NodeStage.DONE,
            status=NodeStatus.SUCCEEDED,
            depth=0,
            contract_json=json.loads(
                (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(
                    encoding="utf-8"
                )
            ),
            created_at=now,
            updated_at=now,
        )
    )


def test_full_plan_review_rank_approve_contract(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    planned = service.plan_next("project_rgbt_003", protocol_id="protocol_rgbt_001")
    plan_id = planned["plan_id"]
    assert planned["valid_candidate_count"] >= 1

    reviewed = service.review_plan(plan_id)
    assert reviewed["status"] == "reviewed"
    assert reviewed["reviews"]

    ranked = service.rank_candidates(plan_id)
    assert ranked["ranking"]
    top = ranked["ranking"][0]["candidate_id"]

    approved = service.approve_candidate(plan_id, top)
    assert approved["status"] == "approved"

    with pytest.raises(ValueError):
        service.approve_candidate(plan_id, top)

    generated = service.generate_contract_from_plan(plan_id, top)
    contract = generated["contract"]
    assert contract["protocol_id"] == "protocol_rgbt_001"
    assert contract["task_config"]["plan_id"] == plan_id
    assert contract["task_config"]["candidate_id"] == top
    assert contract["environment_key"] == "rgbt-detection-v2"
    assert contract["dataset_reference"] == "dataset:rgbt_fast_eval_v1"
    assert Path(generated["contract_path"]).is_file()

    budget = service.show_budget("project_rgbt_003")
    assert budget["used_nodes"] == 1


def test_unapproved_cannot_generate_contract(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    planned = service.plan_next("project_rgbt_003")
    plan_id = planned["plan_id"]
    cand = planned["candidates"][0]["candidate_id"]
    with pytest.raises(ValueError, match="approved"):
        service.generate_contract_from_plan(plan_id, cand)


def test_reject_blocks_approve(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    planned = service.plan_next("project_rgbt_003")
    plan_id = planned["plan_id"]
    cand = next(
        item["candidate_id"]
        for item in planned["candidates"]
        if item.get("status") == "verified"
    )
    service.reject_candidate(plan_id, cand, reason="not useful")
    with pytest.raises(ValueError):
        service.approve_candidate(plan_id, cand)


def test_ranker_orders_and_caps():
    a = CandidateExperiment(
        candidate_id="candidate_a",
        parent_node_id="n",
        title="a",
        hypothesis="h",
        experiment_type="ablation",
        parameter_changes={"input_mode": "rgb", "fusion_method": "none"},
        evidence_gap_addressed=["gap"],
        priority=0.9,
        estimated_cost={"gpu_hours": 1},
    )
    b = CandidateExperiment(
        candidate_id="candidate_b",
        parent_node_id="n",
        title="b",
        hypothesis="h",
        experiment_type="replication",
        parameter_changes={"input_mode": "thermal", "fusion_method": "none"},
        evidence_gap_addressed=[],
        priority=0.4,
        estimated_cost={"gpu_hours": 5},
    )
    critic = MockCritic()
    from scientist_lab.agents.context_builder import build_planning_context

    ctx = build_planning_context(
        project_id="p",
        research_goal="g",
        protocol={"allowed_variables": ["input_mode", "fusion_method"]},
        nodes=[{"node_id": "n", "project_id": "p", "status": "ok", "contract_json": {}}],
    )
    ranked = rank_candidates(
        [(a, critic.review(a, ctx)), (b, critic.review(b, ctx))],
        max_keep=3,
    )
    assert ranked[0]["candidate_id"] == "candidate_a"
    assert ranked[0]["rank"] == 1


def test_contract_generator_preserves_safety_fields():
    parent = json.loads(
        (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
    )
    candidate = CandidateExperiment(
        candidate_id="candidate_x",
        parent_node_id="rgbt_formal_node_003",
        title="RGB ablation",
        hypothesis="h",
        experiment_type="ablation",
        parameter_changes={"input_mode": "rgb", "fusion_method": "none"},
        evidence_gap_addressed=["gap"],
    )
    result = generate_contract_from_candidate(
        parent_contract=parent,
        candidate=candidate,
        plan_id="plan_1",
        existing_node_ids={"rgbt_formal_node_003"},
    )
    assert result["contract"]["parameters"]["input_mode"] == "rgb"
    assert result["contract"]["entrypoint"] == parent["entrypoint"]
    assert result["contract"]["code_reference"] == parent["code_reference"]


def test_iterate_from_plan_waiting_approval(tmp_path: Path):
    from scientist_lab.iteration.service import IterationService

    service = _service(tmp_path)
    _bootstrap(service)
    planned = service.plan_next("project_rgbt_003")
    plan_id = planned["plan_id"]
    service.review_plan(plan_id)
    ranked = service.rank_candidates(plan_id)
    top = ranked["ranking"][0]["candidate_id"]
    service.approve_candidate(plan_id, top)
    iteration = IterationService(service)
    payload = iteration.start_from_plan(plan_id, top)
    assert payload["status"] == "waiting_approval"
    assert payload["proposed_node_id"]
    assert Path(payload["contract_path"]).is_file()
