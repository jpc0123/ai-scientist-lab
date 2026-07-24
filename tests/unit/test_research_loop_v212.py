"""v2.1.2 PlanningContext feedback completeness + round gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.agents.context_builder import build_planning_context
from scientist_lab.agents.models import RoundFeedbackSummary
from scientist_lab.domain import NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExperimentNode, new_id, utc_now_iso
from scientist_lab.llm.context_sanitizer import sanitize_planning_context
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.research_loop.errors import RealLoopValidationError
from scientist_lab.research_loop.feedback import (
    build_round_feedback_summary,
    require_round_feedback_for_planning,
)
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


def _seed(service: ExperimentService) -> dict[str, str]:
    project = service.create_project(
        title="Digits Real Loop FB",
        research_question="feedback completeness",
        research_goal="v2.1.2",
        task_type="general_ml",
        project_id="project_digits_fb",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_openai_fb_v21",
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
    executed = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        node_type=NodeType.IMPROVEMENT,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=96",
        contract_json={"parameters": {"hidden_units": 96}},
        created_at=now,
        updated_at=now,
    )
    service.repo.upsert_node(parent)
    service.repo.upsert_node(executed)
    return {
        "project_id": project["project_id"],
        "protocol_id": protocol.protocol_id,
        "profile_id": profile.profile_id,
        "parent_id": parent.node_id,
        "executed_id": executed.node_id,
    }


def test_build_round_feedback_summary_infers_improved():
    summary = build_round_feedback_summary(
        source_round=1,
        parent_node_id="n1",
        executed_node_id="n2",
        comparison={
            "primary_metric": "accuracy",
            "primary_metric_delta": 0.03,
        },
        evidence_ids=["ev_1"],
        previous_hypothesis="try 96 units",
        executed_parameters={"hidden_units": 96},
    )
    assert summary.outcome_label == "improved"
    assert summary.metric_deltas["accuracy"] == pytest.approx(0.03)
    assert summary.evidence_added == ["ev_1"]


def test_build_round_feedback_summary_infers_costlier_when_time_rises():
    summary = build_round_feedback_summary(
        source_round=1,
        parent_node_id="n1",
        executed_node_id="n2",
        comparison={
            "primary_metric": "accuracy",
            "primary_metric_delta": 0.03,
            "resources": {"duration_seconds": 1.5},
        },
    )
    assert summary.outcome_label == "costlier"


def test_require_feedback_blocks_round_2_without_summary():
    ctx = build_planning_context(
        project_id="p",
        research_goal="g",
        protocol={"protocol_id": "x", "allowed_variables": ["hidden_units"]},
        nodes=[],
        loop_round_number=2,
    )
    with pytest.raises(RealLoopValidationError, match="round_feedback_summary"):
        require_round_feedback_for_planning(ctx, round_number=2)

    require_round_feedback_for_planning(ctx, round_number=1)


def test_sanitize_keeps_round_feedback():
    summary = RoundFeedbackSummary(
        source_round=1,
        parent_node_id="n1",
        executed_node_id="n2",
        metric_deltas={"accuracy": 0.01},
        outcome_label="improved",
    )
    ctx = build_planning_context(
        project_id="p",
        research_goal="g",
        protocol={"protocol_id": "x"},
        nodes=[],
        loop_session_id="realloop_x",
        loop_round_number=2,
        round_feedback_summary=summary,
        recent_execution_summary={"executed_node_id": "n2"},
    )
    cleaned = sanitize_planning_context(ctx)
    assert cleaned.round_feedback_summary is not None
    assert cleaned.round_feedback_summary.executed_node_id == "n2"
    assert cleaned.loop_session_id == "realloop_x"


def test_record_feedback_and_round2_context_gate(service: ExperimentService):
    ids = _seed(service)
    created = service.real_loop_create(
        ids["project_id"],
        profile_id=ids["profile_id"],
        protocol_id=ids["protocol_id"],
        baseline_node_ids=[ids["parent_id"]],
    )
    session_id = created["session_id"]

    # Round 2 without feedback must fail.
    with pytest.raises(RealLoopValidationError, match="feedback_summary missing"):
        service.real_loop_build_context(session_id, round_number=2)

    # Round 1 context does not require feedback.
    r1 = service.real_loop_build_context(session_id, round_number=1)
    assert r1["has_round_feedback"] is False
    assert r1["context"]["loop_round_number"] == 1

    fb = service.real_loop_record_feedback(
        session_id,
        parent_node_id=ids["parent_id"],
        executed_node_id=ids["executed_id"],
        source_round=1,
        comparison={
            "primary_metric": "accuracy",
            "primary_metric_delta": 0.02,
            "duration_seconds": 12.0,
        },
        evidence_ids=["ev_round1"],
        previous_hypothesis="increase hidden_units",
        advance_status=False,
    )
    assert fb["feedback_summary"]["outcome_label"] in {"improved", "costlier"}
    assert fb["round"]["feedback_summary_json"]["evidence_added"] == ["ev_round1"]

    r2 = service.real_loop_build_context(session_id, round_number=2)
    assert r2["has_round_feedback"] is True
    ctx = r2["context"]
    assert ctx["round_feedback_summary"]["source_round"] == 1
    assert ctx["round_feedback_summary"]["executed_node_id"] == ids["executed_id"]
    assert ctx["recent_execution_summary"]["executed_node_id"] == ids["executed_id"]
    assert "accuracy" in (ctx["round_feedback_summary"]["metric_deltas"] or {})
