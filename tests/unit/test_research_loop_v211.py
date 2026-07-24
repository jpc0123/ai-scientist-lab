"""v2.1.1 real research loop session + state machine (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.domain.models import ExperimentNode, new_id, utc_now_iso
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.research_loop.errors import (
    InvalidRealLoopTransition,
    RealLoopValidationError,
)
from scientist_lab.research_loop.models import RealResearchLoopSession
from scientist_lab.research_loop.state_machine import (
    can_enter_round_2,
    require_status,
    transition,
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


def _seed_project_protocol_profile(service: ExperimentService) -> dict[str, str]:
    from scientist_lab.domain import NodeStage, NodeStatus, NodeType

    project = service.create_project(
        title="Digits Real Loop",
        research_question="Can a real LLM iterate on Digits MLP?",
        research_goal="Two-round closed loop",
        task_type="general_ml",
        project_id="project_digits_real_loop",
        mark_ready=True,
    )
    project_id = project["project_id"]
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    protocol_id = protocol.protocol_id

    profile = LLMModelProfile(
        profile_id="profile_openai_digits_v21",
        provider="openai-compatible",
        model="gpt-test",
        api_mode="chat_completions",
        planner_prompt_version="planner_v1",
        critic_prompt_version="critic_v1",
        enabled=True,
        metadata={"note": "offline fixture for v2.1.1"},
    )
    service.llm_evals.upsert_profile(profile)

    now = utc_now_iso()
    node = ExperimentNode(
        node_id=new_id("node"),
        project_id=project_id,
        node_type=NodeType.BASELINE,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hidden_units=64",
        created_at=now,
        updated_at=now,
    )
    service.repo.upsert_node(node)

    return {
        "project_id": project_id,
        "protocol_id": protocol_id,
        "profile_id": profile.profile_id,
        "node_id": node.node_id,
    }


def test_state_machine_happy_path_and_illegal():
    session = RealResearchLoopSession.create(
        project_id="p",
        provider_profile_id="prof",
        protocol_id="proto",
    )
    assert session.status == "created"
    transition(session, "baseline_ready")
    transition(session, "round_1_planning")
    with pytest.raises(InvalidRealLoopTransition):
        transition(session, "completed")
    require_status(session, "round_1_planning")
    with pytest.raises(InvalidRealLoopTransition):
        require_status(session, "created")


def test_cannot_enter_round_2_before_feedback():
    session = RealResearchLoopSession.create(
        project_id="p",
        provider_profile_id="prof",
        protocol_id="proto",
    )
    assert can_enter_round_2(session) is False
    transition(session, "baseline_ready")
    transition(session, "round_1_planning")
    transition(session, "round_1_reviewing")
    transition(session, "round_1_waiting_approval")
    transition(session, "round_1_executing")
    transition(session, "round_1_feedback_ready")
    assert can_enter_round_2(session) is True


def test_create_show_check_persist_and_recover(service: ExperimentService, tmp_path: Path):
    ids = _seed_project_protocol_profile(service)
    baselines = [ids["node_id"]] if ids["node_id"] else []
    created = service.real_loop_create(
        ids["project_id"],
        profile_id=ids["profile_id"],
        protocol_id=ids["protocol_id"],
        rounds=2,
        baseline_node_ids=baselines,
    )
    assert created["status"] == "created"
    assert created["fallback_allowed"] is False
    assert created["fallback_used"] is False
    assert created["real_only"] is True
    assert created["required_rounds"] == 2
    assert len(created["rounds"]) == 2
    session_id = created["session_id"]

    shown = service.real_loop_show(session_id)
    assert shown["session_id"] == session_id
    assert shown["provider_profile_id"] == ids["profile_id"]

    checked = service.real_loop_check(session_id)
    assert checked["network_used"] is False
    assert checked["overall"] == "ok"
    assert checked["issues"] == []

    # Restart: new ExperimentService on same DB recovers session.
    reloaded = ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "lab.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )
    recovered = reloaded.real_loop_show(session_id)
    assert recovered["status"] == "created"
    assert recovered["protocol_id"] == ids["protocol_id"]
    assert len(recovered["rounds"]) == 2


def test_create_rejects_fallback_and_missing_profile(service: ExperimentService):
    ids = _seed_project_protocol_profile(service)
    with pytest.raises(RealLoopValidationError, match="fallback_allowed"):
        service.real_loop_create(
            ids["project_id"],
            profile_id=ids["profile_id"],
            protocol_id=ids["protocol_id"],
            fallback_allowed=True,
        )
    with pytest.raises(RealLoopValidationError, match="llm profile not found"):
        service.real_loop_create(
            ids["project_id"],
            profile_id="missing_profile",
            protocol_id=ids["protocol_id"],
        )


def test_create_rejects_rounds_lt_2(service: ExperimentService):
    ids = _seed_project_protocol_profile(service)
    with pytest.raises(RealLoopValidationError, match="required_rounds"):
        service.real_loop_create(
            ids["project_id"],
            profile_id=ids["profile_id"],
            protocol_id=ids["protocol_id"],
            rounds=1,
        )
