"""Campaign Steer + idea snapshot at register. No GPU."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.core.campaign_steer import (
    active_steer_for_planner,
    needs_protocol_amendment,
    set_steer,
    steer_path,
)
from scientist_lab.core.schema_registry import load_json
from scientist_lab.llm.planner_contract import (
    PlannerContractInput,
    build_planner_request,
)
from scientist_lab.services.autonomous_campaign import AutonomousCampaignService
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.registered_experiments import RegisteredExperimentService
from scientist_lab.settings import Settings

EXAMPLES = Path(__file__).resolve().parents[2] / "schemas" / "examples"


def _client(tmp_path: Path) -> TestClient:
    root = Path(__file__).resolve().parents[2]
    service = ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "api.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    service._autonomous_campaigns = AutonomousCampaignService(project_root=tmp_path)
    return TestClient(create_app(service=service))


def test_needs_protocol_amendment_detects_metric() -> None:
    assert needs_protocol_amendment("下一轮换主指标 APS")
    assert not needs_protocol_amendment("下一轮别再加严 gating，先试多尺度")


def test_steer_active_for_planner(tmp_path: Path) -> None:
    path = steer_path(tmp_path / "camp")
    store = set_steer(path, text="下一轮先试 F3，不要更严 gating")
    active = active_steer_for_planner(store)
    assert active is not None
    assert active["planner_may_use"] is True
    blocked = set_steer(path, text="换数据集再跑")
    active2 = active_steer_for_planner(blocked)
    assert active2 is not None
    assert active2["planner_may_use"] is False
    assert active2["needs_protocol_amendment"] is True


def test_register_freezes_idea_snapshot(tmp_path: Path) -> None:
    proto = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    plan = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    proto = dict(proto)
    plan = dict(plan)
    proto["protocol_id"] = "research_protocol_steer_snap"
    proto["objective"] = {"primary": {"metric": "APS", "direction": "maximize"}, "secondary": []}
    proto.pop("condition_slice", None)
    plan["protocol_id"] = proto["protocol_id"]
    plan["how_id"] = "F0"
    plan["proposed_changes"] = [
        {"target": "fusion", "summary": "F0", "detail": {"how_id": "F0"}}
    ]
    store = RegisteredExperimentService(tmp_path)
    row = store.register(
        protocol=proto,
        seed_plan=plan,
        experiment_id="exp_steer_snap",
        user_intent="full-val APS fusion probe",
        idea_brief={"core_intent": "full-val APS fusion probe", "ready_to_draft": True},
        interview_id="idea_test",
    )
    assert row["user_intent"] == "full-val APS fusion probe"
    snap = tmp_path / ".run" / "experiments" / "exp_steer_snap" / "idea_snapshot.json"
    assert snap.is_file()
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    campaigns._worker = lambda campaign_id: None  # noqa: ARG005
    started = campaigns.start(
        experiment_id="exp_steer_snap",
        confirm_human_gate=True,
        execute=False,
        background=False,
    )
    assert started["idea_snapshot"]["user_intent"] == "full-val APS fusion probe"
    steered = campaigns.set_campaign_steer(
        started["campaign_id"],
        action="human_set",
        text="下一轮优先多尺度，不要更严 gating",
    )
    assert steered["steer"]["steer_intent"]["status"] == "active"
    assert steered["steer"]["active_for_planner"]["planner_may_use"] is True


def test_planner_request_includes_human_steer() -> None:
    req = build_planner_request(
        PlannerContractInput(
            goal={"task_type": "object_detection"},
            protocol={
                "protocol_id": "p",
                "editable_scope": ["fusion"],
                "frozen_scope": ["dataset"],
                "objective": {"primary": {"metric": "APS"}},
            },
            previous_plan={"plan_id": "plan0", "round_index": 0, "budget_class": "probe"},
            human_steer={"text": "try F3 next", "planner_may_use": True},
            idea_snapshot={"user_intent": "full-val APS"},
        )
    )
    blob = str(req.messages[-1]["content"])
    assert "human_steer" in blob
    assert "try F3 next" in blob
    assert "idea_snapshot" in blob


def test_api_steer_endpoint(tmp_path: Path) -> None:
    client = _client(tmp_path)
    # use builtin list then start needs ready experiment — start via service for speed
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    campaigns.experiments.ensure_builtin()
    campaigns._worker = lambda campaign_id: None  # noqa: ARG005
    started = campaigns.start(
        experiment_id="exp_rgbt_dfine_v26_lowlight",
        confirm_human_gate=True,
        execute=False,
        background=False,
        llm_live=False,
    )
    cid = started["campaign_id"]
    # rebind client service campaigns root
    service = ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "api2.db",
            runtime_dir=tmp_path / "runtime2",
            outputs_dir=tmp_path / "outputs2",
            experiment_app_dir=Path(__file__).resolve().parents[2] / "experiment_app",
        ).resolve()
    )
    service._autonomous_campaigns = campaigns
    client = TestClient(create_app(service=service))
    res = client.post(
        f"/api/v1/autonomous-campaigns/{cid}/steer",
        json={"action": "human_set", "text": "下一轮试 F0 对照"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["steer"]["steer_intent"]["text"] == "下一轮试 F0 对照"
