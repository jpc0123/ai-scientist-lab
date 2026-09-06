"""v2.1.8 display mode + real-loop API smoke."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.domain import NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExperimentNode, new_id, utc_now_iso
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.research_loop.display_mode import derive_display_mode
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def test_derive_display_mode_real_mock_replay():
    assert (
        derive_display_mode(
            {"real_only": True, "fallback_allowed": False, "fallback_used": False}
        )
        == "REAL"
    )
    assert (
        derive_display_mode(
            {"real_only": True, "fallback_allowed": False, "fallback_used": True}
        )
        == "MOCK"
    )
    assert (
        derive_display_mode(
            {"real_only": True, "fallback_allowed": False, "fallback_used": False},
            rounds=[
                {
                    "provider_audit_json": {
                        "actual_provider": "replay",
                        "requested_provider": "openai-compatible",
                    }
                }
            ],
        )
        == "REPLAY"
    )


def test_real_loop_api_list_create_show(tmp_path: Path):
    settings = Settings(
        project_root=tmp_path,
        db_path=tmp_path / "lab.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=ROOT / "experiment_app",
    ).resolve()
    service = ExperimentService(settings=settings)
    project = service.create_project(
        title="Web Real Loop",
        research_question="api",
        research_goal="v2.1.8",
        task_type="general_ml",
        project_id="project_web_v218",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_web_v218",
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

    app = create_app(service=service)
    client = TestClient(app)

    created = client.post(
        "/api/v1/real-loops",
        json={
            "project_id": project["project_id"],
            "profile_id": profile.profile_id,
            "protocol_id": protocol.protocol_id,
            "rounds": 2,
            "baseline_node_ids": [parent.node_id],
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["display_mode"] == "REAL"
    sid = body["session_id"]

    listed = client.get(f"/api/v1/real-loops?project_id={project['project_id']}")
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert any(i["session_id"] == sid for i in items)
    assert items[0]["display_mode"] in {"REAL", "MOCK", "REPLAY"}

    shown = client.get(f"/api/v1/real-loops/{sid}")
    assert shown.status_code == 200
    assert shown.json()["mode_badge"] == "REAL"

    checked = client.post(f"/api/v1/real-loops/{sid}/check")
    assert checked.status_code == 200
    assert checked.json()["network_used"] is False
