"""Idea Interview → propose. Not campaign Planner. No GPU."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.models import LLMRequest
from scientist_lab.services.autonomous_campaign import AutonomousCampaignService
from scientist_lab.services.experiment_propose import propose_experiment_draft
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.idea_interview import IdeaInterviewService
from scientist_lab.settings import Settings


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


def test_idea_interview_chat_extracts_brief(tmp_path: Path) -> None:
    svc = IdeaInterviewService(tmp_path)
    created = svc.create()
    iid = created["interview_id"]
    out = svc.chat(
        iid,
        "我想在 full-val 上看 RGB-only F0 是否弱于融合，主指标 APS。",
        live=False,
        provider=FakeProvider(),
    )
    assert out["not_planner"] is True
    assert out["gpu"] is False
    assert out["brief"]["ready_to_draft"] is True
    assert "APS" in (out["user_intent"] or out["brief"]["core_intent"])
    assert len(out["turns"]) >= 2
    assert out["turns"][-1]["role"] == "assistant"


def test_idea_interview_flags_other_task(tmp_path: Path) -> None:
    svc = IdeaInterviewService(tmp_path)
    iid = svc.create()["interview_id"]
    out = svc.chat(iid, "我想做图像分类实验", live=False, provider=FakeProvider())
    assert out["brief"]["ready_to_draft"] is False
    assert out["brief"]["unsupported_task_note"]


def test_propose_honors_user_intent(tmp_path: Path) -> None:
    intent = "Compare seed stability of HOW F3 on full-val APS, not low-light."
    draft = propose_experiment_draft(
        provider=FakeProvider(),
        live=False,
        user_intent=intent,
    )
    assert draft["ok"] is True
    assert draft["gpu"] is False
    assert intent.split()[0] in str(draft["research_question"])
    assert draft["user_intent"] == intent


def test_api_idea_interview_then_propose(tmp_path: Path) -> None:
    client = _client(tmp_path)
    created = client.post("/api/v1/registered-experiments/idea-interview")
    assert created.status_code == 200, created.text
    iid = created.json()["interview_id"]
    chat = client.post(
        f"/api/v1/registered-experiments/idea-interview/{iid}/chat",
        json={
            "message": "我想验证第二检测器迁移，主指标 APS，不要低光切片。",
            "live": False,
        },
    )
    assert chat.status_code == 200, chat.text
    body = chat.json()
    assert body["brief"]["core_intent"]
    proposed = client.post(
        "/api/v1/registered-experiments/propose",
        json={"live": False, "interview_id": iid},
    )
    assert proposed.status_code == 200, proposed.text
    draft = proposed.json()
    assert draft["interview_id"] == iid
    assert draft["user_intent"]
    assert draft["protocol"]["protocol_id"] != "research_protocol_rgbt_dfine_v26"
    session = client.get(f"/api/v1/registered-experiments/idea-interview/{iid}")
    assert session.status_code == 200
    assert session.json()["status"] == "drafted"


def test_fake_provider_idea_interview_contract() -> None:
    response = FakeProvider().complete(
        LLMRequest(
            purpose="other",
            messages=[
                {
                    "role": "user",
                    "content": (
                        '{"dialogue":[{"role":"human","text":"full-val APS fusion check"}],'
                        '"current_brief":{}}'
                    ),
                }
            ],
            metadata={"planner_contract": "idea_interview"},
            response_schema={
                "type": "object",
                "required": ["reply", "brief"],
            },
        )
    )
    assert response.schema_valid is True
    parsed = dict(response.parsed_json or {})
    assert parsed["brief"]["ready_to_draft"] is True
