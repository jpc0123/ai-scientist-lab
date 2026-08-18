"""Command-center snapshot + chat. No GPU, no key echo."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.services.console_chat import (
    build_console_snapshot,
    handle_console_command,
    handle_console_turn,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _client(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    service = ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "api.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    app = create_app(service=service)
    return TestClient(app), service, root


def test_snapshot_has_global_status(tmp_path: Path):
    client, _, _ = _client(tmp_path)
    resp = client.get("/api/v1/console/snapshot")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["llm"]["ready_for_real_calls"] is False
    assert "c1" in data
    assert "packs" in data
    ids = {row["id"] for row in data["packs"]}
    assert "opening" in data
    assert "实验室助手" in data["opening"]["text"]
    assert data["opening"]["suggestions"]


def test_help_and_status_are_local(tmp_path: Path):
    client, _, _ = _client(tmp_path)
    help_resp = client.post("/api/v1/console/chat", json={"message": "/help", "live": True})
    assert help_resp.status_code == 200
    body = help_resp.json()
    assert body["live"] is False
    assert "/status" in body["text"]
    assert body["agent"] == "规划 Agent"
    assert body["agent_id"] == "planner"
    assert body["suggestions"]
    assert "api_key" not in body["text"].lower() or "LLM_API_KEY" not in str(body)

    status = client.post("/api/v1/console/chat", json={"message": "/status"})
    assert status.status_code == 200
    text = status.json()["text"]
    assert "Scientist Lab" in text
    assert "KEEP" in text or "Claim" in text or "C1" in text


def test_live_without_ready_fail_closed(tmp_path: Path):
    client, _, _ = _client(tmp_path)
    resp = client.post(
        "/api/v1/console/chat",
        json={"message": "帮我写一段可以投稿的论文摘要", "live": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["live"] is False
    assert "不会假装" in data["text"] or "尚未就绪" in data["text"]


def test_hello_is_local_agent_turn():
    snap = build_console_snapshot(Path(__file__).resolve().parents[2], llm_status={})
    reply = handle_console_command("你好", snapshot=snap, root=".")
    assert reply is not None
    assert reply["live"] is False
    assert reply["kind"] == "greet"
    assert "实验室助手" in reply["text"]
    assert reply["suggestions"]


def test_claim_question_stays_local():
    snap = {
        "c1": {
            "allowed": True,
            "baseline_aps": 0.0163,
            "candidate_aps": 0.0326,
            "claim_status": "SUPPORTED",
        }
    }
    reply = handle_console_command("现在能声称什么？", snapshot=snap, root=".")
    assert reply is not None
    assert reply["kind"] == "claim"
    assert "0.0326" in reply["text"]
    assert "SOTA" in reply["text"]


def test_unknown_slash_does_not_call_llm(tmp_path: Path):
    snap = build_console_snapshot(Path(__file__).resolve().parents[2], llm_status={})
    reply = handle_console_command("/nuke", snapshot=snap, root=".")
    assert reply is not None
    assert "未知指令" in reply["text"]


def test_c1_command_never_upgrades_claim():
    snap = {
        "c1": {
            "allowed": True,
            "baseline_aps": 0.0163,
            "candidate_aps": 0.0326,
            "claim_status": "SUPPORTED",
        }
    }
    reply = handle_console_command("/c1", snapshot=snap, root=".")
    assert reply is not None
    assert "0.0326" in reply["text"]
    assert "FDPN" in reply["text"]
    assert "SOTA" in reply["text"]


def test_parse_and_validate_skips_plain_chat_without_schema():
    from scientist_lab.llm.schema_parser import parse_and_validate

    data, errors = parse_and_validate("你好，实验室现在能声称什么？", None)
    assert data == {}
    assert errors == []


def test_plain_chat_completion_does_not_require_json():
    from scientist_lab.llm.http_transport import HttpResponse, MockTransport
    from scientist_lab.llm.models import LLMRequest
    from scientist_lab.llm.openai_compatible_provider import OpenAICompatibleProvider
    from scientist_lab.llm.openai_config import OpenAICompatibleConfig
    from pydantic import SecretStr

    body = json.dumps(
        {
            "id": "chatcmpl-hi",
            "choices": [{"message": {"role": "assistant", "content": "你好，我是控制台助手。"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
        },
        ensure_ascii=False,
    )
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            base_url="https://api.example.com/v1",
            api_key=SecretStr("sk-test"),
            model="qwen-test",
            allow_network=False,
        ),
        transport=MockTransport(
            default_response=HttpResponse(status_code=200, body=body)
        ),
    )
    response = provider.complete(
        LLMRequest(
            purpose="other",
            messages=[{"role": "user", "content": "你好"}],
            temperature=0.3,
            max_tokens=64,
            metadata={"console_chat": True},
        )
    )
    assert response.content == "你好，我是控制台助手。"
    assert response.schema_errors == []


def test_live_chat_uses_gateway(tmp_path: Path, monkeypatch):
    from scientist_lab.services import console_chat as mod

    fake = MagicMock()
    fake.content = "实验室当前只有 Formal C1 可以展示对照数字。"
    fake.model = "unit-test"
    fake.provider = "mock"
    fake.request_id = "llmreq_test"
    monkeypatch.setattr(mod, "complete_chat", lambda *args, **kwargs: fake)

    snap = build_console_snapshot(
        Path(__file__).resolve().parents[2],
        llm_status={"ready_for_real_calls": True, "model": "unit-test", "provider": "mock"},
    )
    reply = handle_console_turn(
        message="请用同事口吻解释当前证据边界，不要用指令格式。",
        history=[],
        snapshot=snap,
        runtime_dir=tmp_path / "runtime",
        live=True,
        root=Path(__file__).resolve().parents[2],
    )
    assert reply["live"] is True
    assert "Formal C1" in reply["text"]
    assert reply["model"] == "unit-test"


def test_hello_with_planner_agent():
    snap = build_console_snapshot(Path(__file__).resolve().parents[2], llm_status={})
    reply = handle_console_turn(
        message="你好",
        history=[],
        snapshot=snap,
        runtime_dir=".",
        live=False,
        root=".",
        agent="planner",
    )
    assert reply["agent_id"] == "planner"
    assert "规划 Agent" in reply["text"]
    assert "实验室助手" in reply["text"]


def test_session_chat_persists_and_agent_board(tmp_path: Path):
    client, _, _ = _client(tmp_path)
    agents = client.get("/api/v1/console/agents")
    assert agents.status_code == 200
    board = agents.json()["agents"]
    assert set(board) >= {"planner", "executor", "reviewer"}
    for key in ("planner", "executor", "reviewer"):
        assert board[key]["title"]
        assert "status" in board[key]
        assert "progress" in board[key]

    created = client.post("/api/v1/console/sessions", json={})
    assert created.status_code == 200
    session = created.json()
    sid = session["id"]
    assert sid.startswith("sess_")
    assert set(session["threads"]) >= {"planner", "executor", "reviewer"}
    assert session["threads"]["planner"]["messages"]

    listed = client.get("/api/v1/console/sessions")
    assert listed.status_code == 200
    assert any(row["id"] == sid for row in listed.json()["items"])

    chat = client.post(
        f"/api/v1/console/sessions/{sid}/chat",
        json={"message": "你好", "agent": "planner", "live": False},
    )
    assert chat.status_code == 200
    payload = chat.json()
    assert payload["ok"] is True
    assert payload["reply"]["agent_id"] == "planner"

    detail = client.get(f"/api/v1/console/sessions/{sid}")
    assert detail.status_code == 200
    messages = detail.json()["threads"]["planner"]["messages"]
    roles = [row["role"] for row in messages]
    assert roles[-2:] == ["user", "assistant"]
    assert any(row.get("text") == "你好" for row in messages)

    mem = client.post(
        f"/api/v1/console/sessions/{sid}/memory",
        json={"text": "只讨论 Formal C1，不升级声称"},
    )
    assert mem.status_code == 200
    notes = mem.json()["memory"]
    assert notes and "Formal C1" in notes[-1]["text"]
    note_id = notes[-1]["id"]

    dropped = client.delete(f"/api/v1/console/sessions/{sid}/memory/{note_id}")
    assert dropped.status_code == 200
    assert dropped.json()["memory"] == []

    switched = client.patch(
        f"/api/v1/console/sessions/{sid}",
        json={"active_agent": "reviewer", "pinned": True},
    )
    assert switched.status_code == 200
    assert switched.json()["active_agent"] == "reviewer"
    assert switched.json()["pinned"] is True

    listed2 = client.get("/api/v1/console/sessions").json()["items"]
    assert listed2[0]["id"] == sid
    assert listed2[0]["pinned"] is True
    agents_meta = listed2[0]["agents"]
    assert [row["id"] for row in agents_meta] == ["planner", "executor", "reviewer"]
    assert agents_meta[0]["title"] == "规划 Agent"
    assert agents_meta[0]["turns"] >= 1
    assert "规划 Agent" in listed2[0]["preview"]
    assert any(row.get("agent_id") == "planner" for row in messages)
    assert session.get("workspace_id") == "ws_default"


def test_workspaces_isolate_sessions_and_memory(tmp_path: Path):
    client, _, _ = _client(tmp_path)
    default_sess = client.post("/api/v1/console/sessions", json={}).json()
    client.post(
        f"/api/v1/console/sessions/{default_sess['id']}/memory",
        json={"text": "只属于默认工作区"},
    )

    created_ws = client.post(
        "/api/v1/console/workspaces",
        json={"title": "Formal C1", "kind": "experiment", "pack_id": "formal_c1_aps_early_concat"},
    )
    assert created_ws.status_code == 200
    ws = created_ws.json()
    assert ws["id"].startswith("ws_")
    assert ws["kind"] == "experiment"
    assert ws["title"] == "Formal C1"

    mem = client.post(
        f"/api/v1/console/workspaces/{ws['id']}/memory",
        json={"text": "这个实验只看 C1，不升级声称"},
    )
    assert mem.status_code == 200
    assert any("只看 C1" in str(row.get("text")) for row in mem.json()["memory"])

    new_sess = client.post(
        "/api/v1/console/sessions",
        json={"workspace_id": ws["id"], "title": "C1 对照讨论"},
    )
    assert new_sess.status_code == 200
    assert new_sess.json()["workspace_id"] == ws["id"]

    default_list = client.get("/api/v1/console/sessions?workspace_id=ws_default").json()["items"]
    scoped_list = client.get(f"/api/v1/console/sessions?workspace_id={ws['id']}").json()["items"]
    assert any(row["id"] == default_sess["id"] for row in default_list)
    assert all(row["id"] != new_sess.json()["id"] for row in default_list)
    assert [row["id"] for row in scoped_list] == [new_sess.json()["id"]]

    workspaces = client.get("/api/v1/console/workspaces").json()["items"]
    titles = {row["id"]: row for row in workspaces}
    assert titles[ws["id"]]["session_count"] == 1
    assert titles["ws_default"]["session_count"] >= 1
    assert titles[ws["id"]]["pack_title"]

    unbound_ws = client.post(
        "/api/v1/console/workspaces",
        json={"title": "新课题", "kind": "project"},
    ).json()
    unbound_board = client.get(f"/api/v1/console/agents?workspace_id={unbound_ws['id']}").json()
    assert unbound_board["unbound"] is True
    assert unbound_board["agents"]["planner"]["status"] == "UNBOUND"
    assert unbound_board["pack_id"] in {None, ""}

    scoped_board = client.get(f"/api/v1/console/agents?workspace_id={ws['id']}").json()
    assert scoped_board["scoped"] is True
    assert scoped_board["unbound"] is False
    assert scoped_board["pack_id"] == "formal_c1_aps_early_concat"

    moved = client.patch(
        f"/api/v1/console/sessions/{default_sess['id']}",
        json={"workspace_id": ws["id"], "title": "从默认区挪过来的对照讨论"},
    )
    assert moved.status_code == 200
    assert moved.json()["workspace_id"] == ws["id"]
    assert moved.json()["title"] == "从默认区挪过来的对照讨论"

    client.post(
        f"/api/v1/console/sessions/{default_sess['id']}/chat",
        json={"message": "下一步做什么？", "agent": "planner", "live": False},
    )
    kept = client.get(f"/api/v1/console/sessions/{default_sess['id']}").json()
    assert kept["title"] == "从默认区挪过来的对照讨论"

    still_default = client.get("/api/v1/console/sessions?workspace_id=ws_default").json()["items"]
    assert all(row["id"] != default_sess["id"] for row in still_default)
