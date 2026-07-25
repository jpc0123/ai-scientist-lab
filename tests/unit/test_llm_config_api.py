"""Web LLM model config API — secrets stay in runtime/, never returned."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_ALLOW_NETWORK", raising=False)
    service = ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "lab.db",
            outputs_dir=tmp_path / "outputs",
            runtime_dir=tmp_path / "runtime",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    app = create_app(service=service)
    return TestClient(app), service, tmp_path


def test_llm_config_get_masks_key(api_client):
    client, service, tmp = api_client

    status = client.get("/api/v1/system/llm-config")
    assert status.status_code == 200
    body = status.json()
    assert body["provider"] in {"mock", "openai-compatible", "fake", "replay"}
    assert "api_key" not in body or body.get("api_key") in (None, "", "[REDACTED]")
    assert body["api_key_present"] is False
    text = status.text
    assert "sk-" not in text


def test_llm_config_save_and_reload(api_client):
    client, service, tmp = api_client

    resp = client.post(
        "/api/v1/system/llm-config",
        json={
            "provider": "openai-compatible",
            "base_url": "https://api.example.com/v1",
            "model": "gpt-test",
            "api_key": "sk-test-secret-never-echo",
            "allow_network": True,
            "timeout_seconds": 30,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "openai-compatible"
    assert body["base_url"] == "https://api.example.com/v1"
    assert body["model"] == "gpt-test"
    assert body["api_key_present"] is True
    assert body["allow_network"] is True
    assert body["ready_for_real_calls"] is True
    assert "sk-test-secret" not in resp.text
    assert body.get("api_key_fingerprint") in ("", "[REDACTED]")

    secrets = tmp / "runtime" / "llm_secrets.env"
    assert secrets.is_file()
    raw = secrets.read_text(encoding="utf-8")
    assert "sk-test-secret-never-echo" in raw
    assert "LLM_ALLOW_NETWORK=true" in raw

    again = client.get("/api/v1/system/llm-config")
    assert again.json()["api_key_present"] is True
    assert "sk-test-secret" not in again.text

    # Keep key when omitted
    keep = client.post(
        "/api/v1/system/llm-config",
        json={"model": "gpt-test-2", "api_key": None},
    )
    assert keep.status_code == 200
    assert keep.json()["model"] == "gpt-test-2"
    assert keep.json()["api_key_present"] is True

    cleared = client.post(
        "/api/v1/system/llm-config",
        json={"clear_api_key": True},
    )
    assert cleared.status_code == 200
    assert cleared.json()["api_key_present"] is False


def test_llm_profiles_register_and_select(api_client):
    client, service, _ = api_client
    created = client.post(
        "/api/v1/llm-profiles",
        json={
            "profile_id": "profile_web_test",
            "provider": "openai-compatible",
            "model": "gpt-test",
            "api_mode": "chat_completions",
        },
    )
    assert created.status_code == 200
    assert created.json()["profile"]["profile_id"] == "profile_web_test"

    listed = client.get("/api/v1/llm-profiles")
    assert listed.status_code == 200
    ids = [p["profile_id"] for p in listed.json()["items"]]
    assert "profile_web_test" in ids

    selected = client.post("/api/v1/llm-profiles/profile_web_test/select")
    assert selected.status_code == 200
    assert selected.json()["default_profile_id"] == "profile_web_test"

    blocked = client.post(
        "/api/v1/llm-profiles",
        json={
            "profile_id": "bad",
            "provider": "mock",
            "model": "m",
            "metadata": {"api_key": "sk-should-fail"},
        },
    )
    assert blocked.status_code == 400
