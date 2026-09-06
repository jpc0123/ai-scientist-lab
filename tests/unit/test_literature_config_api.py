"""Web Semantic Scholar config API — secrets stay in runtime/, never returned."""

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
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_BASE_URL", raising=False)
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


def test_literature_config_get_masks_key(api_client):
    client, _service, _tmp = api_client
    status = client.get("/api/v1/system/literature-config")
    assert status.status_code == 200
    body = status.json()
    assert body["provider"] == "fake"
    assert body["api_key_present"] is False
    assert body["ready_for_live_search"] is False
    assert "SEMANTIC_SCHOLAR_API_KEY" in body["missing_for_live"]
    assert "api_key" not in body or body.get("api_key") in (None, "", "[REDACTED]")
    assert "s2-" not in status.text


def test_literature_config_save_and_reload(api_client):
    client, _service, tmp = api_client
    resp = client.post(
        "/api/v1/system/literature-config",
        json={"api_key": "s2-test-secret-never-echo"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_key_present"] is True
    assert body["ready_for_live_search"] is True
    assert body["provider"] == "semantic_scholar"
    assert "s2-test-secret-never-echo" not in resp.text

    secrets = tmp / "runtime" / "literature_secrets.env"
    assert secrets.is_file()
    raw = secrets.read_text(encoding="utf-8")
    assert "s2-test-secret-never-echo" in raw
    assert "SEMANTIC_SCHOLAR_API_KEY=" in raw

    again = client.get("/api/v1/system/literature-config")
    assert again.json()["api_key_present"] is True
    assert "s2-test-secret-never-echo" not in again.text

    keep = client.post("/api/v1/system/literature-config", json={"api_key": None})
    assert keep.status_code == 200
    assert keep.json()["api_key_present"] is True

    cleared = client.post(
        "/api/v1/system/literature-config",
        json={"clear_api_key": True},
    )
    assert cleared.status_code == 200
    assert cleared.json()["api_key_present"] is False
    assert cleared.json()["provider"] == "fake"


def test_literature_probe_without_key_is_conflict(api_client):
    client, _service, _tmp = api_client
    resp = client.post(
        "/api/v1/system/literature-config/probe",
        json={"query": "RGB-T", "limit": 1},
    )
    assert resp.status_code == 409
    assert "s2-" not in resp.text
    assert resp.json()["error"]["code"] == "missing_api_key"


def test_literature_probe_success_masked_and_year_unfiltered(api_client, monkeypatch):
    client, _service, _tmp = api_client
    secret = "s2-test-secret-never-echo"
    saved = client.post("/api/v1/system/literature-config", json={"api_key": secret})
    assert saved.status_code == 200
    captured: dict = {}

    class FakeRetriever:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def search(self, query, **kwargs):
            captured["search"] = {"query": query, **kwargs}
            return {
                "ok": True,
                "papers": [
                    {"paper": {"title": "RGB-T Fusion Probe Paper", "paper_id": "S2:abc"}}
                ],
                "provenance": {"literature_query_id": "q_probe_1"},
            }

    monkeypatch.setattr(
        "scientist_lab.literature.retriever.LiteratureRetriever",
        FakeRetriever,
    )
    resp = client.post(
        "/api/v1/system/literature-config/probe",
        json={"query": "RGB-T", "limit": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["provider"] == "semantic_scholar"
    assert body["hit_count"] == 1
    assert body["titles"] == ["RGB-T Fusion Probe Paper"]
    assert captured["search"]["year_from"] is None
    assert secret not in resp.text
    assert "api_key" not in body or body.get("api_key") in (None, "", "[REDACTED]")


def test_literature_probe_maps_connection_error(api_client, monkeypatch):
    from scientist_lab.llm.errors import LLMConnectionError

    client, _service, _tmp = api_client
    secret = "s2-test-secret-never-echo"
    client.post("/api/v1/system/literature-config", json={"api_key": secret})

    class BoomRetriever:
        def __init__(self, **kwargs):
            pass

        def search(self, *args, **kwargs):
            raise LLMConnectionError("HTTP connection error: ConnectError")

    monkeypatch.setattr(
        "scientist_lab.literature.retriever.LiteratureRetriever",
        BoomRetriever,
    )
    resp = client.post(
        "/api/v1/system/literature-config/probe",
        json={"query": "RGB-T", "limit": 1},
    )
    assert resp.status_code == 502
    payload = resp.json()["error"]
    assert payload["code"] == "literature_probe_failed"
    assert "ConnectError" in payload["message"]
    assert secret not in resp.text
    assert payload.get("details", {}).get("provider") == "semantic_scholar"
