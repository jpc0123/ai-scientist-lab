"""v2.0.9 product error envelope + security posture tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab import API_VERSION
from scientist_lab.api.app import create_app
from scientist_lab.api.errors import error_body, sanitize_message
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.system.security_posture import build_security_posture


@pytest.fixture()
def api_client(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    service = ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "sec.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    return TestClient(app=create_app(service=service), raise_server_exceptions=False), service


def test_error_body_product_shape():
    body = error_body(
        code="protocol_mismatch",
        message="Experiment does not match the selected protocol.",
        details={"field": "seed"},
    )
    err = body["error"]
    assert err["type"] == "protocol_mismatch"
    assert err["code"] == "protocol_mismatch"
    assert err["retryable"] is False
    assert "protocol" in err["suggested_action"].lower() or "协议" in err["suggested_action"]
    assert err["details"]["field"] == "seed"
    assert "Traceback" not in err["message"]


def test_sanitize_strips_traceback():
    raw = 'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: boom'
    assert "Traceback" not in sanitize_message(raw)
    assert "日志" in sanitize_message(raw)


def test_api_not_found_envelope(api_client):
    client, _ = api_client
    resp = client.get("/api/v1/projects/does_not_exist_xyz")
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["type"] == "not_found"
    assert err["code"] == "not_found"
    assert "retryable" in err
    assert "suggested_action" in err
    assert isinstance(err["details"], dict) or err["details"] == {}


def test_api_validation_envelope(api_client):
    client, _ = api_client
    resp = client.post("/api/v1/demo/create", json={"kind": "nope"})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["type"] == "validation_error"
    assert err["retryable"] is False
    assert err["details"]


def test_health_version_v209(api_client):
    client, _ = api_client
    health = client.get("/api/v1/health")
    assert health.json()["version"] == API_VERSION


def test_security_posture_endpoint(api_client):
    client, service = api_client
    resp = client.get("/api/v1/system/security")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == API_VERSION
    assert data["overall"] in {"ok", "warning"}
    ids = {b["id"] for b in data["boundaries"]}
    assert "default_mock_llm" in ids
    assert "no_arbitrary_shell" in ids
    assert "claim_gate" in ids
    assert "Shell 输入框" in data["ui_forbidden"]
    assert data["config_snapshot"]["allow_network_by_default"] is False

    local = service.security_posture()
    assert local["boundaries"]
    assert build_security_posture(project_root=Path(service.settings.project_root))[
        "overall"
    ] in {"ok", "warning"}
