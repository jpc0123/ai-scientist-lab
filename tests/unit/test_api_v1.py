"""v1.7.1 API contract regression tests (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.patching.service import build_mock_unified_diff
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


@pytest.fixture()
def api_client(tmp_path: Path):
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
    return TestClient(app), service


def test_health_and_summary(api_client):
    client, _ = api_client
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True
    assert body["api"] == "v1"

    summary = client.get("/api/v1/system/summary")
    assert summary.status_code == 200
    data = summary.json()
    assert "project_count" in data
    assert "pending_patches" in data


def test_list_endpoints_page_shape(api_client):
    client, _ = api_client
    for path in (
        "/api/v1/projects",
        "/api/v1/executions",
        "/api/v1/trees",
        "/api/v1/plans",
        "/api/v1/iterations",
        "/api/v1/evidence",
        "/api/v1/claims",
        "/api/v1/reports",
        "/api/v1/audits",
        "/api/v1/patches",
    ):
        resp = client.get(path, params={"limit": 10, "offset": 0})
        assert resp.status_code == 200, path
        data = resp.json()
        assert "items" in data
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert "total" in data
        assert "count" in data


def test_error_envelope_not_found(api_client):
    client, _ = api_client
    resp = client.get("/api/v1/projects/does_not_exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "not_found"
    assert "message" in body["error"]


def test_patch_lifecycle_controlled_writes(api_client):
    client, service = api_client
    # Seed a project row indirectly via propose (project_id is free-form).
    proposed = service.patches.propose_mock(
        "project_web_001",
        unified_diff=build_mock_unified_diff(
            relative_path=(
                "experiment_apps/rgbt_detection_real/adapters/web_api_note.md"
            )
        ),
    )
    patch_id = proposed["patch_id"]

    # Unapproved cannot apply-sandbox
    blocked = client.post(f"/api/v1/patches/{patch_id}/apply-sandbox")
    assert blocked.status_code == 409
    assert blocked.json()["error"]["code"] == "conflict"

    shown = client.get(f"/api/v1/patches/{patch_id}")
    assert shown.status_code == 200
    assert shown.json()["patch_id"] == patch_id

    approved = client.post(
        f"/api/v1/patches/{patch_id}/approve", json={"reason": "api test"}
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    applied = client.post(f"/api/v1/patches/{patch_id}/apply-sandbox")
    assert applied.status_code == 200
    assert applied.json()["status"] == "applied_sandbox"
    assert applied.json()["can_apply_main"] is False

    tested = client.post(
        f"/api/v1/patches/{patch_id}/test-sandbox",
        json={"profile": "smoke"},
    )
    assert tested.status_code == 200
    assert tested.json()["sandbox_tests"]["ok"] is True


def test_openapi_contains_v1_paths(api_client):
    client, _ = api_client
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/v1/health" in paths
    assert "/api/v1/system/summary" in paths
    assert "/api/v1/patches/{patch_id}/apply-sandbox" in paths
    assert "/api/v1/trees/{tree_id}/mermaid" in paths


def test_export_openapi_script(tmp_path: Path, api_client):
    client, _ = api_client
    schema = client.get("/openapi.json").json()
    out = tmp_path / "openapi_v1.json"
    out.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    assert out.is_file()
    assert "Scientist Lab" in schema["info"]["title"]
