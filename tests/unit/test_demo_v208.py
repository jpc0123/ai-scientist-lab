"""v2.0.8 demo-create unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


@pytest.fixture()
def service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    return ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "demo.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )


@pytest.fixture()
def api_client(service: ExperimentService):
    app = create_app(service=service)
    return TestClient(app), service


def test_demo_catalog_and_create_digits(api_client):
    client, service = api_client
    catalog = client.get("/api/v1/demo/catalog")
    assert catalog.status_code == 200
    kinds = {item["kind"] for item in catalog.json()["items"]}
    assert kinds == {"digits", "rgbt-debug"}

    created = client.post("/api/v1/demo/create", json={"kind": "digits"})
    assert created.status_code == 200
    body = created.json()
    assert body["status"] == "created"
    assert body["auto_ran_experiments"] is False
    assert body["project"]["project_id"] == "demo_digits_v20"
    assert body["project"]["status"] in {"ready", "active"}
    assert len(body["nodes"]) >= 1

    again = client.post("/api/v1/demo/create", json={"kind": "digits"})
    assert again.status_code == 200
    assert again.json()["status"] == "exists"

    health = client.get("/api/v1/health")
    assert health.json()["version"] == "v2.0.0"

    listed = service.list_projects()
    assert any(p["project_id"] == "demo_digits_v20" for p in listed)


def test_demo_create_rgbt_debug(service: ExperimentService):
    result = service.demos.create("rgbt-debug")
    assert result["status"] == "created"
    assert result["kind"] == "rgbt-debug"
    assert result["auto_ran_experiments"] is False
    project = result["project"]
    assert project["project_id"] == "demo_rgbt_debug_v20"
    assert project["task_type"] == "rgbt_detection"
    assert len(result["nodes"]) >= 3


def test_demo_create_unknown_kind(service: ExperimentService):
    with pytest.raises(ValueError):
        service.demos.create("unknown")
