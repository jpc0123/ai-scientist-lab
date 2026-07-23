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


def test_demo_seed_patch(api_client):
    client, _ = api_client
    seeded = client.post("/api/v1/demo/seed-patch")
    assert seeded.status_code == 200
    body = seeded.json()
    assert body["status"] == "verified"
    assert body["can_apply_main"] is False
    assert body["patch_id"]
    shown = client.get(f"/api/v1/patches/{body['patch_id']}")
    assert shown.status_code == 200


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
        "/api/v1/releases",
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
    assert "/api/v1/system/path-policy" in paths
    assert "/api/v1/patches/{patch_id}/apply-sandbox" in paths
    assert "/api/v1/patches/{patch_id}/record-evidence" in paths
    assert "/api/v1/reports/{report_id}/markdown" in paths
    assert "/api/v1/reports/build" in paths
    assert "/api/v1/audits/build" in paths
    assert "/api/v1/audits/{bundle_id}/export" in paths
    assert "/api/v1/releases" in paths
    assert "/api/v1/workspaces/summary" in paths
    assert "/api/v1/trees/{tree_id}/mermaid" in paths


def test_export_openapi_script(tmp_path: Path, api_client):
    client, _ = api_client
    schema = client.get("/openapi.json").json()
    out = tmp_path / "openapi_v1.json"
    out.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    assert out.is_file()
    assert "Scientist Lab" in schema["info"]["title"]


def test_workspaces_and_release_api(api_client):
    client, _ = api_client
    summary = client.get("/api/v1/workspaces/summary")
    assert summary.status_code == 200
    body = summary.json()
    assert body["can_write_main"] is False
    assert body["main_workspace_modified"] is False

    created = client.post(
        "/api/v1/releases",
        json={"project_id": "project_web_rel", "title": "API release"},
    )
    assert created.status_code == 200
    release_id = created.json()["release_id"]
    assert created.json()["status"] == "draft"

    frozen = client.post(f"/api/v1/releases/{release_id}/freeze", json={"notes": "ok"})
    assert frozen.status_code == 200
    assert frozen.json()["status"] == "frozen"
    assert frozen.json()["main_workspace_modified"] is False

    shown = client.get(f"/api/v1/releases/{release_id}")
    assert shown.status_code == 200
    assert shown.json()["can_write_main"] is False


def test_report_audit_build_and_export_api(api_client, tmp_path: Path):
    from scientist_lab.domain import (
        JobStatus,
        NodeStage,
        NodeStatus,
        NodeType,
        ProjectStatus,
    )
    from scientist_lab.domain.models import (
        ExecutionAttempt,
        ExperimentNode,
        ResearchProject,
    )

    client, service = api_client
    examples = Path(__file__).resolve().parents[2] / "examples"
    now = "2026-01-01T00:00:00+00:00"
    service.protocols.create_from_path(examples / "rgbt_protocol.json")
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="v1.8 api build",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    contract = json.loads(
        (examples / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
    )
    service.repo.upsert_node(
        ExperimentNode(
            node_id="rgbt_formal_node_api",
            project_id="project_rgbt_003",
            node_type=NodeType.BASELINE,
            stage=NodeStage.DONE,
            status=NodeStatus.SUCCEEDED,
            depth=0,
            contract_json=contract,
            feedback_json={
                "aggregate_metrics": {
                    "primary_metric": "mAP50_95",
                    "seed_count": 1,
                    "aggregate_metrics": {
                        "mAP50_95": {
                            "mean": 0.4,
                            "std": 0.0,
                            "min": 0.4,
                            "max": 0.4,
                        }
                    },
                }
            },
            created_at=now,
            updated_at=now,
        )
    )
    service.repo.upsert_attempt(
        ExecutionAttempt(
            execution_id="exec_api_report",
            node_id="rgbt_formal_node_api",
            attempt_index=1,
            runner_profile="local",
            status=JobStatus.COMPLETED,
            image_reference="scientist-rgbt-detection:v2",
            code_version="image:rgbt-detection-v2",
            dataset_version="dataset:rgbt_fast_eval_v1",
            result_json={
                "metrics": {
                    "primary_metric": "mAP50_95",
                    "metrics": {"mAP50_95": 0.4},
                },
                "contract": {**contract, "seed": 1},
            },
            created_at=now,
            updated_at=now,
        )
    )

    report = client.post(
        "/api/v1/reports/build",
        json={"project_id": "project_rgbt_003"},
    )
    assert report.status_code == 200, report.text
    report_id = report.json()["report_id"]

    audit = client.post(
        "/api/v1/audits/build",
        json={"project_id": "project_rgbt_003", "report_id": report_id},
    )
    assert audit.status_code == 200, audit.text
    bundle_id = audit.json()["bundle_id"]

    release = client.post(
        "/api/v1/releases",
        json={
            "project_id": "project_rgbt_003",
            "title": "with audit",
            "report_id": report_id,
            "audit_bundle_id": bundle_id,
        },
    ).json()
    client.post(f"/api/v1/releases/{release['release_id']}/freeze", json={})

    export_dir = tmp_path / "export_out"
    exported = client.post(
        f"/api/v1/audits/{bundle_id}/export",
        json={
            "output_dir": str(export_dir),
            "release_id": release["release_id"],
        },
    )
    assert exported.status_code == 200, exported.text
    body = exported.json()
    assert Path(body["exported_to"]).is_dir()
    assert body["release"]["status"] == "exported"
    assert body["release"]["main_workspace_modified"] is False
