"""Local run inspector API: read-only packs + fail-closed actions."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.local_run_inspector import inspect_local_run, list_local_runs
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
    return TestClient(app), service, root


def test_list_includes_fixtures(api_client):
    client, _, _ = api_client
    resp = client.get("/api/v1/local-runs")
    assert resp.status_code == 200
    data = resp.json()
    ids = {row["id"] for row in data["items"]}
    assert "fixture_m4_rounds3_discard" in ids
    assert "fixture_discard_stub" in ids
    fixture = next(row for row in data["items"] if row["id"] == "fixture_m4_rounds3_discard")
    assert fixture["available"] is True
    assert data["narrative"]["claim"] == "KEEP ≠ Claim"


def test_inspect_fixture_is_engineering_not_c1(api_client):
    client, _, _ = api_client
    resp = client.get("/api/v1/local-runs/fixture_m4_rounds3_discard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["c1"]["allowed"] is False
    assert data["c1"]["engineering_not_claim"] is True
    assert data["c1"].get("show_supported") is False
    assert data["rubric"]["keep_is_not_claim"] is True
    stage_ids = [row["id"] for row in data["loop"]]
    assert stage_ids == [
        "protocol",
        "gate",
        "run",
        "evidence",
        "rubric",
        "memory",
        "next_plan",
    ]
    assert data["rubric"]["review_decision"] == "DISCARD"


def test_inspect_formal_c1_when_present():
    root = Path(__file__).resolve().parents[2]
    pack = root / ".run" / "formal_c1_aps_early_concat"
    if not pack.is_dir():
        pytest.skip(".run/formal_c1_aps_early_concat not on this machine")
    payload = inspect_local_run(root, "formal_c1_aps_early_concat")
    assert payload["c1"]["allowed"] is True
    assert payload["c1"]["show_supported"] is True
    assert payload["c1"]["keep_is_not_claim"] is True
    assert payload["c1"]["baseline_aps_display"] == 0.0163
    assert payload["c1"]["candidate_aps_display"] == 0.0326
    assert payload["claim_gate"]["status"] == "SUPPORTED"
    assert payload["rubric"]["review_decision"] == "KEEP"


def test_inspect_probe_loop_never_paints_supported():
    root = Path(__file__).resolve().parents[2]
    pack = root / ".run" / "v25d_llm_real_loop"
    if not pack.is_dir():
        pytest.skip(".run/v25d_llm_real_loop not on this machine")
    payload = inspect_local_run(root, "v25d_llm_real_loop")
    assert payload["c1"]["allowed"] is False
    assert payload["c1"]["engineering_not_claim"] is True
    assert payload["c1"]["aps_is_zero"] is True
    assert payload["c1"]["show_supported"] is False
    assert payload["claim_gate"]["status"] != "SUPPORTED"
    assert payload["how"]["primary_module"] == "fusion"
    assert payload["llm"]["decision_summary"]


def test_execute_without_confirm_is_rejected(api_client):
    client, _, _ = api_client
    resp = client.post(
        "/api/v1/local-runs/fixture_m4_rounds3_discard/actions",
        json={"action": "manager_run", "execute": True, "confirm_execute": False},
    )
    assert resp.status_code == 403
    body = resp.json()["error"]
    assert "二次确认" in body["message"]


def test_execute_fail_closed_when_doctor_not_ready(api_client, monkeypatch):
    client, service, _ = api_client

    def fake_doctor(*, probe_runtime: bool = False):
        return {"live_ready": False, "overall": "error"}

    monkeypatch.setattr(service, "dfine_cuda_doctor", fake_doctor)
    resp = client.post(
        "/api/v1/local-runs/fixture_m4_rounds3_discard/actions",
        json={
            "action": "manager_run",
            "execute": True,
            "confirm_execute": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert data["fail_closed"] is True
    assert data["metrics_forged"] is False
    assert "live_ready=false" in data["error"]


def test_read_allowlisted_file(api_client):
    client, _, _ = api_client
    resp = client.get(
        "/api/v1/local-runs/fixture_m4_rounds3_discard/file",
        params={"name": "protocol.json"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["protocol_id"]

    denied = client.get(
        "/api/v1/local-runs/fixture_m4_rounds3_discard/file",
        params={"name": "../secrets.env"},
    )
    assert denied.status_code == 403


def test_list_helper_does_not_require_dot_run():
    root = Path(__file__).resolve().parents[2]
    listed = list_local_runs(root)
    ids = {row["id"] for row in listed["items"]}
    assert "fixture_discard_stub" in ids
