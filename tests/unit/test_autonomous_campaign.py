"""P0 autonomous campaign: one Human Gate, two Manager rounds, no GPU."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.api.v1.autonomous_campaigns import _campaigns
from scientist_lab.core.schema_registry import SCHEMA_DIR
from scientist_lab.services.autonomous_campaign import AutonomousCampaignService
from scientist_lab.services.registered_experiments import BUILTIN_EXPERIMENT_ID
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stub_runner(
    calls: list,
    *,
    gate: threading.Event | None = None,
    hold_from_call: int | None = None,
    pause_after_call: int | None = None,
):
    hold = threading.Event()
    if hold_from_call is not None:
        hold.set()

    def runner(contract, output_dir):
        if hold_from_call is not None and len(calls) >= hold_from_call:
            hold.wait(timeout=45)
        if gate is not None:
            gate.wait(timeout=5)
        n = len(calls)
        how = ((contract.get("materialization") or {}).get("how") or {}).get("how_id")
        calls.append(
            {
                "run_id": contract.get("run_id"),
                "plan_id": contract.get("plan_id"),
                "how_id": how,
                "seed": contract.get("seed"),
            }
        )
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        aps = 0.021 if n == 0 else 0.012
        _write_json(
            dest / "metrics.json",
            {
                "APS_lowlight": aps,
                "mAP50_95_lowlight": aps * 0.7,
                "AP50_lowlight": aps * 2,
                "APS": aps,
                "mAP50_95": aps,
            },
        )
        _write_json(dest / "checkpoint_selection.json", {"best_epoch": 1})
        if pause_after_call is not None and len(calls) >= pause_after_call:
            time.sleep(1.5)
        return {"status": "completed"}

    runner.release = hold.set  # type: ignore[attr-defined]
    return runner


@pytest.fixture()
def service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    return ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "api.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )


def test_execute_without_human_gate_is_refused(tmp_path: Path) -> None:
    svc = AutonomousCampaignService(project_root=tmp_path)
    with pytest.raises(PermissionError, match="Human Gate"):
        svc.start(
            experiment_id=BUILTIN_EXPERIMENT_ID,
            execute=True,
            confirm_human_gate=False,
            background=False,
        )


def test_execute_without_live_ready_fail_closed(tmp_path: Path) -> None:
    svc = AutonomousCampaignService(project_root=tmp_path)
    payload = svc.start(
        experiment_id=BUILTIN_EXPERIMENT_ID,
        execute=True,
        confirm_human_gate=True,
        live_ready=False,
        background=False,
    )
    assert payload["ok"] is False
    assert payload["fail_closed"] is True
    assert payload["metrics_forged"] is False
    assert "live_ready" in str(payload["error"])
    assert not list(tmp_path.joinpath(".run", "autonomous").glob("p0_*"))


def test_two_stub_rounds_after_one_gate(tmp_path: Path) -> None:
    calls: list = []
    svc = AutonomousCampaignService(project_root=tmp_path)
    svc.bind_live_runner(_stub_runner(calls))
    payload = svc.start(
        experiment_id=BUILTIN_EXPERIMENT_ID,
        execute=True,
        confirm_human_gate=True,
        max_extra_rounds=1,
        llm_live=False,
        planner_backend="rules",
        reviewer_backend="rules",
        background=False,
        baseline_metrics={"APS_lowlight": 0.00459, "APS": 0.00459},
    )
    assert payload["ok"] is True
    assert payload["status"] == "completed"
    assert payload["experiment_id"] == BUILTIN_EXPERIMENT_ID
    assert payload["metrics_forged"] is False
    assert payload["confirm_human_gate"] is True
    assert len(calls) == 2
    assert payload["gpu_rounds"] == 2
    assert calls[0]["how_id"] == "F1"
    assert calls[1]["how_id"] == "F3"
    assert calls[1]["seed"] == 42
    actions = [row["action"] for row in payload["steps"]]
    assert "NEED_PLAN" in actions
    assert "NEED_EXECUTION" in actions
    assert "NEED_REVIEW" in actions
    assert "NEED_MEMORY" in actions
    assert "NEXT_ROUND" in actions
    reviews = [row for row in payload["steps"] if row["action"] == "NEED_REVIEW"]
    assert len(reviews) == 2
    r0 = reviews[0]["report"]["review"]["objective_check"]["APS_lowlight"]
    r1 = reviews[1]["report"]["review"]["objective_check"]["APS_lowlight"]
    assert r0["delta"] == pytest.approx(0.021 - 0.00459)
    assert r1["delta"] is not None
    assert r1["baseline"] == pytest.approx(0.021)
    assert r1["current"] == pytest.approx(0.012)
    assert r1["delta"] == pytest.approx(0.012 - 0.021)
    work = tmp_path / ".run" / "autonomous"
    campaign_dirs = list(work.glob("exp_rgbt_dfine_v26_lowlight_*"))
    assert campaign_dirs
    previous_round = json.loads(
        (campaign_dirs[0] / "previous_round_metrics.json").read_text(encoding="utf-8")
    )
    assert previous_round["APS_lowlight"] == pytest.approx(0.012)
    scout = (payload.get("how_pending") or {}).get("scout") or {}
    assert scout.get("query")
    assert scout.get("source") in {"human", "llm", "fallback"}
    assert scout.get("intent_source") in {"human", "llm", "fallback"}
    assert scout.get("fail_closed") is False
    assert scout.get("live") is False
    assert scout.get("papers")
    assert scout.get("literature_query_id")
    assert scout.get("can_enter_claim_gate") is False


def test_http_start_returns_before_gpu_finishes(service: ExperimentService) -> None:
    calls: list = []
    gate = threading.Event()
    campaigns = _campaigns(service)
    campaigns.bind_live_runner(_stub_runner(calls, gate=gate))
    client = TestClient(create_app(service=service))

    refused = client.post(
        "/api/v1/autonomous-campaigns",
        json={
            "experiment_id": BUILTIN_EXPERIMENT_ID,
            "execute": True,
            "confirm_human_gate": False,
        },
    )
    assert refused.status_code == 403

    busy = Path(service.settings.project_root) / ".run" / "autonomous" / "p0_busy"
    busy.mkdir(parents=True, exist_ok=True)
    _write_json(
        busy / "campaign.json",
        {
            "campaign_id": "p0_busy",
            "status": "running",
            "execute": True,
            "planner_backend": "llm",
            "gpu_rounds": 1,
            "worker_pid": os.getpid(),
        },
    )
    blocked = client.post(
        "/api/v1/autonomous-campaigns",
        json={
            "experiment_id": BUILTIN_EXPERIMENT_ID,
            "execute": True,
            "confirm_human_gate": True,
            "max_extra_rounds": 1,
            "planner_backend": "rules",
            "reviewer_backend": "rules",
        },
    )
    assert blocked.status_code == 409, blocked.text
    err = blocked.json()["error"]
    assert err["type"] == "live_gpu_busy"
    assert "p0_busy" in json.dumps(err.get("details") or {}, ensure_ascii=False)
    (busy / "campaign.json").unlink()

    started = time.monotonic()
    resp = client.post(
        "/api/v1/autonomous-campaigns",
        json={
            "experiment_id": BUILTIN_EXPERIMENT_ID,
            "execute": True,
            "confirm_human_gate": True,
            "max_extra_rounds": 1,
            "planner_backend": "rules",
            "reviewer_backend": "rules",
        },
    )
    elapsed = time.monotonic() - started
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert elapsed < 1.5
    assert body["status"] in {"queued", "running", "waiting_gpu"}
    campaign_id = body["campaign_id"]
    assert calls == []

    gate.set()
    deadline = time.monotonic() + 20
    final = body
    while time.monotonic() < deadline:
        final = client.get(f"/api/v1/autonomous-campaigns/{campaign_id}").json()
        if final.get("status") in {"completed", "blocked", "failed"}:
            break
        time.sleep(0.05)
    assert final["status"] == "completed", final.get("error")
    assert len(calls) == 2
    scout = (final.get("how_pending") or {}).get("scout") or {}
    assert scout.get("query")
    assert scout.get("source") in {"human", "llm", "fallback"}
    assert "literature_query_id" in scout
    assert scout.get("papers") or scout.get("error")
    assert scout.get("can_enter_claim_gate") is False
    assert final["gpu_rounds"] == 2
    assert final["metrics_forged"] is False


def test_overlapping_live_campaigns_return_409(tmp_path: Path) -> None:
    """Second execute=true start must refuse; CLI manager-run must not silently run."""
    from scientist_lab.core.manager_cli import run_manager_from_files
    from scientist_lab.services.live_gpu_mutex import (
        LiveGpuBusyError,
        acquire_live_execute,
    )

    lease = acquire_live_execute(tmp_path, owner_id="owner-a", inspect_docker=False)
    with pytest.raises(LiveGpuBusyError, match="拒绝"):
        acquire_live_execute(tmp_path, owner_id="owner-b", inspect_docker=False)
    lease.release()

    busy = tmp_path / ".run" / "autonomous" / "p0_busy"
    busy.mkdir(parents=True, exist_ok=True)
    _write_json(
        busy / "campaign.json",
        {
            "campaign_id": "p0_busy",
            "status": "running",
            "execute": True,
            "planner_backend": "llm",
            "gpu_rounds": 1,
            "worker_pid": os.getpid(),
        },
    )
    svc = AutonomousCampaignService(project_root=tmp_path)
    svc.bind_live_runner(_stub_runner([]))
    with pytest.raises(LiveGpuBusyError, match="p0_busy"):
        svc.start(
            experiment_id=BUILTIN_EXPERIMENT_ID,
            execute=True,
            confirm_human_gate=True,
            background=True,
            planner_backend="rules",
            reviewer_backend="rules",
        )

    cli = run_manager_from_files(
        SCHEMA_DIR / "examples" / "research_protocol_rgbt_dfine_v26.json",
        SCHEMA_DIR / "examples" / "experiment_plan_rgbt_dfine_v26_r0.json",
        output_dir=tmp_path / "cli_overlap",
        execute=True,
        confirm_human_gate=True,
        project_root=tmp_path,
        inspect_docker=False,
    )
    assert cli["ok"] is False
    assert cli["exit_code"] == 1
    assert cli["status"] == "blocked"
    assert cli["metrics_forged"] is False


def test_orphan_running_campaign_is_reclaimed_and_does_not_block(tmp_path: Path) -> None:
    """API restart left status=running with a dead/missing worker_pid — not occupancy."""
    from scientist_lab.services.live_gpu_mutex import acquire_live_execute

    dead = tmp_path / ".run" / "autonomous" / "p0_dead"
    dead.mkdir(parents=True, exist_ok=True)
    _write_json(
        dead / "campaign.json",
        {
            "campaign_id": "p0_dead",
            "status": "running",
            "execute": True,
            "planner_backend": "llm",
            "gpu_rounds": 2,
            "last_action": "NEED_MEMORY",
        },
    )
    svc = AutonomousCampaignService(project_root=tmp_path)
    paused = json.loads((dead / "campaign.json").read_text(encoding="utf-8"))
    assert paused["status"] == "paused"
    assert paused["orphan_reclaimed"] is True

    lease = acquire_live_execute(tmp_path, owner_id="owner-after-orphan", inspect_docker=False)
    lease.release()

    svc.bind_live_runner(_stub_runner([]))
    payload = svc.start(
        experiment_id=BUILTIN_EXPERIMENT_ID,
        execute=True,
        confirm_human_gate=True,
        max_extra_rounds=1,
        background=False,
        planner_backend="rules",
        reviewer_backend="rules",
        baseline_metrics={"APS_lowlight": 0.00459, "APS": 0.00459},
    )
    assert payload["status"] == "completed"
    assert payload["campaign_id"] != "p0_dead"


def test_request_stop_on_orphan_goes_to_paused(tmp_path: Path) -> None:
    """Stop must release the mutex when the worker thread is already gone."""
    stuck = tmp_path / ".run" / "autonomous" / "p0_stuck"
    stuck.mkdir(parents=True, exist_ok=True)
    _write_json(
        stuck / "campaign.json",
        {
            "campaign_id": "p0_stuck",
            "status": "running",
            "execute": True,
            "gpu_rounds": 2,
        },
    )
    svc = AutonomousCampaignService(project_root=tmp_path)
    # init already reclaimed; rewrite as running to simulate stop-before-reclaim
    _write_json(
        stuck / "campaign.json",
        {
            "campaign_id": "p0_stuck",
            "status": "running",
            "execute": True,
            "gpu_rounds": 2,
        },
    )
    stopped = svc.request_stop("p0_stuck")
    assert stopped["status"] == "paused"
    assert stopped["stop_requested"] is True
    assert stopped["orphan_reclaimed"] is True


def test_resume_paused_campaign_continues_third_round(tmp_path: Path) -> None:
    calls: list = []
    svc = AutonomousCampaignService(project_root=tmp_path)
    svc.bind_live_runner(_stub_runner(calls))
    payload = svc.start(
        experiment_id=BUILTIN_EXPERIMENT_ID,
        execute=True,
        confirm_human_gate=True,
        max_extra_rounds=1,
        llm_live=False,
        planner_backend="rules",
        reviewer_backend="rules",
        background=False,
        baseline_metrics={"APS_lowlight": 0.00459, "APS": 0.00459},
    )
    assert payload["status"] == "completed"
    assert payload["gpu_rounds"] == 2
    assert len(calls) == 2

    cid = payload["campaign_id"]
    path = tmp_path / ".run" / "autonomous" / cid / "campaign.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["status"] = "paused"
    spec["stop_requested"] = True
    spec["orphan_reclaimed"] = True
    spec["max_extra_rounds"] = 2
    spec["last_action"] = "NEED_MEMORY"
    spec.pop("worker_pid", None)
    _write_json(path, spec)

    resumed = svc.resume(cid, live_ready=True, background=False)
    assert resumed["status"] == "completed", resumed.get("error")
    assert resumed["gpu_rounds"] == 3
    assert len(calls) == 3
    assert resumed.get("resume_count") == 1


def test_http_resume_rejects_non_paused(tmp_path: Path, service: ExperimentService) -> None:
    campaigns = _campaigns(service)
    campaigns.bind_live_runner(_stub_runner([]))
    running = Path(service.settings.project_root) / ".run" / "autonomous" / "p0_running"
    running.mkdir(parents=True, exist_ok=True)
    _write_json(
        running / "campaign.json",
        {
            "campaign_id": "p0_running",
            "status": "running",
            "execute": True,
            "confirm_human_gate": True,
            "gpu_rounds": 1,
        },
    )
    client = TestClient(create_app(service=service))
    resp = client.post("/api/v1/autonomous-campaigns/p0_running/resume")
    assert resp.status_code == 400
    assert "paused" in resp.json()["error"]["message"].lower()

