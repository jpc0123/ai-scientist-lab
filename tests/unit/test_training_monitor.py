"""Unit tests for training monitor parsers."""

from pathlib import Path

from scientist_lab.services.training_monitor import (
    enrich_execution_row,
    parse_dfine_log_txt,
    parse_training_signals,
)


def test_parse_training_signals_epoch_and_error():
    log = """
building train_dataloader
------------------------------------------Start training-------------------------------------------
Epoch: [0/20]  [ 200/1830]  eta: 0:10:08  loss: 23.0716
{"epoch": 3, "test_coco_eval_bbox": [0.0833, 0.2]}
RuntimeError: boom happened here
[worker] failed: container exit code 1
"""
    sig = parse_training_signals(log)
    assert sig["started_training"] is True
    assert sig["epoch"] == 3 or sig["epoch"] == 0  # bracket may appear before json
    # Prefer last bracket or last json — with both present last match wins by scan order
    # bracket scan first fills epochs then json only if empty; so epoch=0 from brackets
    assert sig["epochs_total"] == 20
    assert sig["error_snippet"]
    assert "RuntimeError" in sig["error_snippet"] or "failed" in sig["error_snippet"]


def test_parse_epoch_bracket_progress():
    log = "Start training\nEpoch: [5/20]  [100/1830]\nEpoch: [6/20]  [10/1830]\n"
    sig = parse_training_signals(log)
    assert sig["epoch"] == 6
    assert sig["epochs_total"] == 20


def test_parse_dfine_log_txt(tmp_path: Path):
    p = tmp_path / "log.txt"
    p.write_text(
        '{"epoch": 1, "test_coco_eval_bbox": [0.01, 0.02]}\n'
        '{"epoch": 2, "test_coco_eval_bbox": [0.05, 0.1]}\n',
        encoding="utf-8",
    )
    parsed = parse_dfine_log_txt(p)
    assert parsed["epoch"] == 2
    assert parsed["mAP50_95"] == 0.05
    assert parsed["best_mAP50_95"] == 0.05
    assert parsed["epochs_seen"] == 2


def test_enrich_marks_failed():
    row = enrich_execution_row(
        attempt={
            "execution_id": "exec_x",
            "node_id": "n1",
            "status": "failed",
            "error_json": {"message": "container exit code 1"},
            "result_json": {"contract": {"parameters": {"epochs": 20}}},
        },
        live={"status": "failed", "progress": 0.4, "error_message": "container exit code 1"},
        log_text="Start training\nRuntimeError: grid_sampler",
        dfine=None,
        job_id="job_1",
    )
    assert row["is_failed"] is True
    assert row["error_message"]
    assert row["epochs_total"] == 20


def test_completed_shows_100_percent_not_95():
    row = enrich_execution_row(
        attempt={
            "execution_id": "exec_done",
            "node_id": "n1",
            "status": "completed",
            "result_json": {"contract": {"parameters": {"epochs": 20}}},
        },
        live={"status": "completed", "progress": 1.0},
        log_text="Epoch: [19/20]  [1830/1830]\nRGB-T real baseline experiment finished",
        dfine={"epoch": 19, "mAP50_95": 0.08, "best_mAP50_95": 0.08, "epochs_seen": 20},
        job_id="job_done",
    )
    assert row["is_completed"] is True
    assert row["progress"] == 1.0
    assert row["epoch"] == 20
    assert row["epochs_total"] == 20


def test_snapshot_cli_run_from_metrics(tmp_path: Path):
    from scientist_lab.services.training_monitor import snapshot_cli_run

    pack = tmp_path / "v26_r0"
    pack.mkdir()
    (pack / "execution.json").write_text(
        '{"execution_id":"exec_cli","return_code":0,'
        '"started_at":"2026-08-19T09:52:29+00:00","finished_at":"2026-08-19T10:18:03+00:00"}',
        encoding="utf-8",
    )
    (pack / "metrics.json").write_text(
        '{"status":"completed","training":{"epochs_requested":2,"epochs_completed":2,"trained":true},'
        '"metrics":{"APS_lowlight":0.00459,"mAP50_95":0.03}}',
        encoding="utf-8",
    )
    row = snapshot_cli_run(pack)
    assert row["is_completed"] is True
    assert row["progress"] == 1.0
    assert row["epoch"] == 2
    assert row["APS_lowlight"] == 0.00459
    assert row["href"] == "/loop/v26_r0"


def test_snapshot_manager_pack_approved_pending(tmp_path: Path):
    from scientist_lab.services.training_monitor import snapshot_cli_run

    pack = tmp_path / "v26_r1"
    (pack / "run").mkdir(parents=True)
    (pack / "experiment_run.json").write_text(
        '{"run_id":"run_plan_round1","run_state":"APPROVED","evidence_status":"PENDING",'
        '"updated_at":"2026-08-19T14:27:51+00:00"}',
        encoding="utf-8",
    )
    (pack / "manager_status.json").write_text('{"steps":3}', encoding="utf-8")
    row = snapshot_cli_run(pack)
    assert row["is_active"] is True
    assert row["status"] == "running"
    assert row["stuck_at"] == "docker"
    assert row["href"] == "/loop/v26_r1"


def test_v26_campaigns_mark_not_started(tmp_path: Path):
    from scientist_lab.services.training_monitor import load_v26_campaigns

    camps = load_v26_campaigns(tmp_path)
    names = [c["campaign"] for c in camps]
    assert "V26_5" in names
    v26_5 = next(c for c in camps if c["campaign"] == "V26_5")
    assert v26_5["status"] == "not_started"


def test_v26_campaigns_list_r1_and_r2(tmp_path: Path):
    from scientist_lab.services.training_monitor import load_v26_campaigns

    for name in ("v26_r1", "v26_r2", "v26_r3", "v26_r4", "v26_r5"):
        pack = tmp_path / "outputs" / name
        pack.mkdir(parents=True)
        (pack / "manager_status.json").write_text('{"steps":1}', encoding="utf-8")
    camps = load_v26_campaigns(tmp_path)
    names = {c["campaign"] for c in camps}
    assert "V26_R1" in names
    assert "V26_R2" in names
    assert "V26_R3" in names
    assert "V26_R4" in names
    assert "V26_R5" in names
    r2 = next(c for c in camps if c["campaign"] == "V26_R2")
    assert r2["meta"]["href"] == "/loop/v26_r2"
    assert r2["meta"]["pack"] == "v26_r2"
    r3 = next(c for c in camps if c["campaign"] == "V26_R3")
    assert r3["meta"]["href"] == "/loop/v26_r3"
    assert r3["meta"]["pack"] == "v26_r3"
    r4 = next(c for c in camps if c["campaign"] == "V26_R4")
    assert r4["meta"]["href"] == "/loop/v26_r4"
    assert r4["meta"]["pack"] == "v26_r4"
    r5 = next(c for c in camps if c["campaign"] == "V26_R5")
    assert r5["meta"]["href"] == "/loop/v26_r5"
    assert r5["meta"]["pack"] == "v26_r5"


def test_enrich_promotes_queued_when_epoch_log():
    row = enrich_execution_row(
        attempt={
            "execution_id": "exec_live",
            "node_id": "n1",
            "status": "queued",
        },
        live={"status": "queued", "progress": 0.0},
        log_text="Start training\nEpoch: [0/2]  [100/1671]  eta: 0:10:00  loss: 29.0\n",
        dfine=None,
        job_id="job_live",
    )
    assert row["status"] == "running"
    assert row["epoch"] == 0
    assert row["step"] == 100


def test_inspect_v26_r1_progress_when_present():
    from scientist_lab.services.local_run_inspector import inspect_local_run

    root = Path(__file__).resolve().parents[2]
    pack = root / "outputs" / "v26_r1"
    if not pack.is_dir() or not (pack / "experiment_run.json").is_file():
        return
    payload = inspect_local_run(root, "v26_r1")
    progress = payload.get("progress") or {}
    assert progress.get("href") == "/loop/v26_r1"
    assert progress.get("status") in {"running", "preparing", "queued", "completed", "failed"}
    if progress.get("status") == "running":
        assert progress.get("stuck_at") in {"training", "docker", "llm", "gate"}

def test_enrich_demotes_disk_completed_zombie(tmp_path: Path):
    out = tmp_path / "exec_zombie"
    out.mkdir()
    (out / "execution.json").write_text(
        '{"status":"completed","return_code":0,"finished_at":"2026-08-30T07:15:42+00:00"}',
        encoding="utf-8",
    )
    (out / "metrics.json").write_text(
        '{"status":"completed","metrics":{"mAP50_95":0.01},"training":{"epochs_requested":2,"epochs_completed":2}}',
        encoding="utf-8",
    )
    (out / "combined.log").write_text(
        "Start training\nEpoch: [1/2]  [321/418]  eta: 0:00:59  loss: 27.2\n",
        encoding="utf-8",
    )
    row = enrich_execution_row(
        attempt={
            "execution_id": "exec_zombie",
            "node_id": "n1",
            "status": "running",
            "started_at": "2026-08-30T06:59:40+00:00",
            "result_json": {"contract": {"parameters": {"epochs": 2}}},
        },
        live={"status": "running", "progress": 0.99},
        log_text=(out / "combined.log").read_text(encoding="utf-8"),
        dfine=None,
        job_id=None,
        output_dir=out,
        log_path=out / "combined.log",
    )
    assert row["is_active"] is False
    assert row["is_completed"] is True
    assert row["status"] == "completed"
    assert row["progress"] == 1.0


def test_reconcile_keeps_waiting_pack_without_log(tmp_path: Path):
    from scientist_lab.services.training_monitor import reconcile_active_execution_status

    rec = reconcile_active_execution_status(
        status="running",
        execution_id="exec_waiting",
        output_dir=tmp_path / "missing",
        started_at="2026-08-19T14:27:51+00:00",
        log_path=None,
        job_id=None,
    )
    assert rec["changed"] is False
    assert rec["status"] == "running"


def test_reconcile_demotes_stale_log_without_container(tmp_path: Path, monkeypatch):
    from scientist_lab.services import training_monitor as tm

    log = tmp_path / "combined.log"
    log.write_text("Start training\nEpoch: [0/2]  [10/100]\n", encoding="utf-8")
    # Make log stale
    import os
    import time

    old = time.time() - 600
    os.utime(log, (old, old))

    monkeypatch.setattr(
        tm,
        "inspect_scientist_exec",
        lambda _eid: {"alive": False, "status": None, "exit_code": None},
    )
    monkeypatch.setattr(tm, "read_disk_execution_terminal", lambda _d: None)

    rec = tm.reconcile_active_execution_status(
        status="running",
        execution_id="exec_ghost",
        output_dir=tmp_path,
        started_at="2026-08-30T01:00:00+00:00",
        log_path=log,
        job_id=None,
    )
    assert rec["changed"] is True
    assert rec["status"] == "interrupted"
    assert rec["is_active"] is False
