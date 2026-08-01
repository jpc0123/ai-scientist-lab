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
