"""Unit tests for formal checkpoint selection policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "experiment_apps" / "rgbt_detection_real"
sys.path.insert(0, str(APP))

from checkpoint_policy import (  # noqa: E402
    parse_checkpoint_policy,
    select_best_and_last_from_dfine_log,
)
from disk_gate import assert_disk_for_formal_train, disk_usage_gb  # noqa: E402


def test_parse_checkpoint_policy_defaults_and_rejects_bad_primary():
    policy = parse_checkpoint_policy({})
    assert policy["primary"] == "best_on_validation"
    assert policy["tie_breaker"] == "earliest_epoch"
    with pytest.raises(ValueError, match="best_on_validation"):
        parse_checkpoint_policy({"checkpoint_policy": {"primary": "last_epoch"}})


def test_best_on_val_uses_strict_gt_tie_keeps_earlier(tmp_path: Path):
    log = tmp_path / "log.txt"
    rows = [
        {"epoch": 0, "test_coco_eval_bbox": [0.20, 0.3, 0.1, -1.0], "train_loss": 1.0},
        {"epoch": 1, "test_coco_eval_bbox": [0.25, 0.4, 0.1, -1.0], "train_loss": 0.9},
        {"epoch": 2, "test_coco_eval_bbox": [0.25, 0.5, 0.1, -1.0], "train_loss": 0.8},
        {"epoch": 3, "test_coco_eval_bbox": [0.22, 0.4, 0.1, -1.0], "train_loss": 0.7},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    sel = select_best_and_last_from_dfine_log(tmp_path)
    assert sel["ok"] is True
    assert sel["best_epoch"] == 2  # epoch_0based=1 → display epoch 2
    assert sel["last_epoch"] == 4
    assert sel["best"]["mAP50_95"] == 0.25
    assert sel["best_minus_last_mAP50_95"] == pytest.approx(0.03)


def test_disk_usage_readable():
    info = disk_usage_gb(ROOT)
    assert info["free_gb"] >= 0
    # Should not raise on normal workspace.
    status = assert_disk_for_formal_train(ROOT, min_free_gb_start=0.001, min_free_gb_hard=0.0001)
    assert status["ok_to_start"] is True
