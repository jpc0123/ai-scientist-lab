from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REAL_APP = ROOT / "experiment_apps" / "rgbt_detection_real"
DATASET = ROOT / "datasets" / "rgbt_debug_v1"


pytest.importorskip("torch")


def test_torch_standin_minimal_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REAL_APP))
    from train_minimal import run_minimal_train

    if not DATASET.exists():
        pytest.skip("rgbt_debug_v1 dataset missing")

    metrics = run_minimal_train(
        data_root=DATASET,
        output_dir=tmp_path,
        config={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "image_width": 80,
            "image_height": 64,
            "max_train_images": 8,
            "max_val_images": 4,
        },
        contract={
            "project_id": "project_rgbt_002",
            "node_id": "rgbt_dfine_minimal_001",
            "execution_mode": "smoke_train",
            "task_config": {
                "claim_level": "pipeline_validation_only",
                "evaluation_scope": "debug_subset",
                "primary_metric": "mAP50_95",
            },
            "resources": {"cpu_count": 2, "memory_gb": 4, "gpu_count": 0},
        },
        seed=42,
        input_mode="rgb",
        fusion_method="none",
    )
    assert metrics["status"] == "completed"
    assert metrics["training"]["baseline_key"] == "dfine_s"
    assert metrics["training"]["epochs_completed"] == 1
    assert (tmp_path / "checkpoint" / "last.pt").exists()
    assert (tmp_path / "model_summary.json").exists()
    assert (tmp_path / "resource_usage.json").exists()
    summary = json.loads((tmp_path / "model_summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_implementation"] == "torch_mini_standin_v0_8_1"
