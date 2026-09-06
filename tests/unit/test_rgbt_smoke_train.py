from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scientist_lab.tasks.rgbt_detection.verifier import DetectionVerifier


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_dataset(root: Path, *, n_train: int = 6, n_val: int = 2) -> Path:
    script = _root() / "scripts" / "create_rgbt_debug_dataset.py"
    spec = importlib.util.spec_from_file_location("create_rgbt_debug_dataset", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.create_dataset(root, n_train=n_train, n_val=n_val)
    return root


def _import_train_smoke():
    detector = _root() / "rgbt_detector"
    if str(detector) not in sys.path:
        sys.path.insert(0, str(detector))
    import train_smoke

    return train_smoke


def test_smoke_train_produces_artifacts(tmp_path: Path):
    ds = _make_dataset(tmp_path / "ds", n_train=6, n_val=2)
    out = tmp_path / "out"
    out.mkdir()
    train_smoke = _import_train_smoke()
    metrics = train_smoke.run_smoke_train(
        data_root=ds,
        output_dir=out,
        config={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.05,
            "max_train_images": 6,
            "max_val_images": 2,
            "smoke_image_width": 80,
            "smoke_image_height": 64,
        },
        contract={
            "project_id": "project_rgbt_001",
            "node_id": "rgbt_node_001",
        },
        seed=42,
        input_mode="rgb",
        fusion_method="none",
    )
    assert metrics["status"] == "completed"
    assert metrics["execution_mode"] == "smoke_train"
    assert (out / "metrics.json").exists()
    assert (out / "training_history.csv").exists()
    assert (out / "checkpoint" / "last.npz").exists()
    assert (out / "model_summary.json").exists()
    loss = metrics["training"]["final_train_loss"]
    assert loss == loss  # not NaN
    assert 0.0 <= metrics["metrics"]["mAP50_95"] <= 1.0


def test_smoke_train_thermal_and_fusion(tmp_path: Path):
    ds = _make_dataset(tmp_path / "ds", n_train=4, n_val=1)
    train_smoke = _import_train_smoke()
    for mode, fusion in (("thermal", "none"), ("rgbt", "early_concat")):
        out = tmp_path / f"out_{mode}"
        out.mkdir()
        metrics = train_smoke.run_smoke_train(
            data_root=ds,
            output_dir=out,
            config={
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.05,
                "max_train_images": 4,
                "max_val_images": 1,
                "smoke_image_width": 80,
                "smoke_image_height": 64,
            },
            contract={"project_id": "p", "node_id": f"n_{mode}"},
            seed=7,
            input_mode=mode,
            fusion_method=fusion,
        )
        assert (out / "checkpoint" / "last.npz").exists()
        assert metrics["training"]["epochs_completed"] == 1


def test_verifier_requires_smoke_checkpoint(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    for name in (
        "metrics.json",
        "artifact_manifest.json",
        "dataset_report.json",
        "execution.json",
    ):
        (out / name).write_text("{}", encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "primary_metric": "mAP50_95",
                "metrics": {"mAP50_95": 0.1},
                "execution_mode": "smoke_train",
                "training": {"final_train_loss": 1.2, "nan_loss": False},
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    result = DetectionVerifier().verify(out)
    assert result["valid"] is False
    assert any("checkpoint" in i or "training_history" in i for i in result["issues"])
