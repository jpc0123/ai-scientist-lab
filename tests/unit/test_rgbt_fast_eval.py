from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.adapter import RGBTDetectionAdapter


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_dataset(root: Path) -> Path:
    script = _root() / "scripts" / "create_rgbt_debug_dataset.py"
    spec = importlib.util.spec_from_file_location("create_rgbt_debug_dataset", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.create_dataset(root, n_train=4, n_val=2)
    return root


def _import_modules():
    detector = _root() / "rgbt_detector"
    if str(detector) not in sys.path:
        sys.path.insert(0, str(detector))
    import fast_eval
    import train_smoke

    return train_smoke, fast_eval


def test_adapter_requires_checkpoint_source():
    contract = ExperimentContract.model_validate(
        {
            "schema_version": "1.1",
            "project_id": "p",
            "node_id": "n",
            "task_type": "rgbt_detection",
            "environment_key": "rgbt-detection-v1",
            "code_reference": "local:rgbt_detector",
            "dataset_reference": "dataset:rgbt_debug_v1",
            "entrypoint": "run_detection_experiment.py",
            "execution_mode": "fast_eval",
            "parameters": {"epochs": 0, "input_mode": "rgb"},
        }
    )
    with pytest.raises(ValueError, match="checkpoint_source"):
        RGBTDetectionAdapter().validate_contract(contract)


def test_fast_eval_loads_checkpoint_without_training(tmp_path: Path):
    ds = _make_dataset(tmp_path / "ds")
    train_smoke, fast_eval = _import_modules()
    smoke_out = tmp_path / "smoke"
    smoke_out.mkdir()
    train_smoke.run_smoke_train(
        data_root=ds,
        output_dir=smoke_out,
        config={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.05,
            "max_train_images": 4,
            "max_val_images": 2,
            "smoke_image_width": 80,
            "smoke_image_height": 64,
        },
        contract={"project_id": "p", "node_id": "n_smoke"},
        seed=42,
        input_mode="rgb",
        fusion_method="none",
    )
    ckpt = smoke_out / "checkpoint" / "last.npz"
    assert ckpt.exists()

    eval_out = tmp_path / "eval"
    eval_out.mkdir()
    (eval_out / "checkpoint").mkdir()
    import shutil

    shutil.copy2(ckpt, eval_out / "checkpoint" / "last.npz")
    metrics = fast_eval.run_fast_eval(
        data_root=ds,
        output_dir=eval_out,
        config={
            "max_val_images": 2,
            "smoke_image_width": 80,
            "smoke_image_height": 64,
            "eval_split": "val",
        },
        contract={
            "project_id": "p",
            "node_id": "n_eval",
            "task_config": {"evaluation_scope": "debug_subset"},
        },
        seed=42,
        input_mode="rgb",
        fusion_method="none",
    )
    assert metrics["execution_mode"] == "fast_eval"
    assert metrics["training"]["epochs_completed"] == 0
    assert metrics["training"]["trained"] is False
    assert not (eval_out / "training_history.csv").exists()
    assert (eval_out / "sample_predictions.json").exists()
    summary = json.loads((eval_out / "model_summary.json").read_text(encoding="utf-8"))
    assert summary["loaded_for"] == "fast_eval"
