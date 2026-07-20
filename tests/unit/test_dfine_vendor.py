from __future__ import annotations

import sys
from pathlib import Path

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.baselines.dfine_s import DFineSBaselineAdapter


ROOT = Path(__file__).resolve().parents[2]
REAL_APP = ROOT / "experiment_apps" / "rgbt_detection_real"


def test_dfine_vendor_present():
    vendor = ROOT / "third_party" / "DFINE" / "train.py"
    assert vendor.is_file()
    pin = (ROOT / "third_party" / "VENDOR.md").read_text(encoding="utf-8")
    assert "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6" in pin


def test_dfine_dataset_stage_and_config(tmp_path: Path):
    sys.path.insert(0, str(REAL_APP))
    from dfine_config_builder import write_dfine_fast_config
    from dfine_dataset_stage import count_categories, stage_coco_for_dfine

    data_root = ROOT / "datasets" / "rgbt_fast_eval_v1"
    if not data_root.exists():
        import pytest

        pytest.skip("rgbt_fast_eval_v1 missing")
    stage = stage_coco_for_dfine(data_root, tmp_path / "stage", input_mode="rgb")
    assert stage["train_img"].is_dir()
    assert any(stage["train_img"].glob("*.jpg"))
    n_cls = count_categories(stage["train_ann"])
    assert n_cls >= 1
    cfg = write_dfine_fast_config(
        dfine_root=ROOT / "third_party" / "DFINE",
        config_path=tmp_path / "cfg.yml",
        stage_paths=stage,
        output_dir=tmp_path / "out",
        epochs=1,
        batch_size=2,
        num_workers=0,
        image_size=160,
        learning_rate=2e-4,
        num_classes=n_cls,
        seed=42,
    )
    text = cfg.read_text(encoding="utf-8")
    assert "pretrained: False" in text
    assert "num_classes:" in text


def test_dfine_adapter_reports_vendored():
    adapter = DFineSBaselineAdapter()
    contract = ExperimentContract.model_validate(
        {
            "schema_version": "1.2",
            "project_id": "project_rgbt_002",
            "node_id": "n1",
            "title": "t",
            "research_goal": "g",
            "hypothesis": "h",
            "task_type": "rgbt_detection",
            "runner_profile": "local",
            "environment_key": "rgbt-detection-v2-cuda",
            "code_reference": "local:rgbt_detection_real",
            "dataset_reference": "dataset:rgbt_fast_eval_v1",
            "entrypoint": "run_detection_experiment.py",
            "execution_mode": "fast_eval",
            "parameters": {"baseline": "dfine_s", "epochs": 1, "input_mode": "rgb"},
            "task_config": {"claim_level": "exploratory_comparison"},
            "seed": 42,
            "resources": {
                "gpu_count": 1,
                "cpu_count": 2,
                "memory_gb": 4,
                "timeout_seconds": 600,
            },
            "expected_outputs": ["metrics.json"],
        }
    )
    native = adapter.build_native_config(contract)
    assert native["baseline_key"] == "dfine_s"
    assert "vendored" in native["vendor_status"]
