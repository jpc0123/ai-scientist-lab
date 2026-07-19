from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scientist_lab.datasets.models import DatasetRegistration
from scientist_lab.datasets.rgbt_validator import validate_rgbt_dataset
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.tasks.rgbt_detection.adapter import RGBTDetectionAdapter
from scientist_lab.tasks.rgbt_detection.report_compare import (
    compare_host_container_reports,
)


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


def _load_container_validator():
    path = _root() / "rgbt_detector" / "validate_dataset.py"
    spec = importlib.util.spec_from_file_location("rgbt_container_validate", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_adapter_validate_data_forbids_epochs(tmp_path: Path):
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
            "execution_mode": "validate_data",
            "parameters": {"epochs": 2},
        }
    )
    with pytest.raises(ValueError, match="forbids non-zero epochs"):
        RGBTDetectionAdapter().validate_contract(contract)


def test_adapter_mounts_are_read_only(tmp_path: Path):
    ds = _make_dataset(tmp_path / "ds")
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
            "execution_mode": "validate_data",
            "parameters": {"epochs": 0, "input_mode": "rgb"},
        }
    )
    dataset = DatasetRegistration(
        dataset_key="rgbt_debug_v1",
        task_type="rgbt_detection",
        host_path=str(ds),
        container_path="/datasets/rgbt_debug_v1",
        read_only=True,
        enabled=True,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    plan = RGBTDetectionAdapter().prepare_execution(
        contract,
        dataset,
        code_roots={"local:rgbt_detector": _root() / "rgbt_detector"},
    )
    assert plan.mounts
    assert all(m.read_only for m in plan.mounts)
    assert plan.mounts[0].target == "/datasets/rgbt_debug_v1"
    assert "validate_data" in " ".join(plan.notes)


def test_container_validator_matches_host_counts(tmp_path: Path):
    ds = _make_dataset(tmp_path / "ds", n_train=5, n_val=2)
    host = validate_rgbt_dataset(ds, dataset_key="rgbt_debug_v1", write_previews=False)
    container_mod = _load_container_validator()
    container = container_mod.validate_rgbt_in_container(
        ds, dataset_key="rgbt_debug_v1", probe_read_only=False
    )
    comparison = compare_host_container_reports(host, container)
    assert comparison["consistent"] is True, comparison["mismatches"]


def test_validate_data_script_no_training_artifacts(tmp_path: Path):
    ds = _make_dataset(tmp_path / "ds", n_train=4, n_val=1)
    out = tmp_path / "out"
    out.mkdir()
    (out / "contract.json").write_text(
        json.dumps(
            {
                "project_id": "project_rgbt_001",
                "node_id": "rgbt_node_validate",
                "seed": 42,
                "dataset_reference": "dataset:rgbt_debug_v1",
            }
        ),
        encoding="utf-8",
    )
    (out / "config.json").write_text("{}", encoding="utf-8")

    # Run container entry locally (same code path as Docker workspace).
    detector = _root() / "rgbt_detector"
    sys.path.insert(0, str(detector))
    try:
        import run_detection_experiment as entry

        # emulate argv
        sys.argv = [
            "run_detection_experiment.py",
            "--config",
            str(out / "config.json"),
            "--output-dir",
            str(out),
            "--data-root",
            str(ds),
            "--execution-mode",
            "validate_data",
            "--seed",
            "42",
        ]
        entry.main()
    finally:
        if str(detector) in sys.path:
            sys.path.remove(str(detector))

    assert (out / "dataset_report.json").exists()
    assert (out / "metrics.json").exists()
    assert (out / "artifact_manifest.json").exists()
    assert not (out / "training_history.csv").exists()
    assert not (out / "checkpoint").exists()
    report = json.loads((out / "dataset_report.json").read_text(encoding="utf-8"))
    assert report["trained"] is False
    assert report["execution_mode"] == "validate_data"


def test_disabled_dataset_rejected_for_run(tmp_path: Path):
    root = _root()
    settings = Settings(
        project_root=root,
        db_path=tmp_path / "t.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=root / "experiment_app",
        rgbt_detector_dir=root / "rgbt_detector",
    ).resolve()
    service = ExperimentService(settings=settings)
    ds = _make_dataset(tmp_path / "ds")
    service.register_dataset(
        dataset_key="rgbt_debug_v1",
        task_type="rgbt_detection",
        path=ds,
        container_path="/datasets/rgbt_debug_v1",
    )
    service.disable_dataset("rgbt_debug_v1")
    with pytest.raises(ValueError, match="禁用"):
        service.datasets.require("rgbt_debug_v1")


def test_report_compare_detects_mismatch():
    host = {
        "valid": True,
        "paired_image_count": 10,
        "rgb_image_count": 10,
        "thermal_image_count": 10,
        "claim_level": "pipeline_validation_only",
        "splits": {"train": {"rgb_count": 8, "thermal_count": 8, "paired_count": 8}},
        "missing_rgb": [],
        "missing_thermal": [],
    }
    container = dict(host)
    container["paired_image_count"] = 9
    result = compare_host_container_reports(host, container)
    assert result["consistent"] is False
    assert any("paired_image_count" in m for m in result["mismatches"])
