from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from scientist_lab.datasets.registry import (
    DatasetRegistry,
    is_dangerous_host_path,
    is_linux_absolute_path,
    parse_dataset_reference,
)
from scientist_lab.datasets.rgbt_validator import validate_rgbt_dataset
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.storage.artifact_store import sha256_file
from scientist_lab.tasks.rgbt_detection.feedback_rules import (
    annotate_feedback_for_detection,
    apply_detection_claim_gate,
)
from scientist_lab.tasks.rgbt_detection.result_parser import DetectionResultParser


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    settings = Settings(
        project_root=root,
        db_path=tmp_path / "test.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=root / "experiment_app",
        rgbt_detector_dir=root / "rgbt_detector",
    ).resolve()
    return ExperimentService(settings=settings)


def _make_dataset(
    root: Path,
    *,
    n_train: int = 8,
    n_val: int = 2,
    drop_thermal: str | None = None,
) -> Path:
    script = Path(__file__).resolve().parents[2] / "scripts" / "create_rgbt_debug_dataset.py"
    spec = importlib.util.spec_from_file_location("create_rgbt_debug_dataset", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.create_dataset(root, n_train=n_train, n_val=n_val)
    if drop_thermal:
        path = root / "images" / "train" / "thermal" / f"{drop_thermal}.jpg"
        if path.exists():
            path.unlink()
    return root


def test_parse_dataset_reference():
    assert parse_dataset_reference("dataset:rgbt_debug_v1") == "rgbt_debug_v1"
    assert parse_dataset_reference("sklearn:digits") is None


def test_linux_absolute_container_path():
    assert is_linux_absolute_path("/datasets/rgbt_debug_v1")
    assert not is_linux_absolute_path("datasets/rgbt")
    assert not is_linux_absolute_path("D:/datasets/rgbt")


def test_register_and_recover(tmp_path: Path):
    service = _service(tmp_path)
    ds = tmp_path / "data"
    _make_dataset(ds)
    payload = service.register_dataset(
        dataset_key="rgbt_debug_v1",
        task_type="rgbt_detection",
        path=ds,
        container_path="/datasets/rgbt_debug_v1",
    )
    assert payload["dataset_key"] == "rgbt_debug_v1"
    assert payload["read_only"] is True
    assert payload["enabled"] is True

    restarted = ExperimentService(settings=service.settings)
    listed = restarted.list_datasets()
    assert any(item["dataset_key"] == "rgbt_debug_v1" for item in listed)
    shown = restarted.show_dataset("rgbt_debug_v1")
    assert Path(shown["host_path"]) == ds.resolve()


def test_duplicate_key_rejected(tmp_path: Path):
    service = _service(tmp_path)
    ds = tmp_path / "data"
    _make_dataset(ds)
    service.register_dataset(
        dataset_key="rgbt_debug_v1", task_type="rgbt_detection", path=ds
    )
    with pytest.raises(ValueError, match="already registered"):
        service.register_dataset(
            dataset_key="rgbt_debug_v1", task_type="rgbt_detection", path=ds
        )


def test_missing_host_path_rejected(tmp_path: Path):
    service = _service(tmp_path)
    with pytest.raises(FileNotFoundError):
        service.register_dataset(
            dataset_key="missing",
            task_type="rgbt_detection",
            path=tmp_path / "no_such_dir",
        )


def test_file_path_rejected(tmp_path: Path):
    service = _service(tmp_path)
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="目录"):
        service.register_dataset(
            dataset_key="file_ds",
            task_type="rgbt_detection",
            path=file_path,
        )


def test_non_absolute_container_path_rejected(tmp_path: Path):
    service = _service(tmp_path)
    ds = tmp_path / "data"
    _make_dataset(ds)
    with pytest.raises(ValueError, match="Linux absolute"):
        service.register_dataset(
            dataset_key="bad_container",
            task_type="rgbt_detection",
            path=ds,
            container_path="datasets/relative",
        )


def test_writable_registration_rejected(tmp_path: Path):
    service = _service(tmp_path)
    ds = tmp_path / "data"
    _make_dataset(ds)
    with pytest.raises(ValueError, match="read-only"):
        service.register_dataset(
            dataset_key="writable",
            task_type="rgbt_detection",
            path=ds,
            read_only=False,
        )


def test_dangerous_root_rejected(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    assert is_dangerous_host_path(root, project_root=root)
    assert is_dangerous_host_path(root.parent, project_root=root)
    if Path("C:/").exists():
        assert is_dangerous_host_path(Path("C:/"), project_root=root)


def test_disable_blocks_resolve(tmp_path: Path):
    service = _service(tmp_path)
    ds = tmp_path / "data"
    _make_dataset(ds)
    service.register_dataset(
        dataset_key="rgbt_debug_v1", task_type="rgbt_detection", path=ds
    )
    service.disable_dataset("rgbt_debug_v1")
    with pytest.raises(ValueError, match="禁用"):
        service.datasets.require("rgbt_debug_v1")
    service.enable_dataset("rgbt_debug_v1")
    assert service.datasets.require("rgbt_debug_v1").enabled is True


def test_unregistered_dataset_rejected(tmp_path: Path):
    service = _service(tmp_path)
    with pytest.raises(KeyError):
        service.show_dataset("missing_ds")
    with pytest.raises(KeyError):
        service.datasets.require("missing_ds")


def test_validate_pairing_and_annotations(tmp_path: Path):
    service = _service(tmp_path)
    ds = tmp_path / "ok"
    _make_dataset(ds)
    service.register_dataset(
        dataset_key="rgbt_debug_v1", task_type="rgbt_detection", path=ds
    )
    report = service.validate_dataset("rgbt_debug_v1")
    assert report["valid"] is True
    assert report["paired_image_count"] == 10
    assert report["splits"]["train"]["paired_count"] == 8
    assert report["claim_level"] == "pipeline_validation_only"
    assert Path(report["report_path"]).exists()
    assert report["artifact"]["sha256"] == sha256_file(Path(report["report_path"]))
    assert report["artifact"]["size_bytes"] > 0


def test_missing_thermal_fails(tmp_path: Path):
    ds = tmp_path / "bad"
    _make_dataset(ds, drop_thermal="000001")
    report = validate_rgbt_dataset(ds, dataset_key="bad", write_previews=False)
    assert report["valid"] is False
    assert any("000001" in item for item in report["missing_thermal"])


def test_name_mismatch_same_count_fails(tmp_path: Path):
    ds = tmp_path / "mismatch"
    _make_dataset(ds, n_train=4, n_val=2)
    thermal = ds / "images" / "train" / "thermal" / "000001.jpg"
    renamed = ds / "images" / "train" / "thermal" / "999999.jpg"
    thermal.rename(renamed)
    report = validate_rgbt_dataset(ds, dataset_key="mismatch", write_previews=False)
    assert report["valid"] is False
    assert report["missing_thermal"] or report["missing_rgb"]


def test_corrupt_image_detected(tmp_path: Path):
    ds = tmp_path / "corrupt"
    _make_dataset(ds, n_train=3, n_val=1)
    bad = ds / "images" / "train" / "rgb" / "000001.jpg"
    bad.write_bytes(b"not-an-image")
    report = validate_rgbt_dataset(ds, dataset_key="corrupt", write_previews=False)
    assert report["valid"] is False
    assert report["corrupt_images"]


def test_size_mismatch_detected(tmp_path: Path):
    ds = tmp_path / "size"
    _make_dataset(ds, n_train=3, n_val=1)
    thermal = ds / "images" / "train" / "thermal" / "000001.jpg"
    Image.new("L", (320, 256), color=40).save(thermal)
    report = validate_rgbt_dataset(ds, dataset_key="size", write_previews=False)
    assert report["valid"] is False
    assert report["size_mismatches"]

def test_bbox_oob_detected(tmp_path: Path):
    ds = tmp_path / "bbox"
    _make_dataset(ds, n_train=2, n_val=1)
    ann_path = ds / "annotations" / "instances_train.json"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    data["annotations"][0]["bbox"] = [600.0, 480.0, 80.0, 80.0]
    ann_path.write_text(json.dumps(data), encoding="utf-8")
    report = validate_rgbt_dataset(ds, dataset_key="bbox", write_previews=False)
    assert report["valid"] is False
    assert report["invalid_annotations"]


def test_train_val_stem_overlap_detected(tmp_path: Path):
    ds = tmp_path / "overlap"
    _make_dataset(ds, n_train=2, n_val=1)
    # Force a shared stem across splits (val already has 000003 by default).
    src_rgb = ds / "images" / "train" / "rgb" / "000001.jpg"
    src_th = ds / "images" / "train" / "thermal" / "000001.jpg"
    (ds / "images" / "val" / "rgb" / "000001.jpg").write_bytes(src_rgb.read_bytes())
    (ds / "images" / "val" / "thermal" / "000001.jpg").write_bytes(src_th.read_bytes())
    report = validate_rgbt_dataset(ds, dataset_key="overlap", write_previews=False)
    assert report["valid"] is False
    assert report["split_overlaps"] or report["hash_overlaps"]

def test_preview_generation(tmp_path: Path):
    service = _service(tmp_path)
    ds = tmp_path / "prev"
    _make_dataset(ds)
    service.register_dataset(
        dataset_key="rgbt_debug_v1", task_type="rgbt_detection", path=ds
    )
    payload = service.preview_dataset("rgbt_debug_v1", count=2)
    assert payload["preview_paths"]
    for path in payload["preview_paths"]:
        assert Path(path).exists()


def test_claim_gate_blocks_performance_support():
    gate = apply_detection_claim_gate(execution_mode="smoke_train")
    assert gate["forbid_performance_support"] is True
    feedback = annotate_feedback_for_detection(
        {
            "hypothesis_status": "supported_with_repeated_evidence",
            "evidence_strength": "strong",
            "recommendations": [
                {
                    "recommendation_type": "expand_range",
                    "priority": 0.9,
                    "rationale": "go bigger",
                    "parameter_changes": {"epochs": 10},
                    "status": "active",
                }
            ],
            "uncertainties": [],
        },
        execution_mode="smoke_train",
    )
    assert feedback["hypothesis_status"] == "inconclusive"
    assert feedback["evidence_strength"] == "weak"
    by_type = {r["recommendation_type"]: r for r in feedback["recommendations"]}
    assert by_type["expand_range"]["status"] == "deferred"
    assert by_type["ablation"]["status"] == "active"
    assert by_type["ablation"]["parameter_changes"] == {"epochs": 1}


def test_invalid_map_rejected(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "metrics.json").write_text(
        '{"primary_metric":"mAP50_95","metrics":{"mAP50_95":1.5},"task_type":"rgbt_detection"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid_detection_metrics"):
        DetectionResultParser().parse(out)


def test_read_only_mount_required(tmp_path: Path):
    from scientist_lab.datasets.models import DatasetRegistration
    from scientist_lab.domain.contracts import ExperimentContract
    from scientist_lab.tasks.rgbt_detection.adapter import RGBTDetectionAdapter

    root = Path(__file__).resolve().parents[2]
    contract = ExperimentContract.model_validate(
        {
            "schema_version": "1.1",
            "project_id": "project_rgbt_001",
            "node_id": "rgbt_node_001",
            "task_type": "rgbt_detection",
            "environment_key": "rgbt-detection-v1",
            "code_reference": "local:rgbt_detector",
            "dataset_reference": "dataset:rgbt_debug_v1",
            "entrypoint": "run_detection_experiment.py",
            "execution_mode": "smoke_train",
            "parameters": {"input_mode": "rgb", "fusion_method": "none"},
        }
    )
    dataset = DatasetRegistration(
        dataset_key="rgbt_debug_v1",
        task_type="rgbt_detection",
        host_path=str(tmp_path),
        container_path="/datasets/rgbt_debug_v1",
        read_only=False,
        enabled=True,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="read-only"):
        RGBTDetectionAdapter().prepare_execution(
            contract,
            dataset,
            code_roots={"local:rgbt_detector": root / "rgbt_detector"},
        )
