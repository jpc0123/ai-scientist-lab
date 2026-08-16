from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.baselines.dfine_s import DFineSBaselineAdapter


ROOT = Path(__file__).resolve().parents[2]
REAL_APP = ROOT / "experiment_apps" / "rgbt_detection_real"


def test_dfine_vendor_present():
    vendor = ROOT / "third_party" / "DFINE" / "train.py"
    assert vendor.is_file()
    pin = (ROOT / "third_party" / "VENDOR.md").read_text(encoding="utf-8")
    assert "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6" in pin


def test_eval_spatial_size_tracks_input_size(tmp_path: Path):
    sys.path.insert(0, str(REAL_APP))
    from dfine_config_builder import resolve_input_size, write_dfine_fast_config

    assert resolve_input_size(image_size=160) == (160, 160)
    assert resolve_input_size(image_height=512, image_width=640) == (512, 640)

    stage = {
        "train_img": tmp_path / "train",
        "val_img": tmp_path / "val",
        "train_ann": tmp_path / "train.json",
        "val_ann": tmp_path / "val.json",
    }
    for key in ("train_img", "val_img"):
        stage[key].mkdir()
    for key in ("train_ann", "val_ann"):
        stage[key].write_text("{}", encoding="utf-8")

    cfg = write_dfine_fast_config(
        dfine_root=ROOT / "third_party" / "DFINE",
        config_path=tmp_path / "cfg.yml",
        stage_paths=stage,
        output_dir=tmp_path / "out",
        epochs=1,
        batch_size=2,
        num_workers=0,
        image_height=512,
        image_width=640,
        learning_rate=2e-4,
        num_classes=2,
        seed=42,
        scale_queries_to_tokens=True,
    )
    text = cfg.read_text(encoding="utf-8")
    assert "eval_spatial_size: [512, 640]" in text
    assert "Resize, size: [512, 640]" in text
    budget = json.loads((tmp_path / "dfine_spatial_query_budget.json").read_text(encoding="utf-8"))
    assert budget["eval_spatial_size"] == [512, 640]
    assert budget["input_size"] == [512, 640]


def test_num_queries_scales_only_when_enabled():
    sys.path.insert(0, str(REAL_APP))
    from dfine_config_builder import resolve_query_budget

    smoke = resolve_query_budget(
        input_h=160,
        input_w=160,
        configured_num_queries=300,
        scale_queries_to_tokens=True,
    )
    assert smoke["available_tokens"] == 25
    assert smoke["effective_num_queries"] == 25
    assert smoke["configured_num_queries"] == 300

    formal_ok = resolve_query_budget(
        input_h=640,
        input_w=640,
        configured_num_queries=300,
        scale_queries_to_tokens=False,
    )
    assert formal_ok["effective_num_queries"] == 300

    with pytest.raises(ValueError, match="exceeds coarse feature tokens"):
        resolve_query_budget(
            input_h=160,
            input_w=160,
            configured_num_queries=300,
            scale_queries_to_tokens=False,
        )


def test_coco_category_remap_is_bijective_and_persisted(tmp_path: Path):
    sys.path.insert(0, str(REAL_APP))
    from dfine_dataset_stage import (
        invert_model_label_to_dataset_category,
        remap_categories_zero_based,
        stage_coco_for_dfine,
        validate_category_label_map,
    )

    payload = {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "vehicle"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
        ],
        "images": [],
    }
    remapped, label_map = remap_categories_zero_based(payload)
    validate_category_label_map(label_map)
    assert label_map["dataset_category_to_model_label"] == {"1": 0, "2": 1}
    assert label_map["model_label_to_dataset_category"] == {"0": 1, "1": 2}
    assert {a["category_id"] for a in remapped["annotations"]} == {0, 1}
    assert invert_model_label_to_dataset_category(0, label_map) == 1
    assert invert_model_label_to_dataset_category(1, label_map) == 2

    data_root = ROOT / "datasets" / "rgbt_fast_eval_v1"
    if not data_root.exists():
        pytest.skip("rgbt_fast_eval_v1 missing")
    stage = stage_coco_for_dfine(
        data_root,
        tmp_path / "stage",
        input_mode="rgb",
        label_map_path=tmp_path / "category_label_map.json",
    )
    saved = json.loads(Path(stage["category_label_map_path"]).read_text(encoding="utf-8"))
    validate_category_label_map(saved)
    assert saved["num_classes"] >= 1
    labels = sorted(int(x) for x in saved["model_label_to_dataset_category"])
    assert labels == list(range(saved["num_classes"]))


def test_dfine_dataset_stage_and_config(tmp_path: Path):
    sys.path.insert(0, str(REAL_APP))
    from dfine_config_builder import write_dfine_fast_config
    from dfine_dataset_stage import count_categories, stage_coco_for_dfine

    data_root = ROOT / "datasets" / "rgbt_fast_eval_v1"
    if not data_root.exists():
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
        scale_queries_to_tokens=True,
    )
    text = cfg.read_text(encoding="utf-8")
    assert "pretrained: False" in text
    assert "num_classes:" in text
    assert "eval_spatial_size: [160, 160]" in text
    assert "num_queries: 25" in text
    assert "num_top_queries: 25" in text
    budget = json.loads((tmp_path / "dfine_spatial_query_budget.json").read_text(encoding="utf-8"))
    assert budget["query_budget"]["effective_num_queries"] == 25


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


def test_dfine_s_rejects_unimplemented_fdpn_fusion():
    adapter = DFineSBaselineAdapter()
    with pytest.raises(ValueError, match="not a fusion switch"):
        adapter.validate_parameters(
            {"input_mode": "rgbt", "fusion_method": "fdpn", "epochs": 1}
        )
    with pytest.raises(ValueError, match="unsupported fusion_method"):
        adapter.validate_parameters(
            {"input_mode": "rgbt", "fusion_method": "magic", "epochs": 1}
        )
    adapter.validate_parameters(
        {"input_mode": "rgbt", "fusion_method": "early_concat", "epochs": 1}
    )
    adapter.validate_parameters(
        {
            "input_mode": "rgbt",
            "fusion_method": "gated_multiscale",
            "fusion": {"type": "gated_multiscale", "residual": True},
            "epochs": 1,
        }
    )


def test_rgbt_pair_audit_on_fast_eval_dataset(tmp_path: Path):
    sys.path.insert(0, str(REAL_APP))
    from rgbt_pair_audit import audit_rgbt_pairs, write_rgbt_pair_audit

    data_root = ROOT / "datasets" / "rgbt_fast_eval_v1"
    if not data_root.exists():
        pytest.skip("rgbt_fast_eval_v1 missing")
    report = write_rgbt_pair_audit(data_root, tmp_path / "rgbt_pair_audit.json")
    assert report["ok"] is True
    assert report["splits"]["train"]["n_paired"] >= 1
    assert report["splits"]["val"]["n_paired"] >= 1
    offline = audit_rgbt_pairs(data_root)
    assert offline["ok"] is True
