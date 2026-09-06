"""P4 RT-DETR Adapter + transfer protocol. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.adapters import RTDETRAdapter, adapter_for_protocol, adapter_key_from_protocol
from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.tasks.rgbt_detection.v26_p4 import (
    DATASET_FINGERPRINT,
    PROTOCOL_ID,
    load_plan,
    load_protocol,
    materialize_preview,
    seed_transfer_memory,
)

EXAMPLES = SCHEMA_DIR / "examples"


def test_p4_protocol_validates_and_stays_c0() -> None:
    protocol = load_protocol()
    validate_named("research_protocol", protocol)
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["baseline"]["adapter"] == "rtdetr"
    assert protocol["claim_policy"]["max_claim_strength"] == "C0"
    assert protocol["claim_policy"]["allow_scientific_claims"] is False
    assert protocol["stop_rules"]["max_rounds"] == 2
    assert protocol["condition_slice"]["id"] == "low_light_subset_v1"


def test_p4_plans_are_f1_then_f3() -> None:
    f1 = load_plan("r0")
    f3 = load_plan("transfer")
    validate_named("experiment_plan", f1)
    validate_named("experiment_plan", f3)
    assert f1["how_id"] == "F1"
    assert f3["how_id"] == "F3"
    assert f3["round_index"] == 1


def test_rtdetr_adapter_selected_from_protocol() -> None:
    protocol = load_protocol()
    assert adapter_key_from_protocol(protocol) == "rtdetr"
    adapter = adapter_for_protocol(protocol)
    assert isinstance(adapter, RTDETRAdapter)
    assert adapter.adapter_key == "rtdetr"


def test_rtdetr_f1_how_sets_backend() -> None:
    preview = materialize_preview("r0")
    assert preview["how_id"] == "F1"
    assert preview["fusion_method"] == "early_concat"
    assert preview["decoder_family"] == "RTDETRTransformer"
    assert preview["baseline_key"] == "rtdetr_s"
    assert preview["adapter"] == "rtdetr"
    how = RTDETRAdapter()._resolve_how(load_plan("r0"), load_protocol())
    assert how["legacy_parameters"]["mixed_precision"] is False


def test_rtdetr_f3_how_is_gated_multiscale() -> None:
    preview = materialize_preview("transfer")
    assert preview["how_id"] == "F3"
    assert preview["fusion_method"] == "gated_multiscale"


def test_rtdetr_rejects_n1_fdpn() -> None:
    protocol = load_protocol()
    plan = dict(load_plan("r0"))
    plan["how_id"] = "N1"
    plan["proposed_changes"] = [
        {
            "target": "neck",
            "summary": "illegal FDPN on RT-DETR",
            "detail": {"how_id": "N1"},
        }
    ]
    plan["modification_scope"] = ["neck"]
    with pytest.raises(MaterializeRejected, match="N1/FDPN"):
        RTDETRAdapter().materialize_contract(plan, protocol)


def test_rtdetr_config_yaml_does_not_use_dfine_decoder(tmp_path: Path) -> None:
    import sys

    app = Path(__file__).resolve().parents[2] / "experiment_apps" / "rgbt_detection_real"
    sys.path.insert(0, str(app))
    from dfine_config_builder import dfine_root_from_app, write_dfine_fast_config

    stage = {
        "train_img": tmp_path / "train",
        "val_img": tmp_path / "val",
        "train_ann": tmp_path / "train.json",
        "val_ann": tmp_path / "val.json",
    }
    for path in stage.values():
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        else:
            path.mkdir()
    cfg = write_dfine_fast_config(
        dfine_root=dfine_root_from_app(app),
        config_path=tmp_path / "rtdetr.yml",
        stage_paths=stage,
        output_dir=tmp_path / "out",
        epochs=2,
        batch_size=2,
        num_workers=0,
        image_size=160,
        learning_rate=2e-4,
        num_classes=7,
        seed=42,
        detector="rtdetr",
    )
    text = cfg.read_text(encoding="utf-8")
    assert "model: RTDETR" in text
    assert "decoder: RTDETRTransformer" in text
    assert "criterion: RTDETRCriterion" in text
    assert "num_denoising: 0" in text
    assert "DFINETransformer" not in text
    assert "FDPN" not in text


def test_rtdetr_criterion_accepts_dfine_solver_epoch_kwarg() -> None:
    text = (
        Path(__file__).resolve().parents[2]
        / "experiment_apps"
        / "rgbt_detection_real"
        / "vendor_rtdetr"
        / "rtdetr_criterion.py"
    ).read_text(encoding="utf-8")
    assert "def forward(self, outputs, targets, **kwargs)" in text
    assert "def as_matcher_indices(" in text


def test_seed_transfer_memory_valid(tmp_path: Path) -> None:
    seeded = seed_transfer_memory(tmp_path)
    assert seeded["lesson_id"].startswith("LESSON-V26-P4")


def test_archive_p4_probe_is_inconclusive_keep_not_ban(tmp_path: Path) -> None:
    from scientist_lab.tasks.rgbt_detection.v26_p4 import (
        LESSON_P4_PROBE_INCONCLUSIVE_ID,
        STRATEGY_P4_GATED_MULTISCALE_ID,
        archive_p4_inconclusive_probe,
        seed_transfer_memory,
    )

    mem = tmp_path / "memory"
    seed_transfer_memory(mem)
    (mem / "research_memory.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "lessons": {
                    **json.loads((mem / "research_memory.json").read_text(encoding="utf-8"))[
                        "lessons"
                    ],
                    "LESSON-run_plan_v26_p4_rtdetr_f3-001": {
                        "lesson_id": "LESSON-run_plan_v26_p4_rtdetr_f3-001",
                        "type": "positive_evidence",
                        "statement": "placeholder extra-seed advice",
                        "status": "active",
                        "evidence": [
                            {
                                "run_id": "run_plan_v26_p4_rtdetr_f3",
                                "metric": "APS_lowlight",
                                "delta": 0.0017082151533518684,
                            }
                        ],
                        "created_from": ["run_plan_v26_p4_rtdetr_f3"],
                    },
                    "LESSON-run_plan_v26_p4_rtdetr_f3-semantic-001": {
                        "lesson_id": "LESSON-run_plan_v26_p4_rtdetr_f3-semantic-001",
                        "type": "positive_evidence",
                        "statement": "replicate extra seed",
                        "status": "active",
                        "evidence": [
                            {
                                "run_id": "run_plan_v26_p4_rtdetr_f3",
                                "metric": "APS_lowlight",
                                "delta": 0.0017082151533518684,
                            }
                        ],
                        "created_from": ["run_plan_v26_p4_rtdetr_f3"],
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    from scientist_lab.core.memory_writer import MemoryWriter

    MemoryWriter(mem).persist_strategy(
        {
            "strategy_id": "STRATEGY-run_plan_v26_p4_rtdetr_f3-001",
            "action": "prioritize",
            "target": "fusion",
            "reason_lesson_ids": ["LESSON-run_plan_v26_p4_rtdetr_f3-001"],
            "status": "active",
        }
    )
    report = archive_p4_inconclusive_probe(mem)
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["banned"] is False
    assert report["failed_cross_model"] is False
    assert report["gpu"] is False
    assert report["max_rounds_amended"] is False
    writer = MemoryWriter(mem)
    lessons = writer.load_lessons()
    stop = lessons[LESSON_P4_PROBE_INCONCLUSIVE_ID]
    assert stop["type"] == "inconclusive"
    assert "do not BAN gated_multiscale" in stop["statement"].lower() or "Do not BAN" in stop[
        "statement"
    ]
    assert "failed_cross_model" not in stop["statement"]
    assert lessons["LESSON-run_plan_v26_p4_rtdetr_f3-001"]["status"] == "superseded"
    strategies = writer.load_strategies()
    keep = strategies[STRATEGY_P4_GATED_MULTISCALE_ID]
    assert keep["action"] == "keep"
    assert keep["status"] == "active"
    assert strategies["STRATEGY-run_plan_v26_p4_rtdetr_f3-001"]["status"] == "retired"
    assert strategies["STRATEGY-V26-P4-TRANSFER-001"]["action"] != "ban"
