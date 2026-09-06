"""v2.6-P2: frozen low_light_subset_v1 + APS_lowlight + HOW catalog. No GPU."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scientist_lab.adapters import DFINEAdapter, MaterializeRejected
from scientist_lab.adapters.dfine.how import resolve_adapter_how
from scientist_lab.adapters.dfine.how_catalog import NOT_REGISTERED, catalog_payload, resolve_how_id
from scientist_lab.cli import main
from scientist_lab.core.claim_gate import evaluate_claim
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.datasets.low_light_subset import (
    SLICE_ID,
    SliceAmendmentRequired,
    assert_slice_not_rewritten,
    build_low_light_subset,
    frozen_rule,
    linear_percentile,
    rule_hash,
)
from scientist_lab.metrics.aps_lowlight import evaluate_aps_lowlight, filter_coco_gt

EXAMPLES = SCHEMA_DIR / "examples"


def _write_solid_jpeg(path: Path, rgb: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, rgb).save(path, format="PNG")


def _write_pairs(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "sequence_id",
        "frame_id",
        "split",
        "rgb_path",
        "thermal_path",
        "file_name",
        "width",
        "height",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _toy_rgbt(root: Path) -> Path:
    rows = {
        "train": [
            ("dark_t1", (16, 16, 16)),
            ("dark_t2", (16, 16, 16)),
            ("dark_t3", (16, 16, 16)),
            ("bright_t1", (220, 220, 220)),
            ("bright_t2", (220, 220, 220)),
            ("bright_t3", (220, 220, 220)),
        ],
        "val": [
            ("dark_v1", (16, 16, 16)),
            ("bright_v1", (220, 220, 220)),
        ],
        "test": [
            ("dark_e1", (16, 16, 16)),
            ("bright_e1", (220, 220, 220)),
        ],
    }
    for split, items in rows.items():
        csv_rows = []
        for sample_id, color in items:
            rgb = root / "images" / split / "rgb" / f"{sample_id}.png"
            thermal = root / "images" / split / "thermal" / f"{sample_id}.png"
            _write_solid_jpeg(rgb, color)
            _write_solid_jpeg(thermal, (80, 80, 80))
            csv_rows.append(
                {
                    "sample_id": sample_id,
                    "sequence_id": "seq",
                    "frame_id": "00000",
                    "split": split,
                    "rgb_path": str(rgb),
                    "thermal_path": str(thermal),
                    "file_name": f"{sample_id}.png",
                    "width": "8",
                    "height": "8",
                }
            )
        _write_pairs(root / "manifests" / f"{split}_pairs.csv", csv_rows)
    return root


def _v26_protocol(**overrides) -> dict:
    spec = frozen_rule()
    doc = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    doc["protocol_id"] = "research_protocol_rgbt_dfine_v26"
    doc["fingerprint_id"] = "FP-RGBT-DFINE-V26-LOWLIGHT"
    doc["objective"] = {
        "primary": {"metric": "APS_lowlight", "direction": "maximize"},
        "secondary": [
            {"metric": "mAP50_95_lowlight", "direction": "maximize"},
            {"metric": "APS", "direction": "maximize"},
        ],
    }
    doc["condition_slice"] = {
        "id": SLICE_ID,
        "version": spec["version"],
        "method": spec["method"],
        "official_labels": False,
        "rule_hash": rule_hash(spec),
    }
    doc.update(overrides)
    return doc


def _plan(**overrides) -> dict:
    plan = {
        "schema_version": "1.0.0",
        "plan_id": "plan_v26_p2",
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v26",
        "protocol_version": 1,
        "parent_run_id": None,
        "round_index": 1,
        "observation": "RGB degrades in low light",
        "hypothesis": "thermal fusion can compensate RGB degradation",
        "modification_scope": ["fusion"],
        "proposed_changes": [{"target": "fusion", "summary": "use early concat fusion"}],
        "controlled_variables": ["evaluator"],
        "expected_effect": {"primary_metric": "APS_lowlight", "direction": "increase"},
        "evaluation": {"method": "fast_eval", "seeds": [42]},
        "budget_class": "probe",
        "risk_level": "auto",
        "memory_refs": {"lesson_ids": [], "strategy_ids": []},
        "evidence_runs": [],
    }
    plan.update(overrides)
    return plan


def test_rule_hash_is_stable() -> None:
    assert rule_hash() == rule_hash(frozen_rule())
    assert frozen_rule()["official_labels"] is False
    assert frozen_rule()["slice_id"] == SLICE_ID


def test_linear_percentile_matches_numpy_default() -> None:
    values = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    got = linear_percentile(values, 25)
    np = pytest.importorskip("numpy")
    assert got == pytest.approx(float(np.percentile(values, 25)))


def test_build_subset_uses_train_threshold_only(tmp_path: Path) -> None:
    root = _toy_rgbt(tmp_path / "ds")
    freeze = build_low_light_subset(root)
    validate_named("low_light_subset", freeze)
    assert freeze["llm_may_rewrite"] is False
    assert freeze["official_labels"] is False
    assert set(freeze["membership"]["train"]) == {"dark_t1", "dark_t2", "dark_t3"}
    assert freeze["membership"]["val"] == ["dark_v1"]
    assert freeze["membership"]["test"] == ["dark_e1"]
    assert "bright_v1" not in freeze["membership"]["val"]


def test_llm_cannot_rewrite_slice() -> None:
    with pytest.raises(SliceAmendmentRequired):
        assert_slice_not_rewritten(
            {
                "modification_scope": ["dataset_split"],
                "proposed_changes": [{"target": "dataset_split", "summary": "pick a darker subset"}],
            }
        )


def test_materialize_rejects_slice_rewrite() -> None:
    adapter = DFINEAdapter()
    protocol = _v26_protocol()
    validate_named("research_protocol", protocol)
    with pytest.raises(MaterializeRejected, match="Protocol Amendment"):
        adapter.materialize_contract(
            _plan(
                modification_scope=["fusion"],
                proposed_changes=[{"target": "dataset_split", "summary": "reslice"}],
            ),
            protocol,
        )


def test_unregistered_how_is_rejected() -> None:
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")
    assert "F2" in NOT_REGISTERED
    how = resolve_adapter_how(_plan(how_id="F3", modification_scope=["fusion"]))
    assert how["fusion_method"] == "gated_multiscale"
    assert how["how_id"] == "F3"


def test_v26_fingerprint_binds_slice() -> None:
    adapter = DFINEAdapter()
    protocol = _v26_protocol()
    contract = adapter.materialize_contract(_plan(), protocol)
    assert contract["dataset"]["split_reference"].startswith(f"{SLICE_ID}@")
    fp = adapter.compute_fingerprint(protocol, contract)
    other = dict(protocol)
    other["condition_slice"] = dict(protocol["condition_slice"], rule_hash="0" * 64)
    fp2 = adapter.compute_fingerprint(other, contract)
    assert fp["dataset_split_hash"] != fp2["dataset_split_hash"]


def test_claim_gate_accepts_aps_lowlight_on_baseline_then_blocks_science() -> None:
    protocol = _v26_protocol()
    verdict = evaluate_claim(
        {
            "schema_version": "1.0.0",
            "claim_id": "claim_ll_r1",
            "claim_type": "comparative",
            "claim_text": "F3 improves APS_lowlight vs R0",
            "claim_strength": "C1",
            "metric": "APS_lowlight",
            "asserts": {"outperform": True},
        },
        protocol=protocol,
        evidence={
            "evidence_status": "VALID",
            "run_state": "COMPLETED",
            "run_level": "formal",
            "metrics": {"APS_lowlight": 0.021, "mAP50_95_lowlight": 0.014},
            "result": {
                "metrics": {"APS_lowlight": 0.021, "mAP50_95_lowlight": 0.014},
                "execution": {"status": "success"},
            },
            "fingerprint_comparable": True,
            "baseline": {
                "present": True,
                "matched_fingerprint": True,
                "budget_class": "formal",
                "metrics": {"APS_lowlight": 0.0045926865160844455},
            },
        },
    )
    assert verdict["status"] == "BLOCKED"
    assert "missing APS evidence on baseline" not in verdict["reason"]
    assert "C0" in verdict["reason"] or "forbids scientific" in verdict["reason"]
    verdict = evaluate_claim(
        {
            "schema_version": "1.0.0",
            "claim_id": "claim_ll",
            "claim_type": "comparative",
            "claim_text": "low-light small-object APS improved",
            "claim_strength": "C1",
            "metric": "APS_lowlight",
            "asserts": {"outperform": True},
        },
        protocol=_v26_protocol(),
        evidence={
            "evidence_status": "VALID",
            "run_state": "COMPLETED",
            "run_level": "formal",
            "metrics": {"APS": 0.04, "mAP50_95": 0.12, "mAP50": 0.3},
            "result": {
                "metrics": {"APS": 0.04, "mAP50_95": 0.12, "mAP50": 0.3},
                "execution": {"status": "success"},
            },
            "fingerprint_comparable": True,
        },
    )
    assert verdict["status"] == "BLOCKED"
    assert "APS_lowlight" in verdict["reason"]


def test_canonicalize_does_not_copy_aps_into_aps_lowlight(tmp_path: Path) -> None:
    from scientist_lab.adapters.dfine.metrics_parser import parse_metrics

    (tmp_path / "metrics.json").write_text(
        json.dumps({"APS": 0.08, "mAP50_95": 0.11}),
        encoding="utf-8",
    )
    parsed = parse_metrics(tmp_path)
    assert parsed["metrics"]["APS"] == pytest.approx(0.08)
    assert parsed["metrics"]["APS_lowlight"] is None


def test_evaluate_aps_lowlight_filters_images() -> None:
    freeze = {
        "membership": {"val": ["dark_v1"]},
        "slice_id": SLICE_ID,
    }
    gt = {
        "images": [
            {"id": 1, "file_name": "dark_v1.jpg", "sample_id": "dark_v1"},
            {"id": 2, "file_name": "bright_v1.jpg", "sample_id": "bright_v1"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100},
        ],
        "categories": [{"id": 1, "name": "car"}],
    }
    filtered = filter_coco_gt(gt, ["dark_v1"])
    assert [img["id"] for img in filtered["images"]] == [1]
    metrics = evaluate_aps_lowlight(gt, [], freeze, split="val")
    assert metrics["n_images_lowlight"] == 1
    assert metrics["slice_id"] == SLICE_ID


def test_cli_rule_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / "rule.json"
    code = main(["freeze-lowlight-subset", "--rule-only", "--output", str(dest)])
    assert code == 0
    freeze = json.loads(dest.read_text(encoding="utf-8"))
    validate_named("low_light_subset", freeze)
    assert freeze["rule_hash"] == rule_hash()
    assert freeze["membership_hash"] is None


def test_cli_builds_toy_membership(tmp_path: Path) -> None:
    root = _toy_rgbt(tmp_path / "ds")
    dest = tmp_path / "freeze.json"
    code = main(
        [
            "freeze-lowlight-subset",
            "--dataset-root",
            str(root),
            "--output",
            str(dest),
        ]
    )
    assert code == 0
    freeze = json.loads(dest.read_text(encoding="utf-8"))
    assert freeze["counts"]["val"] == 1
    assert freeze["llm_may_rewrite"] is False


def test_how_catalog_payload_exposes_materializable_how() -> None:
    payload = catalog_payload()
    assert payload["llm_may_invent_how"] is False
    ids = {row["id"] for row in payload["allowed"]}
    assert ids == {"F0", "F1", "F3", "N0", "N1", "A4"}
    assert payload["hidden_from_planner"] == {}
    assert "F2" in payload["not_registered"]


def test_v26_protocol_example_if_present() -> None:
    path = EXAMPLES / "research_protocol_rgbt_dfine_v26.json"
    if not path.is_file():
        pytest.skip("v26 protocol example not written yet")
    doc = load_json(path)
    validate_named("research_protocol", doc)
    assert doc["objective"]["primary"]["metric"] == "APS_lowlight"
    assert doc["condition_slice"]["official_labels"] is False
    assert doc["condition_slice"]["rule_hash"] == rule_hash()
    assert doc["stop_rules"]["max_rounds"] == 12
