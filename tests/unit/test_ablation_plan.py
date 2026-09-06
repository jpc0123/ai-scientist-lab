from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.ablations.models import AblationPlan, AblationVariant
from scientist_lab.ablations.planner import (
    default_modality_ablation_plan,
    plan_variant_contracts,
)
from scientist_lab.ablations.verifier import AblationVerifier
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.protocols.models import ExperimentProtocol
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    settings = Settings(
        project_root=root,
        db_path=tmp_path / "test.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=root / "experiment_app",
    ).resolve()
    return ExperimentService(settings=settings)


def _protocol() -> ExperimentProtocol:
    data = json.loads((EXAMPLES / "rgbt_protocol.json").read_text(encoding="utf-8"))
    data["created_at"] = "2026-07-21T00:00:00+00:00"
    return ExperimentProtocol.model_validate(data)


def test_create_list_show_validate_ablation(tmp_path: Path):
    service = _service(tmp_path)
    created = service.create_ablation(EXAMPLES / "rgbt_ablation_modality_plan.json")
    assert created["ablation_id"] == "ablation_rgbt_modality_001"
    listed = service.list_ablations(project_id="project_rgbt_003")
    assert len(listed) == 1
    shown = service.show_ablation("ablation_rgbt_modality_001")
    assert [v["variant_id"] for v in shown["variants"]] == ["A0", "A1", "A2"]
    report = service.validate_ablation("ablation_rgbt_modality_001")
    assert report["valid"] is True


def test_ablation_rejects_non_controlled_parameter():
    plan = AblationPlan(
        ablation_id="bad",
        project_id="project_rgbt_003",
        reference_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        controlled_variables=["input_mode", "fusion_method"],
        variants=[
            AblationVariant(
                variant_id="A0",
                title="bad",
                parameter_changes={"epochs": 99},
                expected_effect="x",
            )
        ],
    )
    report = AblationVerifier().validate_plan(plan, protocol=_protocol())
    assert report.valid is False
    assert any("epochs" in issue for issue in report.blocking_issues)


def test_ablation_deduplicates_identical_variants():
    plan = AblationPlan(
        ablation_id="dup",
        project_id="project_rgbt_003",
        reference_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        controlled_variables=["input_mode", "fusion_method"],
        variants=[
            AblationVariant(
                variant_id="A0",
                title="RGB",
                parameter_changes={"input_mode": "rgb", "fusion_method": "none"},
                expected_effect="a",
            ),
            AblationVariant(
                variant_id="A0_dup",
                title="RGB again",
                parameter_changes={"input_mode": "rgb", "fusion_method": "none"},
                expected_effect="b",
            ),
        ],
    )
    verifier = AblationVerifier()
    report = verifier.validate_plan(plan)
    assert report.valid is True
    assert report.deduplicated_variant_ids == ["A0_dup"]
    deduped = verifier.deduplicate_plan(plan)
    assert [v.variant_id for v in deduped.variants] == ["A0"]


def test_materialize_ablation_writes_contracts(tmp_path: Path):
    service = _service(tmp_path)
    service.create_ablation(EXAMPLES / "rgbt_ablation_modality_plan.json")
    out = tmp_path / "ablation_out"
    result = service.materialize_ablation(
        "ablation_rgbt_modality_001",
        reference_contract=EXAMPLES / "rgbt_formal_fusion_contract.json",
        output_dir=out,
    )
    assert result["variant_count"] == 3
    node_ids = {item["node_id"] for item in result["variants"]}
    assert node_ids == {
        "rgbt_formal_node_001",
        "rgbt_formal_node_002",
        "rgbt_formal_node_003",
    }
    rgb = next(v for v in result["variants"] if v["variant_id"] == "A0")
    assert rgb["contract"]["parameters"]["input_mode"] == "rgb"
    assert rgb["contract"]["parameters"]["epochs"] == 5  # fixed from reference
    assert len(result["written_paths"]) == 3
    for path in result["written_paths"]:
        assert Path(path).is_file()
        contract = ExperimentContract.model_validate(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
        assert contract.protocol_id == "protocol_rgbt_001"


def test_default_modality_plan_matches_example():
    plan = default_modality_ablation_plan()
    example = AblationPlan.model_validate(
        {
            **json.loads(
                (EXAMPLES / "rgbt_ablation_modality_plan.json").read_text(
                    encoding="utf-8"
                )
            ),
            "created_at": "2026-07-21T00:00:00+00:00",
        }
    )
    assert plan.ablation_id == example.ablation_id
    assert [v.variant_id for v in plan.variants] == [
        v.variant_id for v in example.variants
    ]
    reference = ExperimentContract.model_validate(
        json.loads(
            (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
        )
    )
    items = plan_variant_contracts(plan, reference)
    assert len(items) == 3
