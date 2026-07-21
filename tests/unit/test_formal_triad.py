from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.protocols.formal_triad import (
    DEFAULT_FORMAL_TRIAD_NODES,
    validate_formal_triad,
)
from scientist_lab.protocols.models import ExperimentProtocol
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def _protocol() -> ExperimentProtocol:
    data = _load("rgbt_protocol.json")
    data["created_at"] = "2026-07-21T00:00:00+00:00"
    return ExperimentProtocol.model_validate(data)


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


def test_formal_example_contracts_match_protocol():
    protocol = _protocol()
    contracts = {
        "rgb": _load("rgbt_formal_rgb_contract.json"),
        "thermal": _load("rgbt_formal_thermal_contract.json"),
        "fusion": _load("rgbt_formal_fusion_contract.json"),
    }
    report = validate_formal_triad(protocol, contracts)
    assert report.valid is True, report.blocking_issues
    assert report.protocol_id == "protocol_rgbt_001"


def test_formal_triad_blocks_budget_mismatch():
    protocol = _protocol()
    contracts = {
        "rgb": _load("rgbt_formal_rgb_contract.json"),
        "thermal": _load("rgbt_formal_thermal_contract.json"),
        "fusion": _load("rgbt_formal_fusion_contract.json"),
    }
    contracts["fusion"]["parameters"]["epochs"] = 99
    report = validate_formal_triad(protocol, contracts)
    assert report.valid is False
    assert any("epochs" in issue for issue in report.blocking_issues)


def test_formal_triad_blocks_disallowed_variable_change():
    protocol = _protocol()
    contracts = {
        "rgb": _load("rgbt_formal_rgb_contract.json"),
        "thermal": _load("rgbt_formal_thermal_contract.json"),
        "fusion": _load("rgbt_formal_fusion_contract.json"),
    }
    contracts["thermal"]["parameters"]["learning_rate"] = 0.001
    report = validate_formal_triad(protocol, contracts)
    assert report.valid is False
    assert any("learning_rate" in issue for issue in report.blocking_issues)


def test_formal_triad_blocks_dataset_mismatch():
    protocol = _protocol()
    contracts = {
        "rgb": _load("rgbt_formal_rgb_contract.json"),
        "thermal": _load("rgbt_formal_thermal_contract.json"),
        "fusion": _load("rgbt_formal_fusion_contract.json"),
    }
    contracts["fusion"]["dataset_reference"] = "dataset:other"
    report = validate_formal_triad(protocol, contracts)
    assert report.valid is False
    assert any("dataset_reference" in issue for issue in report.blocking_issues)


def test_formal_node_ids_are_stable():
    assert DEFAULT_FORMAL_TRIAD_NODES == {
        "rgb": "rgbt_formal_node_001",
        "thermal": "rgbt_formal_node_002",
        "fusion": "rgbt_formal_node_003",
    }
    for name, node_id in [
        ("rgbt_formal_rgb_contract.json", "rgbt_formal_node_001"),
        ("rgbt_formal_thermal_contract.json", "rgbt_formal_node_002"),
        ("rgbt_formal_fusion_contract.json", "rgbt_formal_node_003"),
    ]:
        assert _load(name)["node_id"] == node_id
        assert _load(name)["protocol_id"] == "protocol_rgbt_001"


def test_service_validate_formal_triad(tmp_path: Path):
    service = _service(tmp_path)
    data = service.validate_formal_triad_contracts()
    assert data["ok"] is True
    assert data["valid"] is True
    assert data["matched_seeds"] == [42, 43, 44]
    assert set(data["paths"]) == {"rgb", "thermal", "fusion"}


def test_only_allowed_variables_differ_across_formal_nodes():
    rgb = ExperimentContract.model_validate(_load("rgbt_formal_rgb_contract.json"))
    thermal = ExperimentContract.model_validate(
        _load("rgbt_formal_thermal_contract.json")
    )
    fusion = ExperimentContract.model_validate(_load("rgbt_formal_fusion_contract.json"))
    allowed = {"input_mode", "fusion_method"}
    for other in (thermal, fusion):
        for key in set(rgb.parameters) | set(other.parameters):
            if rgb.parameters.get(key) != other.parameters.get(key):
                assert key in allowed
