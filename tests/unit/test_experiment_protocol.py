from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.protocols.models import ExperimentProtocol
from scientist_lab.protocols.service import ProtocolService
from scientist_lab.protocols.verifier import ProtocolVerifier, ProtocolViolationError
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _protocol_payload(**overrides):
    data = {
        "protocol_id": "protocol_rgbt_001",
        "project_id": "project_rgbt_003",
        "title": "RGB-T matched baseline evaluation",
        "task_type": "rgbt_detection",
        "dataset_reference": "dataset:rgbt_fast_eval_v1",
        "dataset_version": "v1",
        "split_reference": "split:fast_eval_fixed_v1",
        "environment_key": "rgbt-detection-v2",
        "code_reference": "image:rgbt-detection-v2",
        "code_version": "v0.8.0",
        "execution_mode": "fast_eval",
        "seeds": [42, 43, 44],
        "primary_metric": "mAP50_95",
        "secondary_metrics": ["mAP50", "AP_small", "precision", "recall"],
        "fixed_parameters": {
            "baseline": "dfine_s",
            "epochs": 5,
            "batch_size": 4,
            "image_width": 640,
            "image_height": 512,
            "learning_rate": 0.0001,
        },
        "allowed_variables": ["input_mode", "fusion_method"],
        "resource_metrics": [
            "duration_seconds",
            "peak_gpu_memory_mb",
            "parameter_count",
        ],
        "claim_level": "exploratory_comparison",
    }
    data.update(overrides)
    return data


def _compliant_contract(**overrides) -> ExperimentContract:
    data = {
        "schema_version": "1.2",
        "project_id": "project_rgbt_003",
        "node_id": "rgbt_formal_node_001",
        "title": "RGB formal",
        "research_goal": "protocol gate",
        "task_type": "rgbt_detection",
        "protocol_id": "protocol_rgbt_001",
        "environment_key": "rgbt-detection-v2",
        "code_reference": "image:rgbt-detection-v2",
        "code_version": "v0.8.0",
        "dataset_reference": "dataset:rgbt_fast_eval_v1",
        "execution_mode": "fast_eval",
        "parameters": {
            "baseline": "dfine_s",
            "input_mode": "rgb",
            "fusion_method": "none",
            "epochs": 5,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "image_width": 640,
            "image_height": 512,
        },
        "task_config": {
            "primary_metric": "mAP50_95",
            "claim_level": "exploratory_comparison",
            "split_reference": "split:fast_eval_fixed_v1",
            "dataset_version": "v1",
        },
        "seed": 42,
    }
    data.update(overrides)
    return ExperimentContract.model_validate(data)


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


def test_protocol_create_list_show(tmp_path: Path):
    service = _service(tmp_path)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(_protocol_payload()), encoding="utf-8")

    created = service.create_protocol(path)
    assert created["protocol_id"] == "protocol_rgbt_001"

    listed = service.list_protocols(project_id="project_rgbt_003")
    assert len(listed) == 1
    assert listed[0]["title"].startswith("RGB-T")

    shown = service.show_protocol("protocol_rgbt_001")
    assert shown["seeds"] == [42, 43, 44]


def test_validate_protocol_ok_and_with_contract(tmp_path: Path):
    service = _service(tmp_path)
    service.protocols.create_from_dict(_protocol_payload())
    report = service.validate_protocol("protocol_rgbt_001")
    assert report["valid"] is True

    contract_path = tmp_path / "c.json"
    contract_path.write_text(
        _compliant_contract().model_dump_json(), encoding="utf-8"
    )
    report2 = service.validate_protocol(
        "protocol_rgbt_001", contract_path=contract_path
    )
    assert report2["valid"] is True


def test_seed_not_in_protocol_blocks(tmp_path: Path):
    service = _service(tmp_path)
    service.protocols.create_from_dict(_protocol_payload())
    contract = _compliant_contract(seed=99)
    with pytest.raises(ProtocolViolationError) as exc:
        service.protocols.enforce_contract(contract)
    assert "seed 99" in str(exc.value)


def test_disallowed_parameter_blocks():
    protocol = ExperimentProtocol.model_validate(
        {**_protocol_payload(), "created_at": "2026-07-21T00:00:00+00:00"}
    )
    contract = _compliant_contract()
    contract.parameters["epochs"] = 10
    report = ProtocolVerifier().verify_contract(protocol, contract)
    assert report.valid is False
    assert any("epochs" in issue for issue in report.blocking_issues)


def test_extra_non_allowed_parameter_blocks():
    protocol = ExperimentProtocol.model_validate(
        {**_protocol_payload(), "created_at": "2026-07-21T00:00:00+00:00"}
    )
    contract = _compliant_contract()
    contract.parameters["dropout"] = 0.1
    report = ProtocolVerifier().verify_contract(protocol, contract)
    assert report.valid is False
    assert any("dropout" in issue for issue in report.blocking_issues)


def test_code_version_mismatch_blocks():
    protocol = ExperimentProtocol.model_validate(
        {**_protocol_payload(), "created_at": "2026-07-21T00:00:00+00:00"}
    )
    contract = _compliant_contract(code_version="v0.7.0")
    report = ProtocolVerifier().verify_contract(protocol, contract)
    assert report.valid is False
    assert any("code_version" in issue for issue in report.blocking_issues)


def test_code_version_missing_warns():
    protocol = ExperimentProtocol.model_validate(
        {**_protocol_payload(), "created_at": "2026-07-21T00:00:00+00:00"}
    )
    contract = _compliant_contract(code_version=None)
    report = ProtocolVerifier().verify_contract(protocol, contract)
    assert report.valid is True
    assert any("code_version missing" in w for w in report.warnings)


def test_dataset_mismatch_blocks():
    protocol = ExperimentProtocol.model_validate(
        {**_protocol_payload(), "created_at": "2026-07-21T00:00:00+00:00"}
    )
    contract = _compliant_contract(dataset_reference="dataset:other")
    report = ProtocolVerifier().verify_contract(protocol, contract)
    assert report.valid is False
    assert any("dataset_reference" in issue for issue in report.blocking_issues)


def test_no_protocol_id_skips_enforce(tmp_path: Path):
    service = _service(tmp_path)
    contract = _compliant_contract(protocol_id=None)
    assert service.protocols.enforce_contract(contract) is None


def test_run_contract_refuses_protocol_violation(tmp_path: Path, monkeypatch):
    service = _service(tmp_path)
    service.protocols.create_from_dict(_protocol_payload())
    contract = _compliant_contract(seed=7)

    def _boom(*_a, **_k):
        raise AssertionError("runner should not be called")

    monkeypatch.setattr(service, "_select_runner", _boom)
    with pytest.raises(ProtocolViolationError):
        service.run_contract(contract, wait=False)


def test_example_protocol_json_loads():
    root = Path(__file__).resolve().parents[2]
    path = root / "examples" / "rgbt_protocol.json"
    assert path.is_file()
    ProtocolService  # noqa: B018 — import kept for clarity
    data = json.loads(path.read_text(encoding="utf-8"))
    data["created_at"] = "2026-07-21T00:00:00+00:00"
    protocol = ExperimentProtocol.model_validate(data)
    assert protocol.protocol_id == "protocol_rgbt_001"

    compliant = root / "examples" / "rgbt_protocol_compliant_rgb_contract.json"
    contract = ExperimentContract.model_validate(
        json.loads(compliant.read_text(encoding="utf-8"))
    )
    report = ProtocolVerifier().verify_contract(protocol, contract)
    assert report.valid is True, report.blocking_issues
