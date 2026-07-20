from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.baseline_adapter import (
    ensure_baselines_loaded,
    get_baseline_adapter,
    is_real_baseline,
    list_baseline_keys,
    resolve_baseline_key,
)
from scientist_lab.tasks.rgbt_detection.baseline_config_builder import (
    build_native_config,
)
from scientist_lab.tasks.rgbt_detection.baseline_result_parser import (
    parse_baseline_results,
)
from scientist_lab.tasks.rgbt_detection.adapter import RGBTDetectionAdapter


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "rgbt_dfine_s_minimal_contract.json"


def test_baseline_registry_contains_dfine_and_tiny():
    ensure_baselines_loaded()
    keys = list_baseline_keys()
    assert "dfine_s" in keys
    assert "tiny_detector" in keys
    assert is_real_baseline("dfine_s")
    assert not is_real_baseline("tiny_detector")


def test_dfine_adapter_builds_native_config():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    contract = ExperimentContract.model_validate(data)
    assert resolve_baseline_key(contract) == "dfine_s"
    native = build_native_config(contract)
    assert native["baseline_key"] == "dfine_s"
    assert native["baseline_implementation"] == "torch_mini_standin_v0_8_1"
    assert native["epochs"] == 2
    notes = get_baseline_adapter("dfine_s").build_command_notes(contract)
    assert any("dfine_s" in n for n in notes)


def test_adapter_routes_real_baseline_workspace(tmp_path: Path):
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    contract = ExperimentContract.model_validate(data)
    from scientist_lab.datasets.models import DatasetRegistration
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    dataset = DatasetRegistration(
        dataset_key="rgbt_debug_v1",
        task_type="rgbt_detection",
        host_path=str(tmp_path / "data"),
        container_path="/data/rgbt_debug_v1",
        read_only=True,
        enabled=True,
        created_at=now,
        updated_at=now,
    )
    (tmp_path / "data").mkdir()
    real_root = ROOT / "experiment_apps" / "rgbt_detection_real"
    plan = RGBTDetectionAdapter().prepare_execution(
        contract,
        dataset,
        code_roots={
            "local:rgbt_detection_real": real_root,
            "local:rgbt_detector": ROOT / "rgbt_detector",
        },
    )
    assert plan.workspace_source == real_root.resolve() or plan.workspace_source == real_root
    assert plan.environment["BASELINE_KEY"] == "dfine_s"


def test_parse_baseline_results(tmp_path: Path):
    for name, payload in {
        "metrics.json": {"status": "completed", "training": {"backend": "torch_mini_standin"}},
        "model_summary.json": {
            "baseline_key": "dfine_s",
            "baseline_implementation": "torch_mini_standin_v0_8_1",
        },
        "resource_usage.json": {"duration_seconds": 1.0},
        "artifact_manifest.json": {"schema_version": "1.0"},
    }.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "checkpoint" / "last.pt").write_bytes(b"fake")
    parsed = parse_baseline_results(tmp_path)
    assert parsed["baseline_key"] == "dfine_s"
    assert parsed["checkpoint_present"] is True


def test_tiny_fast_eval_still_requires_checkpoint_source():
    adapter = RGBTDetectionAdapter()
    with pytest.raises(ValueError, match="checkpoint_source"):
        adapter.validate_contract(
            ExperimentContract.model_validate(
                {
                    "schema_version": "1.1",
                    "project_id": "p",
                    "node_id": "n",
                    "title": "t",
                    "research_goal": "g",
                    "hypothesis": "h",
                    "task_type": "rgbt_detection",
                    "runner_profile": "local",
                    "environment_key": "rgbt-detection-v1",
                    "code_reference": "local:rgbt_detector",
                    "dataset_reference": "dataset:rgbt_debug_v1",
                    "entrypoint": "run_detection_experiment.py",
                    "execution_mode": "fast_eval",
                    "parameters": {"baseline": "tiny_detector", "epochs": 0},
                    "task_config": {"claim_level": "pipeline_validation_only"},
                    "seed": 42,
                    "resources": {
                        "gpu_count": 0,
                        "cpu_count": 1,
                        "memory_gb": 1,
                        "timeout_seconds": 60,
                    },
                    "expected_outputs": ["metrics.json"],
                }
            )
        )
