from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.settings import Settings
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.tasks.rgbt_detection.fast_eval_triad import (
    build_triad_comparison,
    validate_triad_contracts,
)


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def test_validate_local_fast_eval_triad_contracts():
    contracts = {
        "rgb": _load("rgbt_fast_rgb_contract.json"),
        "thermal": _load("rgbt_fast_thermal_contract.json"),
        "fusion": _load("rgbt_fast_fusion_contract.json"),
    }
    result = validate_triad_contracts(contracts)
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_remote_fast_eval_triad_contracts():
    contracts = {
        "rgb": _load("rgbt_remote_docker_fast_rgb_contract.json"),
        "thermal": _load("rgbt_remote_docker_fast_thermal_contract.json"),
        "fusion": _load("rgbt_remote_docker_fast_fusion_contract.json"),
    }
    result = validate_triad_contracts(contracts)
    assert result["ok"] is True


def test_validate_triad_detects_budget_mismatch():
    contracts = {
        "rgb": _load("rgbt_fast_rgb_contract.json"),
        "thermal": _load("rgbt_fast_thermal_contract.json"),
        "fusion": _load("rgbt_fast_fusion_contract.json"),
    }
    contracts["fusion"]["parameters"]["epochs"] = 99
    result = validate_triad_contracts(contracts)
    assert result["ok"] is False
    assert any("epochs" in issue for issue in result["issues"])


def test_build_triad_comparison_report():
    contracts = {
        "rgb": _load("rgbt_fast_rgb_contract.json"),
        "thermal": _load("rgbt_fast_thermal_contract.json"),
        "fusion": _load("rgbt_fast_fusion_contract.json"),
    }
    metrics = {
        "rgb": {"mAP50_95": 0.10, "mAP50": 0.20, "AP_small": 0.05},
        "thermal": {"mAP50_95": 0.08, "mAP50": 0.15, "AP_small": 0.04},
        "fusion": {"mAP50_95": 0.12, "mAP50": 0.22, "AP_small": 0.07},
    }
    report = build_triad_comparison(
        contracts=contracts,
        metrics_by_role=metrics,
        execution_ids={"rgb": "e1", "thermal": "e2", "fusion": "e3"},
    )
    assert report["claim_level"] == "exploratory_comparison"
    assert report["evidence_strength"] == "weak"
    assert report["fairness"]["ok"] is True
    assert report["ranking_by_primary_metric"][0]["role"] == "fusion"
    assert report["pairwise"]["rgb_vs_fusion"]["deltas_candidate_minus_baseline"][
        "mAP50_95"
    ] == 0.02
    assert any("SOTA" in (c or "") or "state-of-the-art" in (c or "").lower() for c in report["forbidden_claims"])


def test_service_validate_fast_eval_triad_cli_path(tmp_path: Path):
    settings = Settings(
        project_root=EXAMPLES.parent,
        db_path=tmp_path / "lab.db",
        outputs_dir=tmp_path / "outputs",
        runtime_dir=tmp_path / "runtime",
        experiment_app_dir=tmp_path / "app",
    ).resolve()
    (tmp_path / "app").mkdir()
    service = ExperimentService(settings)
    result = service.validate_fast_eval_triad_contracts()
    assert result["ok"] is True
    assert "rgb" in result["paths"]
