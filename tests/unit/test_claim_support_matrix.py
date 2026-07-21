from __future__ import annotations

from pathlib import Path

from scientist_lab.evidence.claim_matrix import (
    build_claim_support_matrix,
    evaluate_sota_claim,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

from tests.unit.test_evidence_record import _bootstrap, _service


def test_claim_missing_evidence_is_unsupported():
    matrix = build_claim_support_matrix(
        project_id="project_rgbt_003",
        records=[],
    )
    by_id = {item.claim_id: item for item in matrix.claims}
    assert by_id["claim_fast_eval_ap_small"].support_status == "unsupported"
    assert by_id["claim_fast_eval_map"].support_status == "unsupported"
    assert "Missing" in (by_id["claim_fast_eval_ap_small"].reason or "")


def test_sota_claim_default_blocked():
    claim = evaluate_sota_claim(project_id="project_rgbt_003", records=[])
    assert claim.support_status == "blocked"
    assert claim.claim_type == "sota"
    assert "benchmark" in (claim.reason or "").lower()


def test_full_benchmark_claim_blocked_under_fast_eval_standin(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    service.build_evidence("rgbt_formal_node_001", "rgbt_formal_node_003")
    matrix = service.build_claim_matrix("project_rgbt_003")
    by_id = {item["claim_id"]: item for item in matrix["claims"]}
    assert by_id["claim_full_rgbt_tiny"]["support_status"] == "blocked"
    assert "Fast Eval" in by_id["claim_full_rgbt_tiny"]["reason"]
    assert by_id["claim_sota"]["support_status"] == "blocked"
    assert by_id["claim_formal_dfine"]["support_status"] == "blocked"
    assert by_id["claim_fast_eval_ap_small"]["support_status"] == "supported"
    assert by_id["claim_fast_eval_map"]["support_status"] == "supported"
    assert by_id["claim_fast_eval_ap_small"]["evidence"]


def test_claim_matrix_survives_service_restart(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    service.build_evidence("rgbt_formal_node_001", "rgbt_formal_node_003")
    built = service.build_claim_matrix("project_rgbt_003")
    matrix_path = Path(built["matrix_path"])
    assert matrix_path.is_file()

    # New service instance sharing the same db + outputs (simulates restart).
    root = Path(__file__).resolve().parents[2]
    restarted = ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "test.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    shown = restarted.show_claim_matrix("project_rgbt_003")
    assert shown["project_id"] == "project_rgbt_003"
    assert shown["recovered_from"] == "sqlite"
    assert len(shown["claims"]) == len(built["claims"])
    statuses = {item["claim_id"]: item["support_status"] for item in shown["claims"]}
    assert statuses["claim_sota"] == "blocked"
    assert statuses["claim_full_rgbt_tiny"] == "blocked"
    assert statuses["claim_fast_eval_ap_small"] == "supported"


def test_claim_matrix_filesystem_fallback(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    service.build_evidence("rgbt_formal_node_001", "rgbt_formal_node_003")
    built = service.build_claim_matrix("project_rgbt_003")

    # Wipe SQLite row by pointing to a fresh empty DB while keeping outputs.
    root = Path(__file__).resolve().parents[2]
    orphan = ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "empty.db",
            runtime_dir=tmp_path / "runtime2",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    shown = orphan.show_claim_matrix("project_rgbt_003")
    assert shown["recovered_from"] == "filesystem"
    assert Path(shown["matrix_path"]).is_file()
    assert shown["claims"][0]["claim"] == built["claims"][0]["claim"]


def test_build_matrix_without_evidence_still_emits_blocked_defaults(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    # Do not build evidence — matrix should still mark strong claims blocked
    # and exploratory claims unsupported.
    payload = service.build_claim_matrix("project_rgbt_003")
    by_id = {item["claim_id"]: item for item in payload["claims"]}
    assert by_id["claim_fast_eval_ap_small"]["support_status"] == "unsupported"
    assert by_id["claim_sota"]["support_status"] == "blocked"
    assert by_id["claim_full_rgbt_tiny"]["support_status"] == "blocked"
