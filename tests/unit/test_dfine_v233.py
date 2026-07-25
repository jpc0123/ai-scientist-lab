"""v2.3.3 unit tests — DFINE Fast Eval Evidence annotation (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, new_id, utc_now_iso
from scientist_lab.domain import NodeStage, NodeStatus, NodeType, JobStatus
from scientist_lab.evidence.claim_matrix import evaluate_formal_dfine_claim
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.tasks.rgbt_detection.dfine_evidence import (
    build_dfine_fast_eval_evidence,
    classify_dfine_backend,
    record_dfine_fast_eval_evidence,
)
from scientist_lab.tasks.rgbt_detection.vendor_audit import BASELINE_IMPLEMENTATION

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def service(tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=tmp_path / "ev.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def test_classify_vendor_vs_standin():
    assert (
        classify_dfine_backend(
            execution_metadata={"standin_or_vendor": "vendor"}
        )
        == "vendor"
    )
    assert (
        classify_dfine_backend(
            execution_metadata={"dfine_backend_requested": "standin"}
        )
        == "standin"
    )
    assert (
        classify_dfine_backend(
            execution_metadata={"dfine_backend_requested": "dfine"},
            model_summary={"baseline_implementation": BASELINE_IMPLEMENTATION},
        )
        == "vendor"
    )


def test_vendor_evidence_not_standin_blocked():
    record = build_dfine_fast_eval_evidence(
        project_id="project_v233",
        execution_id="exec_vendor_1",
        node_id="node_1",
        metrics={"mAP50_95": 0.12, "mAP50": 0.2},
        contract={
            "project_id": "project_v233",
            "execution_mode": "fast_eval",
            "environment_key": "rgbt-detection-v2-cuda",
            "parameters": {"dfine_backend": "dfine", "baseline": "dfine_s"},
            "task_config": {
                "claim_level": "exploratory_comparison",
                "evaluation_scope": "fast_eval_subset",
                "primary_metric": "mAP50_95",
            },
        },
        execution_metadata={
            "standin_or_vendor": "vendor",
            "dfine_backend_requested": "dfine",
            "baseline_implementation": BASELINE_IMPLEMENTATION,
            "claim_level": "exploratory_comparison",
        },
    )
    assert record.metric_summary["standin_or_vendor"] == "vendor"
    assert record.evidence_strength == "weak"
    assert not any(
        "Stand-in implementation was used" in item for item in record.limitations
    )
    claim = evaluate_formal_dfine_claim(
        project_id="project_v233", records=[record]
    )
    # Non-stand-in → not blocked; still unsupported (incomplete formal acceptance)
    assert claim.support_status == "unsupported"


def test_standin_evidence_still_blocked():
    record = build_dfine_fast_eval_evidence(
        project_id="project_v233",
        execution_id="exec_standin_1",
        metrics={"mAP50_95": 0.01},
        contract={
            "project_id": "project_v233",
            "execution_mode": "fast_eval",
            "parameters": {"dfine_backend": "standin"},
            "task_config": {"claim_level": "exploratory_comparison"},
        },
        execution_metadata={"standin_or_vendor": "standin"},
    )
    claim = evaluate_formal_dfine_claim(
        project_id="project_v233", records=[record]
    )
    assert claim.support_status == "blocked"


def test_record_evidence_persists(service: ExperimentService):
    now = utc_now_iso()
    project_id = "project_v233_persist"
    node_id = new_id("node")
    execution_id = new_id("exec")
    service.repo.upsert_node(
        ExperimentNode(
            node_id=node_id,
            project_id=project_id,
            node_type=NodeType.SMOKE,
            stage=NodeStage.DONE,
            status=NodeStatus.SUCCEEDED,
            hypothesis="vendor fast eval",
            contract_json={
                "project_id": project_id,
                "node_id": node_id,
                "execution_mode": "fast_eval",
                "environment_key": "rgbt-detection-v2-cuda",
                "parameters": {"dfine_backend": "dfine"},
                "task_config": {
                    "claim_level": "exploratory_comparison",
                    "primary_metric": "mAP50_95",
                },
            },
            created_at=now,
            updated_at=now,
        )
    )
    service.repo.upsert_attempt(
        ExecutionAttempt(
            execution_id=execution_id,
            node_id=node_id,
            attempt_index=0,
            runner_profile="remote_gpu_01",
            status=JobStatus.COMPLETED,
            image_reference="scientist-rgbt-detection:v2-cuda",
            result_json={"metrics": {"mAP50_95": 0.11, "mAP50": 0.18}},
            started_at=now,
            completed_at=now,
            created_at=now,
        )
    )
    meta_dir = (
        Path(service.settings.outputs_dir) / project_id / "_dfine_cuda_runs"
    )
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{execution_id}.json").write_text(
        __import__("json").dumps(
            {
                "execution_id": execution_id,
                "standin_or_vendor": "vendor",
                "dfine_backend_requested": "dfine",
                "baseline_implementation": BASELINE_IMPLEMENTATION,
                "claim_level": "exploratory_comparison",
                "exploratory_only": True,
                "formal_success": False,
            }
        ),
        encoding="utf-8",
    )

    pkg = record_dfine_fast_eval_evidence(
        service,
        execution_id=execution_id,
        refresh_claim_matrix=True,
    )
    assert pkg["non_standin"] is True
    assert pkg["formal_success"] is False
    assert pkg["formal_dfine_claim"]["support_status"] == "unsupported"
    assert Path(pkg["evidence_path"]).is_file()
    shown = service.show_evidence(pkg["evidence_id"])
    assert shown["metric_summary"]["standin_or_vendor"] == "vendor"


def test_service_facade(service: ExperimentService):
    # Minimal: record without prior attempt should fail clearly
    with pytest.raises((ValueError, KeyError)):
        service.dfine_cuda_record_evidence("exec_missing")
