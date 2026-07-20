"""Mock job executor for v0.8.3 (no Docker / no GPU required)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from scientist_worker.artifact_packager import load_job_artifact_policy, package_artifacts
from scientist_worker.log_store import LogStore
from scientist_worker.models import JobRecord


def run_mock_job(record: JobRecord, *, cancelled=None) -> dict[str, Any]:
    output_dir = Path(record.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = LogStore(Path(record.log_path))

    logs.append(f"[worker] mock job start job_id={record.job_id}")
    contract = record.contract_json or {}
    node_id = contract.get("node_id", "mock_node")
    project_id = contract.get("project_id", "project_mock")

    stages = [
        ("preparing", 0.1),
        ("running", 0.5),
        ("collecting", 0.9),
    ]
    for stage, progress in stages:
        if cancelled and cancelled():
            logs.append("[worker] cancel detected")
            raise RuntimeError("cancelled")
        logs.append(f"[worker] stage={stage} progress={progress}")
        time.sleep(0.05)

    metrics = {
        "schema_version": "1.0",
        "project_id": project_id,
        "node_id": node_id,
        "task_type": contract.get("task_type") or "rgbt_detection",
        "primary_metric": "mAP50_95",
        "metrics": {
            "mAP50": 0.01,
            "mAP50_95": 0.005,
            "AP_small": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "duration_seconds": 0.2,
        },
        "training": {
            "seed": int(contract.get("seed") or 42),
            "epochs_requested": int((contract.get("parameters") or {}).get("epochs") or 0),
            "epochs_completed": 0,
            "trained": False,
            "backend": "worker_mock",
        },
        "evaluation_scope": ((contract.get("task_config") or {}).get("evaluation_scope")
                             or "debug_subset"),
        "claim_level": ((contract.get("task_config") or {}).get("claim_level")
                        or "pipeline_validation_only"),
        "execution_mode": contract.get("execution_mode") or "fast_eval",
        "status": "completed",
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "model_summary.json",
        {
            "baseline_key": (contract.get("parameters") or {}).get("baseline", "mock"),
            "baseline_implementation": "worker_mock_v0_8_3",
            "parameter_count": 0,
        },
    )
    _write_json(
        output_dir / "resource_usage.json",
        {
            "gpu_count_requested": 0,
            "peak_gpu_memory_mb": 0.0,
            "duration_seconds": 0.2,
            "worker_id": "mock",
            "executor": "mock",
        },
    )
    _write_json(
        output_dir / "execution.json",
        {
            "execution_id": record.execution_id,
            "job_id": record.job_id,
            "runner_profile": "remote_docker",
            "environment_key": record.environment_key,
            "status": "completed",
            "worker_mode": "mock",
        },
    )
    _write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": "1.0",
            "files": [
                {"path": "metrics.json", "artifact_type": "metrics"},
                {"path": "model_summary.json", "artifact_type": "summary"},
                {"path": "resource_usage.json", "artifact_type": "resource_usage"},
                {"path": "execution.json", "artifact_type": "execution"},
            ],
        },
    )
    (output_dir / "combined.log").write_text(
        logs.log_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    policy = load_job_artifact_policy(record)
    packaged = package_artifacts(
        output_dir,
        job_id=record.job_id,
        execution_id=record.execution_id,
        policy=policy,
    )
    logs.append("[worker] mock job completed")
    return {
        "metrics": metrics,
        "artifact_bundle": packaged,
        "status": "completed",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
