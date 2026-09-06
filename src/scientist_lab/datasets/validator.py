from __future__ import annotations

from pathlib import Path
from typing import Any

from scientist_lab.datasets.models import DatasetRegistration
from scientist_lab.datasets.preview_generator import VALIDATOR_VERSION
from scientist_lab.datasets.rgbt_validator import validate_rgbt_dataset
from scientist_lab.domain.models import utc_now_iso
from scientist_lab.storage.artifact_store import sha256_file, write_json


def validate_dataset(
    registration: DatasetRegistration,
    *,
    outputs_dir: Path,
    write_previews: bool = True,
    preview_count: int = 3,
) -> dict[str, Any]:
    """Dispatch dataset validation by task_type and persist report + artifact meta."""
    out_dir = outputs_dir / "datasets" / registration.dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if registration.task_type == "rgbt_detection":
        report = validate_rgbt_dataset(
            Path(registration.host_path),
            dataset_key=registration.dataset_key,
            write_previews=write_previews,
            preview_dir=out_dir,
            preview_count=preview_count,
        )
    else:
        report = {
            "schema_version": "1.0",
            "dataset_key": registration.dataset_key,
            "valid": True,
            "task_type": registration.task_type,
            "warnings": [
                f"No specialized validator for task_type={registration.task_type}; "
                "only path existence was checked."
            ],
            "claim_level": "pipeline_validation_only",
            "validator_version": VALIDATOR_VERSION,
        }
        if not Path(registration.host_path).exists():
            report["valid"] = False
            report["errors"] = [f"host_path missing: {registration.host_path}"]

    report_path = out_dir / "dataset_report.json"
    write_json(report_path, report)

    artifact = {
        "artifact_type": "dataset_report",
        "path": str(report_path),
        "relative_path": f"datasets/{registration.dataset_key}/dataset_report.json",
        "size_bytes": report_path.stat().st_size,
        "sha256": sha256_file(report_path),
        "dataset_key": registration.dataset_key,
        "validated_at": utc_now_iso(),
        "validator_version": report.get("validator_version", VALIDATOR_VERSION),
    }
    artifact_path = out_dir / "dataset_report_artifact.json"
    write_json(artifact_path, artifact)

    report["report_path"] = str(report_path)
    report["artifact"] = artifact
    return report


def preview_dataset(
    registration: DatasetRegistration,
    *,
    outputs_dir: Path,
    count: int = 5,
) -> dict[str, Any]:
    """Generate pairing previews without re-running full validation semantics."""
    report = validate_dataset(
        registration,
        outputs_dir=outputs_dir,
        write_previews=True,
        preview_count=count,
    )
    return {
        "dataset_key": registration.dataset_key,
        "preview_paths": report.get("preview_paths") or [],
        "report_path": report.get("report_path"),
        "valid": report.get("valid"),
    }
