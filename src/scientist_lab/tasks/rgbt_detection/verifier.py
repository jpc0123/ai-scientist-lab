from __future__ import annotations

from pathlib import Path
from typing import Any

from scientist_lab.tasks.rgbt_detection.result_parser import DetectionResultParser


class DetectionVerifier:
    REQUIRED = (
        "metrics.json",
        "artifact_manifest.json",
        "dataset_report.json",
        "execution.json",
    )

    def verify(self, output_dir: Path) -> dict[str, Any]:
        missing = [name for name in self.REQUIRED if not (output_dir / name).exists()]
        issues: list[str] = []
        if missing:
            issues.append(f"missing artifacts: {missing}")
        try:
            metrics = DetectionResultParser().parse(output_dir)
        except Exception as exc:  # noqa: BLE001
            issues.append(str(exc))
            metrics = {}

        training = (metrics or {}).get("training") or {}
        if training.get("nan_loss"):
            issues.append("nan_loss detected during training")
        if metrics.get("status") == "failed":
            issues.append(str(metrics.get("error") or "evaluation_failed"))

        return {
            "valid": not issues,
            "issues": issues,
            "metrics": metrics,
        }
