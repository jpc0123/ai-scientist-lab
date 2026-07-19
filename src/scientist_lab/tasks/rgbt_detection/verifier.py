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
        mode = str(metrics.get("execution_mode") or "")
        if training.get("nan_loss"):
            issues.append("nan_loss")
        if metrics.get("status") == "failed":
            issues.append(str(metrics.get("error") or "evaluation_failed"))

        if mode == "smoke_train":
            if not (output_dir / "training_history.csv").exists():
                issues.append("missing_artifact: training_history.csv")
            ckpt = output_dir / "checkpoint" / "last.npz"
            ckpt_txt = output_dir / "checkpoint" / "last_smoke.txt"
            if not ckpt.exists() and not ckpt_txt.exists():
                issues.append("checkpoint_missing")
            final_loss = training.get("final_train_loss", training.get("final_loss"))
            if final_loss is not None:
                try:
                    value = float(final_loss)
                except (TypeError, ValueError):
                    issues.append("invalid_detection_metrics: non-numeric loss")
                else:
                    if value != value or value in (float("inf"), float("-inf")):
                        issues.append("nan_loss")

        return {
            "valid": not issues,
            "issues": issues,
            "metrics": metrics,
        }
