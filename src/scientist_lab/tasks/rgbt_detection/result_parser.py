from __future__ import annotations

from pathlib import Path
from typing import Any

from scientist_lab.storage.artifact_store import read_json


class DetectionResultParser:
    def parse(self, output_dir: Path) -> dict[str, Any]:
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError("缺少 metrics.json")
        metrics = read_json(metrics_path)
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json 必须是对象")
        if "primary_metric" not in metrics or "metrics" not in metrics:
            raise ValueError("metrics.json 缺少 primary_metric / metrics")
        values = metrics.get("metrics") or {}
        if not isinstance(values, dict):
            raise ValueError("metrics.metrics 必须是对象")
        for key, value in values.items():
            if key.lower().startswith("map") or key in {"precision", "recall", "AP_small"}:
                if not isinstance(value, (int, float)):
                    raise ValueError(f"invalid metric type for {key}")
                if float(value) < 0.0 or float(value) > 1.0:
                    raise ValueError(
                        f"invalid_detection_metrics: {key}={value} outside [0,1]"
                    )
        return metrics
