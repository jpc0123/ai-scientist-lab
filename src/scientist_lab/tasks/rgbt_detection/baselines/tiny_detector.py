"""Debug tiny_detector baseline metadata (v0.7 path)."""

from __future__ import annotations

from typing import Any

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.baseline_adapter import register_baseline


@register_baseline
class TinyDetectorBaseline:
    baseline_key = "tiny_detector"

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        return

    def build_native_config(self, contract: ExperimentContract) -> dict[str, Any]:
        return {
            "baseline_key": self.baseline_key,
            "backend": "numpy_tiny_detector",
            "parameters": dict(contract.parameters or {}),
        }

    def build_command_notes(self, contract: ExperimentContract) -> list[str]:
        return ["baseline=tiny_detector (v0.7 debug path)"]

    def expected_checkpoint_names(self) -> list[str]:
        return ["checkpoint/last.npz"]
