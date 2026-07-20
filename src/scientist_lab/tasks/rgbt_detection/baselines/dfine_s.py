"""DFINE-S baseline adapter.

v0.8.1 ships a torch mini-detector stand-in under baseline_key=dfine_s.
Replace experiment_apps/rgbt_detection_real torch stand-in with vendored DFINE
when available; keep this adapter's contract surface stable.
"""

from __future__ import annotations

from typing import Any

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.baseline_adapter import register_baseline


@register_baseline
class DFineSBaselineAdapter:
    baseline_key = "dfine_s"

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        epochs = parameters.get("epochs")
        if epochs is not None:
            try:
                value = int(epochs)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid epochs: {epochs!r}") from exc
            if value < 0:
                raise ValueError("epochs must be >= 0")
        batch = parameters.get("batch_size")
        if batch is not None and int(batch) < 1:
            raise ValueError("batch_size must be >= 1")
        mode = str(parameters.get("input_mode", "rgb"))
        if mode not in {"rgb", "thermal", "rgbt"}:
            raise ValueError(f"unsupported input_mode: {mode}")

    def build_native_config(self, contract: ExperimentContract) -> dict[str, Any]:
        params = dict(contract.parameters or {})
        task_config = dict(contract.task_config or {})
        return {
            "baseline_key": self.baseline_key,
            "baseline_implementation": "torch_mini_standin_v0_8_1",
            "vendor_status": "standin_until_dfine_vendored",
            "input_mode": params.get("input_mode", "rgb"),
            "fusion_method": params.get("fusion_method", "none"),
            "epochs": int(params.get("epochs", 2)),
            "batch_size": int(params.get("batch_size", 2)),
            "learning_rate": float(params.get("learning_rate", 1e-3)),
            "image_width": int(params.get("image_width", 160)),
            "image_height": int(params.get("image_height", 128)),
            "max_train_images": int(params.get("max_train_images", 16)),
            "max_val_images": int(params.get("max_val_images", 8)),
            "num_workers": int(params.get("num_workers", 0)),
            "mixed_precision": bool(params.get("mixed_precision", False)),
            "primary_metric": task_config.get("primary_metric", "mAP50_95"),
            "claim_level": task_config.get(
                "claim_level", "pipeline_validation_only"
            ),
            "evaluation_scope": task_config.get(
                "evaluation_scope", "debug_subset"
            ),
        }

    def build_command_notes(self, contract: ExperimentContract) -> list[str]:
        return [
            "baseline=dfine_s",
            "implementation=torch_mini_standin_v0_8_1 (replace with vendored DFINE-S)",
            f"environment_key={contract.environment_key}",
        ]

    def expected_checkpoint_names(self) -> list[str]:
        return ["checkpoint/last.pt"]
