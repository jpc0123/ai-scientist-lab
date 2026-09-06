from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RGBTTaskConfig(BaseModel):
    modalities: list[str] = Field(default_factory=lambda: ["rgb", "thermal"])
    annotation_format: str = "coco"
    primary_metric: str = "mAP50_95"
    metrics: list[str] = Field(
        default_factory=lambda: [
            "mAP50",
            "mAP50_95",
            "AP_small",
            "precision",
            "recall",
        ]
    )
    claim_level: str = "pipeline_validation_only"
    extra: dict[str, Any] = Field(default_factory=dict)
