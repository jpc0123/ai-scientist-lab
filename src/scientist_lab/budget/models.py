from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExperimentBudget(BaseModel):
    project_id: str

    max_new_nodes: int = 3
    max_total_executions: int = 15
    max_gpu_hours: float = 10.0
    max_storage_gb: float = 20.0

    used_nodes: int = 0
    used_executions: int = 0
    used_gpu_hours: float = 0.0
    used_storage_gb: float = 0.0

    updated_at: datetime | None = None

    @field_validator("project_id")
    @classmethod
    def _ok(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("project_id required")
        return text

    def remaining(self) -> dict[str, Any]:
        return {
            "max_new_nodes": max(0, self.max_new_nodes - self.used_nodes),
            "max_total_executions": max(
                0, self.max_total_executions - self.used_executions
            ),
            "max_total_gpu_hours": max(0.0, self.max_gpu_hours - self.used_gpu_hours),
            "max_storage_gb": max(0.0, self.max_storage_gb - self.used_storage_gb),
            "max_seeds_per_node": 3,
        }

    def can_afford(self, *, nodes: int = 1, gpu_hours: float = 0.0) -> bool:
        rem = self.remaining()
        return rem["max_new_nodes"] >= nodes and rem["max_total_gpu_hours"] >= float(
            gpu_hours
        )
