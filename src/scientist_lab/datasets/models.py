from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetRegistration(BaseModel):
    dataset_key: str
    task_type: str

    host_path: str
    container_path: str

    read_only: bool = True
    enabled: bool = True

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime | str
    updated_at: datetime | str
