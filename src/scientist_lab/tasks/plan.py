from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scientist_lab.datasets.models import DatasetRegistration
from scientist_lab.domain.contracts import ExperimentContract


@dataclass
class MountSpec:
    source: str
    target: str
    read_only: bool = True


@dataclass
class TaskExecutionPlan:
    command: list[str]
    mounts: list[MountSpec] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    workspace_source: Path | None = None
    notes: list[str] = field(default_factory=list)


def default_execution_plan(contract: ExperimentContract) -> TaskExecutionPlan:
    return TaskExecutionPlan(
        command=[
            contract.entrypoint,
            "--config",
            "/outputs/config.json",
            "--output-dir",
            "/outputs",
            "--seed",
            str(contract.seed),
        ]
    )


def prepare_execution_plan(
    contract: ExperimentContract,
    *,
    dataset: DatasetRegistration | None,
    code_roots: dict[str, Path],
) -> TaskExecutionPlan:
    if contract.task_type == "rgbt_detection":
        from scientist_lab.tasks.rgbt_detection.adapter import RGBTDetectionAdapter

        return RGBTDetectionAdapter().prepare_execution(
            contract, dataset, code_roots=code_roots
        )
    return default_execution_plan(contract)
