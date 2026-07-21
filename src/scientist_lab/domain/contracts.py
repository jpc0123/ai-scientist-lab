from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResourceRequest(BaseModel):
    gpu_count: int = Field(default=0, ge=0, le=8)
    cpu_count: int = Field(default=4, ge=1, le=64)
    memory_gb: int = Field(default=8, ge=1)
    timeout_seconds: int = Field(default=3600, ge=5)


class ExperimentContract(BaseModel):
    schema_version: Literal["1.0", "1.1", "1.2"] = "1.0"

    project_id: str
    node_id: str
    parent_node_id: str | None = None

    title: str = "Smoke experiment"
    research_goal: str = "Validate local Docker experiment loop"
    hypothesis: str | None = "Mock learning rate affects accuracy"

    task_type: str | None = None
    task_config: dict[str, Any] = Field(default_factory=dict)

    # Optional link to a project-level ExperimentProtocol (v0.9+).
    protocol_id: str | None = None

    runner_profile: str = "local"
    environment_key: str = "scientist-experiment-v1"
    code_reference: str = "local:experiment_app"
    code_version: str | None = None
    dataset_reference: str = "debug-dataset-v1"

    # Free-form so Digits + RGB-T entrypoints/modes coexist.
    entrypoint: str = "run_experiment.py"
    execution_mode: str = "smoke_test"

    parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    resources: ResourceRequest = Field(default_factory=ResourceRequest)

    expected_outputs: list[str] = Field(
        default_factory=lambda: [
            "metrics.json",
            "execution.json",
            "artifact_manifest.json",
            "combined.log",
        ]
    )
