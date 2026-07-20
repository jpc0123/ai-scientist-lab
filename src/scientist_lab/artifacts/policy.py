from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field, field_validator


class ArtifactPolicy(BaseModel):
    """Remote/local artifact download & packaging policy (v0.8.6)."""

    schema_version: str = "1.0"
    include_metrics: bool = True
    include_logs: bool = True
    include_previews: bool = True
    include_best_checkpoint: bool = True
    include_last_checkpoint: bool = True
    include_intermediate_checkpoints: bool = False
    max_bundle_size_gb: float = 2.0
    required_paths: list[str] = Field(
        default_factory=lambda: ["metrics.json", "execution.json"]
    )

    @field_validator("max_bundle_size_gb")
    @classmethod
    def _size_ok(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("max_bundle_size_gb must be > 0")
        return float(value)

    def max_bundle_size_bytes(self) -> int:
        return int(self.max_bundle_size_gb * (1024**3))

    def should_include(self, relative_path: str) -> bool:
        rel = PurePosixPath(relative_path.replace("\\", "/")).as_posix()
        if ".." in rel.split("/"):
            return False

        name = Path(rel).name.lower()
        if rel.startswith("checkpoint/") or "/checkpoint/" in f"/{rel}":
            if _is_intermediate_checkpoint(rel, name):
                return self.include_intermediate_checkpoints
            if "best" in name:
                return self.include_best_checkpoint
            if "last" in name:
                return self.include_last_checkpoint
            # unknown checkpoint files follow intermediate policy
            return self.include_intermediate_checkpoints

        if rel.startswith("previews/") or rel.startswith("preview/"):
            return self.include_previews

        if rel.endswith(".log") or name in {
            "stdout.log",
            "stderr.log",
            "combined.log",
            "worker.log",
        }:
            return self.include_logs

        if name in {
            "metrics.json",
            "training_history.csv",
            "model_summary.json",
            "resource_usage.json",
            "detection_metrics.json",
        }:
            return self.include_metrics

        # Default: keep small control-plane metadata / summaries.
        return True


def default_artifact_policy() -> ArtifactPolicy:
    return ArtifactPolicy()


def _is_intermediate_checkpoint(rel: str, name: str) -> bool:
    lowered = rel.lower()
    if "/intermediate/" in f"/{lowered}":
        return True
    if name.startswith(("epoch_", "step_", "ckpt_", "checkpoint_")):
        return True
    if name.startswith("epoch") and name[5:6].isdigit():
        return True
    return False
