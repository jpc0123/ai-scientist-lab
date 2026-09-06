from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCIENTIST_LAB_")

    project_root: Path = Path(__file__).resolve().parents[2]
    db_path: Path | None = None
    runtime_dir: Path | None = None
    outputs_dir: Path | None = None
    experiment_app_dir: Path | None = None
    image_registry: dict[str, str] = {
        "scientist-experiment-v1": "scientist-experiment:v1",
        "digits-mlp-v1": "scientist-experiment:v2",
        "rgbt-detection-v1": "scientist-rgbt-detection:v1",
        "rgbt-detection-v2": "scientist-rgbt-detection:v2",
        "rgbt-detection-v2-cuda": "scientist-rgbt-detection:v2-cuda",
    }
    poll_interval_seconds: float = 0.5
    rgbt_detector_dir: Path | None = None
    rgbt_detection_real_dir: Path | None = None
    data_workspace_dir: Path | None = None

    def resolve(self) -> "Settings":
        root = self.project_root
        if self.db_path is None:
            self.db_path = root / "scientist_lab.db"
        if self.runtime_dir is None:
            self.runtime_dir = root / "runtime"
        if self.outputs_dir is None:
            self.outputs_dir = root / "outputs"
        if self.experiment_app_dir is None:
            self.experiment_app_dir = root / "experiment_app"
        if self.rgbt_detector_dir is None:
            self.rgbt_detector_dir = root / "rgbt_detector"
        if self.rgbt_detection_real_dir is None:
            self.rgbt_detection_real_dir = (
                root / "experiment_apps" / "rgbt_detection_real"
            )
        if self.data_workspace_dir is None:
            self.data_workspace_dir = root / "data"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self


def get_settings() -> Settings:
    return Settings().resolve()
