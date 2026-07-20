from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCIENTIST_WORKER_")

    worker_id: str = "gpu-worker-01"
    version: str = "0.8.5"
    host: str = "127.0.0.1"
    port: int = 8080

    data_root: Path = Path("runtime/scientist-worker")
    auth_token: str | None = None
    require_auth: bool = False

    supported_environment_keys: list[str] = [
        "rgbt-detection-v2",
        "mock-detection-v1",
    ]
    max_cpu_count: int = 16
    max_memory_gb: int = 64
    max_gpu_count: int = 1
    network_policy: str = "disabled_by_default"

    # mock | docker | auto (mock for mock-detection-v1, docker otherwise)
    executor_mode: str = "auto"
    allow_cpu_fallback: bool = True

    # JSON-friendly maps via env are awkward; use defaults resolved from project.
    dataset_registry: dict[str, str] = Field(default_factory=dict)
    code_roots: dict[str, str] = Field(default_factory=dict)
    image_registry: dict[str, str] = Field(
        default_factory=lambda: {
            "mock-detection-v1": "scientist-mock-detection:v1",
            "rgbt-detection-v2": "scientist-rgbt-detection:v2",
        }
    )

    project_root: Path | None = None

    def resolve(self) -> "WorkerSettings":
        self.data_root = Path(self.data_root).resolve()
        (self.data_root / "jobs").mkdir(parents=True, exist_ok=True)
        (self.data_root / "database").mkdir(parents=True, exist_ok=True)
        (self.data_root / "temp").mkdir(parents=True, exist_ok=True)

        root = Path(self.project_root).resolve() if self.project_root else None
        if root is None:
            # worker/scientist_worker/settings.py -> worker -> scientist-lab
            root = Path(__file__).resolve().parents[2]
            self.project_root = root

        if not self.code_roots:
            self.code_roots = {
                "local:rgbt_detection_real": str(
                    root / "experiment_apps" / "rgbt_detection_real"
                ),
                "local:rgbt_detector": str(root / "rgbt_detector"),
                "image:rgbt-detection-v2": str(
                    root / "experiment_apps" / "rgbt_detection_real"
                ),
            }
        if not self.dataset_registry:
            candidates = {
                "rgbt_debug_v1": root / "datasets" / "rgbt_debug_v1",
                "rgbt_fast_eval_v1": root / "datasets" / "rgbt_fast_eval_v1",
            }
            self.dataset_registry = {
                key: str(path.resolve())
                for key, path in candidates.items()
                if path.exists()
            }
        return self

    @property
    def db_path(self) -> Path:
        return self.data_root / "database" / "worker.db"

    @property
    def jobs_root(self) -> Path:
        return self.data_root / "jobs"
