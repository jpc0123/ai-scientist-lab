from __future__ import annotations

import time
from pathlib import Path

import pytest

from scientist_worker.docker_executor import DockerExecutor, DockerExecutorError
from scientist_worker.models import JobRecord, utc_now_iso
from scientist_worker.service import WorkerService
from scientist_worker.settings import WorkerSettings


ROOT = Path(__file__).resolve().parents[2]


def _docker_v2_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.images.get("scientist-rgbt-detection:v2")
        return (ROOT / "datasets" / "rgbt_fast_eval_v1").exists()
    except Exception:  # noqa: BLE001
        return False


pytestmark_docker = pytest.mark.skipif(
    not _docker_v2_available(),
    reason="需要 Docker 与 scientist-rgbt-detection:v2 以及 rgbt_fast_eval_v1",
)


def test_resolve_executor_mode_auto(tmp_path: Path):
    settings = WorkerSettings(
        data_root=tmp_path / "w",
        executor_mode="auto",
        project_root=ROOT,
    ).resolve()
    service = WorkerService(settings)
    assert service._resolve_executor_mode("mock-detection-v1") == "mock"
    assert service._resolve_executor_mode("rgbt-detection-v2") == "docker"


def test_docker_executor_missing_image(tmp_path: Path):
    if not _docker_v2_available():
        pytest.skip("docker unavailable")
    executor = DockerExecutor(
        image_registry={"rgbt-detection-v2": "scientist-rgbt-detection:does-not-exist"},
        dataset_registry={"rgbt_fast_eval_v1": str(ROOT / "datasets" / "rgbt_fast_eval_v1")},
        code_roots={
            "local:rgbt_detection_real": str(
                ROOT / "experiment_apps" / "rgbt_detection_real"
            )
        },
    )
    record = JobRecord(
        job_id="job_x",
        request_id="req_x",
        execution_id="exec_x",
        status="running",
        contract_json={
            "project_id": "p",
            "node_id": "n",
            "environment_key": "rgbt-detection-v2",
            "code_reference": "local:rgbt_detection_real",
            "dataset_reference": "dataset:rgbt_fast_eval_v1",
            "entrypoint": "run_detection_experiment.py",
            "execution_mode": "fast_eval",
            "parameters": {
                "baseline": "dfine_s",
                "epochs": 1,
                "max_train_images": 4,
                "max_val_images": 2,
                "image_width": 80,
                "image_height": 64,
            },
            "task_config": {
                "claim_level": "exploratory_comparison",
                "evaluation_scope": "fast_eval_subset",
            },
            "resources": {"gpu_count": 0, "cpu_count": 1, "memory_gb": 2, "timeout_seconds": 120},
            "seed": 42,
        },
        environment_key="rgbt-detection-v2",
        output_path=str(tmp_path / "out"),
        log_path=str(tmp_path / "logs" / "combined.log"),
        submitted_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )
    Path(record.log_path).parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(DockerExecutorError, match="image_not_found|missing"):
        executor.run(record)


@pytestmark_docker
def test_docker_executor_runs_rgbt_fast_eval_cpu(tmp_path: Path):
    settings = WorkerSettings(
        data_root=tmp_path / "worker",
        executor_mode="docker",
        project_root=ROOT,
        allow_cpu_fallback=True,
    ).resolve()
    service = WorkerService(settings)
    from scientist_worker.models import JobSubmitRequest

    payload = JobSubmitRequest.model_validate(
        {
            "request_id": "req_docker_fe_1",
            "execution_id": "exec_docker_fe_1",
            "contract": {
                "project_id": "project_rgbt_002",
                "node_id": "rgbt_fast_node_remote_001",
                "task_type": "rgbt_detection",
                "environment_key": "rgbt-detection-v2",
                "code_reference": "local:rgbt_detection_real",
                "dataset_reference": "dataset:rgbt_fast_eval_v1",
                "entrypoint": "run_detection_experiment.py",
                "execution_mode": "fast_eval",
                "parameters": {
                    "baseline": "dfine_s",
                    "dfine_backend": "standin",
                    "input_mode": "rgb",
                    "fusion_method": "none",
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "image_width": 80,
                    "image_height": 64,
                    "max_train_images": 8,
                    "max_val_images": 4,
                },
                "task_config": {
                    "claim_level": "exploratory_comparison",
                    "evaluation_scope": "fast_eval_subset",
                    "primary_metric": "mAP50_95",
                },
                "seed": 42,
                "resources": {
                    "gpu_count": 0,
                    "cpu_count": 2,
                    "memory_gb": 4,
                    "timeout_seconds": 600,
                },
            },
            "environment": {"environment_key": "rgbt-detection-v2"},
        }
    )
    submitted = service.submit(payload)
    deadline = time.time() + 600
    status = None
    while time.time() < deadline:
        status = service.get_status(submitted.job_id)
        if status.status in {"completed", "failed", "cancelled", "timed_out"}:
            break
        time.sleep(1)
    assert status is not None
    assert status.status == "completed", (
        status.error_type,
        status.error_message,
        Path(service._require(submitted.job_id).log_path).read_text(encoding="utf-8")[
            -2000:
        ],
    )
    result = service.get_result(submitted.job_id)
    assert result["status"] == "completed"
    metrics = (result.get("result") or {}).get("metrics") or {}
    assert metrics.get("claim_level") == "exploratory_comparison"
    assert Path(service.artifacts_path(submitted.job_id)).exists()
