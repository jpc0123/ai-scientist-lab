from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from scientist_worker.executor import run_mock_job
from scientist_worker.gpu_inspector import inspect_gpu
from scientist_worker.log_store import LogStore
from scientist_worker.models import (
    JobRecord,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    LogChunkResponse,
    utc_now_iso,
)
from scientist_worker.repository import WorkerJobRepository
from scientist_worker.security import (
    ENVIRONMENT_REGISTRY,
    SecurityError,
    validate_submit_payload,
)
from scientist_worker.settings import WorkerSettings
from scientist_worker.workflow import InvalidWorkerTransition, transition


class WorkerService:
    def __init__(self, settings: WorkerSettings | None = None) -> None:
        self.settings = (settings or WorkerSettings()).resolve()
        self.repo = WorkerJobRepository(self.settings.db_path)
        self._locks: dict[str, threading.Lock] = {}
        self._cancel_flags: dict[str, bool] = {}
        self._threads: dict[str, threading.Thread] = {}

    def health(self) -> dict[str, Any]:
        gpu = inspect_gpu()
        docker_ok = False
        if self.settings.executor_mode in {"docker", "auto"}:
            try:
                import docker

                client = docker.from_env()
                client.ping()
                docker_ok = True
            except Exception:  # noqa: BLE001
                docker_ok = False
        return {
            "status": "ok",
            "worker_id": self.settings.worker_id,
            "version": self.settings.version,
            "docker_available": docker_ok,
            "gpu_available": bool(gpu.get("available")),
            "executor_mode": self.settings.executor_mode,
        }

    def capabilities(self) -> dict[str, Any]:
        gpu = inspect_gpu()
        return {
            "worker_id": self.settings.worker_id,
            "supported_environment_keys": list(self.settings.supported_environment_keys),
            "environment_registry": {
                key: {
                    "image": value["image"],
                    "allowed_entrypoints": value["allowed_entrypoints"],
                    "max_gpu_count": value["max_gpu_count"],
                    "cuda_required": value["cuda_required"],
                }
                for key, value in ENVIRONMENT_REGISTRY.items()
                if key in self.settings.supported_environment_keys
            },
            "gpu": gpu,
            "max_cpu_count": self.settings.max_cpu_count,
            "max_memory_gb": self.settings.max_memory_gb,
            "network_policy": self.settings.network_policy,
            "executor_mode": self.settings.executor_mode,
        }

    def submit(self, request: JobSubmitRequest) -> JobSubmitResponse:
        existing = self.repo.get_by_request_id(request.request_id)
        if existing is not None:
            return JobSubmitResponse(
                job_id=existing.job_id,
                execution_id=existing.execution_id,
                status=existing.status,
                submitted_at=existing.submitted_at,
                request_id=existing.request_id,
            )

        env_key = validate_submit_payload(
            contract=request.contract,
            environment=request.environment,
            supported_environment_keys=self.settings.supported_environment_keys,
        )

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_dir = self.settings.jobs_root / job_id
        output_dir = job_dir / "output"
        log_path = job_dir / "logs" / "combined.log"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        (job_dir / "request").mkdir(parents=True, exist_ok=True)
        (job_dir / "request" / "payload.json").write_text(
            request.model_dump_json(indent=2), encoding="utf-8"
        )

        now = utc_now_iso()
        record = JobRecord(
            job_id=job_id,
            request_id=request.request_id,
            execution_id=request.execution_id,
            status="received",
            contract_json=request.contract,
            environment_key=env_key,
            output_path=str(output_dir),
            log_path=str(log_path),
            submitted_at=now,
            updated_at=now,
            stage="received",
            progress=0.0,
        )
        try:
            self.repo.insert(record)
        except Exception as exc:  # noqa: BLE001
            # race: another insert with same request_id
            existing = self.repo.get_by_request_id(request.request_id)
            if existing is not None:
                return JobSubmitResponse(
                    job_id=existing.job_id,
                    execution_id=existing.execution_id,
                    status=existing.status,
                    submitted_at=existing.submitted_at,
                    request_id=existing.request_id,
                )
            raise SecurityError(f"remote_submission_failed: {exc}") from exc

        self._cancel_flags[job_id] = False
        thread = threading.Thread(
            target=self._run_job, args=(job_id,), daemon=True, name=f"worker-{job_id}"
        )
        self._threads[job_id] = thread
        thread.start()

        return JobSubmitResponse(
            job_id=job_id,
            execution_id=request.execution_id,
            status="queued",
            submitted_at=now,
            request_id=request.request_id,
        )

    def get_status(self, job_id: str) -> JobStatusResponse:
        record = self._require(job_id)
        return JobStatusResponse(
            job_id=record.job_id,
            request_id=record.request_id,
            execution_id=record.execution_id,
            status=record.status,
            progress=record.progress,
            stage=record.stage,
            environment_key=record.environment_key,
            error_type=record.error_type,
            error_message=record.error_message,
            submitted_at=record.submitted_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            updated_at=record.updated_at,
        )

    def get_logs(self, job_id: str, cursor: str | None = None) -> LogChunkResponse:
        record = self._require(job_id)
        store = LogStore(Path(record.log_path))
        complete = record.status in {
            "completed",
            "failed",
            "cancelled",
            "timed_out",
            "validation_failed",
            "prepare_failed",
            "artifact_failed",
        }
        content, next_cursor, _ = store.read_chunk(cursor, complete=complete)
        return LogChunkResponse(
            execution_id=record.execution_id,
            job_id=record.job_id,
            cursor=cursor,
            next_cursor=next_cursor,
            content=content,
            complete=complete,
        )

    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self._require(job_id)
        if record.status in {"completed", "failed", "cancelled", "timed_out"}:
            return {"job_id": job_id, "status": record.status, "cancelled": False}
        self._cancel_flags[job_id] = True
        try:
            record.status = transition(record.status, "cancel_requested")
        except InvalidWorkerTransition:
            record.status = "cancel_requested"
        record.stage = "cancel_requested"
        self.repo.update(record)
        return {"job_id": job_id, "status": record.status, "cancelled": True}

    def get_result(self, job_id: str) -> dict[str, Any]:
        record = self._require(job_id)
        if record.status != "completed":
            return {
                "job_id": job_id,
                "execution_id": record.execution_id,
                "status": record.status,
                "error_type": record.error_type,
                "error_message": record.error_message,
                "result": record.result_json,
            }
        return {
            "job_id": job_id,
            "execution_id": record.execution_id,
            "status": record.status,
            "result": record.result_json,
        }

    def artifacts_path(self, job_id: str) -> Path:
        record = self._require(job_id)
        path = Path(record.output_path).parent / "result" / "artifact_bundle.tar.gz"
        if not path.exists():
            raise FileNotFoundError("remote_artifact_download_failed: bundle missing")
        return path

    def _require(self, job_id: str) -> JobRecord:
        record = self.repo.get_by_job_id(job_id)
        if record is None:
            raise KeyError(f"job not found: {job_id}")
        return record

    def _set_status(self, record: JobRecord, target: str, **kwargs: Any) -> JobRecord:
        record.status = transition(record.status, target)
        for key, value in kwargs.items():
            setattr(record, key, value)
        return self.repo.update(record)

    def _run_job(self, job_id: str) -> None:
        record = self._require(job_id)
        logs = LogStore(Path(record.log_path))
        try:
            record = self._set_status(record, "validating", stage="validating", progress=0.05)
            logs.append("[worker] validating")
            if self._cancel_flags.get(job_id):
                raise RuntimeError("cancelled")

            record = self._set_status(record, "queued", stage="queued", progress=0.1)
            record = self._set_status(
                record,
                "preparing",
                stage="preparing",
                progress=0.2,
                started_at=utc_now_iso(),
            )
            logs.append("[worker] preparing workspace")
            if self._cancel_flags.get(job_id):
                raise RuntimeError("cancelled")

            record = self._set_status(record, "running", stage="running", progress=0.4)

            def cancelled() -> bool:
                return bool(self._cancel_flags.get(job_id))

            mode = self._resolve_executor_mode(record.environment_key)
            logs.append(f"[worker] executor_mode={mode}")
            if mode == "mock":
                result = run_mock_job(record, cancelled=cancelled)
            else:
                from scientist_worker.docker_executor import (
                    DockerExecutor,
                    DockerExecutorError,
                )

                try:
                    executor = DockerExecutor(
                        image_registry=self.settings.image_registry,
                        dataset_registry=self.settings.dataset_registry,
                        code_roots=self.settings.code_roots,
                        allow_cpu_fallback=self.settings.allow_cpu_fallback,
                    )
                    result = executor.run(record, cancelled=cancelled)
                    if result.get("container_id"):
                        record.container_id = str(result["container_id"])
                        self.repo.update(record)
                except DockerExecutorError as exc:
                    record.error_type = exc.error_type
                    if exc.error_type == "timed_out":
                        record.status = "timed_out"
                        record.stage = "timed_out"
                        record.error_message = str(exc)
                        record.finished_at = utc_now_iso()
                        self.repo.update(record)
                        logs.append(f"[worker] timed_out: {exc}")
                        return
                    if exc.error_type == "gpu_unavailable":
                        record.status = "failed"
                        record.stage = "failed"
                        record.error_message = str(exc)
                        record.finished_at = utc_now_iso()
                        self.repo.update(record)
                        logs.append(f"[worker] gpu_unavailable: {exc}")
                        return
                    raise RuntimeError(str(exc)) from exc

            if self._cancel_flags.get(job_id):
                raise RuntimeError("cancelled")

            record = self._set_status(
                record, "collecting", stage="collecting", progress=0.95
            )
            record.result_json = result
            record = self._set_status(
                record,
                "completed",
                stage="completed",
                progress=1.0,
                finished_at=utc_now_iso(),
            )
            logs.append("[worker] completed")
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if message == "cancelled" or self._cancel_flags.get(job_id):
                record.status = "cancelled"
                record.stage = "cancelled"
                record.error_type = "cancelled"
                record.error_message = "job cancelled"
                record.finished_at = utc_now_iso()
                self.repo.update(record)
                logs.append("[worker] cancelled")
                return
            record.status = "failed"
            record.stage = "failed"
            record.error_type = type(exc).__name__
            record.error_message = message
            record.finished_at = utc_now_iso()
            self.repo.update(record)
            logs.append(f"[worker] failed: {message}")

    def _resolve_executor_mode(self, environment_key: str) -> str:
        mode = (self.settings.executor_mode or "auto").strip().lower()
        if mode == "auto":
            if environment_key == "mock-detection-v1":
                return "mock"
            return "docker"
        return mode
