from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound

from scientist_lab.domain import ErrorType, JobStatus
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.domain.models import (
    ExecutionError,
    ExperimentArtifact,
    new_id,
    utc_now_iso,
)
from scientist_lab.domain.results import (
    ArtifactManifest,
    ExecutionResult,
    ExecutionStatus,
    SubmissionResult,
)
from scientist_lab.runners.base import ExperimentRunner
from scientist_lab.storage.artifact_store import (
    ensure_relative_safe,
    read_json,
    sha256_file,
    write_json,
)


@dataclass
class _RuntimeJob:
    execution_id: str
    contract: ExperimentContract
    image_name: str
    runtime_dir: Path
    workspace_dir: Path
    output_dir: Path
    status: JobStatus = JobStatus.CREATED
    container_id: str | None = None
    return_code: int | None = None
    error: ExecutionError | None = None
    thread: threading.Thread | None = field(default=None, repr=False)
    started_at: str | None = None
    completed_at: str | None = None


class LocalDockerRunner(ExperimentRunner):
    """CPU-first local Docker runner with async submit/status/collect API."""

    def __init__(
        self,
        experiment_app_dir: Path,
        runtime_root: Path,
        outputs_root: Path,
        image_registry: dict[str, str],
        poll_interval_seconds: float = 0.5,
        code_roots: dict[str, Path] | None = None,
        dataset_resolver=None,
    ) -> None:
        self.experiment_app_dir = experiment_app_dir.resolve()
        self.runtime_root = runtime_root.resolve()
        self.outputs_root = outputs_root.resolve()
        self.image_registry = image_registry
        self.poll_interval_seconds = poll_interval_seconds
        self.code_roots = {
            key: Path(value).resolve() for key, value in (code_roots or {}).items()
        }
        self.dataset_resolver = dataset_resolver
        self._jobs: dict[str, _RuntimeJob] = {}
        self._lock = threading.Lock()

        try:
            self.client = docker.from_env()
            self.client.ping()
        except DockerException as exc:
            raise RuntimeError(
                "无法连接 Docker。请确认 Docker Desktop 已启动。"
            ) from exc

    def _resolve_workspace_source(self, contract: ExperimentContract) -> Path:
        from scientist_lab.tasks.plan import prepare_execution_plan

        dataset = None
        if self.dataset_resolver is not None:
            dataset = self.dataset_resolver(contract.dataset_reference)
        plan = prepare_execution_plan(
            contract, dataset=dataset, code_roots=self.code_roots
        )
        if plan.workspace_source is not None:
            return Path(plan.workspace_source)
        ref = contract.code_reference or ""
        if ref in self.code_roots:
            return self.code_roots[ref]
        return self.experiment_app_dir

    def _stage_fast_eval_checkpoint(
        self, contract: ExperimentContract, output_dir: Path
    ) -> None:
        if contract.execution_mode != "fast_eval":
            return
        # Real baselines manage their own checkpoints (.pt); only stage for tiny path.
        baseline = str(
            (contract.parameters or {}).get("baseline")
            or (contract.parameters or {}).get("model")
            or "tiny_detector"
        )
        if baseline not in {"", "tiny_detector"}:
            return
        source = contract.parameters.get("checkpoint_source")
        if not source:
            if contract.task_type == "rgbt_detection":
                raise ValueError("fast_eval requires parameters.checkpoint_source")
            return
        src = Path(str(source))
        candidates = [src]
        if not src.is_absolute():
            project_root = Path(self.outputs_root).resolve().parent
            candidates.extend(
                [
                    project_root / src,
                    Path(self.outputs_root) / src,
                    project_root / "outputs" / src,
                ]
            )
        resolved = next((path for path in candidates if path.exists()), None)
        if resolved is None:
            raise FileNotFoundError(
                f"checkpoint_missing: checkpoint_source not found: {source}"
            )
        dest_dir = output_dir / "checkpoint"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "last.npz"
        shutil.copy2(resolved, dest)

    def validate(self, contract: ExperimentContract) -> None:
        if contract.runner_profile != "local":
            raise ValueError("LocalDockerRunner 仅支持 runner_profile=local")
        if contract.environment_key not in self.image_registry:
            raise ValueError(f"未注册实验环境：{contract.environment_key}")

        workspace = self._resolve_workspace_source(contract)
        if not workspace.exists():
            raise FileNotFoundError(f"实验代码目录不存在：{workspace}")
        entry = workspace / contract.entrypoint
        if not entry.exists():
            raise FileNotFoundError(f"入口脚本不存在：{entry}")

        if contract.task_type == "rgbt_detection":
            if self.dataset_resolver is None:
                raise ValueError("RGB-T 任务需要 DatasetRegistry")
            try:
                dataset = self.dataset_resolver(contract.dataset_reference)
            except KeyError as exc:
                raise ValueError(f"dataset_not_registered: {exc}") from exc
            if dataset is None:
                raise ValueError(
                    "rgbt_detection 需要 dataset_reference=dataset:<key>"
                )
            if not Path(dataset.host_path).exists():
                raise FileNotFoundError(
                    f"dataset_not_found: {dataset.host_path}"
                )

        image_name = self.image_registry[contract.environment_key]
        try:
            self.client.images.get(image_name)
        except ImageNotFound as exc:
            raise RuntimeError(
                f"镜像不存在：{image_name}。请先执行 docker build。"
            ) from exc

    def submit(self, contract: ExperimentContract) -> SubmissionResult:
        self.validate(contract)

        execution_id = new_id("exec")
        image_name = self.image_registry[contract.environment_key]

        runtime_dir = self.runtime_root / execution_id
        workspace_dir = runtime_dir / "workspace"
        output_dir = self.outputs_root / contract.project_id / execution_id

        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        output_dir.mkdir(parents=True, exist_ok=False)

        workspace_source = self._resolve_workspace_source(contract)
        shutil.copytree(workspace_source, workspace_dir)
        write_json(output_dir / "contract.json", contract.model_dump())
        config_payload = dict(contract.parameters)
        if contract.task_config:
            config_payload = {
                **config_payload,
                "_task_config": contract.task_config,
                "_task_type": contract.task_type,
                "_execution_mode": contract.execution_mode,
            }
        write_json(output_dir / "config.json", config_payload)
        self._stage_fast_eval_checkpoint(contract, output_dir)

        job = _RuntimeJob(
            execution_id=execution_id,
            contract=contract,
            image_name=image_name,
            runtime_dir=runtime_dir,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            status=JobStatus.QUEUED,
            started_at=utc_now_iso(),
        )

        with self._lock:
            self._jobs[execution_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(execution_id,),
            daemon=True,
            name=f"docker-run-{execution_id}",
        )
        job.thread = thread
        thread.start()

        return SubmissionResult(
            execution_id=execution_id,
            status=JobStatus.QUEUED,
            runner_profile=contract.runner_profile,
        )

    def get_status(self, execution_id: str) -> ExecutionStatus:
        job = self._require_job(execution_id)
        return ExecutionStatus(
            execution_id=execution_id,
            status=job.status,
            container_id=job.container_id,
            message=None if job.error is None else job.error.message,
        )

    def collect_result(self, execution_id: str) -> ExecutionResult:
        job = self._require_job(execution_id)
        if job.status not in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMED_OUT,
        }:
            raise RuntimeError(f"执行尚未结束：{job.status}")

        if job.error is not None and job.status != JobStatus.COMPLETED:
            return ExecutionResult(
                execution_id=execution_id,
                status=job.status,
                return_code=job.return_code,
                error=job.error,
                output_directory=str(job.output_dir),
            )

        try:
            artifacts = self._collect_artifacts(execution_id, job.output_dir)
            metrics = self._load_metrics(job.output_dir)
            return ExecutionResult(
                execution_id=execution_id,
                status=JobStatus.COMPLETED,
                return_code=job.return_code,
                metrics=metrics,
                artifacts=artifacts,
                output_directory=str(job.output_dir),
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            lowered = message.lower()
            if isinstance(exc, FileNotFoundError) or "缺少" in message or "artifact" in lowered:
                error_type = ErrorType.MISSING_ARTIFACT
            else:
                error_type = ErrorType.INVALID_METRICS
            error = ExecutionError(
                error_type=error_type,
                stage="collecting",
                message=message,
                retryable=False,
                log_artifact="combined.log",
            )
            job.status = JobStatus.FAILED
            job.error = error
            return ExecutionResult(
                execution_id=execution_id,
                status=JobStatus.FAILED,
                return_code=job.return_code,
                error=error,
                output_directory=str(job.output_dir),
            )

    def cancel(self, execution_id: str) -> None:
        job = self._require_job(execution_id)
        if job.container_id:
            try:
                container = self.client.containers.get(job.container_id)
                container.stop(timeout=10)
            except (NotFound, DockerException):
                pass
        job.status = JobStatus.CANCELLED
        job.error = ExecutionError(
            error_type=ErrorType.CANCELLED,
            stage="cancelled",
            message="用户取消实验",
            retryable=False,
        )
        job.completed_at = utc_now_iso()

    def wait_until_done(
        self,
        execution_id: str,
        timeout_seconds: float | None = None,
    ) -> ExecutionStatus:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            status = self.get_status(execution_id)
            if status.status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.TIMED_OUT,
            }:
                return status
            if deadline is not None and time.monotonic() > deadline:
                self.cancel(execution_id)
                return self.get_status(execution_id)
            time.sleep(self.poll_interval_seconds)

    def get_job_meta(self, execution_id: str) -> dict[str, Any]:
        job = self._require_job(execution_id)
        return {
            "execution_id": job.execution_id,
            "image_name": job.image_name,
            "container_id": job.container_id,
            "status": str(job.status),
            "return_code": job.return_code,
            "output_directory": str(job.output_dir),
            "runtime_directory": str(job.runtime_dir),
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error": None if job.error is None else job.error.model_dump(),
        }

    def _run_job(self, execution_id: str) -> None:
        job = self._require_job(execution_id)
        job.status = JobStatus.PREPARING
        log_path = job.output_dir / "combined.log"
        container = None

        try:
            from scientist_lab.tasks.plan import prepare_execution_plan

            dataset = None
            if self.dataset_resolver is not None:
                try:
                    dataset = self.dataset_resolver(job.contract.dataset_reference)
                except KeyError as exc:
                    raise RuntimeError(f"dataset_not_registered: {exc}") from exc

            plan = prepare_execution_plan(
                job.contract, dataset=dataset, code_roots=self.code_roots
            )
            command = list(plan.command)

            volumes = {
                str(job.workspace_dir): {"bind": "/workspace", "mode": "ro"},
                str(job.output_dir): {"bind": "/outputs", "mode": "rw"},
            }
            for mount in plan.mounts:
                mode = "ro" if mount.read_only else "rw"
                volumes[str(Path(mount.source).resolve())] = {
                    "bind": mount.target,
                    "mode": mode,
                }

            environment = {
                "PYTHONUNBUFFERED": "1",
                "OUTPUT_ROOT": "/outputs",
                "EXPERIMENT_MODE": job.contract.execution_mode,
            }
            environment.update(plan.environment)

            mem_limit = f"{job.contract.resources.memory_gb}g"
            nano_cpus = int(job.contract.resources.cpu_count * 1_000_000_000)

            job.status = JobStatus.RUNNING
            container = self.client.containers.run(
                image=job.image_name,
                command=command,
                detach=True,
                name=f"scientist-{execution_id}".replace("_", "-").lower(),
                working_dir="/workspace",
                volumes=volumes,
                environment=environment,
                network_disabled=True,
                mem_limit=mem_limit,
                nano_cpus=nano_cpus,
                auto_remove=False,
            )
            job.container_id = container.id

            return_code = self._wait_container(
                container=container,
                timeout_seconds=job.contract.resources.timeout_seconds,
                log_path=log_path,
            )
            job.return_code = return_code

            metadata = {
                "execution_id": execution_id,
                "container_id": container.id,
                "image_name": job.image_name,
                "return_code": return_code,
                "started_at": job.started_at,
                "finished_at": utc_now_iso(),
                "mounts": [
                    {
                        "source": m.source,
                        "target": m.target,
                        "read_only": m.read_only,
                    }
                    for m in plan.mounts
                ],
            }
            write_json(job.output_dir / "execution.json", metadata)

            if job.status == JobStatus.CANCELLED:
                pass
            elif return_code != 0:
                job.status = JobStatus.FAILED
                job.error = ExecutionError(
                    error_type=ErrorType.EXPERIMENT_CODE_ERROR,
                    stage="experiment_running",
                    message=f"容器退出码 {return_code}",
                    retryable=True,
                    suggested_action="inspect_logs",
                    log_artifact="combined.log",
                )
            else:
                job.status = JobStatus.COMPLETED

        except ImageNotFound as exc:
            job.status = JobStatus.FAILED
            job.error = ExecutionError(
                error_type=ErrorType.IMAGE_NOT_FOUND,
                stage="preparing",
                message=str(exc),
                retryable=False,
            )
        except TimeoutError as exc:
            if container is not None:
                self._terminate(container)
            job.status = JobStatus.TIMED_OUT
            job.error = ExecutionError(
                error_type=ErrorType.TIMEOUT,
                stage="experiment_running",
                message=str(exc),
                retryable=True,
                suggested_action="increase_timeout_or_reduce_workload",
                log_artifact="combined.log",
            )
        except APIError as exc:
            job.status = JobStatus.FAILED
            job.error = ExecutionError(
                error_type=ErrorType.CONTAINER_START_FAILED,
                stage="preparing",
                message=str(exc),
                retryable=True,
            )
        except Exception as exc:  # noqa: BLE001
            if container is not None:
                self._terminate(container)
            job.status = JobStatus.FAILED
            job.error = ExecutionError(
                error_type=ErrorType.UNKNOWN_ERROR,
                stage="experiment_running",
                message=str(exc),
                retryable=False,
            )
        finally:
            job.completed_at = utc_now_iso()
            if container is not None:
                self._save_logs(container, log_path)
                try:
                    container.remove(force=True)
                except DockerException:
                    pass

    def _wait_container(
        self,
        container,
        timeout_seconds: int,
        log_path: Path,
    ) -> int:
        deadline = time.monotonic() + timeout_seconds
        log_thread = threading.Thread(
            target=self._stream_logs,
            args=(container, log_path),
            daemon=True,
        )
        log_thread.start()

        while True:
            container.reload()
            if container.status in {"exited", "dead"}:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"实验超过最大运行时间：{timeout_seconds} 秒。"
                )
            time.sleep(self.poll_interval_seconds)

        wait_result = container.wait()
        log_thread.join(timeout=5)
        return int(wait_result.get("StatusCode", -1))

    def _stream_logs(self, container, log_path: Path) -> None:
        try:
            with log_path.open("ab") as log_file:
                for chunk in container.logs(
                    stream=True, follow=True, stdout=True, stderr=True
                ):
                    log_file.write(chunk)
                    log_file.flush()
        except DockerException:
            pass

    def _save_logs(self, container, log_path: Path) -> None:
        try:
            logs = container.logs(stdout=True, stderr=True, timestamps=True)
            log_path.write_bytes(logs)
        except DockerException:
            pass

    @staticmethod
    def _terminate(container) -> None:
        try:
            container.stop(timeout=10)
        except DockerException:
            try:
                container.kill()
            except DockerException:
                pass

    def _collect_artifacts(
        self,
        execution_id: str,
        output_dir: Path,
    ) -> list[ExperimentArtifact]:
        manifest_path = output_dir / "artifact_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("缺少 artifact_manifest.json")

        raw = read_json(manifest_path)
        manifest = ArtifactManifest.model_validate(raw)
        artifacts: list[ExperimentArtifact] = []

        for item in manifest.artifacts:
            path = ensure_relative_safe(output_dir, item.path)
            if not path.exists():
                if item.required:
                    raise FileNotFoundError(f"缺少必需产物：{item.path}")
                continue
            artifacts.append(
                ExperimentArtifact(
                    artifact_id=new_id("art"),
                    execution_id=execution_id,
                    artifact_type=item.type,
                    relative_path=item.path,
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    metadata_json={"required": item.required},
                    created_at=utc_now_iso(),
                )
            )

        # always register manifest / contract / execution if present
        for extra_type, name in [
            ("manifest", "artifact_manifest.json"),
            ("contract", "contract.json"),
            ("execution", "execution.json"),
            ("config", "config.json"),
        ]:
            path = output_dir / name
            if path.exists() and all(a.relative_path != name for a in artifacts):
                artifacts.append(
                    ExperimentArtifact(
                        artifact_id=new_id("art"),
                        execution_id=execution_id,
                        artifact_type=extra_type,
                        relative_path=name,
                        size_bytes=path.stat().st_size,
                        sha256=sha256_file(path),
                        metadata_json=None,
                        created_at=utc_now_iso(),
                    )
                )
        return artifacts

    @staticmethod
    def _load_metrics(output_dir: Path) -> dict[str, Any]:
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError("缺少 metrics.json")
        metrics = read_json(metrics_path)
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json 必须是对象")
        if "primary_metric" not in metrics or "metrics" not in metrics:
            raise ValueError("metrics.json 缺少 primary_metric / metrics")
        if not isinstance(metrics["metrics"], dict):
            raise ValueError("metrics.metrics 必须是对象")
        # Optional detection-specific bound checks.
        if metrics.get("task_type") == "rgbt_detection":
            from scientist_lab.tasks.rgbt_detection.result_parser import (
                DetectionResultParser,
            )

            return DetectionResultParser().parse(output_dir)
        return metrics

    def _require_job(self, execution_id: str) -> _RuntimeJob:
        with self._lock:
            job = self._jobs.get(execution_id)
        if job is None:
            raise KeyError(f"未知 execution_id：{execution_id}")
        return job
