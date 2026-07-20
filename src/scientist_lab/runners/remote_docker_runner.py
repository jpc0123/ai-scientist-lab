from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from scientist_lab.artifacts.bundle import validate_bundle_archive
from scientist_lab.artifacts.policy import ArtifactPolicy, default_artifact_policy
from scientist_lab.domain import ErrorType, JobStatus
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.domain.models import (
    ExecutionError,
    ExperimentArtifact,
    new_id,
    utc_now_iso,
)
from scientist_lab.domain.results import (
    ExecutionResult,
    ExecutionStatus,
    SubmissionResult,
)
from scientist_lab.runners.base import ExperimentRunner
from scientist_lab.runners.profiles import RunnerProfile
from scientist_lab.runners.remote_client import WorkerHttpClient
from scientist_lab.runners.remote_errors import (
    RemoteArtifactDownloadFailed,
    RemoteCapabilityMismatch,
    RemoteRunnerError,
)
from scientist_lab.runners.remote_models import RemoteJobBinding, RunnerLogChunk
from scientist_lab.storage.artifact_store import sha256_file, write_json


WORKER_TO_JOB_STATUS = {
    "received": JobStatus.QUEUED,
    "validating": JobStatus.PREPARING,
    "queued": JobStatus.QUEUED,
    "preparing": JobStatus.PREPARING,
    "running": JobStatus.RUNNING,
    "collecting": JobStatus.COLLECTING,
    "completed": JobStatus.COMPLETED,
    "failed": JobStatus.FAILED,
    "cancelled": JobStatus.CANCELLED,
    "cancel_requested": JobStatus.CANCELLED,
    "timed_out": JobStatus.TIMED_OUT,
    "validation_failed": JobStatus.FAILED,
    "prepare_failed": JobStatus.FAILED,
    "artifact_failed": JobStatus.FAILED,
}


class RemoteDockerRunner(ExperimentRunner):
    """Submit experiments to a Scientist Worker over HTTP."""

    def __init__(
        self,
        profile: RunnerProfile,
        *,
        outputs_root: Path,
        runtime_root: Path,
        poll_interval_seconds: float = 0.5,
        auth_token: str | None = None,
    ) -> None:
        if profile.runner_type != "remote_docker":
            raise ValueError("RemoteDockerRunner requires runner_type=remote_docker")
        if not profile.endpoint:
            raise ValueError("remote profile missing endpoint")
        self.profile = profile
        self.outputs_root = Path(outputs_root).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.poll_interval_seconds = poll_interval_seconds
        self._bindings_dir = self.runtime_root / "remote_jobs"
        self._bindings_dir.mkdir(parents=True, exist_ok=True)
        token = auth_token
        if token is None and profile.auth_token_env:
            token = os.environ.get(profile.auth_token_env)
        self.client = WorkerHttpClient(profile.endpoint, auth_token=token)
        self._lock = threading.Lock()
        self._meta: dict[str, dict[str, Any]] = {}
        self.artifact_policy = default_artifact_policy()

    def validate(self, contract: ExperimentContract) -> None:
        if not self.profile.enabled:
            raise ValueError(f"runner profile disabled: {self.profile.profile_key}")
        if contract.runner_profile != self.profile.profile_key:
            raise ValueError(
                f"contract.runner_profile={contract.runner_profile!r} "
                f"does not match profile {self.profile.profile_key!r}"
            )
        env = contract.environment_key
        allowed = self.profile.allowed_environment_keys
        if allowed and env not in allowed:
            raise RemoteCapabilityMismatch(
                f"environment_key {env!r} not allowed for profile "
                f"{self.profile.profile_key}; allowed={allowed}"
            )
        try:
            health = self.client.health()
        except RemoteRunnerError:
            raise
        if health.get("status") != "ok":
            from scientist_lab.runners.remote_errors import RemoteWorkerUnavailable

            raise RemoteWorkerUnavailable(f"worker unhealthy: {health}")

    def submit(self, contract: ExperimentContract) -> SubmissionResult:
        self.validate(contract)
        execution_id = new_id("exec")
        request_id = f"req_{execution_id.removeprefix('exec_')}"
        output_dir = self.outputs_root / contract.project_id / execution_id
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(output_dir / "contract.json", contract.model_dump())

        payload = {
            "request_id": request_id,
            "execution_id": execution_id,
            "contract": contract.model_dump(),
            "environment": {"environment_key": contract.environment_key},
            "dataset_mounts": [],
            "code_bundle": {},
            "artifact_policy": self.artifact_policy.model_dump(),
        }
        response = self.client.submit_job(payload)
        job_id = response["job_id"]
        now = utc_now_iso()
        binding = RemoteJobBinding(
            execution_id=execution_id,
            request_id=request_id,
            job_id=job_id,
            runner_profile=self.profile.profile_key,
            endpoint=self.profile.endpoint or "",
            project_id=contract.project_id,
            node_id=contract.node_id,
            output_directory=str(output_dir),
            contract=contract.model_dump(),
            submitted_at=now,
            status=response.get("status", "queued"),
        )
        self._save_binding(binding)
        with self._lock:
            self._meta[execution_id] = {
                "started_at": now,
                "completed_at": None,
                "container_id": job_id,
                "job_id": job_id,
                "request_id": request_id,
            }
        return SubmissionResult(
            execution_id=execution_id,
            status=JobStatus.QUEUED,
            runner_profile=self.profile.profile_key,
        )

    def get_status(self, execution_id: str) -> ExecutionStatus:
        binding = self._load_binding(execution_id)
        remote = self.client.get_job(binding.job_id)
        status = WORKER_TO_JOB_STATUS.get(
            str(remote.get("status")), JobStatus.RUNNING
        )
        binding.status = str(remote.get("status"))
        self._save_binding(binding)
        return ExecutionStatus(
            execution_id=execution_id,
            status=status,
            progress=remote.get("progress"),
            message=remote.get("stage"),
            container_id=binding.job_id,
        )

    def get_logs(
        self, execution_id: str, cursor: str | None = None
    ) -> RunnerLogChunk:
        binding = self._load_binding(execution_id)
        payload = self.client.get_logs(binding.job_id, cursor=cursor)
        return RunnerLogChunk(
            execution_id=execution_id,
            cursor=payload.get("cursor"),
            next_cursor=payload.get("next_cursor"),
            content=payload.get("content") or "",
            complete=bool(payload.get("complete")),
        )

    def collect_result(self, execution_id: str) -> ExecutionResult:
        binding = self._load_binding(execution_id)
        status = self.get_status(execution_id)
        output_dir = Path(binding.output_directory)

        if status.status == JobStatus.COMPLETED:
            try:
                self._download_and_unpack(binding, output_dir)
            except RemoteRunnerError as exc:
                return ExecutionResult(
                    execution_id=execution_id,
                    status=JobStatus.FAILED,
                    return_code=1,
                    error=ExecutionError(
                        error_type=ErrorType.UNKNOWN_ERROR,
                        stage="collecting",
                        message=f"{exc.error_type}: {exc}",
                        retryable=exc.retryable,
                    ),
                    output_directory=str(output_dir),
                )

        metrics: dict[str, Any] = {}
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        artifacts = self._scan_artifacts(execution_id, output_dir)
        error = None
        if status.status != JobStatus.COMPLETED:
            remote_result = self.client.get_result(binding.job_id)
            error = ExecutionError(
                error_type=ErrorType.UNKNOWN_ERROR,
                stage=str(remote_result.get("status") or "failed"),
                message=str(
                    remote_result.get("error_message")
                    or remote_result.get("error_type")
                    or status.status
                ),
                retryable=False,
            )

        with self._lock:
            meta = self._meta.setdefault(execution_id, {})
            meta["completed_at"] = utc_now_iso()
            meta["container_id"] = binding.job_id

        return ExecutionResult(
            execution_id=execution_id,
            status=status.status,
            return_code=0 if status.status == JobStatus.COMPLETED else 1,
            metrics=metrics if isinstance(metrics, dict) else {},
            artifacts=artifacts,
            error=error,
            output_directory=str(output_dir),
        )

    def cancel(self, execution_id: str) -> None:
        binding = self._load_binding(execution_id)
        self.client.cancel_job(binding.job_id)

    def wait_until_done(
        self, execution_id: str, timeout_seconds: float = 3600
    ) -> ExecutionStatus:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            status = self.get_status(execution_id)
            if status.status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.TIMED_OUT,
            }:
                return status
            time.sleep(self.poll_interval_seconds)
        self.cancel(execution_id)
        return ExecutionStatus(
            execution_id=execution_id,
            status=JobStatus.TIMED_OUT,
            message="remote wait timeout",
        )

    def get_job_meta(self, execution_id: str) -> dict[str, Any]:
        with self._lock:
            meta = dict(self._meta.get(execution_id) or {})
        if not meta:
            binding = self._load_binding(execution_id)
            meta = {
                "started_at": binding.submitted_at,
                "completed_at": None,
                "container_id": binding.job_id,
                "job_id": binding.job_id,
                "request_id": binding.request_id,
            }
        return meta

    def _download_and_unpack(
        self, binding: RemoteJobBinding, output_dir: Path
    ) -> None:
        result_dir = output_dir / "_remote"
        result_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = result_dir / "artifact_bundle.tar.gz"
        try:
            self.client.download_artifacts(binding.job_id, bundle_path)
        except RemoteArtifactDownloadFailed:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RemoteArtifactDownloadFailed(str(exc)) from exc

        remote_result = self.client.get_result(binding.job_id)
        manifest = (
            ((remote_result.get("result") or {}).get("artifact_bundle") or {}).get(
                "manifest"
            )
            or {}
        )
        policy_payload = manifest.get("policy") or self.artifact_policy.model_dump()
        try:
            policy = ArtifactPolicy.model_validate(policy_payload)
        except Exception:  # noqa: BLE001
            policy = self.artifact_policy

        # Persist remote manifest for audit.
        if manifest:
            write_json(result_dir / "artifact_bundle_manifest.json", manifest)

        validate_bundle_archive(
            bundle_path,
            manifest=manifest,
            policy=policy,
            extract_to=output_dir,
        )

        write_json(
            output_dir / "remote_execution.json",
            {
                "execution_id": binding.execution_id,
                "job_id": binding.job_id,
                "runner_profile": binding.runner_profile,
                "endpoint": binding.endpoint,
                "request_id": binding.request_id,
                "artifact_policy": policy.model_dump(),
            },
        )

    def _scan_artifacts(
        self, execution_id: str, output_dir: Path
    ) -> list[ExperimentArtifact]:
        artifacts: list[ExperimentArtifact] = []
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(output_dir).as_posix()
            if rel.startswith("_remote/"):
                continue
            artifacts.append(
                ExperimentArtifact(
                    artifact_id=new_id("art"),
                    execution_id=execution_id,
                    artifact_type=path.suffix.lstrip(".") or "file",
                    relative_path=rel,
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    created_at=utc_now_iso(),
                )
            )
        return artifacts

    def _binding_path(self, execution_id: str) -> Path:
        return self._bindings_dir / f"{execution_id}.json"

    def _save_binding(self, binding: RemoteJobBinding) -> None:
        path = self._binding_path(binding.execution_id)
        path.write_text(
            binding.model_dump_json(indent=2), encoding="utf-8"
        )

    def _load_binding(self, execution_id: str) -> RemoteJobBinding:
        path = self._binding_path(execution_id)
        if not path.exists():
            raise KeyError(f"remote binding not found: {execution_id}")
        return RemoteJobBinding.model_validate_json(
            path.read_text(encoding="utf-8")
        )
