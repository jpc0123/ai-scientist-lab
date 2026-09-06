"""Docker-based job executor for Scientist Worker (v0.8.5)."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from scientist_worker.artifact_packager import load_job_artifact_policy, package_artifacts
from scientist_worker.gpu_inspector import inspect_gpu
from scientist_worker.log_store import LogStore
from scientist_worker.models import JobRecord
from scientist_worker.security import ENVIRONMENT_REGISTRY


class DockerExecutorError(RuntimeError):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class DockerExecutor:
    def __init__(
        self,
        *,
        image_registry: dict[str, str] | None = None,
        dataset_registry: dict[str, str] | None = None,
        code_roots: dict[str, str] | None = None,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.image_registry = dict(image_registry or {})
        self.dataset_registry = {
            key: str(Path(value).resolve())
            for key, value in (dataset_registry or {}).items()
        }
        self.code_roots = {
            key: Path(value).resolve() for key, value in (code_roots or {}).items()
        }
        self.allow_cpu_fallback = allow_cpu_fallback
        try:
            import docker
        except ImportError as exc:
            raise DockerExecutorError(
                "docker_unavailable", "python docker package not installed"
            ) from exc
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as exc:  # noqa: BLE001
            raise DockerExecutorError(
                "docker_unavailable", f"cannot connect to Docker: {exc}"
            ) from exc

    def _stage_dfine_vendor(self, workspace_dir: Path) -> None:
        roots: list[Path] = []
        for code_root in self.code_roots.values():
            root = Path(code_root).resolve()
            roots.append(root / "third_party" / "DFINE")
            for depth in (1, 2):
                try:
                    roots.append(root.parents[depth] / "third_party" / "DFINE")
                except IndexError:
                    break
        src = next((path for path in roots if (path / "train.py").is_file()), None)
        if src is None:
            return
        dest = Path(workspace_dir) / "third_party" / "DFINE"
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)

    def run(
        self,
        record: JobRecord,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(record.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs = LogStore(Path(record.log_path))
        job_dir = output_dir.parent
        workspace_dir = job_dir / "workspace"
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        contract = dict(record.contract_json or {})
        env_key = record.environment_key
        env_def = ENVIRONMENT_REGISTRY.get(env_key)
        if env_def is None:
            raise DockerExecutorError(
                "remote_capability_mismatch", f"unknown environment_key={env_key}"
            )

        image = self.image_registry.get(env_key) or env_def["image"]
        self._ensure_image(image)

        gpu_requested = int((contract.get("resources") or {}).get("gpu_count") or 0)
        gpu_info = inspect_gpu()
        device_requests = None
        if gpu_requested > 0:
            max_gpu = int(env_def.get("max_gpu_count") or 0)
            if gpu_requested > max_gpu:
                raise DockerExecutorError(
                    "remote_capability_mismatch",
                    f"gpu_count={gpu_requested} exceeds env max_gpu_count={max_gpu}",
                )
            if not gpu_info.get("available"):
                if env_def.get("cuda_required") and not self.allow_cpu_fallback:
                    raise DockerExecutorError(
                        "gpu_unavailable",
                        "GPU requested/required but not available on worker",
                    )
                logs.append(
                    "[worker] GPU requested but unavailable; falling back to CPU"
                )
                gpu_requested = 0
            else:
                available = int(gpu_info.get("count") or 0)
                if gpu_requested > available:
                    raise DockerExecutorError(
                        "gpu_unavailable",
                        f"requested gpu_count={gpu_requested} > available={available}",
                    )
                device_requests = [
                    {
                        "Driver": "nvidia",
                        "Count": gpu_requested,
                        "Capabilities": [["gpu"]],
                    }
                ]

        code_ref = str(contract.get("code_reference") or "")
        code_root = self.code_roots.get(code_ref)
        if code_root is None or not code_root.exists():
            raise DockerExecutorError(
                "prepare_failed",
                f"code_reference not registered on worker: {code_ref}",
            )
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        shutil.copytree(code_root, workspace_dir)
        self._stage_dfine_vendor(workspace_dir)

        dataset_key = str(contract.get("dataset_reference") or "").removeprefix(
            "dataset:"
        )
        dataset_host = self.dataset_registry.get(dataset_key)
        if not dataset_host or not Path(dataset_host).exists():
            raise DockerExecutorError(
                "dataset_not_registered",
                f"dataset not registered on worker: {dataset_key}",
            )
        container_data = f"/data/{dataset_key}"

        # Write contract/config into outputs (same pattern as LocalDockerRunner).
        (output_dir / "contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        config_payload = dict(contract.get("parameters") or {})
        if contract.get("task_config"):
            config_payload = {
                **config_payload,
                "_task_config": contract.get("task_config"),
                "_task_type": contract.get("task_type"),
                "_execution_mode": contract.get("execution_mode"),
            }
        (output_dir / "config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        resources = contract.get("resources") or {}
        mem_gb = max(1, int(resources.get("memory_gb") or 4))
        cpu_count = max(1, int(resources.get("cpu_count") or 2))
        timeout_seconds = max(30, int(resources.get("timeout_seconds") or 3600))
        entrypoint = str(contract.get("entrypoint") or "run_detection_experiment.py")
        command = [
            entrypoint,
            "--config",
            "/outputs/config.json",
            "--output-dir",
            "/outputs",
            "--seed",
            str(contract.get("seed") or 42),
            "--data-root",
            container_data,
            "--execution-mode",
            str(contract.get("execution_mode") or "smoke_train"),
            "--input-mode",
            str((contract.get("parameters") or {}).get("input_mode") or "rgb"),
            "--fusion-method",
            str((contract.get("parameters") or {}).get("fusion_method") or "none"),
        ]

        volumes = {
            str(workspace_dir): {"bind": "/workspace", "mode": "ro"},
            str(output_dir.resolve()): {"bind": "/outputs", "mode": "rw"},
            str(Path(dataset_host).resolve()): {
                "bind": container_data,
                "mode": "ro",
            },
        }
        environment = {
            "PYTHONUNBUFFERED": "1",
            "OUTPUT_ROOT": "/outputs",
            "DATASET_ROOT": container_data,
            "TASK_TYPE": str(contract.get("task_type") or ""),
            "EXECUTION_MODE": str(contract.get("execution_mode") or ""),
            "EXPECT_DATASET_READ_ONLY": "1",
            "BASELINE_KEY": str(
                (contract.get("parameters") or {}).get("baseline") or ""
            ),
        }

        logs.append(f"[worker] docker run image={image} gpu={gpu_requested}")
        run_kwargs: dict[str, Any] = {
            "image": image,
            "command": command,
            "detach": True,
            "name": f"scientist-worker-{record.job_id}".replace("_", "-").lower(),
            "working_dir": "/workspace",
            "volumes": volumes,
            "environment": environment,
            "network_disabled": True,
            "mem_limit": f"{mem_gb}g",
            "nano_cpus": int(cpu_count * 1_000_000_000),
            "auto_remove": False,
        }
        if device_requests:
            # docker-py accepts DeviceRequest objects; dict form works on recent versions.
            from docker.types import DeviceRequest

            run_kwargs["device_requests"] = [
                DeviceRequest(
                    count=gpu_requested,
                    capabilities=[["gpu"]],
                )
            ]

        if cancelled and cancelled():
            raise RuntimeError("cancelled")

        try:
            container = self.client.containers.run(**run_kwargs)
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "nvidia" in message or "gpu" in message:
                raise DockerExecutorError("gpu_unavailable", str(exc)) from exc
            raise DockerExecutorError("prepare_failed", str(exc)) from exc

        record.container_id = container.id
        logs.append(f"[worker] container_id={container.id[:12]}")

        return_code = self._wait_container(
            container,
            timeout_seconds=timeout_seconds,
            log_store=logs,
            cancelled=cancelled,
        )

        inspect = {}
        try:
            container.reload()
            inspect = {
                "Id": container.id,
                "State": (container.attrs or {}).get("State"),
            }
        except Exception:  # noqa: BLE001
            inspect = {"Id": getattr(container, "id", None)}
        (output_dir / "docker_inspect.json").write_text(
            json.dumps(
                {
                    "id": getattr(container, "id", None),
                    "image": image,
                    "return_code": return_code,
                    "gpu_requested": gpu_requested,
                    "status": inspect.get("State"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass

        if return_code != 0:
            raise DockerExecutorError(
                "experiment_code_error", f"container exit code {return_code}"
            )

        metrics_path = output_dir / "metrics.json"
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        # Ensure execution.json exists / enrich.
        execution_payload = {
            "execution_id": record.execution_id,
            "job_id": record.job_id,
            "runner_profile": "remote_docker",
            "environment_key": env_key,
            "image_name": image,
            "container_id": container.id,
            "status": "completed",
            "gpu_requested": gpu_requested,
            "gpu_available": bool(gpu_info.get("available")),
            "worker_mode": "docker",
        }
        (output_dir / "execution.json").write_text(
            json.dumps(execution_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not (output_dir / "combined.log").exists():
            (output_dir / "combined.log").write_text(
                Path(record.log_path).read_text(encoding="utf-8"), encoding="utf-8"
            )

        policy = load_job_artifact_policy(record)
        packaged = package_artifacts(
            output_dir,
            job_id=record.job_id,
            execution_id=record.execution_id,
            policy=policy,
        )
        logs.append("[worker] docker job completed")
        return {
            "metrics": metrics,
            "artifact_bundle": packaged,
            "status": "completed",
            "container_id": container.id,
            "image_name": image,
            "gpu_requested": gpu_requested,
        }

    def _ensure_image(self, image: str) -> None:
        try:
            self.client.images.get(image)
        except Exception as exc:  # noqa: BLE001
            raise DockerExecutorError(
                "image_not_found", f"docker image missing: {image}"
            ) from exc

    def _wait_container(
        self,
        container,
        *,
        timeout_seconds: int,
        log_store: LogStore,
        cancelled: Callable[[], bool] | None,
    ) -> int:
        deadline = time.time() + timeout_seconds
        seen = 0
        while True:
            if cancelled and cancelled():
                try:
                    container.kill()
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError("cancelled")
            container.reload()
            seen = self._append_logs(container, log_store, seen)
            state = container.attrs.get("State") or {}
            if not state.get("Running", False):
                self._append_logs(container, log_store, seen)
                return int(state.get("ExitCode") or 0)
            if time.time() > deadline:
                try:
                    container.kill()
                except Exception:  # noqa: BLE001
                    pass
                raise DockerExecutorError(
                    "timed_out", f"container exceeded timeout_seconds={timeout_seconds}"
                )
            time.sleep(0.5)

    @staticmethod
    def _append_logs(container, log_store: LogStore, previous_len: int) -> int:
        try:
            raw = container.logs(stdout=True, stderr=True)
            text = raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return previous_len
        if len(text) > previous_len:
            chunk = text[previous_len:]
            # Write raw chunk without forcing extra newlines per line.
            with Path(log_store.log_path).open("a", encoding="utf-8") as file:
                file.write(chunk)
            return len(text)
        return previous_len
