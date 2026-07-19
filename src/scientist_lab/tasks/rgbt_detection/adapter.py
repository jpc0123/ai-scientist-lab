from __future__ import annotations

from pathlib import Path

from scientist_lab.datasets.models import DatasetRegistration
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.plan import MountSpec, TaskExecutionPlan


class RGBTDetectionAdapter:
    task_type = "rgbt_detection"

    def validate_contract(self, contract: ExperimentContract) -> None:
        if contract.task_type != self.task_type:
            raise ValueError(f"unexpected task_type: {contract.task_type}")
        if not (contract.dataset_reference or "").startswith("dataset:"):
            raise ValueError(
                "rgbt_detection requires dataset_reference like dataset:<key>"
            )
        if contract.execution_mode not in {
            "validate_data",
            "smoke_train",
            "fast_eval",
            "full_train",
        }:
            raise ValueError(f"unsupported execution_mode: {contract.execution_mode}")
        if contract.execution_mode == "validate_data":
            # Must not request training epochs for validate_data.
            epochs = contract.parameters.get("epochs")
            if epochs not in (None, 0, "0"):
                raise ValueError(
                    "validate_data forbids non-zero epochs "
                    f"(got parameters.epochs={epochs!r})"
                )

    def prepare_execution(
        self,
        contract: ExperimentContract,
        dataset: DatasetRegistration | None,
        *,
        code_roots: dict[str, Path],
    ) -> TaskExecutionPlan:
        self.validate_contract(contract)
        if dataset is None:
            raise ValueError(
                "rgbt_detection requires dataset_reference like dataset:<key>"
            )
        if not dataset.read_only:
            raise ValueError("RGB-T datasets must be mounted read-only")
        if not dataset.enabled:
            raise ValueError(f"dataset is disabled: {dataset.dataset_key}")

        workspace = code_roots.get("local:rgbt_detector") or code_roots.get(
            "rgbt_detector"
        )
        if workspace is None or not workspace.exists():
            raise FileNotFoundError(
                "RGB-T code root not found. Expected local:rgbt_detector directory."
            )

        data_root = dataset.container_path.rstrip("/")
        command = [
            contract.entrypoint,
            "--config",
            "/outputs/config.json",
            "--output-dir",
            "/outputs",
            "--seed",
            str(contract.seed),
            "--data-root",
            data_root,
            "--execution-mode",
            contract.execution_mode,
            "--input-mode",
            str(contract.parameters.get("input_mode", "rgb")),
            "--fusion-method",
            str(contract.parameters.get("fusion_method", "none")),
        ]

        notes = [
            "RGB-T dataset mounted read-only",
            f"execution_mode={contract.execution_mode}",
            "docker network disabled by LocalDockerRunner",
        ]
        if contract.execution_mode == "validate_data":
            notes.append("validate_data: no model training")

        return TaskExecutionPlan(
            command=command,
            mounts=[
                MountSpec(
                    source=dataset.host_path,
                    target=dataset.container_path,
                    read_only=True,
                )
            ],
            environment={
                "DATASET_ROOT": data_root,
                "TASK_TYPE": "rgbt_detection",
                "EXECUTION_MODE": contract.execution_mode,
                "EXPECT_DATASET_READ_ONLY": "1",
            },
            workspace_source=workspace,
            notes=notes,
        )
