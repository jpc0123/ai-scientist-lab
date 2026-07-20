from __future__ import annotations

from pathlib import Path

from scientist_lab.datasets.models import DatasetRegistration
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.plan import MountSpec, TaskExecutionPlan
from scientist_lab.tasks.rgbt_detection.baseline_adapter import (
    ensure_baselines_loaded,
    get_baseline_adapter,
    is_real_baseline,
    resolve_baseline_key,
)
from scientist_lab.tasks.rgbt_detection.baseline_config_builder import (
    build_native_config,
)


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

        ensure_baselines_loaded()
        baseline_key = resolve_baseline_key(contract)
        adapter = get_baseline_adapter(baseline_key)
        adapter.validate_parameters(dict(contract.parameters or {}))

        claim = str((contract.task_config or {}).get("claim_level") or "")
        real = is_real_baseline(baseline_key)

        if not real and contract.execution_mode in {"validate_data", "fast_eval"}:
            epochs = contract.parameters.get("epochs")
            if epochs not in (None, 0, "0"):
                raise ValueError(
                    f"{contract.execution_mode} forbids non-zero epochs "
                    f"(got parameters.epochs={epochs!r})"
                )
        if not real and contract.execution_mode == "fast_eval":
            source = contract.parameters.get("checkpoint_source")
            if not source:
                raise ValueError(
                    "fast_eval requires parameters.checkpoint_source "
                    "(path to an existing checkpoint/last.npz)"
                )
        if real and contract.execution_mode == "fast_eval":
            # Formal exploratory fast_eval (v0.8.2+) may train under budget.
            # Pipeline claim on real baseline still allows short train in v0.8.1.
            if claim not in {
                "",
                "pipeline_validation_only",
                "exploratory_comparison",
            }:
                raise ValueError(
                    f"unsupported claim_level for real baseline fast_eval: {claim}"
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

        ensure_baselines_loaded()
        baseline_key = resolve_baseline_key(contract)
        baseline = get_baseline_adapter(baseline_key)
        real = is_real_baseline(baseline_key)

        workspace = self._resolve_workspace(contract, code_roots, real=real)
        if workspace is None or not workspace.exists():
            raise FileNotFoundError(
                "RGB-T code root not found for "
                f"baseline={baseline_key} code_reference={contract.code_reference}"
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
            f"baseline={baseline_key}",
            "docker network disabled by LocalDockerRunner",
            *baseline.build_command_notes(contract),
        ]
        if contract.execution_mode == "validate_data":
            notes.append("validate_data: no model training")

        # Persist native config hint into notes; runner still writes parameters config.
        try:
            native = build_native_config(contract)
            notes.append(f"native_config_keys={sorted(native.keys())}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"native_config_error={exc}")

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
                "BASELINE_KEY": baseline_key,
                "EXPECT_DATASET_READ_ONLY": "1",
            },
            workspace_source=workspace,
            notes=notes,
        )

    @staticmethod
    def _resolve_workspace(
        contract: ExperimentContract,
        code_roots: dict[str, Path],
        *,
        real: bool,
    ) -> Path | None:
        ref = (contract.code_reference or "").strip()
        if ref in code_roots:
            return code_roots[ref]
        if real:
            return (
                code_roots.get("local:rgbt_detection_real")
                or code_roots.get("rgbt_detection_real")
            )
        return code_roots.get("local:rgbt_detector") or code_roots.get("rgbt_detector")
