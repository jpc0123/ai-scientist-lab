"""Map Freeze ExperimentContract (HOW) onto legacy CUDA Fast Eval contract.

Does not invent hypotheses. Planner already chose budget_class / scope.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.cuda_doctor import CUDA_ENV_KEY

from scientist_lab.adapters.dfine.how import FORMAL_TIMEOUT_SECONDS


EvalFn = Callable[..., dict[str, Any]]


CUDA_FAST_EVAL_DEFAULTS = {
    "baseline": "dfine_s",
    "dfine_backend": "dfine",
    "input_mode": "rgb",
    "fusion_method": "none",
    "epochs": 1,
    # 12GB-class GPUs at 160x160; was 2 (very slow). Do not change mid-run.
    "batch_size": 8,
    "learning_rate": 0.0002,
    "image_width": 160,
    "image_height": 160,
    "max_train_images": 16,
    "max_val_images": 8,
    "num_workers": 0,
    "mixed_precision": False,
}


def freeze_to_legacy_contract(freeze: Mapping[str, Any]) -> dict[str, Any]:
    mat = dict(freeze.get("materialization") or {})
    dataset = dict(freeze.get("dataset") or {})
    metrics = dict(freeze.get("metrics_spec") or {})
    budget = dict(freeze.get("budget") or {})
    params = dict(CUDA_FAST_EVAL_DEFAULTS)
    params.update(mat.get("legacy_parameters") or {})
    primary = metrics.get("primary") or "mAP50_95"
    secondary = list(metrics.get("secondary") or [])
    # Freeze research_protocol_* is not the workbench ProtocolService registry.
    freeze_protocol = str(freeze.get("protocol_id") or "")
    legacy_protocol = mat.get("legacy_protocol_id")
    if not legacy_protocol and freeze_protocol.startswith("protocol_"):
        legacy_protocol = freeze_protocol
    budget_class = str(freeze.get("budget_class") or "").strip().lower()
    exec_mode = str(mat.get("execution_mode") or ("full_train" if budget_class == "formal" else "fast_eval"))
    eval_scope = str(
        mat.get("evaluation_scope")
        or ("rgbt_tiny_v1_full" if exec_mode == "full_train" else "fast_eval_subset")
    )
    claim_level = str(mat.get("claim_level") or "exploratory_comparison")
    timeout_default = FORMAL_TIMEOUT_SECONDS if exec_mode == "full_train" else 1200
    if exec_mode == "full_train":
        params.pop("max_train_images", None)
        params.pop("max_val_images", None)
    payload: dict[str, Any] = {
        "schema_version": "1.2",
        "project_id": freeze["project_id"],
        "node_id": str(freeze.get("run_id") or freeze.get("plan_id") or "dfine_run"),
        "parent_node_id": freeze.get("parent_run"),
        "title": f"Freeze DFINE execute {freeze.get('run_id')}",
        "research_goal": "Execute approved ExperimentContract via CUDA DFINE Adapter HOW path.",
        "hypothesis": freeze.get("hypothesis"),
        "task_type": "rgbt_detection",
        "runner_profile": mat.get("runner_profile", "local"),
        "environment_key": mat.get("environment_key", CUDA_ENV_KEY),
        "code_reference": mat.get("code_reference", "local:rgbt_detection_real"),
        "dataset_reference": dataset.get("reference") or "dataset:rgbt_tiny_v1",
        "entrypoint": "run_detection_experiment.py",
        "execution_mode": exec_mode,
        "parameters": params,
        "task_config": {
            "primary_metric": primary,
            "metrics": [primary, *secondary],
            "evaluation_scope": eval_scope,
            "claim_level": claim_level,
        },
        "seed": int(freeze.get("seed") or 42),
        "resources": {
            "gpu_count": int(budget.get("gpu_count") or 1),
            "cpu_count": 4,
            "memory_gb": 32 if exec_mode == "full_train" else 8,
            "timeout_seconds": int(budget.get("timeout_seconds") or timeout_default),
        },
        "expected_outputs": list(
            freeze.get("expected_artifacts")
            or ["metrics.json", "checkpoint_selection.json"]
        ),
    }
    if legacy_protocol:
        payload["protocol_id"] = legacy_protocol
    return payload


_FREEZE_CONTROL_FILES = {
    "protocol.json",
    "plan.json",
    "contract.json",
    "experiment_run.json",
    "research_events.jsonl",
    "manager_status.json",
    "handle.json",
    "result.json",
    "review.json",
    "cuda_doctor.json",
}


_SKIP_SYNC_DIRS = {"_dfine_run", "_dfine_stage", "__pycache__"}


def _sync_output_dir(payload: Mapping[str, Any], dest: Path) -> None:
    src = (
        (payload.get("run") or {}).get("output_directory")
        or payload.get("output_directory")
    )
    if not src:
        return
    root = Path(str(src))
    if not root.is_dir() or root.resolve() == dest.resolve():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for path in root.iterdir():
        if path.name in _FREEZE_CONTROL_FILES or path.name in _SKIP_SYNC_DIRS:
            continue
        target = dest / path.name
        if path.is_file():
            shutil.copy2(path, target)
        elif path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)


def make_cuda_live_runner(
    experiments: Any,
    *,
    execute: bool,
    require_live_ready: bool = False,
    wait: bool = True,
    probe_runtime: bool = True,
    call_eval: EvalFn | None = None,
) -> Callable[[Mapping[str, Any], Path], dict[str, Any]]:
    """Return Adapter live_runner. execute=False is rejected (Adapter dry_run skips this)."""

    def runner(contract: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
        if not execute:
            raise MaterializeRejected("cuda live_runner requires --execute")
        from scientist_lab.runners.exec_reattach import try_harvest_existing
        from scientist_lab.tasks.rgbt_detection.dfine_cuda_orchestrator import (
            run_dfine_cuda_fast_eval,
        )

        eval_fn = call_eval or run_dfine_cuda_fast_eval
        output_dir.mkdir(parents=True, exist_ok=True)
        legacy_dict = freeze_to_legacy_contract(contract)
        legacy = ExperimentContract.model_validate(legacy_dict)
        from scientist_lab.runners.exec_reattach import contract_identity

        expected = contract_identity(legacy_dict) or contract_identity(contract)
        if not expected.get("node_id"):
            expected["node_id"] = str(
                contract.get("run_id") or contract.get("plan_id") or legacy_dict.get("node_id") or ""
            )

        # Harvest *before* rewriting dest contract. Otherwise a leftover metrics.json
        # would match the freshly written node_id and skip a real GPU job.
        outputs_root = _outputs_root(experiments, output_dir)
        project_id = str(getattr(legacy, "project_id", None) or contract.get("project_id") or "")
        harvested = try_harvest_existing(
            outputs_root=outputs_root,
            dest=output_dir,
            project_id=project_id or None,
            wait=wait,
            timeout_seconds=float(
                getattr(getattr(legacy, "resources", None), "timeout_seconds", None)
                or 86_400
            ),
            expected=expected,
        )
        # Reject harvest when synced artifacts disagree with requested HOW/epochs
        # (seen on LONGTRAIN E8 r3: plan P3@8 harvested F0@2 metrics).
        if harvested is not None:
            from scientist_lab.runners.exec_reattach import (
                identities_match,
                load_contract_identity,
            )

            actual = load_contract_identity(output_dir)
            if not identities_match(expected, actual):
                harvested = None
                # Scrub mismatched leftovers so the fresh GPU job is not confused.
                for name in (
                    "metrics.json",
                    "model_summary.json",
                    "config.json",
                    "dfine_subset.json",
                    "dfine_fast_config.yml",
                    "execution.json",
                    "live_execution.json",
                    "learning_curve.json",
                    "metrics_per_epoch.json",
                    "checkpoint_selection.json",
                    "fusion_module_summary.json",
                ):
                    stale = output_dir / name
                    if stale.is_file():
                        stale.unlink()
        if harvested is not None:
            harvested = dict(harvested)
            contract_path = output_dir / "_legacy_fast_eval_contract.json"
            # Always rewrite expected legacy so the next round cannot latch onto
            # a stale F0 stamp that shares the same node_id.
            contract_path.write_text(
                json.dumps(legacy.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            harvested["legacy_contract_path"] = str(contract_path)
            status = str(harvested.get("status") or "failed")
            return {
                "status": status,
                "orchestrator": harvested.get("orchestrator"),
                "run": harvested.get("run"),
                "dry_run": False,
                "reattached": True,
                "legacy_contract_path": str(contract_path),
            }

        contract_path = output_dir / "_legacy_fast_eval_contract.json"
        contract_path.write_text(
            json.dumps(legacy.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        payload = eval_fn(
            experiments,
            contract_path=contract_path,
            wait=wait,
            dry_run=False,
            require_live_ready=require_live_ready,
            probe_runtime=probe_runtime,
        )
        _sync_output_dir(payload, output_dir)
        status = str(
            (payload.get("run") or {}).get("status") or payload.get("status") or "failed"
        )
        return {
            "status": status,
            "orchestrator": payload.get("orchestrator"),
            "run": payload.get("run"),
            "dry_run": payload.get("dry_run"),
            "legacy_contract_path": str(contract_path),
        }

    return runner


def _outputs_root(experiments: Any, dest: Path) -> Path:
    settings = getattr(experiments, "settings", None)
    if settings is not None:
        out = getattr(settings, "outputs_dir", None) or getattr(settings, "outputs_root", None)
        if out:
            return Path(out)
    for parent in [Path(dest), *Path(dest).parents]:
        if parent.name == "outputs" and parent.is_dir():
            return parent
        cand = parent / "outputs"
        if cand.is_dir():
            return cand
    return Path(dest).resolve().parents[2] / "outputs"
