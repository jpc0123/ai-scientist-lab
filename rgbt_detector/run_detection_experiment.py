from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from artifact_writer import write_manifest
from fast_eval import run_fast_eval
from train_smoke import run_smoke_train
from validate_dataset import validate_rgbt_in_container, write_json


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        import json

        return json.load(file)


def _validate_data(
    data_root: Path,
    output_dir: Path,
    *,
    dataset_key: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    report = validate_rgbt_in_container(
        data_root,
        dataset_key=dataset_key,
        probe_read_only=True,
    )
    write_json(output_dir / "dataset_report.json", report)

    for forbidden in (
        "training_history.csv",
        "checkpoint",
        "sample_predictions.json",
        "model_summary.json",
    ):
        path = output_dir / forbidden
        if path.exists():
            raise RuntimeError(f"validate_data must not create {forbidden}")

    metrics = {
        "schema_version": "1.0",
        "project_id": contract.get("project_id", "project_rgbt_001"),
        "node_id": contract.get("node_id", "rgbt_node"),
        "task_type": "rgbt_detection",
        "primary_metric": "mAP50_95",
        "metrics": {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "AP_small": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "paired_image_count": float(report.get("paired_image_count") or 0),
        },
        "training": {
            "seed": int(contract.get("seed") or 42),
            "epochs_requested": 0,
            "epochs_completed": 0,
            "duration_seconds": 0.0,
        },
        "evaluation_scope": "debug_subset",
        "claim_level": "pipeline_validation_only",
        "execution_mode": "validate_data",
        "status": "completed" if report.get("valid") else "failed",
    }
    write_json(output_dir / "metrics.json", metrics)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--execution-mode", default="smoke_train")
    parser.add_argument("--input-mode", default="rgb")
    parser.add_argument("--fusion-method", default="none")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(Path(args.config))
    contract_path = output_dir / "contract.json"
    contract = read_json(contract_path) if contract_path.exists() else {}
    seed = int(args.seed if args.seed is not None else contract.get("seed", 42))
    data_root = Path(args.data_root)
    mode = args.execution_mode
    dataset_key = str(
        contract.get("dataset_reference", "dataset:unknown")
    ).removeprefix("dataset:")

    print(f"RGB-T detection entry mode={mode}", flush=True)
    print(f"data_root={data_root}", flush=True)

    if mode == "validate_data":
        report = _validate_data(
            data_root,
            output_dir,
            dataset_key=dataset_key or "unknown",
            contract={**contract, "seed": seed},
        )
        if not report.get("valid"):
            raise SystemExit(2)
    elif mode == "smoke_train":
        report = validate_rgbt_in_container(
            data_root,
            dataset_key=dataset_key or "unknown",
            probe_read_only=True,
        )
        write_json(output_dir / "dataset_report.json", report)
        if not report.get("valid"):
            raise RuntimeError("dataset_pair_mismatch inside container")
        run_smoke_train(
            data_root=data_root,
            output_dir=output_dir,
            config=config,
            contract=contract,
            seed=seed,
            input_mode=args.input_mode,
            fusion_method=args.fusion_method,
        )
        if not (output_dir / "checkpoint" / "last.npz").exists():
            raise RuntimeError("checkpoint_missing")
    elif mode == "fast_eval":
        report = validate_rgbt_in_container(
            data_root,
            dataset_key=dataset_key or "unknown",
            probe_read_only=True,
        )
        write_json(output_dir / "dataset_report.json", report)
        if not report.get("valid"):
            raise RuntimeError("dataset_pair_mismatch inside container")
        run_fast_eval(
            data_root=data_root,
            output_dir=output_dir,
            config=config,
            contract=contract,
            seed=seed,
            input_mode=args.input_mode,
            fusion_method=args.fusion_method,
        )
        if (output_dir / "training_history.csv").exists():
            raise RuntimeError("fast_eval must not create training_history.csv")
    else:
        raise SystemExit(f"Unsupported execution_mode: {mode}")

    write_manifest(output_dir, mode=mode)
    print("RGB-T detection experiment finished", flush=True)


if __name__ == "__main__":
    main()
