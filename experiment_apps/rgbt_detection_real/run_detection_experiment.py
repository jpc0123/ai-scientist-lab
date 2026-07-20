from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from artifact_writer import write_json, write_manifest
from dataset_loader import quick_dataset_report
from train_minimal import run_minimal_train


def read_json(path: Path) -> dict[str, Any]:
    import json

    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RGB-T real baseline entry (dfine_s stand-in in v0.8.1)"
    )
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

    print(f"RGB-T real baseline entry mode={mode} baseline=dfine_s", flush=True)
    print(f"data_root={data_root} device_hint=torch", flush=True)

    report = quick_dataset_report(data_root, dataset_key=dataset_key or "unknown")
    write_json(output_dir / "dataset_report.json", report)
    if not report.get("valid"):
        raise RuntimeError(f"dataset_pair_mismatch: {report.get('issues')}")

    if mode == "validate_data":
        metrics = {
            "schema_version": "1.0",
            "project_id": contract.get("project_id", "project_rgbt_002"),
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
                "seed": seed,
                "epochs_requested": 0,
                "epochs_completed": 0,
                "duration_seconds": 0.0,
                "trained": False,
                "baseline_key": "dfine_s",
            },
            "evaluation_scope": "debug_subset",
            "claim_level": "pipeline_validation_only",
            "execution_mode": "validate_data",
            "status": "completed",
        }
        write_json(output_dir / "metrics.json", metrics)
        write_json(
            output_dir / "model_summary.json",
            {
                "baseline_key": "dfine_s",
                "baseline_implementation": "torch_mini_standin_v0_8_1",
                "validate_only": True,
            },
        )
        write_json(
            output_dir / "resource_usage.json",
            {"duration_seconds": 0.0, "validate_only": True},
        )
    elif mode in {"smoke_train", "fast_eval"}:
        # v0.8.1: fast_eval with real baseline currently means short train+eval
        # until dedicated checkpoint-eval path for .pt is added in v0.8.2.
        run_minimal_train(
            data_root=data_root,
            output_dir=output_dir,
            config=config,
            contract={**contract, "execution_mode": mode},
            seed=seed,
            input_mode=args.input_mode,
            fusion_method=args.fusion_method,
        )
        if not (output_dir / "checkpoint" / "last.pt").exists():
            raise RuntimeError("checkpoint_missing")
    else:
        raise SystemExit(f"Unsupported execution_mode: {mode}")

    write_manifest(output_dir, mode=mode)
    print("RGB-T real baseline experiment finished", flush=True)


if __name__ == "__main__":
    main()
