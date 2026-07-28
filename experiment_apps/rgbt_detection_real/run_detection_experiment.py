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


def _select_backend(contract: dict[str, Any], config: dict[str, Any]) -> str:
    params = dict(config.get("parameters") or contract.get("parameters") or {})
    requested = str(
        params.get("dfine_backend") or params.get("baseline_backend") or "auto"
    ).strip().lower()
    if requested in {"standin", "torch_mini", "mini"}:
        return "standin"
    if requested in {"dfine", "dfine_s", "vendor"}:
        return "dfine"
    from dfine_config_builder import dfine_root_from_app

    vendor = dfine_root_from_app(Path(__file__).resolve().parent) / "train.py"
    return "dfine" if vendor.is_file() else "standin"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RGB-T real baseline entry (dfine_s vendored or stand-in)"
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
    backend = _select_backend(contract, config)

    print(
        f"RGB-T real baseline entry mode={mode} baseline=dfine_s backend={backend}",
        flush=True,
    )
    print(f"data_root={data_root}", flush=True)

    fusion = str(args.fusion_method or "none").strip().lower()
    if fusion in {
        "fdpn",
        "full",
        "full_method",
        "complete",
        "mid_fusion",
        "late_fusion",
        "dual_stream",
    }:
        raise RuntimeError(
            f"fusion_method={fusion!r} is not implemented (P00/FDPN blocked); "
            "use early_concat for exploratory fusion smoke only"
        )

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
                "baseline_implementation": "validate_only",
                "validate_only": True,
            },
        )
        write_json(
            output_dir / "resource_usage.json",
            {"duration_seconds": 0.0, "validate_only": True},
        )
    elif mode in {"smoke_train", "fast_eval"}:
        if backend == "dfine":
            from train_dfine import run_dfine_train

            run_dfine_train(
                data_root=data_root,
                output_dir=output_dir,
                config=config,
                contract={**contract, "execution_mode": mode},
                seed=seed,
                input_mode=args.input_mode,
                fusion_method=args.fusion_method,
            )
        else:
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
