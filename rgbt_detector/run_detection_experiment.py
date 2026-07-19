from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def _list_stems(directory: Path) -> list[str]:
    stems = []
    if not directory.is_dir():
        return stems
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            stems.append(path.stem)
    return stems


def _collect_modality_dirs(data_root: Path, modality: str) -> list[Path]:
    """Support split layout images/{train,val}/{rgb,thermal} and legacy flat dirs."""
    split_dirs = [
        data_root / "images" / "train" / modality,
        data_root / "images" / "val" / modality,
    ]
    existing = [path for path in split_dirs if path.is_dir()]
    if existing:
        return existing
    legacy = data_root / modality
    return [legacy] if legacy.is_dir() else []


def _validate_data(data_root: Path, output_dir: Path) -> dict[str, Any]:
    rgb_dirs = _collect_modality_dirs(data_root, "rgb")
    thermal_dirs = _collect_modality_dirs(data_root, "thermal")
    rgb_stems: set[str] = set()
    thermal_stems: set[str] = set()
    for directory in rgb_dirs:
        rgb_stems.update(_list_stems(directory))
    for directory in thermal_dirs:
        thermal_stems.update(_list_stems(directory))
    paired = sorted(rgb_stems & thermal_stems)
    missing_rgb = sorted(thermal_stems - rgb_stems)
    missing_thermal = sorted(rgb_stems - thermal_stems)
    report = {
        "schema_version": "1.0",
        "dataset_root": str(data_root),
        "valid": not missing_rgb and not missing_thermal and bool(paired),
        "rgb_image_count": len(rgb_stems),
        "thermal_image_count": len(thermal_stems),
        "paired_image_count": len(paired),
        "missing_rgb": missing_rgb,
        "missing_thermal": missing_thermal,
        "claim_level": "pipeline_validation_only",
        "warnings": [
            "Dataset is a debug subset and must not be used for publication claims."
        ],
    }
    if not report["valid"]:
        report["errors"] = ["RGB/thermal pairing incomplete inside container."]
    write_json(output_dir / "dataset_report.json", report)
    return report

def _smoke_train(
    *,
    config: dict[str, Any],
    contract: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    seed: int,
    input_mode: str,
    fusion_method: str,
) -> dict[str, Any]:
    started = time.time()
    report = _validate_data(data_root, output_dir)
    if not report.get("valid"):
        raise RuntimeError("dataset_pair_mismatch inside container")

    epochs = int(config.get("epochs", 2))
    train_images = min(int(config.get("max_train_images", 50)), report["paired_image_count"])
    val_images = min(int(config.get("max_val_images", 20)), max(1, report["paired_image_count"] // 5))

    rng = random.Random(seed)
    history_rows: list[tuple[int, float]] = []
    loss = 1.2
    for epoch in range(1, epochs + 1):
        # Deterministic fake optimization curve — validates pipeline only.
        mode_bonus = {"rgb": 0.01, "thermal": 0.008, "rgbt": 0.015}.get(input_mode, 0.01)
        fusion_bonus = 0.005 if fusion_method not in {"none", "", None} else 0.0
        loss = max(0.05, loss * 0.72 - mode_bonus - fusion_bonus + rng.uniform(-0.01, 0.01))
        if math.isnan(loss):
            raise RuntimeError("nan_loss")
        history_rows.append((epoch, float(loss)))
        print(f"epoch={epoch} loss={loss:.4f} mode={input_mode}", flush=True)

    # Debug metrics intentionally low / synthetic.
    base = 0.02 + 0.01 * ({"rgb": 1, "thermal": 0.8, "rgbt": 1.2}.get(input_mode, 1))
    if fusion_method not in {"none", "", None}:
        base += 0.005
    jitter = rng.uniform(-0.002, 0.002)
    map50 = max(0.0, min(1.0, base + 0.03 + jitter))
    map5095 = max(0.0, min(1.0, base + jitter))
    ap_small = max(0.0, min(1.0, map5095 * 0.4))
    precision = max(0.0, min(1.0, map50 * 0.9))
    recall = max(0.0, min(1.0, map50 * 0.85))

    history_path = output_dir / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss"])
        writer.writerows(history_rows)

    ckpt_dir = output_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "last_smoke.txt").write_text(
        f"seed={seed}\ninput_mode={input_mode}\nfusion={fusion_method}\n",
        encoding="utf-8",
    )

    write_json(
        output_dir / "model_summary.json",
        {
            "model": config.get("model", "dfine_s_smoke_stub"),
            "input_mode": input_mode,
            "fusion_method": fusion_method,
            "note": "Pipeline smoke stub — not a real detector.",
        },
    )
    write_json(
        output_dir / "sample_predictions.json",
        {
            "count": min(3, train_images),
            "predictions": [
                {"image_id": i, "boxes": [[10, 10, 40, 40]], "scores": [0.1]}
                for i in range(min(3, train_images))
            ],
        },
    )

    duration = time.time() - started
    metrics = {
        "schema_version": "1.0",
        "project_id": contract.get("project_id", "project_rgbt_001"),
        "node_id": contract.get("node_id", "rgbt_node"),
        "task_type": "rgbt_detection",
        "primary_metric": "mAP50_95",
        "metrics": {
            "mAP50": float(map50),
            "mAP50_95": float(map5095),
            "AP_small": float(ap_small),
            "precision": float(precision),
            "recall": float(recall),
            "duration_seconds": float(duration),
        },
        "training": {
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_completed": epochs,
            "train_images": train_images,
            "validation_images": val_images,
            "duration_seconds": float(duration),
            "final_loss": float(loss),
            "nan_loss": False,
        },
        "evaluation_scope": "debug_subset",
        "claim_level": "pipeline_validation_only",
        "status": "completed",
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


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

    print(f"RGB-T detection entry mode={mode}", flush=True)
    print(f"data_root={data_root}", flush=True)

    if mode == "validate_data":
        report = _validate_data(data_root, output_dir)
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
                "seed": seed,
                "epochs_requested": 0,
                "epochs_completed": 0,
                "duration_seconds": 0.0,
            },
            "evaluation_scope": "debug_subset",
            "claim_level": "pipeline_validation_only",
            "status": "completed" if report.get("valid") else "failed",
        }
        write_json(output_dir / "metrics.json", metrics)
        if not report.get("valid"):
            raise SystemExit(2)
    elif mode in {"smoke_train", "fast_eval"}:
        _smoke_train(
            config=config,
            contract=contract,
            data_root=data_root,
            output_dir=output_dir,
            seed=seed,
            input_mode=args.input_mode,
            fusion_method=args.fusion_method,
        )
        if mode == "smoke_train" and not (output_dir / "dataset_report.json").exists():
            _validate_data(data_root, output_dir)
    else:
        raise SystemExit(f"Unsupported execution_mode: {mode}")

    write_json(
        output_dir / "artifact_manifest.json",
        {
            "artifacts": [
                {"path": "metrics.json", "type": "metrics", "required": True},
                {"path": "dataset_report.json", "type": "dataset_report", "required": True},
                {"path": "training_history.csv", "type": "curve", "required": False},
                {"path": "model_summary.json", "type": "summary", "required": False},
                {"path": "sample_predictions.json", "type": "predictions", "required": False},
                {"path": "checkpoint/last_smoke.txt", "type": "checkpoint", "required": False},
                {"path": "combined.log", "type": "log", "required": False},
            ]
        },
    )
    print("RGB-T detection experiment finished", flush=True)


if __name__ == "__main__":
    main()
