"""fast_eval: load checkpoint and evaluate without training."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dataset import load_split_samples
from metrics import evaluate_detector, predictions_preview
from model import TinyDetector, channels_for_mode
from validate_dataset import write_json


def run_fast_eval(
    *,
    data_root: Path,
    output_dir: Path,
    config: dict[str, Any],
    contract: dict[str, Any],
    seed: int,
    input_mode: str,
    fusion_method: str,
) -> dict[str, Any]:
    started = time.time()
    image_width = int(config.get("smoke_image_width", config.get("image_width", 160)))
    image_height = int(config.get("smoke_image_height", config.get("image_height", 128)))
    max_val = int(config.get("max_val_images", 20))
    split = str(config.get("eval_split", "val"))

    ckpt_path = output_dir / "checkpoint" / "last.npz"
    if not ckpt_path.exists():
        # also accept explicit filename
        alt = output_dir / "checkpoint" / str(config.get("checkpoint_file", "last.npz"))
        if alt.exists():
            ckpt_path = alt
        else:
            raise RuntimeError("checkpoint_missing: fast_eval requires checkpoint/last.npz")

    model = TinyDetector.load(ckpt_path)
    expected_channels = channels_for_mode(input_mode, fusion_method)
    if int(model.in_channels) != expected_channels:
        raise RuntimeError(
            "model_config_error: "
            f"checkpoint in_channels={model.in_channels} != mode channels={expected_channels}"
        )

    samples = load_split_samples(
        data_root,
        split=split,
        input_mode=input_mode,
        fusion_method=fusion_method,
        image_width=image_width,
        image_height=image_height,
        max_images=max_val,
    )
    if not samples:
        raise RuntimeError("evaluation_failed: no evaluation samples")

    eval_metrics = evaluate_detector(model, samples)
    write_json(
        output_dir / "model_summary.json",
        {
            **model.summary(input_mode=input_mode, fusion_method=fusion_method),
            "checkpoint_path": str(ckpt_path),
            "eval_split": split,
            "loaded_for": "fast_eval",
        },
    )
    write_json(
        output_dir / "sample_predictions.json",
        predictions_preview(model, samples, limit=5),
    )

    # Marker that no training occurred.
    (output_dir / "checkpoint" / "fast_eval_loaded.txt").write_text(
        f"loaded={ckpt_path.name}\nseed={seed}\ntrained=false\n",
        encoding="utf-8",
    )

    duration = time.time() - started
    metrics = {
        "schema_version": "1.0",
        "project_id": contract.get("project_id", "project_rgbt_001"),
        "node_id": contract.get("node_id", "rgbt_node"),
        "task_type": "rgbt_detection",
        "primary_metric": "mAP50_95",
        "metrics": {
            **{k: float(v) for k, v in eval_metrics.items()},
            "duration_seconds": float(duration),
        },
        "training": {
            "seed": seed,
            "epochs_requested": 0,
            "epochs_completed": 0,
            "train_images": 0,
            "validation_images": len(samples),
            "duration_seconds": float(duration),
            "final_train_loss": None,
            "nan_loss": False,
            "backend": "numpy_tiny_detector",
            "trained": False,
        },
        "evaluation_scope": str(
            (contract.get("task_config") or {}).get(
                "evaluation_scope", "debug_subset"
            )
        ),
        "claim_level": "pipeline_validation_only",
        "execution_mode": "fast_eval",
        "status": "completed",
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
