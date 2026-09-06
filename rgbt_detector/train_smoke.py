"""Smoke training loop: forward → loss → backward → checkpoint → eval."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from dataset import iter_batches, load_split_samples
from metrics import evaluate_detector, predictions_preview
from model import TinyDetector, channels_for_mode
from validate_dataset import write_json


def run_smoke_train(
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
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Smoke uses smaller spatial size for CPU speed; scale boxes accordingly in loader.
    image_width = int(config.get("smoke_image_width", 160))
    image_height = int(config.get("smoke_image_height", 128))
    epochs = int(config.get("epochs", 2))
    batch_size = max(1, int(config.get("batch_size", 2)))
    lr = float(config.get("learning_rate", 1e-3))
    max_train = int(config.get("max_train_images", 40))
    max_val = int(config.get("max_val_images", 10))

    train_samples = load_split_samples(
        data_root,
        split="train",
        input_mode=input_mode,
        fusion_method=fusion_method,
        image_width=image_width,
        image_height=image_height,
        max_images=max_train,
    )
    val_samples = load_split_samples(
        data_root,
        split="val",
        input_mode=input_mode,
        fusion_method=fusion_method,
        image_width=image_width,
        image_height=image_height,
        max_images=max_val,
    )
    if not train_samples:
        raise RuntimeError("model_config_error: no train samples for smoke_train")

    in_channels = channels_for_mode(input_mode, fusion_method)
    # verify channel match
    if train_samples[0].image.shape[2] != in_channels:
        raise RuntimeError(
            "model_config_error: "
            f"expected {in_channels} channels, got {train_samples[0].image.shape[2]}"
        )

    model = TinyDetector.create(
        in_channels=in_channels,
        num_classes=2,
        feature_dim=64,
        image_width=image_width,
        image_height=image_height,
        seed=seed,
    )

    history_rows: list[list[Any]] = []
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for batch in iter_batches(
            train_samples, batch_size=batch_size, shuffle=True, seed=seed + epoch
        ):
            for sample in batch:
                if sample.boxes.shape[0] == 0:
                    # synthetic tiny box so backward still runs
                    target_box = np.asarray(
                        [
                            image_width * 0.25,
                            image_height * 0.25,
                            image_width * 0.2,
                            image_height * 0.2,
                        ],
                        dtype=np.float32,
                    )
                    target_label = 1
                else:
                    target_box = sample.boxes[0]
                    target_label = int(sample.labels[0])
                loss, _parts = model.loss_and_backward(
                    sample.image, target_box, target_label, lr=lr
                )
                losses.append(loss)
        final_loss = float(np.mean(losses)) if losses else 0.0
        if not np.isfinite(final_loss):
            raise RuntimeError("nan_loss")
        val_metrics = evaluate_detector(model, val_samples)
        history_rows.append(
            [
                epoch,
                final_loss,
                val_metrics["mAP50"],
                val_metrics["mAP50_95"],
                val_metrics["AP_small"],
            ]
        )
        print(
            f"epoch={epoch} train_loss={final_loss:.4f} "
            f"mAP50={val_metrics['mAP50']:.4f} mode={input_mode}",
            flush=True,
        )

    ckpt_dir = output_dir / "checkpoint"
    ckpt_path = ckpt_dir / "last.npz"
    model.save(ckpt_path)
    if not ckpt_path.exists():
        raise RuntimeError("checkpoint_missing")

    # also keep a tiny text marker for older verifiers
    (ckpt_dir / "last_smoke.txt").write_text(
        f"seed={seed}\ninput_mode={input_mode}\nfusion={fusion_method}\n"
        f"checkpoint=last.npz\nbackend=numpy\n",
        encoding="utf-8",
    )

    history_path = output_dir / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "mAP50", "mAP50_95", "AP_small"])
        writer.writerows(history_rows)

    eval_metrics = evaluate_detector(model, val_samples)
    write_json(output_dir / "model_summary.json", model.summary(
        input_mode=input_mode, fusion_method=fusion_method
    ))
    write_json(
        output_dir / "sample_predictions.json",
        predictions_preview(model, val_samples or train_samples, limit=3),
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
            "epochs_requested": epochs,
            "epochs_completed": epochs,
            "train_images": len(train_samples),
            "validation_images": len(val_samples),
            "duration_seconds": float(duration),
            "final_train_loss": float(final_loss),
            "final_loss": float(final_loss),
            "nan_loss": False,
            "batch_size": batch_size,
            "learning_rate": lr,
            "image_width": image_width,
            "image_height": image_height,
            "backend": "numpy_tiny_detector",
        },
        "evaluation_scope": "debug_subset",
        "claim_level": "pipeline_validation_only",
        "execution_mode": "smoke_train",
        "status": "completed",
    }
    write_json(output_dir / "metrics.json", metrics)

    # silence unused rng warning by touching once
    _ = float(rng.random())
    return metrics
