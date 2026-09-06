"""Minimal train / eval loop for dfine_s torch stand-in."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dataset_loader import load_split_samples
from metrics_simple import evaluate_model, predictions_preview
from model_torch import channels_for_mode, create_model, save_checkpoint


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _to_batch(
    samples: list,
    indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = []
    boxes = []
    labels = []
    for idx in indices:
        sample = samples[idx]
        images.append(torch.from_numpy(sample.image).permute(2, 0, 1))
        if sample.boxes.shape[0] == 0:
            boxes.append(torch.zeros(4, dtype=torch.float32))
            labels.append(torch.tensor(0, dtype=torch.int64))
        else:
            box = sample.boxes[0].astype(np.float32)
            # Normalize to [0,1] then logit-space targets via inverse sigmoid-ish clamp.
            target = np.asarray(
                [
                    box[0] / max(sample.width, 1),
                    box[1] / max(sample.height, 1),
                    box[2] / max(sample.width, 1),
                    box[3] / max(sample.height, 1),
                ],
                dtype=np.float32,
            )
            target = np.clip(target, 1e-3, 1.0 - 1e-3)
            logit = np.log(target / (1.0 - target))
            boxes.append(torch.from_numpy(logit.astype(np.float32)))
            labels.append(
                torch.tensor(
                    int(sample.labels[0]) % 10,
                    dtype=torch.int64,
                )
            )
    x = torch.stack(images, dim=0).to(device=device, dtype=torch.float32)
    y_box = torch.stack(boxes, dim=0).to(device=device)
    y_cls = torch.stack(labels, dim=0).to(device=device)
    return x, y_box, y_cls


def run_minimal_train(
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_width = int(config.get("image_width", 160))
    image_height = int(config.get("image_height", 128))
    epochs = int(config.get("epochs", 2))
    batch_size = max(1, int(config.get("batch_size", 2)))
    lr = float(config.get("learning_rate", 1e-3))
    max_train = int(config.get("max_train_images", 16))
    max_val = int(config.get("max_val_images", 8))

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
        raise RuntimeError("model_config_error: no train samples")

    in_channels = channels_for_mode(input_mode, fusion_method)
    if train_samples[0].image.shape[2] != in_channels:
        raise RuntimeError(
            "model_config_error: "
            f"expected {in_channels} channels, got {train_samples[0].image.shape[2]}"
        )

    model = create_model(
        in_channels=in_channels, seed=seed, device=device, num_classes=10
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history_rows: list[list[Any]] = [["epoch", "train_loss", "cls_loss", "box_loss"]]
    final_loss = 0.0
    peak_gpu_mb = 0.0

    model.train()
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_samples)))
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(order)
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_box = 0.0
        steps = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            x, y_box, y_cls = _to_batch(train_samples, batch_idx, device)
            optimizer.zero_grad(set_to_none=True)
            logits, pred_box = model(x)
            cls_loss = F.cross_entropy(logits, y_cls)
            box_loss = F.smooth_l1_loss(pred_box, y_box)
            loss = cls_loss + box_loss
            if not torch.isfinite(loss):
                raise RuntimeError("nan_loss")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_cls += float(cls_loss.item())
            epoch_box += float(box_loss.item())
            steps += 1
            if device.type == "cuda":
                peak_gpu_mb = max(
                    peak_gpu_mb,
                    float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
                )
        final_loss = epoch_loss / max(steps, 1)
        history_rows.append(
            [
                epoch,
                final_loss,
                epoch_cls / max(steps, 1),
                epoch_box / max(steps, 1),
            ]
        )
        progress = {
            "stage": "training",
            "epoch": epoch,
            "total_epochs": epochs,
            "progress": epoch / max(epochs, 1),
            "latest_metrics": {"train_loss": final_loss},
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json(output_dir / "progress.json", progress)

    ckpt_meta = {
        "in_channels": in_channels,
        "num_classes": 10,
        "baseline_key": "dfine_s",
        "baseline_implementation": "torch_mini_standin_v0_8_1",
        "input_mode": input_mode,
        "fusion_method": fusion_method,
        "image_width": image_width,
        "image_height": image_height,
        "seed": seed,
    }
    save_checkpoint(model, output_dir / "checkpoint" / "last.pt", meta=ckpt_meta)

    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(history_rows)

    eval_metrics = evaluate_model(model, val_samples, device)
    preview = predictions_preview(model, val_samples, device)
    duration = time.time() - started

    task_config = contract.get("task_config") or {}
    metrics = {
        "schema_version": "1.0",
        "project_id": contract.get("project_id", "project_rgbt_002"),
        "node_id": contract.get("node_id", "rgbt_fast_node"),
        "task_type": "rgbt_detection",
        "primary_metric": task_config.get("primary_metric", "mAP50_95"),
        "metrics": {
            **eval_metrics,
            "duration_seconds": duration,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "parameter_count": float(model.parameter_count()),
        },
        "training": {
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_completed": epochs,
            "train_images": len(train_samples),
            "validation_images": len(val_samples),
            "duration_seconds": duration,
            "final_train_loss": final_loss,
            "nan_loss": False,
            "backend": "torch_mini_standin",
            "device": str(device),
            "trained": True,
            "baseline_key": "dfine_s",
        },
        "evaluation_scope": task_config.get("evaluation_scope", "debug_subset"),
        "claim_level": task_config.get("claim_level", "pipeline_validation_only"),
        "execution_mode": contract.get("execution_mode", "smoke_train"),
        "status": "completed",
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "model_summary.json",
        {
            "baseline_key": "dfine_s",
            "baseline_implementation": "torch_mini_standin_v0_8_1",
            "vendor_status": "standin_until_dfine_vendored",
            "parameter_count": model.parameter_count(),
            "in_channels": in_channels,
            "input_mode": input_mode,
            "fusion_method": fusion_method,
            "device": str(device),
            "image_width": image_width,
            "image_height": image_height,
        },
    )
    _write_json(
        output_dir / "resource_usage.json",
        {
            "cpu_count_requested": (contract.get("resources") or {}).get("cpu_count"),
            "memory_gb_requested": (contract.get("resources") or {}).get("memory_gb"),
            "gpu_count_requested": (contract.get("resources") or {}).get("gpu_count", 0),
            "peak_gpu_memory_mb": peak_gpu_mb,
            "duration_seconds": duration,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
    )
    _write_json(output_dir / "sample_predictions.json", {"predictions": preview})
    _write_json(
        output_dir / "native_config.json",
        {
            "baseline_key": "dfine_s",
            "config": config,
        },
    )
    return metrics
