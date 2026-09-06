"""Simple metrics for torch mini-detector stand-in."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dataset_loader import DetSample
from model_torch import MiniDetHead


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter + 1e-8
    return float(inter / union)


def _predict_one(
    model: MiniDetHead,
    sample: DetSample,
    device: torch.device,
) -> tuple[int, np.ndarray, float]:
    tensor = (
        torch.from_numpy(sample.image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    with torch.no_grad():
        logits, box = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        score, pred_cls = torch.max(probs, dim=0)
        pred_box = box[0].detach().cpu().numpy()
    # Decode box as sigmoid-normalized xywh relative to image size.
    pred_box = 1.0 / (1.0 + np.exp(-pred_box))
    pred_box = np.asarray(
        [
            pred_box[0] * sample.width,
            pred_box[1] * sample.height,
            pred_box[2] * sample.width,
            pred_box[3] * sample.height,
        ],
        dtype=np.float32,
    )
    return int(pred_cls.item()), pred_box, float(score.item())


def evaluate_model(
    model: MiniDetHead,
    samples: list[DetSample],
    device: torch.device,
    *,
    score_thresh: float = 0.05,
) -> dict[str, float]:
    if not samples:
        return {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "AP_small": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    tp = fp = fn = 0
    ious: list[float] = []
    small_hits = 0
    small_total = 0
    model.eval()
    for sample in samples:
        pred_cls, pred_box, score = _predict_one(model, sample, device)
        if sample.boxes.shape[0] == 0:
            if score >= score_thresh:
                fp += 1
            continue
        gt_box = sample.boxes[0]
        gt_cls = int(sample.labels[0]) if sample.labels.size else 1
        area = float(gt_box[2] * gt_box[3])
        is_small = area < (32 * 32)
        if is_small:
            small_total += 1
        iou = _iou_xywh(pred_box, gt_box)
        ious.append(iou)
        hit = score >= score_thresh and pred_cls == gt_cls and iou >= 0.5
        if hit:
            tp += 1
            if is_small:
                small_hits += 1
        else:
            if score >= score_thresh:
                fp += 1
            fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    map50 = float(np.mean([1.0 if v >= 0.5 else 0.0 for v in ious])) if ious else 0.0
    thr = np.arange(0.5, 1.0, 0.05)
    map5095 = (
        float(np.mean([np.mean([1.0 if v >= t else 0.0 for v in ious]) for t in thr]))
        if ious
        else 0.0
    )
    return {
        "mAP50": map50,
        "mAP50_95": map5095,
        "AP_small": (small_hits / small_total) if small_total else 0.0,
        "precision": float(precision),
        "recall": float(recall),
    }


def predictions_preview(
    model: MiniDetHead,
    samples: list[DetSample],
    device: torch.device,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples[:limit]:
        pred_cls, pred_box, score = _predict_one(model, sample, device)
        rows.append(
            {
                "stem": sample.stem,
                "pred_cls": pred_cls,
                "score": score,
                "pred_box_xywh": [float(x) for x in pred_box.tolist()],
                "gt_boxes": sample.boxes.tolist(),
            }
        )
    return rows
