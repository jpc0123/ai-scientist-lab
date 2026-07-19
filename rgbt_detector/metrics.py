"""Simple detection metrics for smoke evaluation (not COCO-official)."""

from __future__ import annotations

from typing import Any

import numpy as np

from dataset import DetSample
from model import TinyDetector


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter + 1e-8
    return float(inter / union)


def evaluate_detector(
    model: TinyDetector,
    samples: list[DetSample],
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

    for sample in samples:
        pred_cls, pred_box, score = model.predict(sample.image)
        if sample.boxes.shape[0] == 0:
            if score >= score_thresh:
                fp += 1
            continue

        # match first GT (smoke single-object assumption)
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
    # coarse mAP50:95 proxy: average indicator over thresholds
    thr = np.arange(0.5, 1.0, 0.05)
    map5095 = (
        float(np.mean([np.mean([1.0 if v >= t else 0.0 for v in ious]) for t in thr]))
        if ious
        else 0.0
    )
    ap_small = small_hits / (small_total + 1e-8) if small_total else 0.0

    return {
        "mAP50": float(np.clip(map50, 0.0, 1.0)),
        "mAP50_95": float(np.clip(map5095, 0.0, 1.0)),
        "AP_small": float(np.clip(ap_small, 0.0, 1.0)),
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0)),
    }


def predictions_preview(
    model: TinyDetector, samples: list[DetSample], *, limit: int = 3
) -> dict[str, Any]:
    rows = []
    for sample in samples[:limit]:
        cls, box, score = model.predict(sample.image)
        rows.append(
            {
                "stem": sample.stem,
                "pred_category_id": cls,
                "pred_bbox_xywh": [float(v) for v in box.tolist()],
                "score": score,
                "gt_bbox_xywh": (
                    [float(v) for v in sample.boxes[0].tolist()]
                    if sample.boxes.shape[0]
                    else None
                ),
            }
        )
    return {"count": len(rows), "predictions": rows}
