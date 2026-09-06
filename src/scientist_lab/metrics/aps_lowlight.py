"""APS_lowlight: COCO AP_small on frozen low_light_subset_v1 images.

Full-set APS / mAP cannot stand in for this metric.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from scientist_lab.datasets.low_light_subset import SLICE_ID, subset_ids

COCO_SMALL_AREA = 32.0 * 32.0
PRIMARY_METRIC = "APS_lowlight"
SECONDARY_SLICE = ("mAP50_95_lowlight", "AP50_lowlight", "Recall_small_lowlight")
SECONDARY_ALL = ("APS_all", "mAP50_95_all")


def coco_image_key(image: Mapping[str, Any]) -> str:
    for key in ("sample_id", "file_name", "file_name_rgb"):
        value = str(image.get(key) or "").strip()
        if value:
            if key == "file_name" and value.lower().endswith(".jpg"):
                return value.rsplit(".", 1)[0]
            return value
    return str(image.get("id") or "")


def filter_coco_gt(gt: Mapping[str, Any], sample_ids: Sequence[str] | set[str]) -> dict[str, Any]:
    wanted = {str(item) for item in sample_ids}
    images = []
    keep_ids: set[Any] = set()
    for image in gt.get("images") or []:
        if not isinstance(image, Mapping):
            continue
        key = coco_image_key(image)
        image_id = image.get("id")
        file_stem = str(image.get("file_name") or "")
        aliases = {key, str(image_id), file_stem, file_stem.rsplit(".", 1)[0]}
        if wanted.intersection({str(item) for item in aliases if item}):
            images.append(dict(image))
            keep_ids.add(image_id)
    anns = [
        dict(ann)
        for ann in (gt.get("annotations") or [])
        if isinstance(ann, Mapping) and ann.get("image_id") in keep_ids
    ]
    out = dict(gt)
    out["images"] = images
    out["annotations"] = anns
    return out


def filter_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    keep_image_ids: set[Any],
) -> list[dict[str, Any]]:
    return [dict(row) for row in detections if row.get("image_id") in keep_image_ids]


def _small_ann(ann: Mapping[str, Any]) -> bool:
    area = ann.get("area")
    if area is None:
        bbox = list(ann.get("bbox") or [])
        if len(bbox) >= 4:
            area = float(bbox[2]) * float(bbox[3])
    try:
        return float(area) < COCO_SMALL_AREA
    except (TypeError, ValueError):
        return False


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    if union <= 0:
        return 0.0
    return inter / union


def _small_object_recall(
    gt: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    *,
    iou_thr: float = 0.5,
) -> float:
    """Deterministic Recall_small on the filtered GT. Not a substitute for COCO AP."""
    by_image: dict[Any, list[dict[str, Any]]] = {}
    for row in detections:
        by_image.setdefault(row.get("image_id"), []).append(dict(row))
    hits = 0
    total = 0
    for ann in gt.get("annotations") or []:
        if not isinstance(ann, Mapping) or not _small_ann(ann):
            continue
        total += 1
        bbox = [float(x) for x in (ann.get("bbox") or [0, 0, 0, 0])[:4]]
        matched = False
        for det in by_image.get(ann.get("image_id"), []):
            db = [float(x) for x in (det.get("bbox") or [0, 0, 0, 0])[:4]]
            if _iou(bbox, db) >= iou_thr:
                matched = True
                break
        if matched:
            hits += 1
    if total == 0:
        return 0.0
    return hits / float(total)


def _coco_eval_stats(gt: Mapping[str, Any], detections: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None
    if not gt.get("images") or not detections:
        return {
            "APS_lowlight": 0.0,
            "mAP50_95_lowlight": 0.0,
            "AP50_lowlight": 0.0,
        }
    coco = COCO()
    coco.dataset = dict(gt)
    coco.createIndex()
    coco_dt = coco.loadRes(list(detections))
    evaler = COCOeval(coco, coco_dt, "bbox")
    evaler.evaluate()
    evaler.accumulate()
    evaler.summarize()
    stats = list(evaler.stats)
    return {
        "mAP50_95_lowlight": float(stats[0]),
        "AP50_lowlight": float(stats[1]),
        "APS_lowlight": float(stats[3]),
    }


def evaluate_aps_lowlight(
    gt: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    *,
    split: str = "val",
) -> dict[str, Any]:
    ids = subset_ids(freeze, split=split)
    filtered = filter_coco_gt(gt, ids)
    keep_image_ids = {image.get("id") for image in filtered.get("images") or []}
    filtered_dt = filter_detections(detections, keep_image_ids=keep_image_ids)
    coco_stats = _coco_eval_stats(filtered, filtered_dt)
    recall = _small_object_recall(filtered, filtered_dt)
    metrics = {
        "Recall_small_lowlight": recall,
        "n_images_lowlight": len(filtered.get("images") or []),
        "n_annotations_lowlight": len(filtered.get("annotations") or []),
        "slice_id": SLICE_ID,
        "split": split,
        "evaluator": "coco_ap_small_on_low_light_subset_v1",
    }
    if coco_stats is not None:
        metrics.update(coco_stats)
        metrics["evaluator_backend"] = "pycocotools"
    else:
        # Offline tests: Recall_small is recorded; APS_lowlight must not be invented from mAP.
        metrics["APS_lowlight"] = None
        metrics["mAP50_95_lowlight"] = None
        metrics["AP50_lowlight"] = None
        metrics["evaluator_backend"] = "subset_filter_only"
        metrics["note"] = (
            "pycocotools not installed; subset filter is frozen. "
            "APS_lowlight remains unset (do not copy APS_all / mAP)."
        )
    return metrics
