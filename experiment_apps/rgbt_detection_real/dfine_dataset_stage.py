"""Stage Scientist Lab RGB-T data into a flat COCO folder for D-FINE."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fusion_names import (
    is_early_concat,
    needs_paired_thermal,
    normalize_fusion_method,
)


def _subset_coco(payload: dict[str, Any], max_images: int | None) -> dict[str, Any]:
    """Keep a stable prefix of images. Formal runs pass max_images=None."""
    if max_images is None or int(max_images) <= 0:
        return payload
    images = sorted(
        list(payload.get("images") or []),
        key=lambda im: str(im.get("file_name") or im.get("id") or ""),
    )
    keep = images[: int(max_images)]
    keep_ids = {int(im["id"]) for im in keep}
    anns = [
        ann
        for ann in (payload.get("annotations") or [])
        if int(ann.get("image_id")) in keep_ids
    ]
    out = dict(payload)
    out["images"] = keep
    out["annotations"] = anns
    return out


def stage_coco_for_dfine(
    data_root: Path,
    stage_root: Path,
    *,
    input_mode: str = "rgb",
    fusion_method: str = "none",
    label_map_path: Path | None = None,
    max_train_images: int | None = None,
    max_val_images: int | None = None,
) -> dict[str, Path | dict[str, Any]]:
    """Create train/val image folders + annotation copies DFINE can consume."""
    data_root = Path(data_root)
    stage_root = Path(stage_root)
    if stage_root.exists():
        shutil.rmtree(stage_root)
    train_img = stage_root / "train2017"
    val_img = stage_root / "val2017"
    thermal_train_img = stage_root / "thermal_train2017"
    thermal_val_img = stage_root / "thermal_val2017"
    ann_dir = stage_root / "annotations"
    train_img.mkdir(parents=True)
    val_img.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    fusion = normalize_fusion_method(fusion_method)
    early = (
        str(input_mode).lower() in {"rgbt", "rgb_thermal"} and is_early_concat(fusion)
    )
    paired = (
        str(input_mode).lower() in {"rgbt", "rgb_thermal"}
        and needs_paired_thermal(fusion)
    )
    modality = _resolve_modality_dir(input_mode=input_mode, fusion_method=fusion)
    train_map = _copy_split(
        src_images=data_root / "images" / "train" / modality,
        src_ann=data_root / "annotations" / "instances_train.json",
        dst_images=train_img,
        dst_ann=ann_dir / "instances_train2017.json",
        fusion=early,
        data_root=data_root,
        split="train",
        max_images=max_train_images,
    )
    val_map = _copy_split(
        src_images=data_root / "images" / "val" / modality,
        src_ann=data_root / "annotations" / "instances_val.json",
        dst_images=val_img,
        dst_ann=ann_dir / "instances_val2017.json",
        fusion=early,
        data_root=data_root,
        split="val",
        max_images=max_val_images,
    )
    if paired:
        thermal_train_img.mkdir(parents=True, exist_ok=True)
        thermal_val_img.mkdir(parents=True, exist_ok=True)
        _copy_modality_images(
            src_images=data_root / "images" / "train" / "thermal",
            dst_images=thermal_train_img,
            src_ann=data_root / "annotations" / "instances_train.json",
            max_images=max_train_images,
        )
        _copy_modality_images(
            src_images=data_root / "images" / "val" / "thermal",
            dst_images=thermal_val_img,
            src_ann=data_root / "annotations" / "instances_val.json",
            max_images=max_val_images,
        )
    if train_map != val_map:
        raise ValueError(
            "train/val category remap maps differ; refuse inconsistent label spaces"
        )
    validate_category_label_map(train_map)
    map_path = Path(label_map_path or (stage_root / "category_label_map.json"))
    map_path.write_text(
        json.dumps(train_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    out: dict[str, Path | dict[str, Any]] = {
        "train_img": train_img,
        "val_img": val_img,
        "train_ann": ann_dir / "instances_train2017.json",
        "val_ann": ann_dir / "instances_val2017.json",
        "stage_root": stage_root,
        "category_label_map": train_map,
        "category_label_map_path": map_path,
        "fusion_method": fusion,
        "staging_mode": (
            "gated_paired"
            if paired
            else ("early_concat_blend" if early else "single_modality")
        ),
        "max_train_images": max_train_images,
        "max_val_images": max_val_images,
    }
    if paired:
        out["thermal_train_img"] = thermal_train_img
        out["thermal_val_img"] = thermal_val_img
    return out


def _resolve_modality_dir(*, input_mode: str, fusion_method: str) -> str:
    mode = (input_mode or "rgb").strip().lower()
    fusion = normalize_fusion_method(fusion_method)
    if mode == "thermal":
        return "thermal"
    if mode in {"rgbt", "rgb_thermal"} and (
        is_early_concat(fusion) or needs_paired_thermal(fusion)
    ):
        # RGB folder is the primary COCO image root; thermal is paired separately
        # for gated_multiscale / plugin:*, or blended in-place for early_concat.
        return "rgb"
    return "rgb"


def _copy_modality_images(
    *,
    src_images: Path,
    dst_images: Path,
    src_ann: Path,
    max_images: int | None = None,
) -> None:
    if not src_ann.is_file():
        raise FileNotFoundError(f"missing annotation: {src_ann}")
    payload = _subset_coco(json.loads(src_ann.read_text(encoding="utf-8")), max_images)
    for image in payload.get("images") or []:
        name = str(image["file_name"])
        src = src_images / name
        if not src.is_file():
            raise FileNotFoundError(f"missing image: {src}")
        shutil.copy2(src, dst_images / name)


def _copy_split(
    *,
    src_images: Path,
    src_ann: Path,
    dst_images: Path,
    dst_ann: Path,
    fusion: bool,
    data_root: Path,
    split: str,
    max_images: int | None = None,
) -> dict[str, Any]:
    if not src_ann.is_file():
        raise FileNotFoundError(f"missing annotation: {src_ann}")
    payload = _subset_coco(json.loads(src_ann.read_text(encoding="utf-8")), max_images)
    from PIL import Image
    import numpy as np

    for image in payload.get("images") or []:
        name = str(image["file_name"])
        src = src_images / name
        if not src.is_file():
            raise FileNotFoundError(f"missing image: {src}")
        dst = dst_images / name
        if fusion:
            thermal = data_root / "images" / split / "thermal" / name
            if not thermal.is_file():
                raise FileNotFoundError(f"missing thermal pair: {thermal}")
            rgb = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32)
            thr = np.asarray(Image.open(thermal).convert("L"), dtype=np.float32)
            thr3 = np.stack([thr, thr, thr], axis=-1)
            fused = np.clip(0.5 * rgb + 0.5 * thr3, 0, 255).astype(np.uint8)
            Image.fromarray(fused).save(dst)
        else:
            shutil.copy2(src, dst)

    # D-FINE expects class indices in [0, num_classes). Raw COCO ids are often 1-based.
    remapped, label_map = remap_categories_zero_based(payload)
    dst_ann.write_text(json.dumps(remapped, ensure_ascii=False, indent=2), encoding="utf-8")
    return label_map


def remap_categories_zero_based(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remap COCO category ids to contiguous model labels [0, N).

    Returns (remapped_payload, label_map) where label_map contains both directions
    as string-keyed dicts for stable JSON serialization.
    """
    cats = list(payload.get("categories") or [])
    if not cats:
        empty = {
            "dataset_category_to_model_label": {},
            "model_label_to_dataset_category": {},
            "num_classes": 0,
        }
        return payload, empty

    ordered = sorted(cats, key=lambda c: int(c.get("id", 0)))
    id_map = {int(c["id"]): i for i, c in enumerate(ordered)}
    # Reject duplicate category ids.
    if len(id_map) != len(ordered):
        raise ValueError("duplicate category ids in COCO categories")

    new_cats = [{**c, "id": id_map[int(c["id"])]} for c in ordered]
    new_anns = []
    unknown: list[int] = []
    for ann in payload.get("annotations") or []:
        cid = int(ann.get("category_id", 0))
        if cid not in id_map:
            unknown.append(cid)
            continue
        new_anns.append({**ann, "category_id": id_map[cid]})
    if unknown:
        raise ValueError(f"annotations reference unknown category_id(s): {sorted(set(unknown))}")

    out = dict(payload)
    out["categories"] = new_cats
    out["annotations"] = new_anns

    dataset_to_model = {str(k): int(v) for k, v in id_map.items()}
    model_to_dataset = {str(v): int(k) for k, v in id_map.items()}
    label_map = {
        "dataset_category_to_model_label": dataset_to_model,
        "model_label_to_dataset_category": model_to_dataset,
        "num_classes": len(id_map),
    }
    validate_category_label_map(label_map)
    return out, label_map


def validate_category_label_map(label_map: dict[str, Any]) -> None:
    forward = dict(label_map.get("dataset_category_to_model_label") or {})
    inverse = dict(label_map.get("model_label_to_dataset_category") or {})
    if len(forward) != len(inverse):
        raise ValueError("category label map is not bijective (size mismatch)")
    # Bijection check.
    for src, dst in forward.items():
        back = inverse.get(str(dst))
        if back is None or str(back) != str(src):
            raise ValueError(f"category map not bijective at dataset_id={src}")
    for dst, src in inverse.items():
        fwd = forward.get(str(src))
        if fwd is None or str(fwd) != str(dst):
            raise ValueError(f"category map not bijective at model_label={dst}")
    labels = sorted(int(x) for x in inverse.keys())
    n = int(label_map.get("num_classes") or len(labels))
    if labels and (labels[0] != 0 or labels[-1] != n - 1 or labels != list(range(n))):
        raise ValueError(
            f"model labels must be contiguous 0..num_classes-1; got {labels} num_classes={n}"
        )


def invert_model_label_to_dataset_category(
    model_label: int,
    label_map: dict[str, Any],
) -> int:
    inverse = label_map.get("model_label_to_dataset_category") or {}
    key = str(int(model_label))
    if key not in inverse:
        raise KeyError(f"unknown model_label={model_label}")
    return int(inverse[key])


def count_categories(ann_path: Path) -> int:
    payload = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    cats = payload.get("categories") or []
    return max(len(cats), 1)
