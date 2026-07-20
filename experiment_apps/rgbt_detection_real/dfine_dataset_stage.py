"""Stage Scientist Lab RGB-T data into a flat COCO folder for D-FINE."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def stage_coco_for_dfine(
    data_root: Path,
    stage_root: Path,
    *,
    input_mode: str = "rgb",
    fusion_method: str = "none",
) -> dict[str, Path]:
    """Create train/val image folders + annotation copies DFINE can consume."""
    data_root = Path(data_root)
    stage_root = Path(stage_root)
    if stage_root.exists():
        shutil.rmtree(stage_root)
    train_img = stage_root / "train2017"
    val_img = stage_root / "val2017"
    ann_dir = stage_root / "annotations"
    train_img.mkdir(parents=True)
    val_img.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    modality = _resolve_modality_dir(input_mode=input_mode, fusion_method=fusion_method)
    _copy_split(
        src_images=data_root / "images" / "train" / modality,
        src_ann=data_root / "annotations" / "instances_train.json",
        dst_images=train_img,
        dst_ann=ann_dir / "instances_train2017.json",
        fusion=(input_mode == "rgbt" and fusion_method == "early_concat"),
        data_root=data_root,
        split="train",
    )
    _copy_split(
        src_images=data_root / "images" / "val" / modality,
        src_ann=data_root / "annotations" / "instances_val.json",
        dst_images=val_img,
        dst_ann=ann_dir / "instances_val2017.json",
        fusion=(input_mode == "rgbt" and fusion_method == "early_concat"),
        data_root=data_root,
        split="val",
    )
    return {
        "train_img": train_img,
        "val_img": val_img,
        "train_ann": ann_dir / "instances_train2017.json",
        "val_ann": ann_dir / "instances_val2017.json",
        "stage_root": stage_root,
    }


def _resolve_modality_dir(*, input_mode: str, fusion_method: str) -> str:
    mode = (input_mode or "rgb").strip().lower()
    if mode == "thermal":
        return "thermal"
    if mode in {"rgbt", "rgb_thermal"} and fusion_method == "early_concat":
        # Early fusion is materialized as RGB-shaped tensors during copy.
        return "rgb"
    return "rgb"


def _copy_split(
    *,
    src_images: Path,
    src_ann: Path,
    dst_images: Path,
    dst_ann: Path,
    fusion: bool,
    data_root: Path,
    split: str,
) -> None:
    if not src_ann.is_file():
        raise FileNotFoundError(f"missing annotation: {src_ann}")
    payload = json.loads(src_ann.read_text(encoding="utf-8"))
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

    # Keep categories as-is; DFINE will use num_classes from config.
    dst_ann.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def count_categories(ann_path: Path) -> int:
    payload = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    cats = payload.get("categories") or []
    return max(len(cats), 1)
