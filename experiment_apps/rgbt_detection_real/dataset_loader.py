"""Minimal RGB-T loader for real-baseline stand-in (self-contained for Docker)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DetSample:
    stem: str
    image: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    width: int
    height: int


def channels_for_mode(input_mode: str, fusion_method: str) -> int:
    mode = (input_mode or "rgb").strip().lower()
    if mode in {"rgb", "thermal"}:
        return 3
    if mode == "rgbt":
        return 4 if (fusion_method or "early_concat") == "early_concat" else 3
    raise ValueError(f"unsupported input_mode: {input_mode}")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _load_thermal_as_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return np.stack([gray, gray, gray], axis=-1)


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8), mode="L")
        pil = pil.resize((width, height), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return arr
    if image.shape[2] == 1:
        return _resize(image[:, :, 0], width, height)[:, :, None]
    if image.shape[2] == 4:
        rgb = Image.fromarray((np.clip(image[:, :, :3], 0, 1) * 255).astype(np.uint8))
        t = Image.fromarray((np.clip(image[:, :, 3], 0, 1) * 255).astype(np.uint8), mode="L")
        rgb = rgb.resize((width, height), Image.BILINEAR)
        t = t.resize((width, height), Image.BILINEAR)
        out = np.asarray(rgb, dtype=np.float32) / 255.0
        thermal = np.asarray(t, dtype=np.float32) / 255.0
        return np.concatenate([out, thermal[:, :, None]], axis=-1)
    pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((width, height), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _list_stems(directory: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not directory.is_dir():
        return mapping
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            mapping[path.stem] = path
    return mapping


def _resolve_split_dirs(root: Path, split: str) -> tuple[Path, Path, Path]:
    rgb = root / "images" / split / "rgb"
    thermal = root / "images" / split / "thermal"
    ann = root / "annotations" / f"instances_{split}.json"
    return rgb, thermal, ann


def load_split_samples(
    root: Path,
    *,
    split: str,
    input_mode: str,
    fusion_method: str,
    image_width: int,
    image_height: int,
    max_images: int | None = None,
) -> list[DetSample]:
    rgb_dir, thermal_dir, ann_path = _resolve_split_dirs(root, split)
    rgb_map = _list_stems(rgb_dir)
    thermal_map = _list_stems(thermal_dir)
    paired = sorted(set(rgb_map) & set(thermal_map))
    if max_images is not None:
        paired = paired[: max(0, int(max_images))]

    coco: dict[str, Any] = {"images": [], "annotations": [], "categories": []}
    if ann_path.is_file():
        with ann_path.open("r", encoding="utf-8") as file:
            coco = json.load(file)
    images_by_stem = {
        Path(str(img.get("file_name", ""))).stem: img for img in coco.get("images") or []
    }
    anns_by_id: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations") or []:
        anns_by_id.setdefault(int(ann["image_id"]), []).append(ann)

    samples: list[DetSample] = []
    mode = (input_mode or "rgb").strip().lower()
    for stem in paired:
        rgb = _load_rgb(rgb_map[stem])
        thermal = _load_thermal_as_rgb(thermal_map[stem])
        if mode == "rgb":
            image = rgb
        elif mode == "thermal":
            image = thermal
        else:
            image = np.concatenate([rgb, thermal[:, :, :1]], axis=-1)

        ow, oh = image.shape[1], image.shape[0]
        image = _resize(image, image_width, image_height)
        sx = image_width / max(ow, 1)
        sy = image_height / max(oh, 1)

        boxes: list[list[float]] = []
        labels: list[int] = []
        meta = images_by_stem.get(stem)
        if meta is not None:
            for ann in anns_by_id.get(int(meta["id"]), []):
                x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
                boxes.append([x * sx, y * sy, w * sx, h * sy])
                labels.append(int(ann.get("category_id", 1)))

        samples.append(
            DetSample(
                stem=stem,
                image=image.astype(np.float32),
                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.asarray(labels, dtype=np.int64),
                width=image_width,
                height=image_height,
            )
        )
    return samples


def quick_dataset_report(root: Path, *, dataset_key: str) -> dict[str, Any]:
    issues: list[str] = []
    paired_train = 0
    paired_val = 0
    for split in ("train", "val"):
        rgb_dir, thermal_dir, _ = _resolve_split_dirs(root, split)
        if not rgb_dir.is_dir():
            issues.append(f"missing {split} rgb dir")
            continue
        if not thermal_dir.is_dir():
            issues.append(f"missing {split} thermal dir")
            continue
        paired = len(set(_list_stems(rgb_dir)) & set(_list_stems(thermal_dir)))
        if split == "train":
            paired_train = paired
        else:
            paired_val = paired
    return {
        "valid": not issues and (paired_train + paired_val) > 0,
        "dataset_key": dataset_key,
        "paired_image_count": paired_train + paired_val,
        "paired_train": paired_train,
        "paired_val": paired_val,
        "issues": issues,
        "validator": "rgbt_detection_real_quick",
    }
