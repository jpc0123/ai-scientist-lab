"""Minimal RGB-T detection dataset loader for smoke_train."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DetSample:
    stem: str
    image: np.ndarray  # H,W,C float32 in [0,1]
    boxes: np.ndarray  # N,4 xywh absolute
    labels: np.ndarray  # N int64
    width: int
    height: int


def _load_image(path: Path, *, mode: str) -> np.ndarray:
    with Image.open(path) as img:
        if mode == "L":
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
            return arr[:, :, None]
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return arr


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    if image.ndim == 3 and image.shape[2] == 1:
        pil = Image.fromarray((np.clip(image[:, :, 0], 0, 1) * 255).astype(np.uint8), mode="L")
        pil = pil.resize((width, height), Image.BILINEAR)
        out = np.asarray(pil, dtype=np.float32) / 255.0
        return out[:, :, None]
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
    if rgb.is_dir() and thermal.is_dir():
        return rgb, thermal, ann
    # legacy flat
    return root / "rgb", root / "thermal", ann


def load_split_samples(
    root: Path,
    *,
    split: str,
    input_mode: str,
    fusion_method: str,
    image_width: int,
    image_height: int,
    max_images: int | None = None,
    stem_prefix: str | None = None,
) -> list[DetSample]:
    rgb_dir, thermal_dir, ann_path = _resolve_split_dirs(root, split)
    rgb_map = _list_stems(rgb_dir)
    thermal_map = _list_stems(thermal_dir)
    if stem_prefix:
        rgb_map = {k: v for k, v in rgb_map.items() if k.startswith(stem_prefix)}
        thermal_map = {k: v for k, v in thermal_map.items() if k.startswith(stem_prefix)}

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
    for stem in paired:
        rgb = _load_image(rgb_map[stem], mode="RGB")
        thermal = _load_image(thermal_map[stem], mode="L")
        # spatial match assumed; resize thermal to rgb if needed
        if thermal.shape[:2] != rgb.shape[:2]:
            thermal = _resize(thermal, rgb.shape[1], rgb.shape[0])

        if input_mode == "rgb":
            image = rgb
        elif input_mode == "thermal":
            image = np.repeat(thermal, 3, axis=2)
        elif input_mode == "rgbt":
            if fusion_method in {"early_concat", "concat", "early"}:
                image = np.concatenate([rgb, thermal], axis=2)  # H,W,4
            else:
                image = 0.5 * rgb + 0.5 * np.repeat(thermal, 3, axis=2)
        else:
            raise ValueError(f"model_config_error: unsupported input_mode={input_mode}")

        scale_x = image_width / float(image.shape[1])
        scale_y = image_height / float(image.shape[0])
        image = _resize(image, image_width, image_height)

        meta = images_by_stem.get(stem) or {}
        image_id = meta.get("id")
        boxes: list[list[float]] = []
        labels: list[int] = []
        if image_id is not None:
            for ann in anns_by_id.get(int(image_id), []):
                bbox = ann.get("bbox") or []
                if len(bbox) != 4:
                    continue
                x, y, w, h = [float(v) for v in bbox]
                boxes.append([x * scale_x, y * scale_y, w * scale_x, h * scale_y])
                labels.append(int(ann.get("category_id", 1)))

        samples.append(
            DetSample(
                stem=stem,
                image=image.astype(np.float32),
                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.asarray(labels, dtype=np.int64).reshape(-1),
                width=image_width,
                height=image_height,
            )
        )
    return samples


def iter_batches(
    samples: list[DetSample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[list[DetSample]]:
    order = list(range(len(samples)))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield [samples[i] for i in idx]
