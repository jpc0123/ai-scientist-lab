"""RGB | thermal pair previews. Thermal is grayscale IR, not a second RGB camera."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

DEFAULT_DATASET_ID = "rgbt_tiny_v1"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SPLITS = ("train", "val", "test")
_MODALITIES = ("rgb", "thermal")


def resolve_processed_root(project_root: Path | str, dataset_id: str) -> Path:
    root = Path(project_root)
    key = dataset_id.replace("dataset:", "")
    registry = root / "data" / "registry" / f"{key}.json"
    processed: Path | None = None
    if registry.is_file():
        import json

        doc = json.loads(registry.read_text(encoding="utf-8"))
        rel = str(doc.get("processed_root") or "").strip()
        if rel:
            processed = Path(rel)
            if not processed.is_absolute():
                processed = root / processed
    if processed is None or not processed.is_dir():
        fallback = root / "datasets" / "registered" / key
        if fallback.is_dir():
            processed = fallback
    if processed is None or not processed.is_dir():
        raise FileNotFoundError(f"processed root missing for {key}")
    return processed


def list_pair_stems(
    processed_root: Path,
    *,
    split: str = "train",
    limit: int = 3,
) -> list[str]:
    if split not in _SPLITS:
        raise ValueError(f"illegal split: {split}")
    rgb_dir = processed_root / "images" / split / "rgb"
    thermal_dir = processed_root / "images" / split / "thermal"
    if not rgb_dir.is_dir() or not thermal_dir.is_dir():
        return []
    names: list[str] = []
    for path in sorted(rgb_dir.iterdir()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if (thermal_dir / path.name).is_file():
            names.append(path.name)
        if len(names) >= limit:
            break
    return names


def modality_path(
    processed_root: Path,
    *,
    split: str,
    modality: str,
    name: str,
) -> Path:
    if split not in _SPLITS:
        raise ValueError(f"illegal split: {split}")
    if modality not in _MODALITIES:
        raise ValueError(f"illegal modality: {modality}")
    safe = Path(name).name
    if safe != name or ".." in name or "/" in name or "\\" in name:
        raise ValueError("illegal image name")
    path = processed_root / "images" / split / modality / safe
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def compose_pair_png(
    rgb_path: Path,
    thermal_path: Path,
    *,
    max_height: int = 220,
    caption: str | None = None,
) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    rgb = Image.open(rgb_path).convert("RGB")
    thermal = Image.open(thermal_path).convert("L").convert("RGB")
    scale = max_height / max(rgb.height, 1)
    if scale < 1:
        rgb = rgb.resize((max(1, int(rgb.width * scale)), max_height))
        thermal = thermal.resize((max(1, int(thermal.width * scale)), max_height))
    gap = 8
    label_h = 22
    width = rgb.width + gap + thermal.width
    height = max(rgb.height, thermal.height) + label_h + 18
    canvas = Image.new("RGB", (width, height), (18, 24, 28))
    canvas.paste(rgb, (0, label_h))
    canvas.paste(thermal, (rgb.width + gap, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    draw.text((6, 4), "RGB", fill=(240, 240, 240), font=font)
    draw.text((rgb.width + gap + 6, 4), "thermal (grayscale IR)", fill=(180, 220, 180), font=font)
    foot = caption or f"{rgb_path.name}  |  thermal is not a second color camera"
    draw.text((6, height - 16), foot, fill=(200, 200, 200), font=font)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def preview_manifest(
    project_root: Path | str,
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    split: str = "train",
    limit: int = 3,
) -> dict[str, Any]:
    key = dataset_id.replace("dataset:", "")
    processed = resolve_processed_root(project_root, key)
    names = list_pair_stems(processed, split=split, limit=limit)
    items = []
    for idx, name in enumerate(names):
        items.append(
            {
                "index": idx,
                "split": split,
                "stem": Path(name).stem,
                "file_name": name,
                "composite_url": (
                    f"/api/v1/dataset-workspace/datasets/{key}/pair-previews/{idx}/png"
                    f"?split={split}"
                ),
                "rgb_url": (
                    f"/api/v1/dataset-workspace/datasets/{key}/images/{split}/rgb/{name}"
                ),
                "thermal_url": (
                    f"/api/v1/dataset-workspace/datasets/{key}/images/{split}/thermal/{name}"
                ),
            }
        )
    return {
        "dataset_id": key,
        "processed_root": str(processed),
        "split": split,
        "thermal_is_grayscale_ir": True,
        "note": (
            "RGBT-Tiny 是 RGB + 单通道热红外配对，不是两路彩色照片。"
            "F1 early_concat 会把两路各 50% 混成一张 3 通道图再喂检测器，"
            "所以训练缓存看起来仍是彩色；原始热红外在 thermal/。"
        ),
        "items": items,
        "count": len(items),
    }
