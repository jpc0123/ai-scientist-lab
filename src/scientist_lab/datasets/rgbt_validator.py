from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scientist_lab.datasets.preview_generator import (
    VALIDATOR_VERSION,
    generate_rgbt_previews,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96


def _list_images(directory: Path) -> tuple[dict[str, Path], list[str]]:
    if not directory.exists():
        return {}, []
    mapping: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        if stem in mapping:
            duplicates.append(stem)
        mapping[stem] = path
    return mapping, duplicates

def _load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _image_meta(path: Path) -> tuple[int, int, int] | None:
    """Return (width, height, channels) or None if corrupt."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = int(image.width), int(image.height)
            channels = len(image.getbands())
            if width <= 0 or height <= 0:
                return None
            if channels <= 0:
                return None
            return width, height, channels
    except Exception:  # noqa: BLE001
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_layout(root: Path) -> dict[str, Any]:
    """Support dataset.yaml layout and legacy flat rgb/thermal layout."""
    yaml_path = root / "dataset.yaml"
    if yaml_path.exists():
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML required to read dataset.yaml") from exc
        with yaml_path.open("r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file) or {}
        splits_cfg = cfg.get("splits") or {}
        layout = {"style": "split", "yaml": cfg, "splits": {}}
        for split_name, spec in splits_cfg.items():
            layout["splits"][split_name] = {
                "rgb_dir": root / spec["rgb_dir"],
                "thermal_dir": root / spec["thermal_dir"],
                "annotation": root / spec["annotation"],
            }
        return layout

    # Legacy: root/rgb + root/thermal + annotations/instances_{train,val}.json
    rgb = root / "rgb"
    thermal = root / "thermal"
    ann = root / "annotations"
    if rgb.is_dir() and thermal.is_dir():
        layout = {"style": "legacy_flat", "yaml": None, "splits": {}}
        for split_name in ("train", "val"):
            ann_path = ann / f"instances_{split_name}.json"
            layout["splits"][split_name] = {
                "rgb_dir": rgb,
                "thermal_dir": thermal,
                "annotation": ann_path,
                "stem_prefix": f"{split_name}_",
            }
        return layout

    return {"style": "unknown", "yaml": None, "splits": {}}


def validate_rgbt_dataset(
    root: Path,
    *,
    dataset_key: str,
    write_previews: bool = True,
    preview_dir: Path | None = None,
    preview_count: int = 3,
    max_invalid_samples: int = 50,
) -> dict[str, Any]:
    """Validate paired RGB/thermal COCO-style debug dataset (four+ layers)."""
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    missing_rgb: list[str] = []
    missing_thermal: list[str] = []
    corrupt_images: list[str] = []
    size_mismatches: list[str] = []
    invalid_annotations: list[dict[str, Any]] = []
    split_overlaps: list[str] = []
    hash_overlaps: list[str] = []
    extension_mismatches: list[str] = []
    duplicate_stems: list[str] = []
    class_distribution: Counter[str] = Counter()
    scale_distribution: Counter[str] = Counter()
    empty_annotation_images = 0
    total_objects = 0
    total_images_with_ann = 0

    layout = _resolve_layout(root)
    if layout["style"] == "unknown":
        errors.append(
            "Unrecognized dataset layout. Expect dataset.yaml or legacy rgb/thermal dirs."
        )

    if layout["style"] == "split" and not (root / "dataset.yaml").exists():
        errors.append("Missing dataset.yaml")

    splits_report: dict[str, Any] = {}
    splits_runtime: dict[str, dict[str, Any]] = {}
    stem_to_split: dict[str, str] = {}
    hash_to_refs: dict[str, list[str]] = {}

    for split_name, spec in (layout.get("splits") or {}).items():
        rgb_dir: Path = spec["rgb_dir"]
        thermal_dir: Path = spec["thermal_dir"]
        ann_path: Path = spec["annotation"]
        stem_prefix = spec.get("stem_prefix")

        if not rgb_dir.is_dir():
            errors.append(f"Missing RGB directory for {split_name}: {rgb_dir}")
        if not thermal_dir.is_dir():
            errors.append(f"Missing thermal directory for {split_name}: {thermal_dir}")
        if not ann_path.is_file():
            # For legacy flat layout, annotations may filter by prefix; still require file.
            errors.append(f"Missing annotation file for {split_name}: {ann_path}")
            coco: dict[str, Any] = {"images": [], "annotations": [], "categories": []}
        else:
            try:
                coco = _load_coco(ann_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Invalid annotation JSON for {split_name}: {exc}")
                coco = {"images": [], "annotations": [], "categories": []}

        rgb_map, rgb_dups = (
            _list_images(rgb_dir) if rgb_dir.is_dir() else ({}, [])
        )
        thermal_map, thermal_dups = (
            _list_images(thermal_dir) if thermal_dir.is_dir() else ({}, [])
        )
        if stem_prefix:
            rgb_map = {k: v for k, v in rgb_map.items() if k.startswith(stem_prefix)}
            thermal_map = {
                k: v for k, v in thermal_map.items() if k.startswith(stem_prefix)
            }

        for stem in rgb_dups + thermal_dups:
            duplicate_stems.append(f"{split_name}:{stem}")

        for stem in sorted(set(thermal_map) - set(rgb_map)):
            missing_rgb.append(f"{split_name}:{stem}")
        for stem in sorted(set(rgb_map) - set(thermal_map)):
            missing_thermal.append(f"{split_name}:{stem}")

        paired = sorted(set(rgb_map) & set(thermal_map))
        for stem in paired:
            rgb_path = rgb_map[stem]
            thermal_path = thermal_map[stem]
            if rgb_path.suffix.lower() != thermal_path.suffix.lower():
                extension_mismatches.append(f"{split_name}:{stem}")

            rgb_meta = _image_meta(rgb_path)
            thermal_meta = _image_meta(thermal_path)
            if rgb_meta is None:
                corrupt_images.append(f"{split_name}:rgb:{stem}")
                continue
            if thermal_meta is None:
                corrupt_images.append(f"{split_name}:thermal:{stem}")
                continue
            if rgb_meta[0] != thermal_meta[0] or rgb_meta[1] != thermal_meta[1]:
                size_mismatches.append(
                    f"{split_name}:{stem}:{rgb_meta[0]}x{rgb_meta[1]}!="
                    f"{thermal_meta[0]}x{thermal_meta[1]}"
                )

            for modality, path in (("rgb", rgb_path), ("thermal", thermal_path)):
                digest = _file_sha256(path)
                ref = f"{split_name}:{modality}:{stem}"
                hash_to_refs.setdefault(digest, []).append(ref)

            if stem in stem_to_split and stem_to_split[stem] != split_name:
                split_overlaps.append(stem)
            stem_to_split[stem] = split_name

        # Annotation checks
        categories = {
            int(cat["id"]): str(cat.get("name", cat["id"]))
            for cat in coco.get("categories") or []
        }
        if not categories and (coco.get("annotations") or []):
            errors.append(f"{split_name}: categories missing while annotations exist")

        images = coco.get("images") or []
        annotations = coco.get("annotations") or []
        if not isinstance(images, list) or not isinstance(annotations, list):
            errors.append(f"{split_name}: images/annotations must be lists")
            images, annotations = [], []

        images_by_id: dict[int, dict[str, Any]] = {}
        for img in images:
            try:
                images_by_id[int(img["id"])] = img
            except Exception:  # noqa: BLE001
                invalid_annotations.append(
                    {"split": split_name, "reason": "illegal images entry"}
                )

        anns_by_image: dict[int, int] = Counter()
        for ann in annotations:
            try:
                image_id = int(ann["image_id"])
                category_id = int(ann["category_id"])
            except Exception:  # noqa: BLE001
                item = {"split": split_name, "reason": "illegal annotation ids"}
                if len(invalid_annotations) < max_invalid_samples:
                    invalid_annotations.append(item)
                continue

            if image_id not in images_by_id:
                item = {
                    "split": split_name,
                    "reason": f"annotation references missing image_id={image_id}",
                }
                if len(invalid_annotations) < max_invalid_samples:
                    invalid_annotations.append(item)
                continue

            img = images_by_id[image_id]
            width = float(img.get("width") or 0)
            height = float(img.get("height") or 0)
            stem = Path(str(img.get("file_name", ""))).stem
            if stem and stem not in paired and layout["style"] == "split":
                item = {
                    "split": split_name,
                    "image": stem,
                    "reason": "annotation image file not in paired set",
                }
                if len(invalid_annotations) < max_invalid_samples:
                    invalid_annotations.append(item)

            if category_id not in categories:
                item = {
                    "split": split_name,
                    "image": stem,
                    "reason": f"illegal category_id={category_id}",
                }
                if len(invalid_annotations) < max_invalid_samples:
                    invalid_annotations.append(item)
                continue

            bbox = ann.get("bbox") or []
            if len(bbox) != 4:
                item = {
                    "split": split_name,
                    "image": stem,
                    "reason": "bbox must have 4 values",
                }
                if len(invalid_annotations) < max_invalid_samples:
                    invalid_annotations.append(item)
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                item = {
                    "split": split_name,
                    "image": stem,
                    "reason": "bbox width/height must be > 0",
                }
                if len(invalid_annotations) < max_invalid_samples:
                    invalid_annotations.append(item)
                continue
            if x < 0 or y < 0 or x + w > width + 1e-3 or y + h > height + 1e-3:
                item = {
                    "split": split_name,
                    "image": stem,
                    "reason": f"bbox out of bounds: {[x, y, w, h]} vs {width}x{height}",
                }
                if len(invalid_annotations) < max_invalid_samples:
                    invalid_annotations.append(item)
                continue

            area = ann.get("area", w * h)
            try:
                area_f = float(area)
            except Exception:  # noqa: BLE001
                area_f = -1.0
            if area_f < 0:
                item = {
                    "split": split_name,
                    "image": stem,
                    "reason": "area must be non-negative",
                }
                if len(invalid_annotations) < max_invalid_samples:
                    invalid_annotations.append(item)
                continue

            class_distribution[categories[category_id]] += 1
            if area_f < SMALL_AREA:
                scale_distribution["small"] += 1
            elif area_f < MEDIUM_AREA:
                scale_distribution["medium"] += 1
            else:
                scale_distribution["large"] += 1
            anns_by_image[image_id] += 1
            total_objects += 1

        for image_id in images_by_id:
            total_images_with_ann += 1
            if anns_by_image.get(image_id, 0) == 0:
                empty_annotation_images += 1

        splits_report[split_name] = {
            "rgb_count": len(rgb_map),
            "thermal_count": len(thermal_map),
            "paired_count": len(paired),
            "annotation_count": len(annotations),
        }
        splits_runtime[split_name] = {
            "rgb_map": rgb_map,
            "thermal_map": thermal_map,
            "paired_stems": paired,
            "coco": coco,
        }

    if missing_rgb or missing_thermal:
        errors.append(
            "dataset_pair_mismatch: "
            f"missing_rgb={missing_rgb[:10]} missing_thermal={missing_thermal[:10]}"
        )
    if corrupt_images:
        errors.append(f"Found {len(corrupt_images)} corrupt image(s).")
    if size_mismatches:
        errors.append(f"Found {len(size_mismatches)} RGB/thermal size mismatch(es).")
    if extension_mismatches:
        warnings.append(
            f"Extension differs between modalities for {len(extension_mismatches)} pair(s)."
        )
    if duplicate_stems:
        errors.append(f"Duplicate stems detected: {duplicate_stems[:10]}")
    if split_overlaps:
        errors.append(
            f"Train/val filename overlap detected ({len(set(split_overlaps))} stems)."
        )
    for digest, refs in hash_to_refs.items():
        splits_touched = {ref.split(":", 1)[0] for ref in refs}
        if len(splits_touched) > 1:
            hash_overlaps.append(digest[:12])
    if hash_overlaps:
        errors.append(
            f"Train/val file-hash overlap detected ({len(hash_overlaps)} hashes)."
        )
    if invalid_annotations:
        errors.append(f"Found {len(invalid_annotations)} invalid annotation issue(s).")

    preview_paths: list[str] = []
    if write_previews and preview_dir is not None and splits_runtime:
        preview_paths = generate_rgbt_previews(
            root=root,
            splits_info=splits_runtime,
            preview_dir=preview_dir / "previews"
            if preview_dir.name != "previews"
            else preview_dir,
            count_per_split=preview_count,
        )

    warnings.append("This is a debug subset.")
    warnings.append(
        "The dataset must not support publication-level performance claims."
    )

    paired_total = sum(int(s.get("paired_count", 0)) for s in splits_report.values())
    avg_objects = (
        float(total_objects) / float(total_images_with_ann)
        if total_images_with_ann
        else 0.0
    )

    valid = not errors
    return {
        "schema_version": "1.0",
        "dataset_key": dataset_key,
        "task_type": "rgbt_detection",
        "valid": valid,
        "layout": layout.get("style"),
        "splits": splits_report,
        "rgb_image_count": sum(int(s.get("rgb_count", 0)) for s in splits_report.values()),
        "thermal_image_count": sum(
            int(s.get("thermal_count", 0)) for s in splits_report.values()
        ),
        "paired_image_count": paired_total,
        "missing_rgb": missing_rgb[:max_invalid_samples],
        "missing_thermal": missing_thermal[:max_invalid_samples],
        "corrupt_images": corrupt_images[:max_invalid_samples],
        "size_mismatches": size_mismatches[:max_invalid_samples],
        "invalid_annotations": invalid_annotations,
        "split_overlaps": sorted(set(split_overlaps))[:max_invalid_samples],
        "hash_overlaps": hash_overlaps[:max_invalid_samples],
        "duplicate_stems": duplicate_stems[:max_invalid_samples],
        "extension_mismatches": extension_mismatches[:max_invalid_samples],
        "class_distribution": dict(class_distribution),
        "object_scale_distribution": dict(scale_distribution),
        "empty_annotation_images": empty_annotation_images,
        "mean_objects_per_image": avg_objects,
        "preview_paths": preview_paths,
        "errors": errors,
        "warnings": warnings,
        "claim_level": "pipeline_validation_only",
        "validator_version": VALIDATOR_VERSION,
    }
