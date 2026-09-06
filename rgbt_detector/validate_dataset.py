"""Container-side RGB-T dataset validation (aligned with host report schema)."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VALIDATOR_VERSION = "0.7.3-container"
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def _list_images(directory: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not directory.is_dir():
        return mapping
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            mapping[path.stem] = path
    return mapping


def _load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0:
                return None
            return int(image.width), int(image.height)
    except Exception:  # noqa: BLE001
        return None


def _resolve_splits(root: Path) -> dict[str, dict[str, Path]]:
    yaml_path = root / "dataset.yaml"
    if yaml_path.exists():
        try:
            import yaml
        except ImportError:
            yaml = None
        if yaml is not None:
            with yaml_path.open("r", encoding="utf-8") as file:
                cfg = yaml.safe_load(file) or {}
            splits = {}
            for name, spec in (cfg.get("splits") or {}).items():
                splits[name] = {
                    "rgb_dir": root / spec["rgb_dir"],
                    "thermal_dir": root / spec["thermal_dir"],
                    "annotation": root / spec["annotation"],
                }
            if splits:
                return splits

    # Fallback split layout without YAML
    candidate = {
        "train": {
            "rgb_dir": root / "images" / "train" / "rgb",
            "thermal_dir": root / "images" / "train" / "thermal",
            "annotation": root / "annotations" / "instances_train.json",
        },
        "val": {
            "rgb_dir": root / "images" / "val" / "rgb",
            "thermal_dir": root / "images" / "val" / "thermal",
            "annotation": root / "annotations" / "instances_val.json",
        },
    }
    if candidate["train"]["rgb_dir"].is_dir():
        return candidate

    # Legacy flat
    return {
        "train": {
            "rgb_dir": root / "rgb",
            "thermal_dir": root / "thermal",
            "annotation": root / "annotations" / "instances_train.json",
        },
        "val": {
            "rgb_dir": root / "rgb",
            "thermal_dir": root / "thermal",
            "annotation": root / "annotations" / "instances_val.json",
        },
    }


def _probe_read_only(data_root: Path) -> dict[str, Any]:
    probe = data_root / ".scientist_lab_write_probe"
    try:
        probe.write_text("should-fail", encoding="utf-8")
    except OSError as exc:
        return {
            "dataset_writable": False,
            "write_probe_blocked": True,
            "write_probe_error": type(exc).__name__,
        }
    try:
        probe.unlink(missing_ok=True)
    except OSError:
        pass
    return {
        "dataset_writable": True,
        "write_probe_blocked": False,
        "write_probe_error": None,
    }


def validate_rgbt_in_container(
    data_root: Path,
    *,
    dataset_key: str = "unknown",
    probe_read_only: bool = True,
) -> dict[str, Any]:
    root = data_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    missing_rgb: list[str] = []
    missing_thermal: list[str] = []
    corrupt_images: list[str] = []
    size_mismatches: list[str] = []
    invalid_annotations: list[dict[str, Any]] = []
    split_overlaps: list[str] = []
    class_distribution: Counter[str] = Counter()
    scale_distribution: Counter[str] = Counter()
    splits_report: dict[str, Any] = {}
    stem_owner: dict[str, str] = {}

    splits = _resolve_splits(root)
    for split_name, spec in splits.items():
        rgb_map = _list_images(spec["rgb_dir"])
        thermal_map = _list_images(spec["thermal_dir"])
        if split_name in {"train", "val"} and spec["rgb_dir"].name == "rgb":
            # legacy shared dirs: filter by prefix if present
            if (root / "rgb").is_dir() and not (root / "images").is_dir():
                prefix = f"{split_name}_"
                rgb_map = {k: v for k, v in rgb_map.items() if k.startswith(prefix)}
                thermal_map = {
                    k: v for k, v in thermal_map.items() if k.startswith(prefix)
                }

        for stem in sorted(set(thermal_map) - set(rgb_map)):
            missing_rgb.append(f"{split_name}:{stem}")
        for stem in sorted(set(rgb_map) - set(thermal_map)):
            missing_thermal.append(f"{split_name}:{stem}")

        paired = sorted(set(rgb_map) & set(thermal_map))
        for stem in paired:
            if stem in stem_owner and stem_owner[stem] != split_name:
                split_overlaps.append(stem)
            stem_owner[stem] = split_name
            rgb_size = _image_size(rgb_map[stem])
            th_size = _image_size(thermal_map[stem])
            if rgb_size is None:
                corrupt_images.append(f"{split_name}:rgb:{stem}")
                continue
            if th_size is None:
                corrupt_images.append(f"{split_name}:thermal:{stem}")
                continue
            if rgb_size != th_size:
                size_mismatches.append(f"{split_name}:{stem}")

        ann_path = spec["annotation"]
        annotations: list[dict[str, Any]] = []
        if ann_path.is_file():
            try:
                coco = _load_coco(ann_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{split_name}: invalid annotation JSON: {exc}")
                coco = {"images": [], "annotations": [], "categories": []}
            categories = {
                int(c["id"]): str(c.get("name", c["id"]))
                for c in coco.get("categories") or []
            }
            images_by_id = {
                int(img["id"]): img for img in coco.get("images") or [] if "id" in img
            }
            annotations = list(coco.get("annotations") or [])
            for ann in annotations:
                try:
                    image_id = int(ann["image_id"])
                    category_id = int(ann["category_id"])
                    bbox = ann.get("bbox") or []
                except Exception:  # noqa: BLE001
                    invalid_annotations.append(
                        {"split": split_name, "reason": "illegal annotation ids"}
                    )
                    continue
                if image_id not in images_by_id:
                    invalid_annotations.append(
                        {
                            "split": split_name,
                            "reason": f"missing image_id={image_id}",
                        }
                    )
                    continue
                if category_id not in categories:
                    invalid_annotations.append(
                        {
                            "split": split_name,
                            "reason": f"illegal category_id={category_id}",
                        }
                    )
                    continue
                if len(bbox) != 4:
                    invalid_annotations.append(
                        {"split": split_name, "reason": "bbox length != 4"}
                    )
                    continue
                x, y, w, h = [float(v) for v in bbox]
                img = images_by_id[image_id]
                width = float(img.get("width") or 0)
                height = float(img.get("height") or 0)
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1e-3 or y + h > height + 1e-3:
                    invalid_annotations.append(
                        {"split": split_name, "reason": "bbox invalid/oob"}
                    )
                    continue
                area = float(ann.get("area", w * h))
                class_distribution[categories[category_id]] += 1
                if area < SMALL_AREA:
                    scale_distribution["small"] += 1
                elif area < MEDIUM_AREA:
                    scale_distribution["medium"] += 1
                else:
                    scale_distribution["large"] += 1
        else:
            errors.append(f"Missing annotation: {ann_path}")

        splits_report[split_name] = {
            "rgb_count": len(rgb_map),
            "thermal_count": len(thermal_map),
            "paired_count": len(paired),
            "annotation_count": len(annotations),
        }

    if missing_rgb or missing_thermal:
        errors.append("dataset_pair_mismatch")
    if corrupt_images:
        errors.append(f"corrupt_images={len(corrupt_images)}")
    if size_mismatches:
        errors.append(f"size_mismatches={len(size_mismatches)}")
    if split_overlaps:
        errors.append(f"split_overlaps={len(set(split_overlaps))}")
    if invalid_annotations:
        errors.append(f"invalid_annotations={len(invalid_annotations)}")

    mount_probe = _probe_read_only(root) if probe_read_only else {}
    expect_ro = os.environ.get("EXPECT_DATASET_READ_ONLY", "").strip() in {
        "1",
        "true",
        "yes",
    }
    if mount_probe.get("dataset_writable"):
        if expect_ro:
            errors.append("dataset_mount_is_writable")
        else:
            warnings.append(
                "Dataset path is writable in this process; "
                "Docker mounts must set EXPECT_DATASET_READ_ONLY=1."
            )
    warnings.extend(
        [
            "This is a debug subset.",
            "The dataset must not support publication-level performance claims.",
            "Container validate_data does not train models.",
        ]
    )

    paired_total = sum(int(s["paired_count"]) for s in splits_report.values())
    report = {
        "schema_version": "1.0",
        "dataset_key": dataset_key,
        "task_type": "rgbt_detection",
        "valid": not errors,
        "layout": "split" if (root / "images").is_dir() else "legacy_or_unknown",
        "splits": splits_report,
        "rgb_image_count": sum(int(s["rgb_count"]) for s in splits_report.values()),
        "thermal_image_count": sum(
            int(s["thermal_count"]) for s in splits_report.values()
        ),
        "paired_image_count": paired_total,
        "missing_rgb": missing_rgb[:50],
        "missing_thermal": missing_thermal[:50],
        "corrupt_images": corrupt_images[:50],
        "size_mismatches": size_mismatches[:50],
        "invalid_annotations": invalid_annotations[:50],
        "split_overlaps": sorted(set(split_overlaps))[:50],
        "class_distribution": dict(class_distribution),
        "object_scale_distribution": dict(scale_distribution),
        "errors": errors,
        "warnings": warnings,
        "claim_level": "pipeline_validation_only",
        "validator_version": VALIDATOR_VERSION,
        "execution_mode": "validate_data",
        "trained": False,
        "mount_probe": mount_probe,
        "dataset_root": str(root),
    }
    return report
