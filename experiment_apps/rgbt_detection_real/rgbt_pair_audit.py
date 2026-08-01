"""RGB-T pairing and tensor-shape smoke checks for S01 data-chain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_rgbt_pairs(
    data_root: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, Any]:
    """Verify 1:1 RGB↔Thermal filename pairing and annotation image coverage."""
    data_root = Path(data_root)
    split_reports: dict[str, Any] = {}
    ok = True
    errors: list[str] = []

    for split in splits:
        rgb_dir = data_root / "images" / split / "rgb"
        thr_dir = data_root / "images" / split / "thermal"
        ann_path = data_root / "annotations" / f"instances_{split}.json"
        if not rgb_dir.is_dir():
            ok = False
            errors.append(f"missing rgb dir: {rgb_dir}")
            continue
        if not thr_dir.is_dir():
            ok = False
            errors.append(f"missing thermal dir: {thr_dir}")
            continue
        if not ann_path.is_file():
            ok = False
            errors.append(f"missing annotation: {ann_path}")
            continue

        rgb_names = {p.name for p in rgb_dir.iterdir() if p.is_file()}
        thr_names = {p.name for p in thr_dir.iterdir() if p.is_file()}
        only_rgb = sorted(rgb_names - thr_names)
        only_thr = sorted(thr_names - rgb_names)
        paired = sorted(rgb_names & thr_names)

        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        ann_files = {str(img.get("file_name")) for img in payload.get("images") or []}
        missing_rgb = sorted(ann_files - rgb_names)
        missing_thr = sorted(ann_files - thr_names)
        orphan_rgb = sorted(rgb_names - ann_files)

        split_ok = not (only_rgb or only_thr or missing_rgb or missing_thr)
        if not split_ok:
            ok = False
            if only_rgb:
                errors.append(f"{split}: rgb without thermal: {only_rgb[:5]}")
            if only_thr:
                errors.append(f"{split}: thermal without rgb: {only_thr[:5]}")
            if missing_rgb:
                errors.append(f"{split}: ann images missing rgb: {missing_rgb[:5]}")
            if missing_thr:
                errors.append(f"{split}: ann images missing thermal: {missing_thr[:5]}")

        # Sample shape check on first paired file.
        sample: dict[str, Any] | None = None
        if paired:
            from PIL import Image

            name = paired[0]
            rgb = Image.open(rgb_dir / name)
            thr = Image.open(thr_dir / name)
            sample = {
                "sample_id": name,
                "rgb_size": list(rgb.size),  # W,H
                "thermal_size": list(thr.size),
                "rgb_mode": rgb.mode,
                "thermal_mode": thr.mode,
                "same_spatial_size": list(rgb.size) == list(thr.size),
            }
            if list(rgb.size) != list(thr.size):
                ok = False
                errors.append(
                    f"{split}: rgb/thermal size mismatch for {name}: "
                    f"{rgb.size} vs {thr.size}"
                )

        cats = payload.get("categories") or []
        cat_ids = sorted(int(c["id"]) for c in cats)
        cat_id_set = set(cat_ids)
        annotations = list(payload.get("annotations") or [])
        ann_cat_ids = sorted({int(a.get("category_id", -1)) for a in annotations})
        unknown_cats = sorted(cid for cid in ann_cat_ids if cid not in cat_id_set)

        illegal_bbox = 0
        for ann in annotations:
            bbox = ann.get("bbox") or []
            if len(bbox) != 4:
                illegal_bbox += 1
                continue
            try:
                x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except (TypeError, ValueError):
                illegal_bbox += 1
                continue
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                illegal_bbox += 1

        image_ids = [img.get("id") for img in payload.get("images") or []]
        file_names = [str(img.get("file_name")) for img in payload.get("images") or []]
        dup_image_ids = sorted({i for i in image_ids if image_ids.count(i) > 1})
        dup_file_names = sorted({n for n in file_names if file_names.count(n) > 1})

        quality_ok = (
            split_ok
            and (sample is None or sample.get("same_spatial_size"))
            and not unknown_cats
            and illegal_bbox == 0
            and not dup_image_ids
            and not dup_file_names
        )
        if not quality_ok:
            ok = False
            if unknown_cats:
                errors.append(f"{split}: unknown category_id: {unknown_cats[:10]}")
            if illegal_bbox:
                errors.append(f"{split}: illegal_bbox_count={illegal_bbox}")
            if dup_image_ids:
                errors.append(f"{split}: duplicate image ids: {dup_image_ids[:10]}")
            if dup_file_names:
                errors.append(f"{split}: duplicate file_name: {dup_file_names[:10]}")

        split_reports[split] = {
            "n_rgb": len(rgb_names),
            "n_thermal": len(thr_names),
            "n_paired": len(paired),
            "n_ann_images": len(ann_files),
            "only_rgb": only_rgb,
            "only_thermal": only_thr,
            "missing_rgb_for_ann": missing_rgb,
            "missing_thermal_for_ann": missing_thr,
            "orphan_rgb_not_in_ann": orphan_rgb[:20],
            "category_ids": cat_ids,
            "annotation_category_ids": ann_cat_ids,
            "unknown_category_ids": unknown_cats,
            "illegal_bbox_count": illegal_bbox,
            "duplicate_image_ids": dup_image_ids,
            "duplicate_file_names": dup_file_names,
            "pairing_errors": len(only_rgb) + len(only_thr),
            "missing_images": len(missing_rgb) + len(missing_thr),
            "sample": sample,
            "ok": quality_ok,
        }

    return {
        "ok": ok,
        "errors": errors,
        "splits": split_reports,
    }


def write_rgbt_pair_audit(data_root: Path, out_path: Path) -> dict[str, Any]:
    report = audit_rgbt_pairs(data_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["ok"]:
        raise ValueError("RGB-T pairing audit failed: " + "; ".join(report["errors"][:5]))
    return report
