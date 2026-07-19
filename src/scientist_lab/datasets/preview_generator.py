from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALIDATOR_VERSION = "0.7.2"


def generate_rgbt_previews(
    *,
    root: Path,
    splits_info: dict[str, dict[str, Any]],
    preview_dir: Path,
    count_per_split: int = 3,
) -> list[str]:
    """Write side-by-side RGB/thermal previews with shared boxes."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return []

    preview_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for split_name, info in splits_info.items():
        rgb_map: dict[str, Path] = info.get("rgb_map") or {}
        thermal_map: dict[str, Path] = info.get("thermal_map") or {}
        coco: dict[str, Any] = info.get("coco") or {}
        paired = list(info.get("paired_stems") or [])
        if not paired:
            continue

        categories = {
            int(cat["id"]): str(cat.get("name", cat["id"]))
            for cat in coco.get("categories") or []
        }
        images_by_file = {
            Path(img["file_name"]).stem: img for img in coco.get("images") or []
        }
        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco.get("annotations") or []:
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        selected = paired[: max(1, count_per_split)]
        for stem in selected:
            if stem not in rgb_map or stem not in thermal_map:
                continue
            rgb = Image.open(rgb_map[stem]).convert("RGB")
            thermal = Image.open(thermal_map[stem]).convert("RGB")
            height = max(rgb.height, thermal.height)
            canvas = Image.new(
                "RGB", (rgb.width + thermal.width, height), color=(24, 24, 24)
            )
            canvas.paste(rgb, (0, 0))
            canvas.paste(thermal, (rgb.width, 0))
            draw = ImageDraw.Draw(canvas)

            img_meta = images_by_file.get(stem) or {}
            image_id = img_meta.get("id")
            boxes = anns_by_image.get(int(image_id), []) if image_id is not None else []
            for ann in boxes:
                bbox = ann.get("bbox") or []
                if len(bbox) != 4:
                    continue
                x, y, w, h = [float(v) for v in bbox]
                label = categories.get(int(ann.get("category_id", -1)), "?")
                for offset in (0, rgb.width):
                    box = [x + offset, y, x + w + offset, y + h]
                    draw.rectangle(box, outline=(255, 64, 64), width=2)
                    draw.text((box[0] + 2, max(0, box[1] - 12)), label, fill=(255, 220, 80))

            caption = (
                f"{split_name}/{stem}  "
                f"RGB {rgb.width}x{rgb.height} | thermal {thermal.width}x{thermal.height}"
            )
            draw.rectangle([0, height - 22, canvas.width, height], fill=(0, 0, 0))
            try:
                font = ImageFont.load_default()
            except Exception:  # noqa: BLE001
                font = None
            draw.text((6, height - 18), caption, fill=(230, 230, 230), font=font)

            out = preview_dir / f"{split_name}_{stem}.png"
            canvas.save(out)
            written.append(str(out))

    return written


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
