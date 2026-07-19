from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "vehicle"},
]


def _make_pair(path_rgb: Path, path_thermal: Path, *, seed: int, label: int) -> list[dict]:
    rng_color = ((40 + seed * 3) % 200, (80 + seed * 5) % 200, (120 + seed * 7) % 200)
    rgb = Image.new("RGB", (640, 512), color=rng_color)
    thermal = Image.new(
        "L", (640, 512), color=int((rng_color[0] + rng_color[1]) // 2)
    )
    draw_r = ImageDraw.Draw(rgb)
    draw_t = ImageDraw.Draw(thermal)
    boxes = []
    x0 = 40 + (seed * 13) % 200
    y0 = 30 + (seed * 17) % 150
    w = 80 + (seed * 3) % 60
    h = 60 + (seed * 5) % 40
    draw_r.rectangle([x0, y0, x0 + w, y0 + h], outline=(255, 0, 0), width=3)
    draw_t.rectangle([x0, y0, x0 + w, y0 + h], outline=255, width=3)
    boxes.append(
        {
            "bbox": [float(x0), float(y0), float(w), float(h)],
            "category_id": 1 if label % 2 == 0 else 2,
            "area": float(w * h),
            "iscrowd": 0,
        }
    )
    path_rgb.parent.mkdir(parents=True, exist_ok=True)
    path_thermal.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(path_rgb)
    thermal.save(path_thermal)
    return boxes


def create_dataset(root: Path, *, n_train: int = 40, n_val: int = 10) -> Path:
    """Create split-layout RGB-T debug dataset (images/{train,val}/{rgb,thermal})."""
    root = root.resolve()
    ann = root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)

    def build_split(split: str, count: int, start_id: int, stem_start: int) -> dict:
        rgb_dir = root / "images" / split / "rgb"
        thermal_dir = root / "images" / split / "thermal"
        images = []
        annotations = []
        ann_id = start_id * 1000
        for index in range(count):
            image_id = start_id + index
            # Globally unique stems across train/val to avoid leakage by filename.
            stem = f"{stem_start + index:06d}"
            boxes = _make_pair(
                rgb_dir / f"{stem}.jpg",
                thermal_dir / f"{stem}.jpg",
                seed=image_id,
                label=index,
            )
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{stem}.jpg",
                    "width": 640,
                    "height": 512,
                }
            )
            for box in boxes:
                ann_id += 1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": box["category_id"],
                        "bbox": box["bbox"],
                        "area": box["area"],
                        "iscrowd": 0,
                    }
                )
        return {
            "images": images,
            "annotations": annotations,
            "categories": CATEGORIES,
        }

    train = build_split("train", n_train, 1, stem_start=1)
    val = build_split("val", n_val, 1000, stem_start=1 + n_train)
    (ann / "instances_train.json").write_text(
        json.dumps(train, indent=2), encoding="utf-8"
    )
    (ann / "instances_val.json").write_text(
        json.dumps(val, indent=2), encoding="utf-8"
    )

    dataset_yaml = """dataset_key: rgbt_debug_v1
task_type: rgbt_detection
annotation_format: coco

modalities:
  - rgb
  - thermal

splits:
  train:
    rgb_dir: images/train/rgb
    thermal_dir: images/train/thermal
    annotation: annotations/instances_train.json

  val:
    rgb_dir: images/val/rgb
    thermal_dir: images/val/thermal
    annotation: annotations/instances_val.json

claim_level: pipeline_validation_only
"""
    (root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic RGB-T debug dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/rgbt_debug_v1"),
        help="Output dataset root",
    )
    parser.add_argument("--train", type=int, default=40)
    parser.add_argument("--val", type=int, default=10)
    args = parser.parse_args()
    root = create_dataset(args.output, n_train=args.train, n_val=args.val)
    print(f"Created RGB-T debug dataset at {root}")


if __name__ == "__main__":
    main()
