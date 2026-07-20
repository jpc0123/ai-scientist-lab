"""Create a fixed Fast Eval subset by copying a capped slice of rgbt_debug_v1."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _copy_split(
    src_root: Path,
    dst_root: Path,
    *,
    split: str,
    max_images: int,
) -> int:
    rgb_src = src_root / "images" / split / "rgb"
    thermal_src = src_root / "images" / split / "thermal"
    rgb_dst = dst_root / "images" / split / "rgb"
    thermal_dst = dst_root / "images" / split / "thermal"
    rgb_dst.mkdir(parents=True, exist_ok=True)
    thermal_dst.mkdir(parents=True, exist_ok=True)

    stems = sorted(
        {
            p.stem
            for p in rgb_src.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        }
        & {
            p.stem
            for p in thermal_src.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        }
    )[: max(0, max_images)]

    for stem in stems:
        for folder_src, folder_dst in (
            (rgb_src, rgb_dst),
            (thermal_src, thermal_dst),
        ):
            matches = list(folder_src.glob(f"{stem}.*"))
            if not matches:
                continue
            shutil.copy2(matches[0], folder_dst / matches[0].name)
    return len(stems)


def _filter_annotations(
    src_root: Path,
    dst_root: Path,
    *,
    split: str,
    kept_stems: set[str],
) -> None:
    src = src_root / "annotations" / f"instances_{split}.json"
    dst_dir = dst_root / "annotations"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"instances_{split}.json"
    if not src.exists():
        return
    coco = json.loads(src.read_text(encoding="utf-8"))
    images = []
    keep_ids: set[int] = set()
    for img in coco.get("images") or []:
        stem = Path(str(img.get("file_name", ""))).stem
        if stem in kept_stems:
            images.append(img)
            keep_ids.add(int(img["id"]))
    anns = [
        ann
        for ann in (coco.get("annotations") or [])
        if int(ann.get("image_id", -1)) in keep_ids
    ]
    payload = {
        "info": coco.get("info") or {"description": "rgbt_fast_eval_v1"},
        "licenses": coco.get("licenses") or [],
        "categories": coco.get("categories") or [],
        "images": images,
        "annotations": anns,
    }
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("datasets/rgbt_debug_v1"),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("datasets/rgbt_fast_eval_v1"),
    )
    parser.add_argument("--max-train", type=int, default=24)
    parser.add_argument("--max-val", type=int, default=12)
    args = parser.parse_args()

    if args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True)

    train_n = _copy_split(args.src, args.dst, split="train", max_images=args.max_train)
    val_n = _copy_split(args.src, args.dst, split="val", max_images=args.max_val)

    train_stems = {
        p.stem for p in (args.dst / "images" / "train" / "rgb").glob("*") if p.is_file()
    }
    val_stems = {
        p.stem for p in (args.dst / "images" / "val" / "rgb").glob("*") if p.is_file()
    }
    _filter_annotations(args.src, args.dst, split="train", kept_stems=train_stems)
    _filter_annotations(args.src, args.dst, split="val", kept_stems=val_stems)

    yaml_src = args.src / "dataset.yaml"
    if yaml_src.exists():
        text = yaml_src.read_text(encoding="utf-8")
        text = text.replace("rgbt_debug_v1", "rgbt_fast_eval_v1")
        (args.dst / "dataset.yaml").write_text(text, encoding="utf-8")

    meta = {
        "dataset_key": "rgbt_fast_eval_v1",
        "source": str(args.src),
        "evaluation_scope": "fast_eval_subset",
        "max_train": args.max_train,
        "max_val": args.max_val,
        "train_images": train_n,
        "val_images": val_n,
        "claim_level_default": "exploratory_comparison",
    }
    (args.dst / "fast_eval_budget.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
