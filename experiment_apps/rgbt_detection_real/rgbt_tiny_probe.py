"""Gate F1: small read-only + optional CUDA one-batch probe on registered RGBT-Tiny."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load_manifest(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[: max(0, int(limit))]


def probe_readonly(
    registered_root: Path,
    *,
    train_pairs: int = 40,
    val_pairs: int = 15,
    report_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(registered_root)
    report: dict[str, Any] = {
        "gate": "F1",
        "registered_root": str(root),
        "checks": {},
        "errors": [],
    }
    freeze = root / "DATASET_FREEZE.json"
    if not freeze.is_file():
        report["errors"].append("missing DATASET_FREEZE.json — run F0 first")
        report["status"] = "blocked"
        return report

    train_rows = _load_manifest(root / "manifests" / "train_pairs.csv", train_pairs)
    val_rows = _load_manifest(root / "manifests" / "val_pairs.csv", val_pairs)
    report["checks"]["train_rows_loaded"] = len(train_rows)
    report["checks"]["val_rows_loaded"] = len(val_rows)

    from PIL import Image

    decode_ok = 0
    align_mismatch = 0
    missing = 0
    for row in train_rows + val_rows:
        rgb_p = Path(row["rgb_path"])
        thr_p = Path(row["thermal_path"])
        # Prefer linked registered files when present
        split = row.get("split") or "train"
        linked_rgb = root / "images" / split / "rgb" / row["file_name"]
        linked_thr = root / "images" / split / "thermal" / row["file_name"]
        if linked_rgb.is_file():
            rgb_p = linked_rgb
        if linked_thr.is_file():
            thr_p = linked_thr
        if not rgb_p.is_file() or not thr_p.is_file():
            missing += 1
            continue
        try:
            with Image.open(rgb_p) as im_r, Image.open(thr_p) as im_t:
                im_r.load()
                im_t.load()
                if im_r.size != im_t.size:
                    align_mismatch += 1
                decode_ok += 1
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"decode failed {row.get('sample_id')}: {exc}")

    report["checks"]["decode_ok"] = decode_ok
    report["checks"]["align_mismatch"] = align_mismatch
    report["checks"]["missing_files"] = missing

    # bbox/category from COCO
    illegal = 0
    unknown = 0
    boxes = 0
    cat_ids = set()
    for split, rows in (("train", train_rows), ("val", val_rows)):
        ann_path = root / "annotations" / f"instances_{split}.json"
        if not ann_path.is_file():
            continue
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        allowed = {int(c["id"]) for c in payload.get("categories") or []}
        file_set = {r["file_name"] for r in rows}
        img_by_name = {
            str(im["file_name"]): im for im in payload.get("images") or [] if im.get("file_name") in file_set
        }
        img_ids = {int(im["id"]) for im in img_by_name.values()}
        for ann in payload.get("annotations") or []:
            if int(ann["image_id"]) not in img_ids:
                continue
            boxes += 1
            cid = int(ann["category_id"])
            cat_ids.add(cid)
            if cid not in allowed:
                unknown += 1
            x, y, w, h = [float(v) for v in ann["bbox"]]
            im = next(im for im in img_by_name.values() if int(im["id"]) == int(ann["image_id"]))
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                illegal += 1
            if x + w > float(im["width"]) + 1.0 or y + h > float(im["height"]) + 1.0:
                illegal += 1

    report["checks"]["boxes_in_probe"] = boxes
    report["checks"]["illegal_or_oob_bbox"] = illegal
    report["checks"]["unknown_category"] = unknown
    report["checks"]["category_ids_seen"] = sorted(cat_ids)

    # dataloader one-batch (numpy path used by lab loader)
    try:
        import sys

        app = Path(__file__).resolve().parent
        if str(app) not in sys.path:
            sys.path.insert(0, str(app))
        from dataset_loader import load_split_samples

        samples = load_split_samples(
            root,
            split="train",
            input_mode="rgbt",
            fusion_method="early_concat",
            image_width=640,
            image_height=640,
            max_images=min(4, max(1, len(train_rows))),
        )
        report["checks"]["dataloader_samples"] = len(samples)
        if samples:
            report["checks"]["dataloader_shape"] = list(samples[0].image.shape)
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(f"dataloader probe failed: {exc}")
        report["checks"]["dataloader_samples"] = 0

    cuda_probe: dict[str, Any] = {"attempted": False}
    try:
        import torch

        cuda_probe["attempted"] = True
        cuda_probe["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available() and report["checks"].get("dataloader_samples", 0) > 0:
            from dataset_loader import load_split_samples

            samples = load_split_samples(
                root,
                split="train",
                input_mode="rgbt",
                fusion_method="early_concat",
                image_width=640,
                image_height=640,
                max_images=1,
            )
            tensor = torch.from_numpy(samples[0].image).permute(2, 0, 1).unsqueeze(0).float()
            tensor = tensor.to("cuda")
            # trivial forward: channel reduce (not the detector) — proves host→device path
            out = tensor.mean(dim=1, keepdim=True)
            cuda_probe["forward_ok"] = True
            cuda_probe["tensor_shape"] = list(tensor.shape)
            cuda_probe["out_finite"] = bool(torch.isfinite(out).all().item())
            del tensor, out
            torch.cuda.empty_cache()
        else:
            cuda_probe["forward_ok"] = False
            cuda_probe["reason"] = "cuda unavailable or no samples"
    except Exception as exc:  # noqa: BLE001
        cuda_probe["forward_ok"] = False
        cuda_probe["error"] = str(exc)
    report["cuda_probe"] = cuda_probe

    ok = (
        missing == 0
        and align_mismatch == 0
        and illegal == 0
        and unknown == 0
        and report["checks"].get("dataloader_samples", 0) > 0
        and not report["errors"]
    )
    report["status"] = "passed" if ok else "failed"
    report["claim_authority"] = "probe_only"
    report["forbidden"] = [
        "formal metrics",
        "A4/P01",
        "full-corpus training",
    ]
    target = Path(report_path) if report_path is not None else (root / "GATE_F1_PROBE.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
