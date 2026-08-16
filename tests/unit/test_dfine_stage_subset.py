"""fast_eval/smoke staging must honor max_train_images / max_val_images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REAL_APP = ROOT / "experiment_apps" / "rgbt_detection_real"

# 1x1 PNG so copy2 has a real file without depending on a dataset checkout.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


def _write_split(root: Path, split: str, n_images: int) -> None:
    img_dir = root / "images" / split / "rgb"
    img_dir.mkdir(parents=True, exist_ok=True)
    images = []
    anns = []
    for i in range(n_images):
        name = f"img_{i:03d}.png"
        (img_dir / name).write_bytes(_PNG)
        images.append(
            {"id": i + 1, "file_name": name, "width": 1, "height": 1}
        )
        anns.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [0, 0, 1, 1],
                "area": 1,
                "iscrowd": 0,
            }
        )
    payload = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": 1, "name": "person"}],
    }
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"instances_{split}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_stage_coco_honors_max_images(tmp_path: Path):
    sys.path.insert(0, str(REAL_APP))
    from dfine_dataset_stage import stage_coco_for_dfine

    data_root = tmp_path / "data"
    _write_split(data_root, "train", 6)
    _write_split(data_root, "val", 4)

    stage = stage_coco_for_dfine(
        data_root,
        tmp_path / "stage",
        input_mode="rgb",
        max_train_images=2,
        max_val_images=1,
    )
    train_ann = json.loads(Path(stage["train_ann"]).read_text(encoding="utf-8"))
    val_ann = json.loads(Path(stage["val_ann"]).read_text(encoding="utf-8"))
    assert len(train_ann["images"]) == 2
    assert len(val_ann["images"]) == 1
    train_ids = {int(im["id"]) for im in train_ann["images"]}
    val_ids = {int(im["id"]) for im in val_ann["images"]}
    assert len(list(Path(stage["train_img"]).glob("*.png"))) == 2
    assert len(list(Path(stage["val_img"]).glob("*.png"))) == 1
    assert {int(a["image_id"]) for a in train_ann["annotations"]} <= train_ids
    assert {int(a["image_id"]) for a in val_ann["annotations"]} <= val_ids
    assert len(train_ann["annotations"]) == 2
    assert len(val_ann["annotations"]) == 1
    assert stage["max_train_images"] == 2
    assert stage["max_val_images"] == 1


def test_stage_coco_none_keeps_full_split(tmp_path: Path):
    sys.path.insert(0, str(REAL_APP))
    from dfine_dataset_stage import stage_coco_for_dfine

    data_root = tmp_path / "data"
    _write_split(data_root, "train", 5)
    _write_split(data_root, "val", 3)
    stage = stage_coco_for_dfine(
        data_root,
        tmp_path / "stage",
        input_mode="rgb",
        max_train_images=None,
        max_val_images=None,
    )
    train_ann = json.loads(Path(stage["train_ann"]).read_text(encoding="utf-8"))
    val_ann = json.loads(Path(stage["val_ann"]).read_text(encoding="utf-8"))
    assert len(train_ann["images"]) == 5
    assert len(val_ann["images"]) == 3
    assert len(train_ann["annotations"]) == 5
    assert len(val_ann["annotations"]) == 3


def test_probe_caps_formal_keeps_full():
    sys.path.insert(0, str(REAL_APP))
    from train_dfine import resolve_stage_image_caps

    assert resolve_stage_image_caps("fast_eval", {}) == (16, 8)
    assert resolve_stage_image_caps(
        "fast_eval", {"max_train_images": 16, "max_val_images": 8}
    ) == (16, 8)
    assert resolve_stage_image_caps("smoke_train", {"max_train_images": 4}) == (4, 8)
    assert resolve_stage_image_caps(
        "full_train", {"max_train_images": 16, "max_val_images": 8}
    ) == (None, None)
    assert resolve_stage_image_caps("validate_data", {}) == (None, None)
    entry = (REAL_APP / "run_detection_experiment.py").read_text(encoding="utf-8")
    assert 'mode in {"smoke_train", "fast_eval", "full_train"}' in entry


def test_sync_copies_checkpoint_dir_skips_freeze_contract(tmp_path: Path):
    from scientist_lab.adapters.dfine.cuda_runner import _sync_output_dir

    src = tmp_path / "exec"
    dest = tmp_path / "run"
    src.mkdir()
    dest.mkdir()
    (src / "metrics.json").write_text('{"ok": true}', encoding="utf-8")
    (src / "contract.json").write_text('{"stolen": true}', encoding="utf-8")
    ckpt = src / "checkpoint"
    ckpt.mkdir()
    (ckpt / "last.pt").write_bytes(b"ckpt")
    (dest / "contract.json").write_text('{"freeze": true}', encoding="utf-8")
    _sync_output_dir({"run": {"output_directory": str(src)}}, dest)
    assert (dest / "metrics.json").is_file()
    assert (dest / "checkpoint" / "last.pt").read_bytes() == b"ckpt"
    assert json.loads((dest / "contract.json").read_text(encoding="utf-8")) == {
        "freeze": True
    }


def test_gpu_device_requests_attached_when_requested():
    from scientist_lab.runners.local_docker import build_gpu_device_requests

    assert build_gpu_device_requests(0) is None
    reqs = build_gpu_device_requests(1)
    assert reqs is not None
    assert len(reqs) == 1
    assert int(reqs[0].count) == 1
    assert reqs[0].capabilities == [["gpu"]]
