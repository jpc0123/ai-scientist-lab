"""Build a short-budget D-FINE YAML config for Scientist Lab Fast Eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_dfine_fast_config(
    *,
    dfine_root: Path,
    config_path: Path,
    stage_paths: dict[str, Path],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    image_size: int,
    learning_rate: float,
    num_classes: int,
    seed: int,
) -> Path:
    dfine_root = Path(dfine_root).resolve()
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # Paths must be POSIX for Linux containers / YAMLConfig.
    train_img = Path(stage_paths["train_img"]).as_posix()
    val_img = Path(stage_paths["val_img"]).as_posix()
    train_ann = Path(stage_paths["train_ann"]).as_posix()
    val_ann = Path(stage_paths["val_ann"]).as_posix()
    out = Path(output_dir).as_posix()

    # Relative includes resolve from this file's directory; use absolute includes
    # via copied relative links under a config workspace next to DFINE configs.
    text = f"""
# Auto-generated Scientist Lab Fast Eval config for DFINE-S
__include__:
  - { (dfine_root / 'configs' / 'runtime.yml').as_posix() }
  - { (dfine_root / 'configs' / 'dfine' / 'include' / 'dataloader.yml').as_posix() }
  - { (dfine_root / 'configs' / 'dfine' / 'include' / 'optimizer.yml').as_posix() }
  - { (dfine_root / 'configs' / 'dfine' / 'include' / 'dfine_hgnetv2.yml').as_posix() }

task: detection
evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']

num_classes: {int(num_classes)}
remap_mscoco_category: False

output_dir: {out}
print_freq: 1
checkpoint_freq: {max(1, int(epochs))}
epoches: {int(epochs)}
epochs: {int(epochs)}
seed: {int(seed)}
use_amp: False
sync_bn: False
find_unused_parameters: True

DFINE:
  backbone: HGNetv2

HGNetv2:
  name: 'B0'
  return_idx: [1, 2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True
  pretrained: False

DFINETransformer:
  num_layers: 3
  eval_idx: -1

HybridEncoder:
  in_channels: [256, 512, 1024]
  hidden_dim: 256
  depth_mult: 0.34
  expansion: 0.5

optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: {float(learning_rate) * 0.5}
    -
      params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: {float(learning_rate) * 0.5}
      weight_decay: 0.
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.
  lr: {float(learning_rate)}
  betas: [0.9, 0.999]
  weight_decay: 0.0001

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {train_img}
    ann_file: {train_ann}
    return_masks: False
    transforms:
      type: Compose
      ops:
        - {{type: Resize, size: [{int(image_size)}, {int(image_size)}]}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy:
        name: stop_epoch
        epoch: 9999
        ops: []
  shuffle: True
  num_workers: {int(num_workers)}
  drop_last: False
  total_batch_size: {int(batch_size)}
  collate_fn:
    type: BatchImageCollateFunction
    base_size: {int(image_size)}
    base_size_repeat: 1
    stop_epoch: 9999

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {val_img}
    ann_file: {val_ann}
    return_masks: False
    transforms:
      type: Compose
      ops:
        - {{type: Resize, size: [{int(image_size)}, {int(image_size)}]}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
  shuffle: False
  num_workers: {int(num_workers)}
  drop_last: False
  total_batch_size: {int(batch_size)}
  collate_fn:
    type: BatchImageCollateFunction
"""
    config_path.write_text(text.strip() + "\n", encoding="utf-8")
    return config_path


def dfine_root_from_app(app_dir: Path) -> Path:
    app_dir = Path(app_dir).resolve()
    candidates = [
        app_dir / "third_party" / "DFINE",
        app_dir.parents[1] / "third_party" / "DFINE",  # repo: experiment_apps/../third_party
        app_dir.parents[2] / "third_party" / "DFINE",  # safety for nested layouts
    ]
    for candidate in candidates:
        if (candidate / "train.py").is_file():
            return candidate.resolve()
    return candidates[0].resolve()
