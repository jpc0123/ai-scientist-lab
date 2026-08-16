"""Build a short-budget D-FINE YAML config for Scientist Lab Fast Eval."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CONFIGURED_NUM_QUERIES = 300
DEFAULT_CONFIGURED_NUM_DENOISING = 100
COARSE_STRIDE = 32


def resolve_input_size(
    *,
    image_size: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[int, int]:
    """Resolve (H, W) from experiment params. Prefer explicit H/W over square size."""
    if image_height is not None or image_width is not None:
        if image_height is None or image_width is None:
            raise ValueError("image_height and image_width must be set together")
        h, w = int(image_height), int(image_width)
    else:
        side = int(image_size if image_size is not None else 160)
        h = w = side
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid input size: {(h, w)}")
    return h, w


def resolve_query_budget(
    *,
    input_h: int,
    input_w: int,
    configured_num_queries: int = DEFAULT_CONFIGURED_NUM_QUERIES,
    configured_num_denoising: int = DEFAULT_CONFIGURED_NUM_DENOISING,
    stride: int = COARSE_STRIDE,
    scale_queries_to_tokens: bool = True,
) -> dict[str, Any]:
    """Bound Fast Eval num_queries by coarse-map token count.

    Formal / full-resolution runs should set scale_queries_to_tokens=False so the
    model keeps its configured query count (structure must not silently change).
    """
    feature_h = max(1, math.ceil(int(input_h) / int(stride)))
    feature_w = max(1, math.ceil(int(input_w) / int(stride)))
    available_tokens = int(feature_h * feature_w)
    configured = max(1, int(configured_num_queries))
    if scale_queries_to_tokens:
        effective = min(configured, available_tokens)
    else:
        if configured > available_tokens:
            raise ValueError(
                "configured_num_queries exceeds coarse feature tokens; "
                f"queries={configured} tokens={available_tokens} "
                f"input=[{input_h}, {input_w}] stride={stride}. "
                "Increase input size or enable Fast Eval query scaling."
            )
        effective = configured
    denoising_cap = max(1, effective // 2)
    effective_denoising = min(max(1, int(configured_num_denoising)), denoising_cap)
    return {
        "input_size": [int(input_h), int(input_w)],
        "stride": int(stride),
        "feature_h": feature_h,
        "feature_w": feature_w,
        "available_tokens": available_tokens,
        "configured_num_queries": configured,
        "effective_num_queries": int(effective),
        "configured_num_denoising": int(configured_num_denoising),
        "effective_num_denoising": int(effective_denoising),
        "scale_queries_to_tokens": bool(scale_queries_to_tokens),
    }


def write_dfine_fast_config(
    *,
    dfine_root: Path,
    config_path: Path,
    stage_paths: dict[str, Path],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    image_size: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
    learning_rate: float,
    num_classes: int,
    seed: int,
    configured_num_queries: int = DEFAULT_CONFIGURED_NUM_QUERIES,
    configured_num_denoising: int = DEFAULT_CONFIGURED_NUM_DENOISING,
    scale_queries_to_tokens: bool = True,
    budget_record_path: Path | None = None,
    pretrained: bool = False,
    local_model_dir: str | None = None,
    disable_multiscale_collate: bool = False,
    warmup_duration: int | None = None,
    lr_scheduler_milestones: list[int] | None = None,
    lr_scheduler_gamma: float | None = None,
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

    input_h, input_w = resolve_input_size(
        image_size=image_size,
        image_height=image_height,
        image_width=image_width,
    )
    eval_spatial_size = [input_h, input_w]
    assert tuple(eval_spatial_size) == (input_h, input_w)

    query_budget = resolve_query_budget(
        input_h=input_h,
        input_w=input_w,
        configured_num_queries=configured_num_queries,
        configured_num_denoising=configured_num_denoising,
        scale_queries_to_tokens=scale_queries_to_tokens,
    )
    num_queries = int(query_budget["effective_num_queries"])
    num_denoising = int(query_budget["effective_num_denoising"])

    # Prefer vendored local weights (Docker jobs run with network_disabled).
    # Use a cwd-relative path so the same YAML works on host and in /workspace.
    weight_dir = local_model_dir
    if not weight_dir:
        weight_dir = "third_party/DFINE/weight/hgnetv2/"
    weight_dir = str(weight_dir).replace("\\", "/")
    if not weight_dir.endswith("/"):
        weight_dir = weight_dir + "/"
    # Sanity: refuse Windows drive paths that would break Linux containers.
    if len(weight_dir) >= 2 and weight_dir[1] == ":":
        raise ValueError(
            f"local_model_dir must be container-portable, got {weight_dir!r}"
        )

    # Default matches DFINE include/optimizer.yml (LinearWarmup warmup_duration: 500).
    resolved_warmup = (
        int(warmup_duration) if warmup_duration is not None else 500
    )
    if resolved_warmup < 0:
        raise ValueError(f"warmup_duration must be >= 0, got {resolved_warmup}")

    # Default MultiStep milestones come from include/optimizer.yml ([500]).
    # When overridden, emit an explicit lr_scheduler block (YAML merge replaces).
    resolved_milestones: list[int] | None = None
    if lr_scheduler_milestones is not None:
        resolved_milestones = [int(x) for x in lr_scheduler_milestones]
        if not resolved_milestones:
            raise ValueError("lr_scheduler_milestones must be non-empty when set")
        if any(m < 0 for m in resolved_milestones):
            raise ValueError(f"lr_scheduler_milestones must be >= 0, got {resolved_milestones}")
    resolved_gamma = (
        float(lr_scheduler_gamma) if lr_scheduler_gamma is not None else 0.1
    )
    if resolved_gamma <= 0:
        raise ValueError(f"lr_scheduler_gamma must be > 0, got {resolved_gamma}")

    record = {
        "eval_spatial_size": eval_spatial_size,
        "input_size": [input_h, input_w],
        "resize_size": [input_h, input_w],
        "collate_base_size_h": input_h,
        "collate_base_size_w": input_w,
        "query_budget": query_budget,
        "pretrained": bool(pretrained),
        "local_model_dir": weight_dir,
        "warmup_duration": resolved_warmup,
        "a2_reference_warmup_duration": 500,
        "lr_scheduler_type": "MultiStepLR",
        "lr_scheduler_milestones": resolved_milestones
        if resolved_milestones is not None
        else [500],
        "lr_scheduler_gamma": resolved_gamma,
        "a2_reference_lr_scheduler_milestones": [500],
    }
    record_path = Path(
        budget_record_path
        if budget_record_path is not None
        else Path(config_path).with_name("dfine_spatial_query_budget.json")
    )
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Relative includes resolve from this file's directory; use absolute includes
    # via copied relative links under a config workspace next to DFINE configs.
    # D-FINE selects encoder top-k over the coarsest map (stride 32). At small
    # Fast Eval sizes, default num_queries=300 exceeds available tokens unless
    # scale_queries_to_tokens=True (Fast Eval only).
    scheduler_override = ""
    if resolved_milestones is not None:
        ms = ", ".join(str(m) for m in resolved_milestones)
        scheduler_override = f"""
# Override include/optimizer.yml MultiStepLR milestones (default [500]).
lr_scheduler:
  type: MultiStepLR
  milestones: [{ms}]
  gamma: {format(float(resolved_gamma), ".8f")}
"""
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
# Must match Resize/collate size derived from experiment input_size.
# Included dfine_hgnetv2.yml defaults to 640x640 and would desync pos_embed.
eval_spatial_size: [{input_h}, {input_w}]

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
  pretrained: {str(bool(pretrained))}
  local_model_dir: {weight_dir}

DFINETransformer:
  num_layers: 3
  eval_idx: -1
  num_queries: {int(num_queries)}
  num_denoising: {int(num_denoising)}

DFINEPostProcessor:
  num_top_queries: {int(num_queries)}

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
      lr: {format(float(learning_rate) * 0.5, ".8f")}
    -
      params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: {format(float(learning_rate) * 0.5, ".8f")}
      weight_decay: 0.
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.
  lr: {format(float(learning_rate), ".8f")}
  betas: [0.9, 0.999]
  weight_decay: 0.0001

# Override include/optimizer.yml LinearWarmup (default 500). Integer only — avoid YAML float coercion.
lr_warmup_scheduler:
  type: LinearWarmup
  warmup_duration: {int(resolved_warmup)}
{scheduler_override}
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
        - {{type: Resize, size: [{input_h}, {input_w}]}}
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
    base_size: {input_h if input_h == input_w else input_w}
    base_size_repeat: {("null" if disable_multiscale_collate else 1)}
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
        - {{type: Resize, size: [{input_h}, {input_w}]}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
  shuffle: False
  num_workers: {int(num_workers)}
  drop_last: False
  total_batch_size: {int(batch_size)}
  collate_fn:
    type: BatchImageCollateFunction
"""
    # BatchImageCollateFunction historically takes a single base_size; for non-square
    # inputs Resize ops carry the true HxW and eval_spatial_size matches them.
    config_path.write_text(text.strip() + "\n", encoding="utf-8")
    return config_path


def dfine_root_from_app(app_dir: Path) -> Path:
    app_dir = Path(app_dir).resolve()
    candidates: list[Path] = [app_dir / "third_party" / "DFINE"]
    # parents[i] can IndexError when the workspace is shallow (e.g. /workspace).
    for depth in (1, 2):
        try:
            candidates.append(app_dir.parents[depth] / "third_party" / "DFINE")
        except IndexError:
            break
    for candidate in candidates:
        if (candidate / "train.py").is_file():
            return candidate.resolve()
    return candidates[0].resolve()
