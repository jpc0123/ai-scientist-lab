"""Attach a HOW backbone_wrap plugin onto a built DFINE YAMLConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apply_gated_fusion import _patch_vendor_visualizer_for_6ch
from models.dual_stream_backbone import DualStreamGatedBackbone
from rgbt_paired_dataset import wrap_loader_dataset_with_thermal


def apply_plugin_backbone_wrap(
    yaml_cfg: Any,
    wrap: Any,
    *,
    how_id: str,
    input_mode: str = "rgb",
    thermal_train_img: Path | None = None,
    thermal_val_img: Path | None = None,
) -> dict[str, Any]:
    """Replace ``model.backbone``. Must run before optimizer/EMA are first accessed."""
    model = yaml_cfg.model
    if not hasattr(model, "backbone"):
        raise TypeError(f"model has no backbone attribute: {type(model)!r}")
    if isinstance(model.backbone, DualStreamGatedBackbone) and not isinstance(
        wrap, DualStreamGatedBackbone
    ):
        raise RuntimeError("backbone already dual-stream wrapped; one slot per plugin")
    replaced = type(model.backbone).__name__
    model.backbone = wrap
    mode = str(input_mode or "rgb").strip().lower()
    if mode == "rgbt":
        if thermal_train_img is None or thermal_val_img is None:
            raise RuntimeError("backbone_wrap input_mode=rgbt needs thermal_* folders")
        wrap_loader_dataset_with_thermal(yaml_cfg.train_dataloader, thermal_train_img)
        wrap_loader_dataset_with_thermal(yaml_cfg.val_dataloader, thermal_val_img)
        _patch_vendor_visualizer_for_6ch()
    return {
        "plugin_how_id": str(how_id),
        "plugin_kind": "backbone_wrap",
        "input_mode": mode,
        "replaced_backbone_class": replaced,
        "backbone_class": type(wrap).__name__,
        "backbone_params": int(sum(p.numel() for p in wrap.parameters())),
    }
