"""Attach dual-stream gated fusion onto a built DFINE YAMLConfig (config-driven)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from models.dual_stream_backbone import (
    DualStreamGatedBackbone,
    build_dual_stream_backbone,
    dual_stream_state_summary,
)
from models.fusion_factory import FusionConfig
from rgbt_paired_dataset import wrap_loader_dataset_with_thermal


def apply_gated_multiscale_fusion(
    yaml_cfg: Any,
    fusion_config: FusionConfig,
    *,
    thermal_train_img: Path,
    thermal_val_img: Path,
) -> dict[str, Any]:
    """Replace backbone + wrap train/val datasets for 6ch dual-stream A3.

    Must run after ``yaml_cfg.model`` is constructible and **before** optimizer/EMA
    are first accessed so new fusion parameters enter the optimizer.
    """
    model = yaml_cfg.model
    if not hasattr(model, "backbone"):
        raise TypeError(f"model has no backbone attribute: {type(model)!r}")
    if isinstance(model.backbone, DualStreamGatedBackbone):
        raise RuntimeError("gated multiscale fusion already applied")

    model.backbone = build_dual_stream_backbone(model.backbone, fusion_config)

    train_loader = yaml_cfg.train_dataloader
    val_loader = yaml_cfg.val_dataloader
    wrap_loader_dataset_with_thermal(train_loader, thermal_train_img)
    wrap_loader_dataset_with_thermal(val_loader, thermal_val_img)
    _patch_vendor_visualizer_for_6ch()

    summary = dual_stream_state_summary(model.backbone)
    summary["fusion_config"] = fusion_config.to_dict()
    summary["thermal_train_img"] = str(thermal_train_img)
    summary["thermal_val_img"] = str(thermal_val_img)
    return summary


def _patch_vendor_visualizer_for_6ch() -> None:
    """Vendor save_samples assumes ≤4 channels; show RGB plane for 6ch batches."""
    import torch

    try:
        from src.misc import visualizer as _visualizer  # type: ignore
        from src.solver import det_engine as _det_engine  # type: ignore
    except Exception:  # noqa: BLE001
        return
    if getattr(_visualizer.save_samples, "_rgbt_6ch_patched", False):
        return
    original = _visualizer.save_samples

    def _save_samples_rgb_plane(samples, targets, output_dir, split, normalized, box_fmt):
        if torch.is_tensor(samples) and samples.ndim == 4 and samples.shape[1] > 3:
            samples = samples[:, :3].contiguous()
        return original(samples, targets, output_dir, split, normalized, box_fmt)

    _save_samples_rgb_plane._rgbt_6ch_patched = True  # type: ignore[attr-defined]
    _visualizer.save_samples = _save_samples_rgb_plane
    _det_engine.save_samples = _save_samples_rgb_plane
