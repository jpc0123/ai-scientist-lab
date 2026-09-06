"""Example HOW backbone_wrap plugin. Not registered. Not a Claim.

Wraps the vendor RGB backbone at model.backbone. Does not edit third_party/DFINE.
A wrap that changes the detection frontend needs a new R0; do not reuse the
old fusion-baseline run as the KEEP anchor.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

PLUGIN_KIND = "backbone_wrap"
PLUGIN_INPUT_MODE = "rgb"


def _infer_channels(backbone: nn.Module) -> list[int]:
    if hasattr(backbone, "out_channels"):
        value = getattr(backbone, "out_channels")
        if callable(value):
            value = value()
        return [int(c) for c in value]
    if hasattr(backbone, "_out_channels"):
        raw = list(getattr(backbone, "_out_channels"))
        return_idx = list(getattr(backbone, "return_idx", range(len(raw))))
        return [int(raw[i]) for i in return_idx]
    return [256, 512, 1024]


class ResidualRefineWrap(nn.Module):
    """1x1 residual refine on each backbone pyramid level."""

    def __init__(self, rgb_backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = rgb_backbone
        self.out_channels = _infer_channels(rgb_backbone)
        if not self.out_channels:
            raise ValueError("backbone_wrap could not infer out_channels")
        self.refine = nn.ModuleList(
            nn.Conv2d(ch, ch, kernel_size=1) for ch in self.out_channels
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        rgb = x[:, :3] if x.ndim == 4 and int(x.shape[1]) >= 3 else x
        feats = self.backbone(rgb)
        if not isinstance(feats, (list, tuple)):
            raise TypeError("wrapped backbone must return a list/tuple of feature maps")
        maps = list(feats)
        if len(maps) != len(self.refine):
            raise ValueError(
                f"backbone_wrap expected {len(self.refine)} levels, got {len(maps)}"
            )
        return [conv(feat) + feat for conv, feat in zip(self.refine, maps)]


def build_backbone_wrap(
    rgb_backbone: nn.Module,
    *,
    fusion: Any | None = None,
    share_backbone: bool = False,
    **_: object,
) -> nn.Module:
    del fusion, share_backbone
    return ResidualRefineWrap(rgb_backbone)
