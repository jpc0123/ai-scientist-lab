"""HOW plugin P1 invent. Thermal-guided spatial attention then concat fuse."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class _SpatialAttention(nn.Module):
    """Spatial attention mask from thermal features using GAP + 1x1 conv + sigmoid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, thermal: torch.Tensor) -> torch.Tensor:
        pooled = self.gap(thermal)
        attended = self.conv(pooled)
        return self.sigmoid(attended)


class P1SpatialAttentionFusion(FeatureFusion):
    """Thermal-guided attention on RGB, then concatenate and 1x1 fuse."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        self.attention_modules = nn.ModuleList([_SpatialAttention(c) for c in chans])
        self.fusion_convs = nn.ModuleList(
            [nn.Conv2d(c * 2, c, kernel_size=1, bias=True) for c in chans]
        )

    def forward(
        self, rgb_feats: Sequence[torch.Tensor], thermal_feats: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        if len(rgb_feats) != len(thermal_feats):
            raise ValueError("rgb and thermal feature lists must have same length")
        if len(rgb_feats) != len(self.channels):
            raise ValueError("number of feature levels must match channels")
        fused: list[torch.Tensor] = []
        for attn, conv, r, t in zip(
            self.attention_modules, self.fusion_convs, rgb_feats, thermal_feats
        ):
            mask = attn(t)
            attended_rgb = r * mask
            cat_feats = torch.cat([attended_rgb, t], dim=1)
            out = conv(cat_feats)
            if self.residual:
                out = out + r
            fused.append(out)
        return fused


def build_plugin(cfg) -> P1SpatialAttentionFusion:
    """Factory used by the HOW runner."""
    channels = cfg.get("channels", [64, 128, 256, 512])
    residual = cfg.get("residual", True)
    return P1SpatialAttentionFusion(channels=channels, residual=residual)


def build_fusion(channels=(256, 512, 1024), residual=True, **kwargs):
    """Lab contract export; wraps LLM alias build_plugin."""
    try:
        return build_plugin(channels=channels, residual=residual, **kwargs)
    except TypeError:
        return build_plugin({"channels": list(channels), "residual": residual})
