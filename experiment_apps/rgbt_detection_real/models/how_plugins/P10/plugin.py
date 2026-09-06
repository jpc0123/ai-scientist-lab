"""HOW plugin P10: Thermal-guided spatial attention fusion."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class ThermalSpatialAttentionFusion(FeatureFusion):
    """Generates spatial attention masks from thermal features to gate RGB features."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        
        # Per-level 1x1 conv to generate spatial attention mask from thermal features
        self.spatial_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
            for ch in chans
        ])

    def forward(
        self,
        rgb_features: list[torch.Tensor],
        thermal_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(rgb_features) != len(thermal_features):
            raise ValueError("rgb/thermal feature list lengths must match")
        if len(rgb_features) != len(self.channels):
            raise ValueError(
                f"expected {len(self.channels)} levels, got {len(rgb_features)}"
            )
        
        fused: list[torch.Tensor] = []
        for rgb, thermal, spatial_module, ch in zip(
            rgb_features, thermal_features, self.spatial_conv, self.channels
        ):
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            
            # Generate spatial attention mask (N, 1, H, W) from thermal features
            spatial_mask = spatial_module(thermal)  # Shape: (N, 1, H, W)
            
            # Apply spatial mask to RGB features via element-wise multiplication
            gated_rgb = rgb * spatial_mask  # Broadcasting: (N, C, H, W) * (N, 1, H, W)
            
            # Residual connection: add gated RGB back to original RGB
            if self.residual:
                fused.append(rgb + gated_rgb)
            else:
                fused.append(gated_rgb)
        
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    """Build thermal-guided spatial attention fusion plugin.
    
    Args:
        channels: Sequence of channel dimensions for each feature level (P3, P4, P5)
        residual: If True, adds gated RGB features back to original RGB features
    
    Returns:
        ThermalSpatialAttentionFusion module
    """
    return ThermalSpatialAttentionFusion(channels, residual=residual)
