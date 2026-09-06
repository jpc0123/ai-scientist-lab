"""HOW plugin P3: Thermal-guided spatial attention fusion."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class ThermalSpatialAttentionFusion(FeatureFusion):
    """Applies thermal-derived spatial attention to RGB features before concatenation."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        
        # 1x1 conv + sigmoid for spatial attention per level
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
            for ch in chans
        ])
        
        # 1x1 conv to reduce concatenated features back to original channels
        self.reduce_conv = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, kernel_size=1, stride=1, padding=0)
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
        for i, (rgb, thermal, attn_layer, reduce_layer, ch) in enumerate(
            zip(rgb_features, thermal_features, self.spatial_attention, 
                self.reduce_conv, self.channels)
        ):
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"level {i}: expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            
            # Compute spatial attention mask from thermal: (N, 1, H, W)
            spatial_mask = attn_layer(thermal)
            
            # Apply spatial attention to RGB features
            gated_rgb = rgb * spatial_mask
            
            # Concatenate gated RGB with thermal along channel dimension
            concatenated = torch.cat([gated_rgb, thermal], dim=1)
            
            # Reduce back to original channel count
            reduced = reduce_layer(concatenated)
            
            # Add residual connection if enabled
            if self.residual:
                output = reduced + rgb
            else:
                output = reduced
            
            fused.append(output)
        
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return ThermalSpatialAttentionFusion(channels, residual=residual)
