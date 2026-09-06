"""HOW plugin P8: Thermal-guided spatial attention fusion."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class ThermalSpatialAttentionFusion(FeatureFusion):
    """Generates spatial saliency mask from thermal features to modulate RGB."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        
        # Per-level 1x1 conv + sigmoid for spatial attention from thermal
        self.spatial_attention = nn.ModuleList([
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
        for rgb, thermal, attn_layer, ch in zip(
            rgb_features, thermal_features, self.spatial_attention, self.channels
        ):
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            
            # Generate spatial attention mask (N, 1, H, W) from thermal
            spatial_mask = attn_layer(thermal)  # Shape: (N, 1, H, W)
            
            # Apply spatial mask to RGB features (broadcasts across C dimension)
            rgb_modulated = rgb * spatial_mask
            
            # Concatenate modulated RGB with thermal along channel dimension
            concatenated = torch.cat([rgb_modulated, thermal], dim=1)  # (N, 2*C, H, W)
            
            # Reduce back to original channel count via 1x1 conv
            # (This is implicit in the fusion - we'll use a simple average for now)
            # Actually, let's just add them with equal weight after modulation
            mixed = rgb_modulated + thermal
            
            if self.residual:
                fused.append(mixed + rgb)
            else:
                fused.append(mixed)
        
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return ThermalSpatialAttentionFusion(channels, residual=residual)
